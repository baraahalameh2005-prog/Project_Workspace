"""Turn two photos of a garment (front + back) into a 3D GLB fitted to an avatar.

Pipeline:
  1. Cut the garment out of each photo (rembg, model u2net).
  2. Read the garment's outline row by row: collar, shoulders, body edges,
     sleeves and hem.
  3. Measure the avatar with cross-sections: torso, shoulders, armpits, arms.
  4. Build the garment surface around the avatar: the body plus a little ease,
     widened wherever the photo shows the garment flaring (skirt, cuffs).
  5. Project the front photo onto the front half and the back photo onto the
     back half, then export a GLB in the avatar's frame (Y up, metres,
     feet at y = 0, facing +Z).

Size comes from the avatar, shape and look come from the photos. The result is
a static shell: no cloth simulation and no skeleton.

Example:
  python backend/photo_to_garment.py --front samples/dress01_front.jpg ^
      --back samples/dress01_back.jpg --avatar avatar.glb --name dress01
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.spatial import ConvexHull

# Gap between body and garment, in metres.
EASE_M = 0.015
# Mesh resolution.
BODY_RINGS = 180
SLEEVE_RINGS = 60
PANEL_COLUMNS = 48  # per half (front or back)
# Below the hips the skirt cross-section blends from the body shape to an
# ellipse this deep (depth / width), so it hangs round instead of flat.
SKIRT_DEPTH_RATIO = 0.8


# ---------------------------------------------------------------------------
# 1-2. Photo: cut-out and outline
# ---------------------------------------------------------------------------

def garment_mask(image, session):
    """Boolean mask of the garment: largest cut-out region, holes filled."""
    from rembg import remove

    alpha = np.asarray(remove(image, session=session))[:, :, 3] > 128
    labels, count = ndimage.label(alpha)
    if count == 0:
        raise SystemExit("No garment found in the photo.")
    sizes = ndimage.sum(alpha, labels, range(1, count + 1))
    mask = labels == (int(np.argmax(sizes)) + 1)
    mask = ndimage.binary_closing(mask, iterations=2)
    return ndimage.binary_fill_holes(mask)


def row_runs(row, min_len=6):
    """Horizontal runs of garment pixels in one image row: [(start, end), ...]."""
    cols = np.flatnonzero(row)
    if cols.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(cols) > 1)
    starts = np.r_[cols[0], cols[breaks + 1]]
    ends = np.r_[cols[breaks], cols[-1]]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a + 1 >= min_len]


@dataclass
class Sleeve:
    top: int  # image row of the shoulder seam
    cuff: int  # image row of the sleeve end
    start: int  # first row of the sleeve cap (the rounded top above the seam)
    center: np.ndarray  # image x of the sleeve centre, per row from start to cuff
    half_width: np.ndarray  # per row from start to cuff

    def edges(self, row):
        i = int(np.clip(round(row - self.start), 0, len(self.center) - 1))
        return self.center[i] - self.half_width[i], self.center[i] + self.half_width[i]


@dataclass
class PhotoLayout:
    top: int  # collar row
    shoulder: int
    hem: int
    torso_left: np.ndarray  # image x per row (NaN outside top..hem)
    torso_right: np.ndarray
    sleeves: dict  # "left" / "right" (image side) -> Sleeve
    shoulder_width: float  # seam to seam, pixels

    def torso_width(self, row):
        row = int(np.clip(round(row), self.top, self.hem))
        return self.torso_right[row] - self.torso_left[row]


def analyze_photo(mask):
    """Find collar, shoulders, hem, torso edges and sleeves in a garment mask."""
    height = mask.shape[0]
    span_left = np.full(height, np.nan)
    span_right = np.full(height, np.nan)
    for r in range(height):
        cols = np.flatnonzero(mask[r])
        if cols.size:
            span_left[r], span_right[r] = cols[0], cols[-1]
    width = np.nan_to_num(span_right - span_left + 1)

    rows = np.flatnonzero(width > 0)
    first, last = rows[0], rows[-1]
    upper_max = width[first:first + int(0.3 * (last - first)) + 1].max()
    # The hanger hook is narrow; the collar is the first clearly wide row.
    top = first + int(np.argmax(width[first:] >= 0.3 * upper_max))
    # The hem is the last wide, solid row (the very bottom is ragged folds).
    coverage = mask.sum(axis=1) / np.maximum(width, 1)
    hem = int(np.flatnonzero((width >= 0.5 * width.max()) & (coverage > 0.9))[-1])
    ref = top + int(0.1 * (hem - top))
    shoulder = top + int(np.argmax(width[top:] >= 0.9 * width[ref]))

    # Rows where a sleeve hangs free of the body show up as separate runs.
    sleeve_zone_end = top + int(0.6 * (hem - top))
    separated = {"left": {}, "right": {}}
    torso_known = {"left": {}, "right": {}}
    for r in range(shoulder, sleeve_zone_end):
        runs = row_runs(mask[r])
        if len(runs) < 2:
            continue
        body = max(runs, key=lambda run: run[1] - run[0])
        limit = 0.35 * width[r]
        if runs[0] != body and runs[0][1] - runs[0][0] < limit:
            separated["left"][r] = runs[0]
            torso_known["left"][r] = body[0]
        if runs[-1] != body and runs[-1][1] - runs[-1][0] < limit:
            separated["right"][r] = runs[-1]
            torso_known["right"][r] = body[1]

    sleeves = {}
    for side in ("left", "right"):
        if len(separated[side]) < 10:
            continue
        sep_rows = np.array(sorted(separated[side]))
        cuff = int(sep_rows.max())
        first_sep = int(sep_rows.min())
        sep_hw = np.array([(separated[side][r][1] - separated[side][r][0]) / 2 for r in sep_rows[:10]])
        hw_merged = float(np.median(sep_hw))
        center, half_width = [], []
        for r in range(shoulder, cuff + 1):
            if r in separated[side]:
                a, b = separated[side][r]
                center.append((a + b) / 2)
                half_width.append((b - a) / 2)
            else:
                # Sleeve still touching the body: keep its outer edge and the
                # width it has where it first hangs free.
                outer = span_left[r] if side == "left" else span_right[r]
                if np.isnan(outer):
                    outer = center[-1] if center else span_left[shoulder]
                center.append(outer + hw_merged if side == "left" else outer - hw_merged)
                half_width.append(hw_merged if r < first_sep else (half_width[-1] if half_width else hw_merged))
                if r < first_sep:
                    # The hanging sleeve hides the body's edge: it is the sleeve's inner edge.
                    torso_known[side][r] = outer + 2 * hw_merged if side == "left" else outer - 2 * hw_merged
        # Sleeve cap: above the seam the outline still bulges out past the
        # sleeve's inner edge; that bulge is the rounded top of the sleeve.
        inner = center[0] + half_width[0] if side == "left" else center[0] - half_width[0]
        start = shoulder
        cap_center, cap_half_width = [], []
        for r in range(shoulder - 1, top - 1, -1):
            outer = span_left[r] if side == "left" else span_right[r]
            reach = inner - outer if side == "left" else outer - inner
            if np.isnan(outer) or reach < 4:
                break
            cap_center.insert(0, (outer + inner) / 2)
            cap_half_width.insert(0, reach / 2)
            start = r
        center = cap_center + center
        half_width = cap_half_width + half_width
        sleeves[side] = Sleeve(
            top=shoulder,
            cuff=cuff,
            start=start,
            center=ndimage.gaussian_filter1d(np.array(center, float), 4),
            half_width=ndimage.gaussian_filter1d(np.array(half_width, float), 4),
        )
        for r in range(cuff + 1, hem + 1):
            torso_known[side][r] = span_left[r] if side == "left" else span_right[r]

    torso = {}
    for side, span in (("left", span_left), ("right", span_right)):
        known = dict(torso_known[side])
        for r in range(top, shoulder):
            known[r] = span[r]
        if side not in sleeves:
            for r in range(shoulder, hem + 1):
                known[r] = span[r]
        known_rows = np.array(sorted(r for r in known if not np.isnan(known[r])))
        edge = np.full(height, np.nan)
        all_rows = np.arange(top, hem + 1)
        edge[all_rows] = np.interp(all_rows, known_rows, [known[r] for r in known_rows])
        edge[all_rows] = ndimage.gaussian_filter1d(edge[all_rows], 3)
        # Between collar and shoulder the outline includes the sleeve caps;
        # the body there is no wider than at the shoulder seam.
        upper = np.arange(top, shoulder)
        if side == "left":
            edge[upper] = np.maximum(edge[upper], edge[shoulder])
        else:
            edge[upper] = np.minimum(edge[upper], edge[shoulder])
        torso[side] = edge

    return PhotoLayout(top, shoulder, hem, torso["left"], torso["right"], sleeves, float(width[shoulder]))


def fill_background(rgb, mask):
    """Paint the background with the nearest garment colour, so texture
    samples that land just outside the outline never show the wall. The
    outline itself is pulled in a few pixels: those pixels mix in the wall."""
    inner = ndimage.binary_erosion(mask, iterations=4)
    indices = ndimage.distance_transform_edt(~inner, return_distances=False, return_indices=True)
    return rgb[indices[0], indices[1]]


def draw_layout(rgb, layout, path):
    """Debug image: the detected collar/shoulder/hem lines, torso edges and sleeves."""
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image)
    w = image.width
    for row, colour in ((layout.top, "yellow"), (layout.shoulder, "orange"), (layout.hem, "yellow")):
        draw.line([(0, row), (w, row)], fill=colour, width=3)
    rows = range(layout.top, layout.hem + 1, 4)
    draw.line([(layout.torso_left[r], r) for r in rows], fill="lime", width=3)
    draw.line([(layout.torso_right[r], r) for r in rows], fill="lime", width=3)
    for sleeve in layout.sleeves.values():
        pts = [(sleeve.center[i], sleeve.start + i) for i in range(0, len(sleeve.center), 4)]
        draw.line(pts, fill="red", width=3)
        for i in range(0, len(sleeve.center), 40):
            c, hw, r = sleeve.center[i], sleeve.half_width[i], sleeve.start + i
            draw.line([(c - hw, r), (c + hw, r)], fill="cyan", width=2)
    image.save(path)


# ---------------------------------------------------------------------------
# 3. Avatar: cross-sections
# ---------------------------------------------------------------------------

def section_loops(mesh, y):
    path = mesh.section(plane_origin=[0.0, y, 0.0], plane_normal=[0.0, 1.0, 0.0])
    if path is None:
        return []
    return [loop for loop in path.discrete if len(loop) >= 4]


@dataclass
class AvatarFrame:
    height: float
    hip_limit: float  # loops reaching inside +-hip_limit belong to the body
    armpit_y: float
    shoulder_y: float
    shoulder_half: float  # half-width of the shoulders (where sleeves begin)


def body_points(loops, frame, y):
    """Torso (or both legs) cross-section points at height y, as (x, z)."""
    parts = [l for l in loops if l[:, 0].min() < frame.hip_limit and l[:, 0].max() > -frame.hip_limit]
    if not parts:
        return None
    pts = np.concatenate(parts)[:, [0, 2]]
    if y > frame.armpit_y:
        # Arms join the torso above the armpits; cut them at the shoulder.
        pts = pts.copy()
        pts[:, 0] = np.clip(pts[:, 0], -frame.shoulder_half, frame.shoulder_half)
    return pts


def measure_avatar(mesh):
    lo, hi = mesh.bounds
    height = float(hi[1] - lo[1])

    def crossing_half_width(y):
        halves = [np.abs(l[:, 0]).max() for l in section_loops(mesh, y) if l[:, 0].min() < 0 < l[:, 0].max()]
        return max(halves) if halves else 0.0

    hip_half = max(crossing_half_width(y) for y in np.linspace(0.45 * height, 0.62 * height, 12))
    hip_limit = 0.6 * hip_half

    armpit_y = None
    for y in np.arange(0.80 * height, 0.55 * height, -0.005):
        loops = section_loops(mesh, y)
        left = any(l[:, 0].max() < -hip_limit for l in loops)
        right = any(l[:, 0].min() > hip_limit for l in loops)
        if left and right:
            armpit_y = float(y)
            break
    if armpit_y is None:
        raise SystemExit("Could not find the armpits: are the arms touching the body?")

    shoulder_half = crossing_half_width(armpit_y - 0.02) + 0.035
    frame = AvatarFrame(height, hip_limit, armpit_y, armpit_y, shoulder_half)
    shoulder_y = armpit_y
    for y in np.arange(armpit_y, 0.95 * height, 0.005):
        pts = body_points(section_loops(mesh, y), frame, y)
        if pts is None or np.abs(pts[:, 0]).max() < 0.8 * shoulder_half:
            break
        shoulder_y = float(y)
    frame.shoulder_y = shoulder_y
    return frame


def arm_centerline(mesh, frame, side):
    """Centre line of the arm on one side (+1 = +X), from inside the shoulder
    down to the hand, as a polyline of points (n, 3)."""
    points = []
    for y in np.arange(frame.armpit_y - 0.01, 0.3 * frame.height, -0.01):
        loops = [
            l for l in section_loops(mesh, y)
            if (side > 0 and l[:, 0].min() > frame.hip_limit) or (side < 0 and l[:, 0].max() < -frame.hip_limit)
        ]
        if not loops:
            if points:
                break
            continue
        if points:
            loop = min(loops, key=lambda l: np.linalg.norm(l.mean(axis=0) - points[-1]))
            if np.linalg.norm(loop.mean(axis=0) - points[-1]) > 0.06:
                break  # fingers or a different body part
        else:
            loop = min(loops, key=lambda l: side * l.mean(axis=0)[0])
        points.append(loop.mean(axis=0))
    if len(points) < 5:
        raise SystemExit("Could not trace the avatar's arm.")
    points = ndimage.gaussian_filter1d(np.array(points), 2, axis=0, mode="nearest")

    # Extend up the upper arm into the shoulder so the sleeve meets the bodice.
    up = points[0] - points[min(8, len(points) - 1)]
    up /= np.linalg.norm(up)
    if abs(up[0]) > 0.2:
        # Start well inside the bodice so the sleeve's open end stays hidden.
        reach = (points[0][0] - side * (frame.shoulder_half - 0.05)) / -up[0]
    else:
        reach = 0.06
    start = points[0] + max(reach, 0.0) * up
    return np.vstack([start, points])


def polyline_at(points, s):
    """Position and unit tangent at arc length s (extrapolated past the end)."""
    seg = np.diff(points, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(seg_len)]
    i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(seg) - 1))
    tangent = seg[i] / seg_len[i]
    return points[i] + (s - cum[i]) * tangent, tangent


# ---------------------------------------------------------------------------
# 4. Garment geometry
# ---------------------------------------------------------------------------

def ring_angles(columns=PANEL_COLUMNS):
    """Full-circle angles: front half pi -> 0 (-X -> front -> +X), then back
    half 0 -> -pi (+X -> back -> -X). Index `columns` is shared by both halves."""
    return np.pi - np.arange(2 * columns + 1) * (np.pi / columns)


def ray_polygon_radius(center, polygon, angles):
    """Distance from center to a convex polygon's edge along each angle."""
    a = polygon
    e = np.roll(polygon, -1, axis=0) - a
    d = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    ac = a - center
    det = -d[:, None, 0] * e[None, :, 1] + d[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-ac[None, :, 0] * e[None, :, 1] + e[None, :, 0] * ac[None, :, 1]) / det
        s = (d[:, None, 0] * ac[None, :, 1] - d[:, None, 1] * ac[None, :, 0]) / det
    valid = (np.abs(det) > 1e-12) & (s >= -1e-9) & (s <= 1 + 1e-9) & (t > 0)
    return np.where(valid, t, np.inf).min(axis=1)


def hull_ring(points2d, angles, center=None):
    """Convex outline of 2D points sampled at `angles`: (center, radii)."""
    hull = ConvexHull(points2d)
    polygon = points2d[hull.vertices]
    if center is None:
        center = polygon.mean(axis=0)
    return center, ray_polygon_radius(center, polygon, angles)


def perimeter(center, radii, angles):
    pts = center + radii[:, None] * np.stack([np.cos(angles), np.sin(angles)], axis=1)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def garment_scale(body_perimeter, photo_width, fitted):
    """Garment circumference per ring: never smaller than the body, and as
    much bigger as the photo shows the garment flaring out. `fitted` marks the
    rings whose photo width/body ratio sets the scale."""
    ratio = np.median(body_perimeter[fitted] / photo_width[fitted])
    target = ndimage.gaussian_filter1d(np.maximum(body_perimeter, ratio * photo_width), 2)
    return np.maximum(target, body_perimeter) / body_perimeter


def build_body(mesh, frame, front, back, y_top, y_hem, photo_rows):
    angles = ring_angles()
    ys = np.linspace(y_top, y_hem, BODY_RINGS)
    centers, radii = [], []
    for y in ys:
        pts = body_points(section_loops(mesh, y), frame, min(y, frame.shoulder_y + 0.04))
        if pts is None or len(pts) < 6:
            centers.append(centers[-1])
            radii.append(radii[-1])
            continue
        c, r = hull_ring(pts, angles)
        centers.append(c)
        radii.append(r + EASE_M)
    centers = ndimage.gaussian_filter1d(np.array(centers), 1.5, axis=0)
    radii = np.array(radii)

    body_perimeter = np.array([perimeter(c, r, angles) for c, r in zip(centers, radii)])
    photo_width = np.array([front.torso_width(photo_rows["front"](y)) for y in ys])
    fitted = (ys < frame.armpit_y - 0.05) & (ys > frame.armpit_y - 0.25)
    scale = garment_scale(body_perimeter, photo_width, fitted)
    garment = radii * scale[:, None]

    # Skirt: blend towards a rounder ellipse of the same circumference.
    hip_y = ys[np.argmax(np.where(ys < frame.armpit_y - 0.2, body_perimeter, 0))]
    for i, y in enumerate(ys):
        blend = np.clip((hip_y - y) / 0.3, 0.0, 1.0)
        if blend == 0:
            continue
        circumference = body_perimeter[i] * scale[i]
        a = circumference / (np.pi * (1 + SKIRT_DEPTH_RATIO))
        b = SKIRT_DEPTH_RATIO * a
        ellipse = a * b / np.sqrt((b * np.cos(angles)) ** 2 + (a * np.sin(angles)) ** 2)
        garment[i] = np.maximum((1 - blend) * garment[i] + blend * ellipse, radii[i])
    garment = ndimage.gaussian_filter(garment, sigma=(1.0, 0.0))
    garment = np.maximum(garment, radii)

    positions = np.zeros((len(ys), len(angles), 3))
    positions[:, :, 0] = centers[:, None, 0] + garment * np.cos(angles)
    positions[:, :, 1] = ys[:, None]
    positions[:, :, 2] = centers[:, None, 1] + garment * np.sin(angles)

    photo_y = {
        "front": np.array([photo_rows["front"](y) for y in ys]),
        "back": np.array([photo_rows["back"](y) for y in ys]),
    }
    edges = {
        "front": lambda row: (front.torso_left[row], front.torso_right[row]),
        "back": lambda row: (back.torso_left[row], back.torso_right[row]),
    }
    # Above the armpits the photo's outline (sloped shoulders, narrow collar)
    # does not match a horizontal ring, so the photo is projected straight on
    # there instead of being spread around the ring.
    projection = {
        "weight": np.clip((ys - (frame.armpit_y - 0.03)) / 0.06, 0.0, 1.0),
        "center_x": centers[:, 0],
    }
    return positions, photo_y, edges, projection


def build_sleeve(mesh, frame, side, sleeve_front, sleeve_back, length):
    """Sleeve tube along the avatar's arm. side: +1 = +X arm, -1 = -X arm."""
    angles = ring_angles()
    centerline = arm_centerline(mesh, frame, side)

    # Arm vertices, including the shoulder cap above the armpit; the torso
    # inside the bodice and the neck are left out.
    arm = (side * mesh.vertices[:, 0] > frame.shoulder_half - 0.04) & (mesh.vertices[:, 1] < frame.shoulder_y + 0.02)

    us = np.linspace(0.0, length, SLEEVE_RINGS)
    origins, rights, fronts, centers, radii = [], [], [], [], []
    for ui in us:
        origin, axis = polyline_at(centerline, ui)
        front_dir = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
        front_dir /= np.linalg.norm(front_dir)
        right_dir = np.cross(front_dir, axis)
        origins.append(origin)
        rights.append(right_dir)
        fronts.append(front_dir)

        rel = mesh.vertices - origin
        along = rel @ axis
        radial = rel - along[:, None] * axis
        pick = arm & (np.abs(along) < 0.015) & (np.linalg.norm(radial, axis=1) < 0.09)
        if pick.sum() < 6:
            if centers:
                centers.append(centers[-1])
                radii.append(radii[-1])
            continue
        coords = np.stack([radial[pick] @ right_dir, radial[pick] @ front_dir], axis=1)
        c, r = hull_ring(coords, angles)
        centers.append(c)
        radii.append(r + EASE_M)
    if not centers:
        raise SystemExit("Could not measure the avatar's arm.")
    while len(centers) < len(us):  # rings before the first measured one
        centers.insert(0, centers[0])
        radii.insert(0, radii[0])
    centers = ndimage.gaussian_filter1d(np.array(centers), 1.5, axis=0)
    radii = np.array(radii)

    t = us / length
    below_seam = sleeve_front.half_width[sleeve_front.top - sleeve_front.start:]
    photo_width = 2 * np.interp(t, np.linspace(0, 1, len(below_seam)), below_seam)
    body_perimeter = np.array([perimeter(c, r, angles) for c, r in zip(centers, radii)])
    scale = garment_scale(body_perimeter, photo_width, (t > 0.2) & (t < 0.6))
    # Sleeve cap: a little extra room where the sleeve is sewn to the shoulder.
    cap = 1.0 + 0.08 * np.clip(1.0 - us / 0.10, 0.0, 1.0)
    garment = radii * (scale * cap)[:, None]

    origins, rights, fronts = np.array(origins), np.array(rights), np.array(fronts)
    ring_centers = origins + centers[:, :1] * rights + centers[:, 1:] * fronts
    offsets = (np.cos(angles)[None, :, None] * rights[:, None, :]
               + np.sin(angles)[None, :, None] * fronts[:, None, :])
    positions = ring_centers[:, None, :] + garment[:, :, None] * offsets

    photo_y = {
        "front": sleeve_front.top + t * (sleeve_front.cuff - sleeve_front.top),
        "back": sleeve_back.top + t * (sleeve_back.cuff - sleeve_back.top),
    }

    edges = {"front": sleeve_front.edges, "back": sleeve_back.edges}
    return positions, photo_y, edges


# ---------------------------------------------------------------------------
# 5. Texture and export
# ---------------------------------------------------------------------------

def panel_mesh(positions, photo_y, edges, half, atlas_offset, atlas_size, projection=None, planar=False):
    """One half (front or back) of a tube as a textured grid.

    positions: (rings, 2*PANEL_COLUMNS+1, 3) full rings.
    The panel's columns run from the photo's left edge to its right edge,
    spaced by arc length so the print is not stretched at the sides, or by
    x position when `planar` (the shape was taken straight from the photo).
    projection (optional): per-ring weight for projecting the photo straight
    on instead, with the ring centre's x and the photo's metres per pixel."""
    cols = PANEL_COLUMNS
    part = positions[:, :cols + 1] if half == "front" else positions[:, cols:]
    rings = part.shape[0]

    if planar:
        x = part[:, :, 0]
        frac = (x - x[:, :1]) / (x[:, -1:] - x[:, :1])
    else:
        seg = np.linalg.norm(np.diff(part, axis=1), axis=2)
        arc = np.concatenate([np.zeros((rings, 1)), np.cumsum(seg, axis=1)], axis=1)
        frac = arc / np.maximum(arc[:, -1:], 1e-9)

    uv = np.zeros((rings, cols + 1, 2))
    width, height = atlas_size
    for i in range(rings):
        row = float(photo_y[half][i])
        left, right = edges[half](int(round(row)))
        x = left + frac[i] * (right - left)
        if projection is not None and projection["weight"][i] > 0:
            offset = (part[i, :, 0] - projection["center_x"][i]) / projection["metres_per_px"][half]
            projected = (left + right) / 2 + (offset if half == "front" else -offset)
            w = projection["weight"][i]
            x = (1 - w) * x + w * projected
        uv[i, :, 0] = (atlas_offset + x) / width
        uv[i, :, 1] = 1.0 - row / height

    idx = np.arange(rings * (cols + 1)).reshape(rings, cols + 1)
    v00, v01 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel()
    v10, v11 = idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    faces = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    return part.reshape(-1, 3), faces, uv.reshape(-1, 2)


def assemble(panels, atlas):
    vertices, faces, uvs, offset = [], [], [], 0
    for v, f, uv in panels:
        mesh = trimesh.Trimesh(v, f, process=False)
        # Faces must point away from the tube's axis.
        center = v.mean(axis=0)
        outward = ((mesh.triangles_center - center) * mesh.face_normals).sum(axis=1)
        if np.mean(outward) < 0:
            f = f[:, ::-1]
        vertices.append(v)
        faces.append(f + offset)
        uvs.append(uv)
        offset += len(v)
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=atlas, metallicFactor=0.0, roughnessFactor=0.9, doubleSided=True
    )
    return trimesh.Trimesh(
        np.concatenate(vertices),
        np.concatenate(faces),
        visual=trimesh.visual.TextureVisuals(uv=np.concatenate(uvs), material=material),
        process=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True, help="photo of the garment's front")
    parser.add_argument("--back", required=True, help="photo of the garment's back")
    parser.add_argument("--avatar", required=True, help="avatar GLB to fit the garment to")
    parser.add_argument("--name", required=True, help="output name, e.g. dress01")
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    parser.add_argument("--length-cm", type=float, help="garment length from shoulder seam to hem; "
                        "if left out it is estimated from the avatar's shoulder width")
    args = parser.parse_args()

    from rembg import new_session

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = new_session("u2net")

    photos, layouts, masks = {}, {}, {}
    for half, path in (("front", args.front), ("back", args.back)):
        image = Image.open(path).convert("RGB")
        photos[half] = np.asarray(image)
        masks[half] = garment_mask(image, session)
        layouts[half] = analyze_photo(masks[half])
        draw_layout(photos[half], layouts[half], out / f"{args.name}_{half}_layout.png")
    front, back = layouts["front"], layouts["back"]
    for half in ("front", "back"):
        if set(layouts[half].sleeves) != {"left", "right"}:
            raise SystemExit(f"Could not find both sleeves in the {half} photo "
                             f"(see {args.name}_{half}_layout.png).")

    scene = trimesh.load(args.avatar, force="scene")
    avatar = scene.to_geometry()
    frame = measure_avatar(avatar)

    # Vertical scale: metres per photo row on the front photo.
    shoulder_px = front.shoulder_width
    if args.length_cm:
        metres_per_px = args.length_cm / 100 / (front.hem - front.shoulder)
        length_source = "measured"
    else:
        metres_per_px = 2 * frame.shoulder_half / shoulder_px
        length_source = "estimated from avatar shoulder width"
    y_shoulder = frame.shoulder_y
    y_hem = max(y_shoulder - (front.hem - front.shoulder) * metres_per_px, 0.02)
    y_top = min(y_shoulder + (front.shoulder - front.top) * metres_per_px, y_shoulder + 0.07)

    def rows_for(layout):
        def row(y):
            t = (y_shoulder - y) / (y_shoulder - y_hem)
            return float(np.clip(layout.shoulder + t * (layout.hem - layout.shoulder), layout.top, layout.hem))
        return row

    photo_rows = {"front": rows_for(front), "back": rows_for(back)}

    # Texture atlas: front photo | back photo, background filled.
    fronts = fill_background(photos["front"], masks["front"])
    backs = fill_background(photos["back"], masks["back"])
    height = max(fronts.shape[0], backs.shape[0])
    atlas = np.zeros((height, fronts.shape[1] + backs.shape[1], 3), np.uint8)
    atlas[:fronts.shape[0], :fronts.shape[1]] = fronts
    atlas[:backs.shape[0], fronts.shape[1]:] = backs
    atlas_size = (atlas.shape[1], atlas.shape[0])
    offsets = {"front": 0, "back": fronts.shape[1]}

    panels = []
    positions, photo_y, edges, projection = build_body(avatar, frame, front, back, y_top, y_hem, photo_rows)
    projection["metres_per_px"] = {
        "front": metres_per_px,
        "back": (y_shoulder - y_hem) / (back.hem - back.shoulder),
    }
    for half in ("front", "back"):
        panels.append(panel_mesh(positions, photo_y, edges, half, offsets[half], atlas_size, projection))

    sleeve_lengths = {}
    for side in (-1, 1):
        # Seen from the front, the -X arm is on the photo's left; from the back, on its right.
        sleeve_front = front.sleeves["left" if side < 0 else "right"]
        sleeve_back = back.sleeves["right" if side < 0 else "left"]
        length = (sleeve_front.cuff - sleeve_front.top) * metres_per_px
        sleeve_lengths["-X" if side < 0 else "+X"] = round(length, 3)
        positions, photo_y, edges = build_sleeve(avatar, frame, side, sleeve_front, sleeve_back, length)
        for half in ("front", "back"):
            panels.append(panel_mesh(positions, photo_y, edges, half, offsets[half], atlas_size))

    garment = assemble(panels, Image.fromarray(atlas))
    garment_path = out / f"{args.name}.glb"
    garment.export(garment_path)

    combined = scene.copy()
    combined.add_geometry(garment, geom_name=args.name)
    combined.export(out / f"{args.name}_on_avatar.glb")

    info = {
        "name": args.name,
        "modelSource": "generated_from_image",
        "generationTool": "photo_to_garment.py (silhouette + photo projection on avatar)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "photos": {"front": str(args.front), "back": str(args.back)},
        "avatar": str(args.avatar),
        "lengthM": round(y_shoulder - y_hem, 3),
        "lengthSource": length_source,
        "sleeveLengthM": sleeve_lengths,
        "avatarFrame": {
            "height": round(frame.height, 3),
            "shoulderY": round(frame.shoulder_y, 3),
            "armpitY": round(frame.armpit_y, 3),
            "shoulderHalfWidth": round(frame.shoulder_half, 3),
        },
        "vertices": len(garment.vertices),
        "faces": len(garment.faces),
    }
    (out / f"{args.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
