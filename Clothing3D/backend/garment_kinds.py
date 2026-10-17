"""Skirts, trousers and scarves on the avatar, from flat-lay photos measured
with an A4 sheet (measure_photo.py): the photos give the size at every
level and the fabric; Blender's cloth simulation lets it hang
(blender_drape.py, the same as for dresses).

  skirt  - rings around the body from the waistband to the hem, as wide as
           the photo shows them (twice the flat width), the extra fabric in
           folds; the waistband sits where the body is as wide as it.
  pants  - rings around the hips from the waistband to the crotch, then a
           tube around each leg down to the hem.
  scarf  - an open sheet of the scarf's size, laid over the shoulders.

Measurements typed over the photo's (the page lets the user correct them)
rescale the photo's widths so the given levels match exactly.

The result is a template shape sized from the photos, not a scan: info's
modelSource is "template_parametric".
"""

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from photo_to_garment import body_points, fill_background, hull_ring, measure_avatar, row_runs, section_loops
from template_dress import AROUND, BLENDER, RING_STEP_M, fold_ring, neck_points, photo_uvs, ring_dirs, ring_length

EASE_M = 0.012  # gap between the body and the fabric
WAIST_EASE = 1.03  # a waistband this much larger than the body rests there
LEG_EASE_M = 0.01
FRAMES = 20


class InputError(ValueError):
    pass


# ---------------------------------------------------------------- the avatar

def torso_ring(avatar, frame, y, angles):
    """(center, radii) of the body's outline at height y (torso, or both legs together)."""
    pts = body_points(section_loops(avatar, y), frame, y)
    if pts is None or len(pts) < 6:
        return None
    return hull_ring(pts, angles)


def leg_rings(avatar, frame, y, angles):
    """{-1: (center, radii), 1: ...}: each leg's outline at y, or None above the crotch."""
    loops = [l for l in section_loops(avatar, y) if l[:, 0].min() < frame.hip_limit and l[:, 0].max() > -frame.hip_limit]
    sides = {}
    for side in (-1, 1):
        pts = [l[:, [0, 2]] for l in loops if np.sign(l[:, 0].mean()) == side and l[:, 0].min() * l[:, 0].max() > 0]
        if not pts:
            return None
        sides[side] = hull_ring(np.concatenate(pts), angles)
    return sides


def crotch_y(avatar, frame):
    """Highest height where the legs are apart."""
    for y in np.arange(0.6 * frame.height, 0.3 * frame.height, -0.005):
        if leg_rings(avatar, frame, y, np.linspace(0, 2 * np.pi, 8, endpoint=False)) is not None:
            return float(y)
    raise InputError("Could not find where this avatar's legs part.")


def body_girth(avatar, frame, y, angles):
    ring = torso_ring(avatar, frame, y, angles)
    if ring is None:
        return None
    c, r = ring
    return ring_length(c + r[:, None] * np.stack([np.cos(angles), np.sin(angles)], 1))


def natural_waist_y(avatar, frame, angles):
    """The narrowest torso level between 40 and 10 cm below the armpits."""
    ys = np.arange(frame.armpit_y - 0.10, frame.armpit_y - 0.40, -0.01)
    girths = [body_girth(avatar, frame, y, angles) or np.inf for y in ys]
    return float(ys[int(np.argmin(girths))])


def rest_height(avatar, frame, angles, girth, lo, hi):
    """Where a waistband of this girth rests when pulled on from above: it
    stays at `hi` if the body there is at least as wide as the band (a snug or
    tight band), else it slides down until the body is (a loose band on the
    hips), but not below `lo`."""
    for y in np.arange(hi, lo, -0.005):
        g = body_girth(avatar, frame, y, angles)
        if g is not None and g * WAIST_EASE >= girth:
            return float(y)
    return float(lo)


# ---------------------------------------------------------------- the photos

class FlatPhoto:
    """A straightened photo, its garment mask and scale."""

    def __init__(self, image, mask, ppcm):
        self.rgb = np.asarray(image)
        self.mask = mask
        self.ppcm = ppcm
        rows = np.flatnonzero(mask.any(1))
        cols = np.flatnonzero(mask.any(0))
        self.top, self.bottom = int(rows[0]), int(rows[-1])
        self.cx = (cols[0] + cols[-1]) / 2
        waist_run = self.run(self.top + int(1.5 * ppcm), self.cx)
        if waist_run:
            self.cx = (waist_run[0] + waist_run[1]) / 2

    def run(self, row, x):
        row = int(np.clip(row, 0, self.mask.shape[0] - 1))
        for a, b in row_runs(self.mask[row], 3):
            if a <= x <= b:
                return a, b
        return None

    def runs(self, row):
        row = int(np.clip(row, 0, self.mask.shape[0] - 1))
        return row_runs(self.mask[row], 3)

    def width_cm(self, row, x=None):
        r = self.run(row, self.cx if x is None else x)
        return (r[1] - r[0]) / self.ppcm if r else np.nan

    def crotch_row(self):
        for r in range(self.top + int(3 * self.ppcm), self.bottom):
            if not self.mask[r, int(self.cx)]:
                return r
        return None


def atlas(photos):
    """Front | back photo side by side (background painted in fabric colour), and offsets."""
    textures = {h: fill_background(p.rgb, p.mask) for h, p in photos.items()}
    if "back" not in textures:
        textures["back"] = textures["front"][:, ::-1]  # no back photo: the front, mirrored
    fh, fw = textures["front"].shape[:2]
    bh, bw = textures["back"].shape[:2]
    img = np.zeros((max(fh, bh), fw + bw, 3), np.uint8)
    img[:fh, :fw] = textures["front"]
    img[:bh, fw:] = textures["back"]
    return Image.fromarray(img), {"front": 0, "back": fw}, (img.shape[1], img.shape[0])


def anchored(profile_cm, depth_cm, anchors):
    """Scale a girth profile so it passes through the given (depth, girth)
    anchors, linearly in between (and beyond, the nearest anchor's ratio)."""
    good = [(d, g / np.interp(d, depth_cm, profile_cm)) for d, g in anchors if g and np.isfinite(g)]
    if not good:
        return profile_cm
    good.sort()
    return profile_cm * np.interp(depth_cm, [d for d, _ in good], [k for _, k in good])


# ---------------------------------------------------------------- the kinds

def skirt_parts(avatar, frame, photos, cm, atlas_info):
    front = photos["front"]
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    length_cm = cm.get("length") or (front.bottom - front.top) / front.ppcm
    waist_girth = (cm.get("waist") or 2 * front.width_cm(front.top + 1.5 * front.ppcm)) / 100
    top_y = rest_height(avatar, frame, angles, waist_girth, crotch_y(avatar, frame) + 0.05,
                        natural_waist_y(avatar, frame, angles))
    ys = np.arange(top_y, max(top_y - length_cm / 100, 0.01), -RING_STEP_M)
    depth = (top_y - ys) * 100
    # The photo's girth at each level (front + back flat widths), then fitted to the measurements.
    scale = (front.bottom - front.top) / front.ppcm / length_cm  # photo cm per skirt cm
    girth = []
    for d in depth:
        w = [p.width_cm(p.top + d * scale * p.ppcm) for p in photos.values()]
        girth.append(np.nansum(w) if len(w) == 2 else 2 * w[0])
    girth = ndimage.gaussian_filter1d(np.array(girth, float), 2)
    girth = anchored(girth, depth, [(1.5, cm.get("waist")), (20.0, cm.get("hip")), (depth[-1], cm.get("hem"))]) / 100
    rings = []
    last = None
    for y, g in zip(ys, girth):
        ring = torso_ring(avatar, frame, y, angles) or last
        last = ring
        c, r = ring
        oval = c + (r + EASE_M)[:, None] * dirs
        rings.append(fold_ring(oval, c, g) if g > ring_length(oval) else oval)
    rings = np.array(rings)
    rings3 = np.stack([rings[:, :, 0], np.repeat(ys[:, None], AROUND, 1), rings[:, :, 1]], axis=2)
    rows = {h: p.top + depth * scale * p.ppcm for h, p in photos.items()}
    pin = (depth < 3).astype(float)
    return {"skirt": (rings3, uv_pair(rings, rows, photos, atlas_info), pin)}, {"waistY": top_y}


def uv_pair(rings2d, rows, photos, atlas_info, pick=None):
    """Front and back texture coordinates for rings (n, AROUND, 2)."""
    img, offsets, size = atlas_info
    out = []
    for half in ("front", "back"):
        p = photos.get(half) or photos["front"]
        r = rows.get(half, rows["front"])

        def edges(row, p=p, half=half):
            if pick is not None:
                run = pick(p, row, half)
            else:
                run = p.run(row, p.cx)
            if run is None:
                run = (p.cx - 1, p.cx + 1)
            if half == "back" and "back" not in photos:  # mirrored front
                w = p.rgb.shape[1]
                run = (w - 1 - run[1], w - 1 - run[0])
            return run

        out.append(photo_uvs(rings2d, r, edges, offsets[half], size))
    return out


def star_radii(pts, center, angles):
    """Farthest point in each direction from `center` (a star-shaped outline)."""
    rel = pts - center
    a = np.arctan2(rel[:, 1], rel[:, 0])
    r = np.linalg.norm(rel, axis=1)
    step = 2 * np.pi / len(angles)
    radii = np.full(len(angles), np.nan)
    for k, ang in enumerate(angles):
        near = np.abs((a - ang + np.pi) % (2 * np.pi) - np.pi) < step
        if near.any():
            radii[k] = r[near].max()
    ok = ~np.isnan(radii)
    radii = np.interp(np.arange(len(angles)), np.flatnonzero(ok), radii[ok], period=len(angles))
    return ndimage.gaussian_filter1d(radii, 1.0, mode="wrap")


def pants_parts(avatar, frame, photos, cm, atlas_info):
    front = photos["front"]
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    c_px = front.crotch_row()
    if c_px is None:
        raise InputError("Could not find the crotch in the photo.")
    rise_cm = cm.get("rise") or (c_px - front.top) / front.ppcm
    inseam_cm = cm.get("inseam") or (front.bottom - c_px) / front.ppcm
    body_crotch = crotch_y(avatar, frame)
    waist_girth = (cm.get("waist") or 2 * front.width_cm(front.top + 1.5 * front.ppcm)) / 100
    # The rise sets how high they come; a waistband wider than the body there slides down.
    top_y = min(body_crotch + rise_cm / 100, natural_waist_y(avatar, frame, angles))
    top_y = rest_height(avatar, frame, angles, waist_girth, body_crotch + 0.06, top_y)
    parts = {}

    # Legs first: from the crotch down the inseam.
    ys = np.arange(body_crotch, max(body_crotch - inseam_cm / 100, 0.01), -RING_STEP_M)
    depth = (body_crotch - ys) * 100
    leg_tops = []
    for side in (-1, 1):
        # The avatar's right leg (-X) is the front photo's left leg and the back photo's right one.
        def pick(p, row, half, side=side):
            runs = [r for r in p.runs(row) if r[1] - r[0] > 2 * p.ppcm]
            if not runs:
                return None
            left_in_photo = (side < 0) == (half == "front")
            return runs[0] if left_in_photo else runs[-1]

        rows = {h: (p.crotch_row() or c_px) + depth * p.ppcm for h, p in photos.items()}
        girth = []
        for i in range(len(ys)):
            w = []
            for h, p in photos.items():
                run = pick(p, rows[h][i], h)
                w.append((run[1] - run[0]) / p.ppcm if run else np.nan)
            girth.append(np.nansum(w) if len(w) == 2 else 2 * w[0])
        girth = ndimage.gaussian_filter1d(np.array(girth, float), 1.5)
        girth = anchored(girth, depth, [(2.5, cm.get("thigh")), (depth[-1], cm.get("opening"))]) / 100
        rings, last = [], None
        for y, g in zip(ys, girth):
            legs = leg_rings(avatar, frame, y, angles)
            ring = legs[side] if legs else last
            last = ring
            c, r = ring
            oval = c + (r + LEG_EASE_M)[:, None] * dirs
            rings.append(fold_ring(oval, c, g) if g > ring_length(oval) else oval)
        rings = np.array(rings)
        leg_tops.append(rings[0])
        parts["leg_r" if side < 0 else "leg_l"] = (
            np.stack([rings[:, :, 0], np.repeat(ys[:, None], AROUND, 1), rings[:, :, 1]], 2),
            uv_pair(rings, rows, photos, atlas_info, pick), (depth < 2).astype(float))

    # Hips: waistband down to the crotch. Over the last 10 cm the outline turns
    # into that of the two leg tubes' tops together, so the hips meet the legs
    # without a step (no skirt-like edge at the crotch).
    legs_outline = np.concatenate(leg_tops)
    ys = np.arange(top_y, body_crotch - 0.005, -RING_STEP_M)
    t = (top_y - ys) / max(top_y - body_crotch, 1e-6)  # 0 at the waistband, 1 at the crotch
    rows = {h: p.top + np.clip(t, 0, 1) * ((p.crotch_row() or c_px) - p.top) for h, p in photos.items()}
    girth = []
    for i in range(len(ys)):
        w = [p.width_cm(rows[h][i]) for h, p in photos.items()]
        girth.append(np.nansum(w) if len(w) == 2 else 2 * w[0])
    girth = ndimage.gaussian_filter1d(np.array(girth, float), 1.5)
    girth = anchored(girth, t * rise_cm, [(1.5, cm.get("waist")), (0.75 * rise_cm, cm.get("hip"))]) / 100
    rings, last = [], None
    for y, g in zip(ys, girth):
        ring = torso_ring(avatar, frame, y, angles) or last
        last = ring
        c, r = ring
        r = r + EASE_M
        if g > ring_length(c + r[:, None] * dirs):
            r = r * g / ring_length(c + r[:, None] * dirs)
        blend = float(np.clip(1 - (y - body_crotch) / 0.10, 0, 1)) ** 1.5
        if blend > 0:
            r = (1 - blend) * r + blend * star_radii(legs_outline, c, angles)
        rings.append(c + r[:, None] * dirs)
    rings = np.array(rings)
    parts["hips"] = (np.stack([rings[:, :, 0], np.repeat(ys[:, None], AROUND, 1), rings[:, :, 1]], 2),
                     uv_pair(rings, rows, photos, atlas_info), (t < 0.1).astype(float))
    return parts, {"waistY": top_y, "crotchY": body_crotch}


def scarf_parts(avatar, frame, photos, cm, atlas_info):
    """A stole: folded in half along its length, its middle lying round the
    back of the neck (held there), both ends hanging down the front."""
    length = (cm.get("length") or 180) / 100
    width = (cm.get("width") or 70) / 100 / 2  # folded in half
    y0 = frame.shoulder_y + 0.03
    neck = neck_points(avatar, frame, y0)
    nz = float(neck[:, 1].mean()) if neck is not None else 0.0
    neck_r = float(np.abs(neck[:, 0]).max()) + 0.015 if neck is not None else 0.07
    inner = 0.3  # share of the folded width inside the path line (the rest lies outwards)
    R = neck_r + inner * width

    def front_z(y):
        loops = [l for l in section_loops(avatar, y) if np.abs(l[:, 0]).min() < 0.2]
        if not loops:
            return nz + 0.12
        p = np.concatenate(loops)
        return float(p[np.abs(p[:, 0]) < 0.2, 2].max()) + 0.03

    arc_len = np.pi * R
    hang = max((length - arc_len) / 2, 0.05)
    n_arc, n_hang = 16, max(6, int(hang / RING_STEP_M / 2))
    path, across = [], []
    # Right end, bottom to top (the avatar's right is -X).
    for k in range(n_hang, 0, -1):
        y = y0 - hang * k / n_hang
        path.append([-R, y, front_z(y)])
        across.append([-1.0, 0, 0])
    for k in range(n_arc + 1):  # round the back of the neck, from -X to +X
        a = np.pi + np.pi * k / n_arc
        d = np.array([np.cos(a), 0, np.sin(a)])
        path.append([R * d[0], y0, nz + R * d[2]])
        across.append(d)
    for k in range(1, n_hang + 1):
        y = y0 - hang * k / n_hang
        path.append([R, y, front_z(y)])
        across.append([1.0, 0, 0])
    path, across = np.array(path), np.array(across)
    cols = 8
    sheet = np.stack([path + across * width * (j / (cols - 1) - inner) for j in range(cols)], axis=1)  # (along, across, 3)
    # Only its edge round the back of the neck stays put; the rest falls onto the shoulders.
    pin = np.zeros((len(path), cols))
    pin[n_hang + 2:n_hang + n_arc - 1, 0] = 1.0

    # Texture: the scarf's rectangle in the front photo, its long side along the stole.
    img, offsets, size = atlas_info
    p = photos["front"]
    rows = np.flatnonzero(p.mask.any(1))
    cs = np.flatnonzero(p.mask.any(0))
    long_x = (cs[-1] - cs[0]) >= (rows[-1] - rows[0])
    uv = np.zeros((len(path), cols + 1, 2))
    for i in range(len(path)):
        for j in range(cols):
            a, b = i / (len(path) - 1), j / (cols - 1) / 2  # half its width shows (folded)
            if long_x:
                px, py = cs[0] + a * (cs[-1] - cs[0]), rows[0] + b * (rows[-1] - rows[0])
            else:
                px, py = cs[0] + b * (cs[-1] - cs[0]), rows[0] + a * (rows[-1] - rows[0])
            uv[i, j] = (px / size[0], 1 - py / size[1])
    return {"scarf": (sheet, (uv, uv), pin, False)}, {"shoulderY": frame.shoulder_y}


BUILDERS = {"skirt": skirt_parts, "pants": pants_parts, "scarf": scarf_parts}


def make(kind, avatar_path, photos, cm, out, name="dress", frames=FRAMES):
    """photos: {"front": FlatPhoto, "back"?: FlatPhoto}; cm: measurements
    (from the photo, possibly corrected). Writes <name>.glb (the garment) and
    <name>_on_avatar.glb, <name>.json; returns the info dict."""
    if kind not in BUILDERS:
        raise InputError(f"kind must be one of {tuple(BUILDERS)}")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    avatar = trimesh.load(avatar_path, force="scene").to_geometry()
    frame = measure_avatar(avatar)
    atlas_info = atlas(photos)
    atlas_path = out / f"{name}_texture.jpg"
    atlas_info[0].save(atlas_path, quality=92)
    parts, where = BUILDERS[kind](avatar, frame, photos, cm, atlas_info)

    data = {"parts": np.array(list(parts))}
    for part, spec in parts.items():
        rings, (uvf, uvb), pin = spec[:3]
        data[part] = rings
        data[part + "_uvf"], data[part + "_uvb"] = uvf, uvb
        data[part + "_trim"] = np.zeros(len(rings), bool)
        data[part + "_pin"] = pin
        if len(spec) > 3:
            data[part + "_closed"] = np.array(spec[3])
    rest = out / f"{name}_rest.npz"
    np.savez(rest, **data)
    glb = out / f"{name}.glb"
    subprocess.run([
        str(BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).with_name("blender_drape.py")), "--",
        "--avatar", str(Path(avatar_path).resolve()), "--rest", str(rest.resolve()), "--fabric", str(atlas_path.resolve()),
        "--trim", "0.9,0.9,0.9", "--frames", str(frames), "--out", str(glb.resolve()),
        "--out-with-avatar", str((out / f"{name}_on_avatar.glb").resolve()),
    ], check=True, capture_output=True)
    info = {
        "name": name, "kind": kind, "modelSource": "template_parametric",
        "generationTool": f"garment_kinds.py ({kind}: rings sized from flat-lay photos measured with an A4 sheet, "
                          "photos as texture, Blender cloth simulation)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "avatar": str(avatar_path), "measurementsCm": cm,
        "placement": {k: round(v, 3) for k, v in where.items()},
    }
    (out / f"{name}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info
