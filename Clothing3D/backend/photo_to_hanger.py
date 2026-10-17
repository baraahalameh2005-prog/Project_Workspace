"""Turn two photos of a garment (front + back) into a 3D model hanging on a
hanger, the way it was photographed. No avatar involved.

The outline comes straight from the photos: every row of the front photo
becomes an elliptical cross-section exactly as wide as the garment there,
with an assumed depth (a garment on a hanger is fairly flat). The front photo
is projected onto the front half and the back photo onto the back half.

With --hem-cm (the hem's width laid flat, measured along its curve) the
skirt gets the fabric the photo does not show: from the waist down the
cross-sections fold, so that at the hem their length is twice that width.

Example:
  python backend/photo_to_hanger.py --front samples/dress01_front.jpg ^
      --back samples/dress01_back.jpg --length-cm 131 --hem-cm 118 --name dress01
"""

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

import photo_to_garment
from photo_to_garment import (
    BODY_RINGS,
    SLEEVE_RINGS,
    analyze_photo,
    assemble,
    draw_layout,
    fill_background,
    garment_mask,
    panel_mesh,
    ring_angles,
)

HEM_Y = 0.10  # height of the hem above the floor, metres
BODY_DEPTH = (0.35, 0.55)  # cross-section depth/width at the collar and at the hem
SLEEVE_DEPTH = 0.75
HANGER_COLOUR = [0.12, 0.12, 0.14, 1.0]
WAIST_DOWN_M = 0.38  # the skirt's folds start this far below the shoulder seam
FOLDS = (10, 0.75, 7, 0.25)  # fold count/weight, plus an irregular second wave

# Deep skirt folds need more points around each cross-section.
photo_to_garment.PANEL_COLUMNS = 96


def ellipse_rings(center_x, ys, half_widths, depth_ratios):
    """Elliptical cross-sections, one per y, in the order of ring_angles()."""
    angles = ring_angles(photo_to_garment.PANEL_COLUMNS)
    a = half_widths[:, None]
    b = (half_widths * depth_ratios)[:, None]
    positions = np.zeros((len(ys), len(angles), 3))
    positions[:, :, 0] = center_x[:, None] + a * np.cos(angles)
    positions[:, :, 1] = ys[:, None]
    positions[:, :, 2] = b * np.sin(angles)
    return positions


def add_folds(positions, extra):
    """Fold each ring (in place) until its length is (1 + extra) times what it was."""
    angles = ring_angles(photo_to_garment.PANEL_COLUMNS)
    k1, w1, k2, w2 = FOLDS
    fold = w1 * np.sin(k1 * angles) + w2 * np.sin(k2 * angles + 1.3)

    def length(ring):
        return np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()

    for i, ring in enumerate(positions):
        if extra[i] <= 0:
            continue
        xz = ring[:, [0, 2]]
        center = xz.mean(axis=0)
        normal = xz - center
        normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-9)
        target = (1 + extra[i]) * length(xz)
        lo, hi = 0.0, 0.5 * np.ptp(xz[:, 1])
        for _ in range(30):
            mid = (lo + hi) / 2
            if length(xz + mid * fold[:, None] * normal) < target:
                lo = mid
            else:
                hi = mid
        xz = xz + lo * fold[:, None] * normal
        positions[i, :, 0], positions[i, :, 2] = xz[:, 0], xz[:, 1]


def make_hanger(neck, shoulder_y, half_span):
    """A simple wire hanger: two shoulder arms, a bottom bar, a stem and a hook."""
    def rod(p, q, radius=0.006):
        return trimesh.creation.cylinder(radius=radius, segment=[p, q], sections=16)

    top = np.array([neck[0], neck[1], 0.0])
    left = np.array([neck[0] - half_span, shoulder_y, 0.0])
    right = np.array([neck[0] + half_span, shoulder_y, 0.0])
    stem_top = top + [0.0, 0.05, 0.0]
    parts = [rod(top, left), rod(top, right), rod(left, right, 0.004), rod(top, stem_top, 0.004)]

    radius = 0.035
    center = stem_top + [radius, 0.0, 0.0]
    phis = np.linspace(np.pi, -np.pi / 4, 24)
    hook = center + radius * np.stack([np.cos(phis), np.sin(phis), np.zeros_like(phis)], axis=1)
    for p, q in zip(hook[:-1], hook[1:]):
        parts.append(rod(p, q, 0.004))
    for p in (top, left, right, *hook):
        joint = trimesh.creation.icosphere(subdivisions=1, radius=0.0045)
        joint.apply_translation(p)
        parts.append(joint)

    hanger = trimesh.util.concatenate(parts)
    hanger.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            baseColorFactor=HANGER_COLOUR, metallicFactor=0.3, roughnessFactor=0.5
        )
    )
    return hanger


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--front", required=True, help="photo of the garment's front")
    parser.add_argument("--back", required=True, help="photo of the garment's back")
    parser.add_argument("--length-cm", type=float, required=True, help="garment length from shoulder seam to hem")
    parser.add_argument("--hem-cm", type=float, help="hem width laid flat, measured along its curve")
    parser.add_argument("--name", required=True, help="output name, e.g. dress01")
    parser.add_argument("--out", default="output", help="output folder (default: output)")
    args = parser.parse_args()

    from rembg import new_session

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = new_session("u2net")

    photos, masks, layouts = {}, {}, {}
    for half, path in (("front", args.front), ("back", args.back)):
        image = Image.open(path).convert("RGB")
        photos[half] = np.asarray(image)
        masks[half] = garment_mask(image, session)
        layouts[half] = analyze_photo(masks[half])
        draw_layout(photos[half], layouts[half], out / f"{args.name}_{half}_layout.png")
    front, back = layouts["front"], layouts["back"]

    metres_per_px = args.length_cm / 100 / (front.hem - front.shoulder)
    x0 = (front.torso_left[front.shoulder] + front.torso_right[front.shoulder]) / 2

    def y_of(row):
        return HEM_Y + (front.hem - row) * metres_per_px

    def back_row(row):
        t = (row - front.shoulder) / (front.hem - front.shoulder)
        return np.clip(back.shoulder + t * (back.hem - back.shoulder), back.top, back.hem)

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

    # Body: one cross-section per photo row.
    rows = np.linspace(front.top, front.hem, BODY_RINGS)
    ri = np.round(rows).astype(int)
    left, right = front.torso_left[ri], front.torso_right[ri]
    t = (rows - front.top) / (front.hem - front.top)
    positions = ellipse_rings(
        ((left + right) / 2 - x0) * metres_per_px,
        y_of(rows),
        (right - left) / 2 * metres_per_px,
        BODY_DEPTH[0] + (BODY_DEPTH[1] - BODY_DEPTH[0]) * t,
    )
    hem_ring = positions[-1][:, [0, 2]]
    hem_built = np.linalg.norm(np.diff(hem_ring, axis=0), axis=1).sum()
    if args.hem_cm:
        down = (rows - front.shoulder) * metres_per_px
        skirt = np.clip((down - WAIST_DOWN_M) / (down[-1] - WAIST_DOWN_M), 0, 1)
        add_folds(positions, skirt * max(2 * args.hem_cm / 100 / hem_built - 1, 0))
    photo_y = {"front": rows, "back": back_row(rows)}
    edges = {
        "front": lambda row: (front.torso_left[row], front.torso_right[row]),
        "back": lambda row: (back.torso_left[row], back.torso_right[row]),
    }
    hem_ring_final = positions[-1][:, [0, 2]]
    for half in ("front", "back"):
        panels.append(panel_mesh(positions, photo_y, edges, half, offsets[half], atlas_size, planar=True))

    # Sleeves hang beside the body, as in the photo.
    for side, other in (("left", "right"), ("right", "left")):
        if side not in front.sleeves or other not in back.sleeves:
            continue
        sf, sb = front.sleeves[side], back.sleeves[other]
        rows = np.linspace(sf.start, sf.cuff, SLEEVE_RINGS)
        i = np.clip(np.round(rows - sf.start).astype(int), 0, len(sf.center) - 1)
        positions = ellipse_rings(
            (sf.center[i] - x0) * metres_per_px,
            y_of(rows),
            sf.half_width[i] * metres_per_px,
            np.full(len(rows), SLEEVE_DEPTH),
        )
        t = (rows - sf.start) / (sf.cuff - sf.start)
        photo_y = {"front": rows, "back": sb.start + t * (sb.cuff - sb.start)}
        edges = {"front": sf.edges, "back": sb.edges}
        for half in ("front", "back"):
            panels.append(panel_mesh(positions, photo_y, edges, half, offsets[half], atlas_size, planar=True))

    garment = assemble(panels, Image.fromarray(atlas))

    top_row = front.top
    neck_x = ((front.torso_left[top_row] + front.torso_right[top_row]) / 2 - x0) * metres_per_px
    hanger = make_hanger(
        neck=(neck_x, y_of(top_row) + 0.01),
        shoulder_y=y_of(front.shoulder) - 0.01,
        # Inside the body, short of where the sleeves start.
        half_span=(front.torso_right[front.shoulder] - front.torso_left[front.shoulder]) / 2 * metres_per_px - 0.03,
    )

    scene = trimesh.Scene()
    scene.add_geometry(garment, geom_name=args.name)
    scene.add_geometry(hanger, geom_name="hanger")
    path = out / f"{args.name}_on_hanger.glb"
    scene.export(path)

    info = {
        "file": str(path),
        "lengthM": round(args.length_cm / 100, 3),
        "shoulderWidthM": round(front.shoulder_width * metres_per_px, 3),
        "hemWidthM": round((front.torso_right[front.hem] - front.torso_left[front.hem]) * metres_per_px, 3),
        "hemCircumferenceM": round(float(np.linalg.norm(np.diff(hem_ring_final, axis=0), axis=1).sum()), 3),
        "sleeves": sorted(front.sleeves),
        "vertices": len(garment.vertices),
    }
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
