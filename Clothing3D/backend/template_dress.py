"""Dress from a template + the garment's measurements + its photos, draped on
the avatar with Blender's cloth simulation.

The shape is a dress template (fitted bodice, skirt that widens to the
measured hem, long sleeves, ruffle collar and tiered ruffle cuffs) sized by
the flat-lay measurements, with the skirt's outline taken from the front
photo. The look is the photos themselves: the front photo is laid around the
front of the dress and the back photo around the back, so the fabric, buttons
and trim are the real ones. The ruffles are template geometry. The product
record says so (modelSource "template_parametric").

Steps:
  1. Read the photos: garment outline, shoulders, hem, sleeves (rembg u2net).
  2. Measure the avatar (cross-sections): shoulders, armpits, waist, arms.
  3. Build the garment's starting shape around it: bodice = body + ease;
     skirt as wide as the photo shows it, with the rest of the measured hem
     gathered into folds; sleeves along the arms.
  4. Blender (blender_drape.py) lets the skirt fall onto the avatar under
     gravity and exports the result as GLB (Y up, metres, facing +Z).

Example:
  python backend/template_dress.py --avatar output/avatar.glb --front samples/dress01_front.jpg ^
      --back samples/dress01_back.jpg --trim-box 280,760,330,820 --length-cm 130 --waist-cm 34 ^
      --hem-cm 118 --sleeve-cm 60 --name dress01_template
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage

from photo_to_garment import (
    analyze_photo,
    arm_centerline,
    body_points,
    fill_background,
    garment_mask,
    hull_ring,
    measure_avatar,
    polyline_at,
    section_loops,
)

BLENDER = Path.home() / "tools" / "blender-4.2.23-windows-x64" / "blender.exe"
EASE_M = 0.015  # gap between body and bodice
SLEEVE_EASE_M = 0.02  # less and the elbow shows through the sleeve
SLEEVE_HEAD_EASE_M = 0.006  # over the shoulder
AROUND = 48  # vertices around each ring (multiple of 4: back, +X, front, -X)
RING_STEP_M = 0.015  # distance between rings
COLLAR_M = 0.05  # high collar above the shoulder line
CUFF_M = 0.09  # ruffle cuff at the end of each sleeve
PIN_BELOW_ARMPIT_M = 0.04  # the bodice is held in place down to here
PUFF_M = 0.0  # extra radius of the slightly puffed sleeve head
SKIRT_DEPTH = 0.8  # skirt cross-section depth / width
FOLDS = (10, 0.75, 7, 0.25)  # skirt fold count/weight, plus an irregular second wave
# Ruffles: white frills, kept as built (not simulated).
RUFFLE_AROUND = 96
COLLAR_RUFFLE = dict(width=0.015, drop=0.03, waves=16, wave=0.01)  # stands up around the neck
CUFF_TIERS = 3
CUFF_RUFFLE = dict(width=0.025, drop=0.035, waves=12, wave=0.006)


def ring_dirs():
    """Unit directions around a ring, starting at the back centre and going
    round through +X (index AROUND/4), the front (AROUND/2), -X (3*AROUND/4)."""
    a = -np.pi / 2 + np.arange(AROUND) * 2 * np.pi / AROUND
    return np.stack([np.cos(a), np.sin(a)], axis=1)


def ring_length(points):
    return float(np.linalg.norm(np.diff(np.vstack([points, points[:1]]), axis=0), axis=1).sum())


def frill(center, radius_at, right, front, down, width, drop, waves, wave, rings=4, around=RUFFLE_AROUND,
          flute=0.6, edge=1.0):
    """A ruffle: a strip that starts on a ring (centre, radius per direction,
    in the plane of `right`/`front`) and spreads outwards and along `down`,
    with a fluted edge: a sine wave whose height grows smoothly from nothing
    at the seam to `wave` at the edge, `flute` times that in and out (soft
    vertical folds) and `edge` times that along `down` (a wavy hem).
    Returns (rings, around, 3)."""
    a = np.arange(around) * 2 * np.pi / around
    dirs = np.stack([np.cos(a), np.sin(a)], axis=1)
    base = radius_at(a)
    out = []
    for t in np.linspace(0, 1, rings):
        grow = t ** 1.5
        r = base + width * t + flute * wave * grow * np.cos(waves * a)
        along = drop * t + edge * wave * grow * np.sin(waves * a)
        out.append(center + (r * dirs[:, 0])[:, None] * right + (r * dirs[:, 1])[:, None] * front
                   + along[:, None] * down)
    return np.array(out)


def neck_points(avatar, frame, y):
    """The neck's cross-section at y (no shoulders), as (x, z), or None."""
    loops = [l for l in section_loops(avatar, y)
             if l[:, 0].min() < 0 < l[:, 0].max() and np.abs(l[:, 0]).max() < 0.6 * frame.shoulder_half]
    return np.concatenate(loops)[:, [0, 2]] if loops else None


def fold_ring(oval, center, target):
    """Wave an oval (AROUND, 2) along its normals until its length is `target`."""
    if ring_length(oval) >= target:
        return oval
    a = np.arange(AROUND) * 2 * np.pi / AROUND
    k1, w1, k2, w2 = FOLDS
    fold = w1 * np.sin(k1 * a) + w2 * np.sin(k2 * a + 1.3)
    normal = oval - center
    normal /= np.linalg.norm(normal, axis=1, keepdims=True)
    lo, hi = 0.0, 0.15
    for _ in range(30):
        mid = (lo + hi) / 2
        if ring_length(oval + mid * fold[:, None] * normal) < target:
            lo = mid
        else:
            hi = mid
    return oval + lo * fold[:, None] * normal


def build_body(avatar, frame, ys, photo_half_width, waist_cm, hem_cm):
    """Rings from the collar to the hem: (rings, AROUND, 3), and the waist's y."""
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    centers, radii = [], []
    for y in ys:
        pts = None
        if y > frame.shoulder_y:
            # Above the shoulder line: the neck and the slope of the shoulders
            # up to it (the central section), not the tops of the arms.
            central = [l for l in section_loops(avatar, y) if l[:, 0].min() < 0 < l[:, 0].max()]
            if central:
                pts = np.concatenate(central)[:, [0, 2]]
                pts[:, 0] = np.clip(pts[:, 0], -frame.shoulder_half, frame.shoulder_half)
        if pts is None:
            pts = body_points(section_loops(avatar, y), frame, min(y, frame.shoulder_y + 0.04))
        if pts is None or len(pts) < 6:
            centers.append(centers[-1])
            radii.append(radii[-1])
            continue
        c, r = hull_ring(pts, angles)
        centers.append(c)
        radii.append(r + EASE_M)
    centers = ndimage.gaussian_filter1d(np.array(centers), 1.5, axis=0)
    # Smooth the step at the armpits (above them the outline includes the top
    # of the arms, which the sleeves cover), but stay outside the torso below.
    raw = np.array(radii)
    radii = ndimage.gaussian_filter1d(raw, 3, axis=0)
    keep_out = (ys < frame.armpit_y) | (ys > frame.armpit_y + 0.06)  # only the armpit step is smoothed
    radii = np.where(keep_out[:, None], np.maximum(radii, raw), radii)
    girth = np.array([ring_length(c + r[:, None] * dirs) for c, r in zip(centers, radii)])

    # Waist: the narrowest girth between the armpits and the hips.
    torso = (ys < frame.armpit_y - 0.1) & (ys > frame.armpit_y - 0.4)
    waist_y = float(ys[torso][np.argmin(girth[torso])])
    waist_girth = float(np.interp(waist_y, ys[::-1], girth[::-1]))
    if waist_cm:
        waist_girth = max(waist_girth, 2 * waist_cm / 100)
    hem_girth = 2 * hem_cm / 100

    # Skirt: as wide as the photo shows it (and never inside the body), with
    # the rest of the fabric gathered into folds that the simulation lets fall.
    rings = []
    for y, c, r, g, half in zip(ys, centers, radii, girth, photo_half_width):
        ring = c + r[:, None] * dirs
        if y < waist_y:
            t = (waist_y - y) / (waist_y - ys[-1])
            target = waist_girth + t * (hem_girth - waist_girth)
            a = max(half, r.max())
            oval = c + np.stack([a * dirs[:, 0], SKIRT_DEPTH * a * dirs[:, 1]], axis=1)
            body_r = np.linalg.norm(ring - c, axis=1)
            oval_r = np.maximum(np.linalg.norm(oval - c, axis=1), body_r)
            blend = np.clip((waist_y - y) / 0.15, 0, 1)
            shape = c + ((1 - blend) * body_r + blend * oval_r)[:, None] * dirs
            ring = fold_ring(shape, c, max(target, g))
        rings.append(np.column_stack([ring[:, 0], np.full(AROUND, y), ring[:, 1]]))
    return np.array(rings), waist_y


def build_sleeve(avatar, frame, side, length):
    """Rings along the arm, from inside the shoulder to the wrist, their arc
    length, the seam's arc length and the cuff's ruffle tiers."""
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    line = arm_centerline(avatar, frame, side)
    # Arc length where the centre line leaves the shoulder: the sleeve seam.
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    seam = float(np.interp(frame.shoulder_half, side * line[:, 0], cum))
    arm = (side * avatar.vertices[:, 0] > frame.shoulder_half - 0.04) & (avatar.vertices[:, 1] < frame.shoulder_y + 0.02)

    tiers_at = seam + length - CUFF_M + np.arange(CUFF_TIERS) * CUFF_M / CUFF_TIERS
    ring_angles = np.r_[angles, angles[0] + 2 * np.pi]
    rings, us, tiers, last = [], [], [], None
    for u in np.arange(0.0, seam + length, RING_STEP_M):
        origin, axis = polyline_at(line, u)
        front = np.array([0.0, 0.0, 1.0]) - axis[2] * axis
        front /= np.linalg.norm(front)
        right = np.cross(front, axis)
        rel = avatar.vertices - origin
        along = rel @ axis
        radial = rel - along[:, None] * axis
        pick = arm & (np.abs(along) < 0.015) & (np.linalg.norm(radial, axis=1) < 0.09)
        if pick.sum() >= 6:
            coords = np.stack([radial[pick] @ right, radial[pick] @ front], axis=1)
            c, r = hull_ring(coords, angles)
            # Close over the shoulder (no boxy top), roomier from the elbow down.
            ease = np.interp(u - seam, [0.0, 0.15], [SLEEVE_HEAD_EASE_M, SLEEVE_EASE_M])
            last = (c, r + ease)
        elif last is None:
            continue
        c, r = last
        # Slightly puffed sleeve head, gone 8 cm past the seam.
        r = r + PUFF_M * np.clip((seam + 0.08 - u) / 0.08, 0, 1)
        p = c + r[:, None] * dirs
        rings.append(origin + p[:, :1] * right + p[:, 1:] * front)
        us.append(u)
        if len(tiers) < CUFF_TIERS and u >= tiers_at[len(tiers)]:
            center = origin + c[0] * right + c[1] * front
            radius_at = lambda a, r=r: np.interp(a, ring_angles, np.r_[r, r[0]], period=2 * np.pi) + 0.004
            tiers.append(frill(center, radius_at, right, front, axis, **CUFF_RUFFLE))
    return np.array(rings), np.array(us), seam, tiers


def photo_uvs(rings, rows, edges, atlas_offset, atlas_size):
    """Texture coordinates for one photo around one half of each ring.

    Front half (the front photo, left to right): -X, front, +X. Back half (the
    back photo, left to right): +X, back, -X. Returned as (rings, AROUND + 1, 2)
    indexed by ring position j, where j = AROUND is j = 0 again (for the
    face that closes the ring); only the half's own entries are filled."""
    q = AROUND // 4
    order = np.arange(3 * q, q - 1, -1) if atlas_offset == 0 else np.r_[np.arange(q, -1, -1), np.arange(AROUND - 1, 3 * q - 1, -1)]
    width, height = atlas_size
    uv = np.zeros((len(rings), AROUND + 1, 2))
    for i, ring in enumerate(rings):
        pts = ring[order % AROUND]
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        left, right = edges(int(round(rows[i])))
        x = left + arc / arc[-1] * (right - left)
        uv[i, order, 0] = (atlas_offset + x) / width
        uv[i, order, 1] = 1.0 - rows[i] / height
    if atlas_offset != 0:
        uv[:, AROUND] = uv[:, 0]
    return uv


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avatar", required=True, help="avatar GLB")
    parser.add_argument("--front", required=True, help="photo of the garment's front")
    parser.add_argument("--back", required=True, help="photo of the garment's back")
    parser.add_argument("--trim-box", help="x0,y0,x1,y1: area of the white trim in the front photo, for the ruffles")
    parser.add_argument("--length-cm", type=float, required=True, help="shoulder seam to hem")
    parser.add_argument("--waist-cm", type=float, help="waist width laid flat")
    parser.add_argument("--hem-cm", type=float, required=True, help="hem width laid flat, along its curve")
    parser.add_argument("--sleeve-cm", type=float, required=True, help="sleeve length from the shoulder seam")
    parser.add_argument("--frames", type=int, default=60, help="cloth simulation frames")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    from rembg import new_session, remove

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    session = new_session("u2net")
    photos, masks, textures, layouts = {}, {}, {}, {}
    for half, path in (("front", args.front), ("back", args.back)):
        image = Image.open(path).convert("RGB")
        photos[half] = np.asarray(image)
        masks[half] = garment_mask(image, session)
        # For the texture, wall showing through closed gaps (between a
        # sleeve and the body) is not garment, though the outline fills it.
        textures[half] = masks[half] & (np.asarray(remove(image, session=session))[:, :, 3] > 128)
        layouts[half] = analyze_photo(masks[half])
        if set(layouts[half].sleeves) != {"left", "right"}:
            raise SystemExit(f"Could not find both sleeves in the {half} photo.")
    front, back = layouts["front"], layouts["back"]

    # Texture atlas: front photo | back photo, background filled.
    fronts = fill_background(photos["front"], textures["front"])
    backs = fill_background(photos["back"], textures["back"])
    atlas = np.zeros((max(fronts.shape[0], backs.shape[0]), fronts.shape[1] + backs.shape[1], 3), np.uint8)
    atlas[:fronts.shape[0], :fronts.shape[1]] = fronts
    atlas[:backs.shape[0], fronts.shape[1]:] = backs
    atlas_path = out / f"{args.name}_texture.jpg"
    Image.fromarray(atlas).save(atlas_path, quality=92)
    atlas_size = (atlas.shape[1], atlas.shape[0])
    offsets = {"front": 0, "back": fronts.shape[1]}

    avatar = trimesh.load(args.avatar, force="scene").to_geometry()
    frame = measure_avatar(avatar)
    y_top = frame.shoulder_y + COLLAR_M
    y_hem = max(frame.shoulder_y - args.length_cm / 100, 0.02)
    metres_per_px = args.length_cm / 100 / (front.hem - front.shoulder)

    def body_rows(layout, ys):
        t = (frame.shoulder_y - ys) / (frame.shoulder_y - y_hem)
        # Above the shoulder line the photo is mostly wall beside the collar
        # (painted white from the ruffle), so the collar takes the shoulder row.
        return np.clip(layout.shoulder + t * (layout.hem - layout.shoulder), layout.shoulder, layout.hem)

    ys = np.arange(y_top, y_hem, -RING_STEP_M)
    rows = {"front": body_rows(front, ys), "back": body_rows(back, ys)}
    photo_half = np.array([front.torso_width(r) for r in rows["front"]]) * metres_per_px / 2
    body, waist_y = build_body(avatar, frame, ys, photo_half, args.waist_cm, args.hem_cm)
    parts = {"body": body}
    uvs = {"body": (
        photo_uvs(body, rows["front"], lambda r: (front.torso_left[r], front.torso_right[r]), 0, atlas_size),
        photo_uvs(body, rows["back"], lambda r: (back.torso_left[r], back.torso_right[r]), offsets["back"], atlas_size),
    )}

    ruffles = {}
    for side, name in ((-1, "sleeve_r"), (1, "sleeve_l")):  # the avatar's right arm is at -X
        rings, us, seam, tiers = build_sleeve(avatar, frame, side, args.sleeve_cm / 100)
        parts[name] = rings
        # Seen from the front the -X arm is on the photo's left; from the back, on its right.
        sf = front.sleeves["left" if side < 0 else "right"]
        sb = back.sleeves["right" if side < 0 else "left"]
        sleeve_uvs = []
        for s, half in ((sf, "front"), (sb, "back")):
            cap = np.full(len(us), s.top + 10.0)  # the cap in the photo is mostly background
            arm_rows = s.top + np.clip((us - seam) / (args.sleeve_cm / 100), 0, 1) * (s.cuff - s.top)
            sleeve_rows = np.where(us < seam, cap, arm_rows)
            sleeve_uvs.append(photo_uvs(rings, sleeve_rows, s.edges, offsets[half], atlas_size))
        uvs[name] = tuple(sleeve_uvs)
        for k, tier in enumerate(tiers):
            ruffles[f"{name}_cuff{k}"] = tier

    # Collar ruffle: standing up around the high collar, from its base.
    on_neck = [i for i, y in enumerate(ys) if y > frame.shoulder_y and neck_points(avatar, frame, y) is not None]
    neck = body[max(on_neck) if on_neck else 0]
    center = neck.mean(axis=0)
    rel = neck - center
    neck_angles = np.arctan2(rel[:, 2], rel[:, 0]) % (2 * np.pi)
    order = np.argsort(neck_angles)
    neck_r = np.linalg.norm(rel[:, [0, 2]], axis=1)[order]
    ruffles["collar"] = frill(
        center, lambda a: np.interp(a, neck_angles[order], neck_r, period=2 * np.pi) + 0.004,
        np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), **COLLAR_RUFFLE)

    # Per part: rings, photo UVs (front half, back half), white trim rings and
    # how firmly each ring is pinned. Only the bodice below the armpits and
    # the skirt are simulated.
    data = {"parts": np.array(list(parts) + list(ruffles))}
    for name, rings in {**parts, **ruffles}.items():
        data[name] = rings
        if name in ruffles:
            data[name + "_trim"] = np.ones(len(rings), bool)
            data[name + "_pin"] = np.ones(len(rings))
            continue
        data[name + "_uvf"], data[name + "_uvb"] = uvs[name]
        data[name + "_trim"] = np.zeros(len(rings), bool)
        if name == "body":
            data[name + "_pin"] = np.clip((ys - (frame.armpit_y - PIN_BELOW_ARMPIT_M)) / 0.08, 0, 1)
        else:
            data[name + "_pin"] = np.ones(len(rings))
    rest_path = out / f"{args.name}_rest.npz"
    np.savez(rest_path, **data)

    if args.trim_box:
        x0, y0, x1, y1 = [int(v) for v in args.trim_box.split(",")]
        px = photos["front"][y0:y1, x0:x1].reshape(-1, 3).astype(float)
        bright = px.sum(axis=1)
        trim_rgb = px[bright >= np.percentile(bright, 75)].mean(axis=0) / 255  # the lit side, not the shadows
    else:
        trim_rgb = np.array([0.92, 0.92, 0.9])

    glb = out / f"{args.name}.glb"
    cmd = [
        str(BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).with_name("blender_drape.py")), "--",
        "--avatar", str(Path(args.avatar).resolve()), "--rest", str(rest_path.resolve()),
        "--fabric", str(atlas_path.resolve()),
        "--trim", ",".join(f"{v:.3f}" for v in trim_rgb), "--frames", str(args.frames),
        "--out", str(glb.resolve()), "--out-with-avatar", str((out / f"{args.name}_on_avatar.glb").resolve()),
    ]
    subprocess.run(cmd, check=True)

    info = {
        "name": args.name,
        "modelSource": "template_parametric",
        "generationTool": "template_dress.py (dress template sized by measurements and the photo's outline, "
                          "photos as texture, Blender cloth simulation)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "photos": {"front": str(args.front), "back": str(args.back)},
        "avatar": str(args.avatar),
        "measurementsCm": {"length": args.length_cm, "waist": args.waist_cm, "hem": args.hem_cm,
                           "sleeve": args.sleeve_cm},
        "avatarFrame": {"height": round(frame.height, 3), "shoulderY": round(frame.shoulder_y, 3),
                        "armpitY": round(frame.armpit_y, 3), "waistY": round(waist_y, 3)},
    }
    (out / f"{args.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
