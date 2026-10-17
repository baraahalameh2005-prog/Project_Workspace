"""Dress from two flat-lay photos (front and back, laid on a plain cloth and
shot from above) + its length and shoulder width, draped on the avatar with
Blender's cloth simulation.

Laid flat, a garment shows its true flat widths, so the fabric around the
body at any level is the front's width plus the back's. That sets the size
of every cross-section: bodice, the skirt down to its hem, and the sleeves
along their length. The skirt is laid out as a fan, so its levels are curves
(side seam - centre - side seam; the last one is the hem). Each photo is laid
around its half of the dress along those same curves, so the fabric, buttons
and trim are the real ones. The ruffles (collar, cuff tiers) are template
geometry. Shape from the template + flat-lay, look from the photos:
modelSource "template_parametric".

Example:
  python backend/flat_dress.py --avatar output/avatar.glb --front samples/flat_front.jpg ^
      --back samples/flat_back.jpg --length-cm 130 --shoulder-cm 42 --name dress01_flat
"""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from flat_lay import PhotoError, analyze_flat, draw_flat, load_photo, sleeve_length, summary
from photo_to_garment import body_points, fill_background, hull_ring, measure_avatar, section_loops
from template_dress import (
    AROUND,
    BLENDER,
    COLLAR_M,
    EASE_M,
    PIN_BELOW_ARMPIT_M,
    RING_STEP_M,
    build_sleeve,
    fold_ring,
    frill,
    ring_dirs,
    ring_length,
)

CURVE_SAMPLES = 200
# The skirt's starting shape holds this share of the extra fabric (beyond
# the body) as width; the rest starts as folds and the simulation settles it.
SKIRT_OPEN = 0.45
SKIRT_DEPTH = 0.8
COLLAR_SOURCE_CM, COLLAR_HALF_CM = 1.0, 7.0  # where the collar's fabric is taken from
SLEEVE_CAP_SOURCE_CM = 2.0  # the sleeve head takes the fabric this far down the sleeve
# The collar and shoulders (above this depth) and the sleeve heads are mostly
# horizontal on the body, which the photo's rows cannot cover without
# stretching. They continue the fabric of the first ring below them instead,
# at its real scale, reflected back and forth over the next REFLECT_CM of the
# photo (so the texture is continuous where they meet).
PATCHED_ABOVE_CM = 4.0
SLEEVE_PATCHED_CM = 3.0  # past the sleeve seam
REFLECT_CM = 6.0
EDGE_INSET_CM = 1.5  # the photo's outline is shaded or mixed with the background: sample this far inside
COLLAR_RUFFLE_AT_M = -0.028  # height of the ruffle's base: above the shoulder line, at the collar stand
SHOULDER_SLOPE_M = 0.03  # the shoulders rise this much from the shoulder points to the collar stand
NECK_HALF_M = 0.065  # half-width of the collar stand
# Soft folds that fall over the chest, with a gently wavy edge (not a zigzag).
# A ruff round the collar stand that lies outwards over the shoulders, with a few big loose waves.
COLLAR_RUFFLE_FLAT = dict(width=0.032, drop=0.032, waves=16, wave=0.009, rings=12, around=320, flute=1.0, edge=0.35)


# ---------------------------------------------------------------------------
# Levels in a flat-lay photo
# ---------------------------------------------------------------------------

def side_polyline(layout, side):
    """The body's side edge from the armpit down to the hem corner, (n, 2)."""
    edge = layout.left_edge if side == "left" else layout.right_edge
    start = int(layout.armpit[side][1])
    rows = np.arange(start, int(layout.hem_center) + 1)
    xs = edge[rows]
    ok = ~np.isnan(xs)
    rows, xs = rows[ok], xs[ok]
    # The corner: where the edge is furthest out (below that it is the hem).
    corner = int(np.argmin(xs) if side == "left" else np.argmax(xs))
    return np.stack([xs[:corner + 1], rows[:corner + 1]], axis=1).astype(float)


def at_fraction(polyline, f):
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = f * cum[-1]
    return np.stack([np.interp(s, cum, polyline[:, 0]), np.interp(s, cum, polyline[:, 1])], axis=-1)


class FlatLevels:
    """Level curves on one flat-lay photo, by depth d (cm below the shoulder
    line, measured down the centre). Above the armpits a level is the image
    row between the body's edges; below, a curve from side to side through
    the centre line, ending as the hem."""

    def __init__(self, layout):
        self.l = layout
        self.ppc = layout.px_per_cm
        self.shoulder_y = (layout.shoulder["left"][1] + layout.shoulder["right"][1]) / 2
        self.armpit_y = max(layout.armpit["left"][1], layout.armpit["right"][1])
        self.d_armpit = (self.armpit_y - self.shoulder_y) / self.ppc
        self.d_hem = (layout.hem_center - self.shoulder_y) / self.ppc
        self.sides = {s: side_polyline(layout, s) for s in ("left", "right")}

    def curve(self, d):
        """(CURVE_SAMPLES, 2) image points from the photo's left to its right."""
        l = self.l
        row = float(np.clip(self.shoulder_y + d * self.ppc, l.top, l.hem_center))
        if d < 0:
            # The high collar: above the shoulder line the photo's rows run
            # through the ruffle and the shoulder puffs, so the collar takes
            # the fabric just below the neck, as wide as the collar stand.
            row = self.shoulder_y + COLLAR_SOURCE_CM * self.ppc
            half = COLLAR_HALF_CM * self.ppc
            a, b = np.array([l.center_x - half, row]), np.array([l.center_x + half, row])
            u = np.linspace(0, 1, CURVE_SAMPLES)[:, None]
            return a + u * (b - a)
        if d <= self.d_armpit:
            r = int(round(row))
            a, b = np.array([l.left_edge[r], row]), np.array([l.right_edge[r], row])
            u = np.linspace(0, 1, CURVE_SAMPLES)[:, None]
            return a + u * (b - a)
        f = (d - self.d_armpit) / (self.d_hem - self.d_armpit)
        a, b = at_fraction(self.sides["left"], f), at_fraction(self.sides["right"], f)
        c = np.array([l.center_x, row])
        # Quadratic through a (u=0), c (u=0.5), b (u=1).
        q = 2 * c - (a + b) / 2
        u = np.linspace(0, 1, CURVE_SAMPLES)[:, None]
        return (1 - u) ** 2 * a + 2 * u * (1 - u) * q + u ** 2 * b

    def width_cm(self, d):
        return ring_length_open(self.curve(d)) / self.ppc


def ring_length_open(points):
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def curve_point(curve, frac):
    """Points at arc-length fractions along an image curve."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    s = np.asarray(frac) * cum[-1]
    return np.stack([np.interp(s, cum, curve[:, 0]), np.interp(s, cum, curve[:, 1])], axis=-1)


def ring_fractions(ring, order):
    """Arc-length fractions (0..1) of ring points taken in `order`."""
    pts = ring[order % len(ring)]
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return arc / arc[-1]


def halves_order():
    q = AROUND // 4
    front = np.arange(3 * q, q - 1, -1)  # -X, front, +X = photo left to right
    back = np.r_[np.arange(q, -1, -1), np.arange(AROUND - 1, 3 * q - 1, -1)]  # +X, back, -X
    return front, back


def to_uv(points, offset, atlas_size):
    w, h = atlas_size
    return np.stack([(offset + points[:, 0]) / w, 1.0 - points[:, 1] / h], axis=1)


# ---------------------------------------------------------------------------
# Garment
# ---------------------------------------------------------------------------

def build_body(avatar, frame, ds, girths, waist_d):
    """Rings at depths ds (m below the shoulder line), each `girths` around
    (never inside the body)."""
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    ys = frame.shoulder_y - ds
    rings = []
    last = None
    for y, d, g in zip(ys, ds, girths):
        pts = None
        if y > frame.shoulder_y:
            central = [lp for lp in section_loops(avatar, y) if lp[:, 0].min() < 0 < lp[:, 0].max()]
            if central:
                pts = np.concatenate(central)[:, [0, 2]]
                # Sloping shoulders: the dress narrows from the shoulder
                # points to the collar stand instead of ending square.
                t = min((y - frame.shoulder_y) / SHOULDER_SLOPE_M, 1.0)
                limit = frame.shoulder_half + t * (NECK_HALF_M - frame.shoulder_half)
                pts[:, 0] = np.clip(pts[:, 0], -limit, limit)
        if pts is None:
            pts = body_points(section_loops(avatar, y), frame, min(y, frame.shoulder_y + 0.04))
        if pts is None or len(pts) < 6:
            c, r = last
        else:
            c, r = hull_ring(pts, angles)
            r = r + EASE_M
            last = (c, r)
        body = c + r[:, None] * dirs
        body_g = ring_length(body)
        if d <= waist_d or g <= body_g:
            # Bodice: the body's shape, widened evenly if the garment is bigger.
            ring = c + (r * max(1.0, g / body_g))[:, None] * dirs
        else:
            # Skirt: an oval holding part of the extra fabric, the rest as folds.
            blend = min((d - waist_d) / 0.15, 1.0)
            open_g = body_g + SKIRT_OPEN * blend * (g - body_g)
            unit = np.stack([dirs[:, 0], SKIRT_DEPTH * dirs[:, 1]], axis=1)
            oval = c + unit * open_g / ring_length(unit)
            body_r = np.linalg.norm(body - c, axis=1)
            oval_r = np.maximum(np.linalg.norm(oval - c, axis=1), body_r)
            shape = c + ((1 - blend) * body_r + blend * oval_r)[:, None] * dirs
            ring = fold_ring(shape, c, g)
        rings.append(np.column_stack([ring[:, 0], np.full(AROUND, y), ring[:, 1]]))
    return np.array(rings)


def mirror_wrap(x, size):
    """Fold x back and forth into [0, size], so a patch repeats without seams."""
    x = np.mod(x, 2 * size)
    return np.where(x > size, 2 * size - x, x)


def reflected_points(rings, order, base, direction, ppc):
    """Image points for rings[:-1] that continue the last ring's image points
    (`base`, one per column): each column moves through the photo along
    `direction` (away from the patched area) by its real distance from the
    last ring, reflected back and forth within REFLECT_CM."""
    pts = rings[:, order % AROUND]
    step = np.linalg.norm(np.diff(pts, axis=0), axis=2)  # (n - 1, columns)
    up = np.cumsum(step[::-1], axis=0)[::-1]  # distance from each ring to the last one
    shift = mirror_wrap(up * 100, REFLECT_CM) * ppc
    return base[None] + shift[..., None] * direction


def body_uvs(rings, levels, ds_cm, offsets, atlas_size):
    """Front photo around the front half, back photo around the back half;
    the collar and shoulders get plain fabric (see PATCHED_ABOVE_CM).

    Where the dress is narrower than the photo's level (a full skirt that
    did not lie flat, so the photo also shows some of the other side), the
    fabric is taken from the middle of the level at its real size instead of
    being squeezed in: the print keeps its scale."""
    front_order, back_order = halves_order()
    uvf = np.zeros((len(rings), AROUND + 1, 2))
    uvb = np.zeros_like(uvf)
    first = int(np.argmax(ds_cm >= PATCHED_ABOVE_CM))
    for table, order, half in ((uvf, front_order, "front"), (uvb, back_order, "back")):
        for i, (ring, d) in enumerate(zip(rings, ds_cm)):
            if i < first:
                continue
            curve = levels[half].curve(d)
            frac = ring_fractions(ring, order)
            pts_ring = ring[order % AROUND]
            dress_cm = np.linalg.norm(np.diff(pts_ring, axis=0), axis=1).sum() * 100
            open_len = ring_length_open(curve)
            keep = min(1.0, dress_cm / (open_len / levels[half].ppc))
            keep = min(keep, max(0.0, 1 - 2 * EDGE_INSET_CM * levels[half].ppc / open_len))
            pts = curve_point(curve, 0.5 + (frac - 0.5) * keep)
            table[i, order] = to_uv(pts, offsets[half], atlas_size)
            if i == first:
                base = pts
        top = reflected_points(rings[:first + 1], order, base, np.array([0.0, 1.0]), levels[half].ppc)
        for i in range(first):
            table[i, order] = to_uv(top[i], offsets[half], atlas_size)
    uvb[:, AROUND] = uvb[:, 0]
    return uvf, uvb


def sleeve_uvs(rings, s_px, sleeves, sl_ppc, head, offsets, atlas_size):
    """Each sleeve photo around its half of the tube; the ring position that
    is uppermost on the avatar maps to the sleeve's upper edge in the photo.
    The sleeve head (the first `head` rings) continues the fabric below it."""
    q = AROUND // 4
    front_order, back_order = halves_order()
    uvf = np.zeros((len(rings), AROUND + 1, 2))
    uvb = np.zeros_like(uvf)
    image_pts = {"front": {}, "back": {}}
    for i, ring in enumerate(rings):
        if i < head:
            continue
        q_is_up = ring[q, 1] > ring[3 * q, 1]
        for table, order, half in ((uvf, front_order, "front"), (uvb, back_order, "back")):
            sl = sleeves[half]
            a, b = sl.edges(np.clip(s_px[half][i], SLEEVE_CAP_SOURCE_CM * sl_ppc[half], sl.length))
            upper, lower = (a, b) if a[1] < b[1] else (b, a)
            # order runs from 3q to q (front) or q to 3q (back).
            start, end = (lower, upper) if (order[0] == 3 * q) == q_is_up else (upper, lower)
            frac = ring_fractions(ring, order)[:, None]
            image_pts[half][i] = start + frac * (end - start)
            table[i, order] = to_uv(image_pts[half][i], offsets[half], atlas_size)
    for table, order, half in ((uvf, front_order, "front"), (uvb, back_order, "back")):
        if head > 0:
            top = reflected_points(rings[:head + 1], order, image_pts[half][head], sleeves[half].axis, sl_ppc[half])
            for i in range(head):
                table[i, order] = to_uv(top[i], offsets[half], atlas_size)
    uvb[:, AROUND] = uvb[:, 0]
    return uvf, uvb


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avatar", required=True)
    parser.add_argument("--front", required=True, help="flat-lay photo, front up")
    parser.add_argument("--back", required=True, help="flat-lay photo, back up")
    parser.add_argument("--length-cm", type=float, required=True, help="shoulder seam to hem")
    parser.add_argument("--shoulder-cm", type=float, required=True, help="shoulder seam to shoulder seam")
    parser.add_argument("--hem-cm", type=float, help="hem width laid flat, measured along its curve; "
                        "if given, it overrides the photos' hem (a full skirt rarely lies exactly flat)")
    parser.add_argument("--sleeve-cm", type=float, help="sleeve length by tape; if left out, the photos' length is used")
    parser.add_argument("--sleeve-from", choices=("shoulder", "neck"), default="shoulder",
                        help="where the sleeve tape started: the shoulder seam, or the side of the neck "
                             "(then the shoulder seam's distance from the neck is taken off)")
    parser.add_argument("--frames", type=int, default=20,
                        help="cloth simulation frames (20 is ~15 s; 60 settles a wide skirt a little more)")
    parser.add_argument("--collar-ruffle", action="store_true", help="add a ruffle around the collar stand")
    parser.add_argument("--cuff-ruffle", action="store_true", help="add ruffle tiers at the cuffs")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    try:
        info = run(args)
    except PhotoError as error:
        raise SystemExit(f"PHOTO_ERROR {error}")
    print(json.dumps(info, indent=2))


def options(**kw):
    """Arguments for run() without the command line (the server's jobs)."""
    defaults = dict(hem_cm=None, sleeve_cm=None, sleeve_from="shoulder", frames=20,
                    collar_ruffle=False, cuff_ruffle=False, out="output")
    return argparse.Namespace(**{**defaults, **kw})


def run(args, session=None, masks=None):
    """Build the dress and put it on the avatar; returns the info dict.
    masks: {"front": bool array, "back": ...} of the photos as given (they are
    then not cut out again). Raises PhotoError ("front: ..." / "back: ...")."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if session is None and not masks:
        from rembg import new_session

        session = new_session("u2net")
    photos, layouts, levels, texture = {}, {}, {}, {}
    for half, path in (("front", args.front), ("back", args.back)):
        image = load_photo(path)
        photos[half] = np.asarray(image)
        try:
            given = None
            if masks and half in masks:
                given = np.asarray(Image.fromarray(np.asarray(masks[half], np.uint8) * 255).resize(
                    image.size, Image.NEAREST)) > 127
            layouts[half], mask = analyze_flat(image, session, args.length_cm, args.shoulder_cm, mask=given)
        except PhotoError as error:
            raise PhotoError(f"{half}: {error}", error.code) from error
        draw_flat(image, layouts[half], out / f"{args.name}_{half}_flat.png")
        levels[half] = FlatLevels(layouts[half])
        texture[half] = fill_background(photos[half], mask)

    # Texture atlas: front photo | back photo.
    fh, fw = texture["front"].shape[:2]
    bh, bw = texture["back"].shape[:2]
    atlas = np.zeros((max(fh, bh), fw + bw, 3), np.uint8)
    atlas[:fh, :fw] = texture["front"]
    atlas[:bh, fw:] = texture["back"]
    atlas_path = out / f"{args.name}_texture.jpg"
    Image.fromarray(atlas).save(atlas_path, quality=92)
    atlas_size = (atlas.shape[1], atlas.shape[0])
    offsets = {"front": 0, "back": fw}

    avatar = trimesh.load(args.avatar, force="scene").to_geometry()
    frame = measure_avatar(avatar)

    # Body levels: from the top of the collar to the hem, by depth below the shoulder line.
    ds = np.arange(-COLLAR_M, args.length_cm / 100 + 1e-6, RING_STEP_M)
    ds_cm = ds * 100
    widths = np.array([[levels[h].width_cm(d) for h in ("front", "back")] for d in ds_cm])
    girths = widths.sum(axis=1) / 100
    flat = summary(layouts["front"])
    waist_d = flat["waistBelowShoulderCm"] / 100
    if args.hem_cm:
        # Scale the skirt gradually from the waist so the hem matches the measurement.
        t = np.clip((ds - waist_d) / (ds[-1] - waist_d), 0, 1)
        girths = girths * (1 + t * (2 * args.hem_cm / 100 / girths[-1] - 1))
    body = build_body(avatar, frame, ds, girths, waist_d)
    parts = {"body": body}
    uvs = {"body": body_uvs(body, levels, ds_cm, offsets, atlas_size)}

    # Sleeve length from the shoulder seam. A tape run from the side of the
    # neck also covers the shoulder seam: from the neckline (the collar's
    # half-width) out to the shoulder point (half the shoulder width).
    neckline_half_cm = float(np.abs(body[0][:, 0]).max()) * 100
    sleeve_from_shoulder = None
    if args.sleeve_cm:
        offset = args.shoulder_cm / 2 - neckline_half_cm if args.sleeve_from == "neck" else 0.0
        sleeve_from_shoulder = args.sleeve_cm - offset
    ruffles = {}
    sleeve_len_cm = {}
    for side, name in ((-1, "sleeve_r"), (1, "sleeve_l")):  # the avatar's right arm is at -X
        # Seen from the front the -X arm is the photo's left sleeve; from the back, its right one.
        sleeves = {"front": layouts["front"].sleeves["left" if side < 0 else "right"],
                   "back": layouts["back"].sleeves["right" if side < 0 else "left"]}
        photo_cm = np.mean([sleeve_length(sleeves[h]) / layouts[h].px_per_cm for h in sleeves])
        length_cm = sleeve_from_shoulder if sleeve_from_shoulder else photo_cm
        sleeve_len_cm[name] = {"photo": round(float(photo_cm), 1), "used": round(float(length_cm), 1)}
        rings, us, seam, tiers = build_sleeve(avatar, frame, side, length_cm / 100)
        # Distance along each photo sleeve, from its shoulder point.
        s_px = {}
        for h, sl in sleeves.items():
            # The 3D sleeve runs from the armhole (the photo's seam line, s = 0)
            # to the end of the cuff (the photo sleeve's tip).
            s_px[h] = (us - seam) / (us[-1] - seam) * sl.length
        # Size: the sleeve's flat widths (front + back), never inside the arm.
        for i, u in enumerate(us):
            g = sum(float(np.interp(np.clip(s_px[h][i], 0, sl.length), sl.s, sl.hi - sl.lo)) / layouts[h].px_per_cm
                    for h, sl in sleeves.items()) / 100
            c = rings[i].mean(axis=0)
            scale = max(1.0, g / ring_length(rings[i]))
            rings[i] = c + (rings[i] - c) * scale
        parts[name] = rings
        head = int(np.argmax(us >= seam + SLEEVE_PATCHED_CM / 100))
        uvs[name] = sleeve_uvs(rings, s_px, sleeves, {h: layouts[h].px_per_cm for h in sleeves}, head,
                               offsets, atlas_size)
        if args.cuff_ruffle:
            for k, tier in enumerate(tiers):
                ruffles[f"{name}_cuff{k}"] = tier

    # Collar ruffle (when the garment has one): around the base of the high
    # collar, falling outwards over the shoulders and chest.
    ys = frame.shoulder_y - ds
    if args.collar_ruffle:
        neck = body[int(np.argmin(np.abs(ds - COLLAR_RUFFLE_AT_M)))]
        center = neck.mean(axis=0)
        rel = neck - center
        neck_angles = np.arctan2(rel[:, 2], rel[:, 0]) % (2 * np.pi)
        order = np.argsort(neck_angles)
        neck_r = np.linalg.norm(rel[:, [0, 2]], axis=1)[order]
        ruffles["collar"] = frill(
            center, lambda a: np.interp(a, neck_angles[order], neck_r, period=2 * np.pi) + 0.004,
            np.array([1.0, 0, 0]), np.array([0, 0, 1.0]), np.array([0, -1.0, 0]), **COLLAR_RUFFLE_FLAT)

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

    # Ruffle colour: the lit side of the collar in the front photo.
    l = layouts["front"]
    collar = photos["front"][l.top:int(l.top + 6 * l.px_per_cm), int(l.center_x - 8 * l.px_per_cm):int(l.center_x + 8 * l.px_per_cm)]
    px = collar.reshape(-1, 3).astype(float)
    bright = px.sum(axis=1)
    trim_rgb = px[bright >= np.percentile(bright, 85)].mean(axis=0) / 255

    glb = out / f"{args.name}.glb"
    subprocess.run([
        str(BLENDER), "-b", "--factory-startup", "--python", str(Path(__file__).with_name("blender_drape.py")), "--",
        "--avatar", str(Path(args.avatar).resolve()), "--rest", str(rest_path.resolve()),
        "--fabric", str(atlas_path.resolve()), "--trim", ",".join(f"{v:.3f}" for v in trim_rgb),
        "--frames", str(args.frames),
        "--out", str(glb.resolve()), "--out-with-avatar", str((out / f"{args.name}_on_avatar.glb").resolve()),
    ], check=True, capture_output=True)

    info = {
        "name": args.name,
        "modelSource": "template_parametric",
        "generationTool": "flat_dress.py (sizes from flat-lay photos, photos as texture, "
                          + ("template ruffles, " if args.collar_ruffle or args.cuff_ruffle else "")
                          + "Blender cloth simulation)",
        "ruffles": {"collar": args.collar_ruffle, "cuff": args.cuff_ruffle},
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "photos": {"front": str(args.front), "back": str(args.back)},
        "avatar": str(args.avatar),
        "measuredCm": {"length": args.length_cm, "shoulder": args.shoulder_cm, "hem": args.hem_cm},
        "fromPhotosCm": {
            "front": flat, "back": summary(layouts["back"]),
            "hemFlatFront": round(levels["front"].width_cm(args.length_cm), 1),
            "hemFlatBack": round(levels["back"].width_cm(args.length_cm), 1),
            "sleeveLength": sleeve_len_cm,
            "sleeveTape": {"cm": args.sleeve_cm, "from": args.sleeve_from,
                           "neckToShoulderCm": round(args.shoulder_cm / 2 - neckline_half_cm, 1),
                           "fromShoulderCm": round(sleeve_from_shoulder, 1) if sleeve_from_shoulder else None},
        },
    }
    (out / f"{args.name}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info


if __name__ == "__main__":
    main()
