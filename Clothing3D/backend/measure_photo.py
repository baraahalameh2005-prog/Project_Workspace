"""Measure a garment from one photo of it laid flat, with nothing typed in
and nothing else in the picture.

  1. cut the garment out (rembg) and read its outline;
  2. measure it by kind, in pixels:
       dress / top (with or without sleeves) - Dress3D's measurements
         (dress3d/schema.py: dress_length from the shoulder at the neck,
         bust, waist, hip, hem sweep, shoulder width, neck width, armhole
         depth, waist/hip positions, sleeve length, upper arm, cuff);
       skirt  - waist, hip, hem sweep, length;
       pants  - waist, hip, rise, inseam, outseam, thigh, leg opening;
       scarf  - length and width;
       accessory - width and height.
     A laid-flat garment shows half of each girth (front and back lie on
     top of each other), so girths are twice the flat width.
  3. turn pixels into centimetres. A photo alone holds no absolute size (a
     small dress close to the camera looks like a big one further away), so
     the scale comes from the dimension that varies least between sizes:
     the shoulder width of dresses and tops (women's 37-43 cm: about +-7 %),
     the waist of skirts and trousers, the length of a scarf. The result
     says so (scaleSource "estimate", with its uncertainty); one tape
     measurement (rescale()) makes every other one exact.

All proportions come from the photo; every measurement comes with the line
it was taken along, so the page can draw it and the user can check it.
What cannot be found raises PhotoError with what to change in the photo.

  python backend/measure_photo.py photo.jpg [--kind dress] [--debug out.png]
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2  # noqa: F401  (minAreaRect, contours)
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from flat_lay import PhotoError, clean_mask, load_photo, measure_sleeve, sleeve_length
from photo_to_garment import garment_mask, row_runs

KINDS = ("dress", "top", "skirt", "pants", "scarf", "accessory")
# Front and back photos of one garment that disagree by more than this are flagged.
AGREE_CM = 2.0


@dataclass
class Measured:
    kind: str
    px_per_cm: float
    cm: dict  # name -> centimetres
    lines: dict  # name -> [[x, y], ...] in the photo, pixels
    estimated: list = field(default_factory=list)  # names that are estimates, not read off the outline
    notes: list = field(default_factory=list)


# ---------------------------------------------------------------- helpers

def _run_through(mask, r, x):
    """The horizontal run of row r that contains column x, or None."""
    for a, b in row_runs(mask[r], 3):
        if a <= x <= b:
            return a, b
    return None


def _smooth(values, size):
    return ndimage.median_filter(values, size=max(3, int(size) | 1), mode="nearest")


def _piecewise3(u, y, i_range=None, j_range=None):
    """Best fit of three straight pieces to y(u): the two breakpoints
    (indices), by least squares over a coarse grid. i_range / j_range limit
    where each breakpoint may be, in u units."""
    n = len(u)
    cu, cy = np.cumsum(np.r_[0, u]), np.cumsum(np.r_[0, y])
    cuu, cuy, cyy = np.cumsum(np.r_[0, u * u]), np.cumsum(np.r_[0, u * y]), np.cumsum(np.r_[0, y * y])

    def sse(i, j):  # residual of a line through points i..j-1
        k = j - i
        if k < 2:
            return 0.0
        su, sy, suu, suy, syy = cu[j] - cu[i], cy[j] - cy[i], cuu[j] - cuu[i], cuy[j] - cuy[i], cyy[j] - cyy[i]
        var_u = suu - su * su / k
        cov = suy - su * sy / k
        var_y = syy - sy * sy / k
        return var_y - (cov * cov / var_u if var_u > 1e-9 else 0.0)

    step = max(1, n // 120)
    best, best_ij = np.inf, (n // 3, 2 * n // 3)
    def allowed(k, rng):
        return rng is None or rng[0] <= u[k] <= rng[1]

    for i in range(2 * step, n - 4 * step, step):
        if not allowed(i, i_range):
            continue
        for j in range(i + 2 * step, n - 2 * step, step):
            if not allowed(j, j_range):
                continue
            cost = sse(0, i) + sse(i, j) + sse(j, n)
            if cost < best:
                best, best_ij = cost, (i, j)
    return best_ij


def _hem_sweep(mask, x0, x1, ppcm):
    """Length of the bottom outline between columns x0 and x1 (the hem as it lies)."""
    cols = np.arange(int(x0), int(x1) + 1)
    bottom = np.array([np.flatnonzero(mask[:, c])[-1] if mask[:, c].any() else np.nan for c in cols], float)
    ok = ~np.isnan(bottom)
    cols, bottom = cols[ok], _smooth(bottom[ok], ppcm)
    pts = np.stack([cols, bottom], 1)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()) / ppcm, pts


def _bounds(mask):
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    return rows[0], rows[-1], cols[0], cols[-1]


def _armpit(mask, cx, side, start, stop, ppcm):
    """Topmost row where a sleeve lies clear of the body on this side and stays
    clear for 4 cm further down, with the body there at least 15 cm wide
    (gaps around a ruffled collar close again within a few rows). Returns (x, y)."""
    def gap(r):
        runs = row_runs(mask[r], 3)
        body = [run for run in runs if run[0] <= cx <= run[1]]
        if not body or body[0][1] - body[0][0] < 15 * ppcm:
            return None
        b0, b1 = body[0]
        outside = [run for run in runs
                   if (run[1] < b0 if side == "left" else run[0] > b1) and run[1] - run[0] >= 3 * ppcm]
        if not outside:
            return None
        return (outside[-1][1], b0) if side == "left" else (b1, outside[0][0])

    hold = int(4 * ppcm)
    for r in range(int(start), int(stop)):
        g = gap(r)
        if g and all(gap(r + k) for k in range(hold // 4, hold + 1, max(1, hold // 4))):
            return np.array([(g[0] + g[1]) / 2, float(r)])
    raise PhotoError(f"Could not find the {side} armpit: lay the sleeves a little away from the body.", "armpit")


# ---------------------------------------------------------------- kinds

def guess_kind(mask, ppcm):
    """A first guess from the outline; the user confirms it."""
    top, last, left, right = _bounds(mask)
    height_cm = (last - top) / ppcm
    center = (left + right) / 2
    # Two legs: low rows split into two wide runs either side of the centre.
    low = range(int(top + 0.7 * (last - top)), int(last))
    split = sum(1 for r in low if len([ab for ab in row_runs(mask[r], int(6 * ppcm))]) >= 2)
    if split > 0.6 * len(low):
        return "pants"
    try:
        body = _run_through(mask, int(top + 0.6 * (last - top)), int(center))
        mid = (body[0] + body[1]) / 2 if body else center
        _armpit(mask, mid, "left", top + 6 * ppcm, top + 0.5 * (last - top), ppcm)
        _armpit(mask, mid, "right", top + 6 * ppcm, top + 0.5 * (last - top), ppcm)
        return "dress" if height_cm > 85 else "top"
    except PhotoError:
        pass
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    if cv2.contourArea(c) > 0.9 * rw * rh:
        return "scarf"
    widths = mask.sum(1)
    top_w = np.median(widths[top + int(0.05 * (last - top)):top + int(0.15 * (last - top))])
    return "skirt" if height_cm < 95 and top_w < 0.8 * widths[top:last].max() else "dress"


def measure_bodice(mask, ppcm, kind):
    """Dress or top, with or without sleeves: Dress3D's measurements."""
    top, last, left, right = _bounds(mask)
    rows = np.arange(mask.shape[0])
    estimated, notes, lines = [], [], {}

    # Centre: middle of the body run at 60% down (below any sleeve), then of the bodice.
    probe = _run_through(mask, int(top + 0.6 * (last - top)), int((left + right) / 2))
    if probe is None:
        raise PhotoError("Could not find the garment's middle; lay it flat and straight.")
    cx = (probe[0] + probe[1]) / 2

    try:
        armpit = {s: _armpit(mask, cx, s, top + 6 * ppcm, top + 0.6 * (last - top), ppcm)
                  for s in ("left", "right")}
        sleeves = True
    except PhotoError:
        sleeves = False
        # Sleeveless: the armhole bottom is where the width stops growing fast.
        widths = np.array([(lambda run: run[1] - run[0] if run else 0)(_run_through(mask, r, int(cx)))
                           for r in range(top, last)], float)
        grow = np.gradient(_smooth(widths, 2 * ppcm))
        start = int(5 * ppcm)
        peak = start + int(np.argmax(grow[start:int(45 * ppcm)]))
        flat = peak + int(np.argmax(grow[peak:int(50 * ppcm)] < 0.3))
        r = top + flat
        run = _run_through(mask, r, int(cx))
        armpit = {"left": np.array([run[0], float(r)]), "right": np.array([run[1], float(r)])}
    mids = []
    for r in range(int(max(a[1] for a in armpit.values()) + 2 * ppcm), int(min(last, top + 0.45 * (last - top)))):
        run = _run_through(mask, r, int(cx))
        if run:
            mids.append((run[0] + run[1]) / 2)
    if mids:
        cx = float(np.median(mids))

    # Top outline, centre outwards on each side: collar/neck, shoulder line, then the sleeve.
    top_y = np.array([np.flatnonzero(mask[:, c])[0] if mask[:, c].any() else np.nan for c in range(mask.shape[1])])
    hps, shoulder = {}, {}
    for side, sign in (("left", -1), ("right", 1)):
        reach = abs(armpit[side][0] - cx)
        span = reach + 24 * ppcm if sleeves else reach
        xs = np.arange(int(cx), int(cx + sign * span), sign)
        ys = top_y[xs]
        ok = ~np.isnan(ys)
        xs, ys = xs[ok], _smooth(ys[ok], ppcm / 2)
        u = np.abs(xs - cx).astype(float)
        # The neck point lies between 3 cm out and 4 cm short of the armpit;
        # the shoulder point (where the sleeve starts) near above the armpit.
        # Without these limits a ruffled collar or puffed shoulder is taken for the kink.
        i_range = (3 * ppcm, reach - 4 * ppcm)
        if sleeves:
            i, j = _piecewise3(u, ys, i_range, (reach - 4 * ppcm, reach + 6 * ppcm))
        else:
            i, j = _piecewise3(u, ys, i_range)
            j = len(u) - 1
        hps[side], shoulder[side] = u[i], u[j]
        if sleeves:
            # The shoulder point is where the shoulder line meets the line of
            # the sleeve's top edge: both are long straight stretches, so a
            # puff or a gather at the seam does not move it. Shoulder line:
            # from 3 cm past the neck point to 3 cm short of the armpit;
            # sleeve top: from 6 to 22 cm past the armpit.
            def fit(lo, hi):
                sel = (u >= lo) & (u <= hi)
                return np.polyfit(u[sel], ys[sel], 1) if sel.sum() > ppcm else None
            line_s = fit(u[i] + 3 * ppcm, reach - 3 * ppcm)
            line_v = fit(reach + 6 * ppcm, reach + 22 * ppcm)
            if line_s is not None and line_v is not None and abs(line_v[0] - line_s[0]) > 0.15:
                cross = (line_s[1] - line_v[1]) / (line_v[0] - line_s[0])
                if reach - 4 * ppcm <= cross <= reach + 6 * ppcm:
                    shoulder[side] = cross
    # A garment is symmetric: one distance from the centre for both sides
    # (a puffed shoulder or a sleeve laid differently moves one side's kink).
    for points in (hps, shoulder):
        d = (points["left"] + points["right"]) / 2
        for side, sign in (("left", -1), ("right", 1)):
            x = int(round(cx + sign * d))
            points[side] = np.array([float(x), float(_smooth(top_y[x - int(ppcm):x + int(ppcm) + 1], 3)[int(ppcm)])])
    hps_y = (hps["left"][1] + hps["right"][1]) / 2
    armpit_y = max(a[1] for a in armpit.values())

    # Hem at the centre and its sweep between the lowest side corners.
    hem_y = float(np.flatnonzero(mask[:, int(cx)])[-1])
    low = int(hem_y - 0.12 * (hem_y - hps_y))
    xs_low = np.flatnonzero(mask[low:int(hem_y) + 1].any(0))
    hem_flat, hem_pts = _hem_sweep(mask, xs_low[0], xs_low[-1], ppcm)

    def width_at(r):
        run = _run_through(mask, int(r), int(cx))
        if not run:
            raise PhotoError("The garment's sides could not be followed; lay it flat with nothing on it.")
        return run, (run[1] - run[0]) / ppcm

    cm = {}
    cm["dress_length"] = (hem_y - hps_y) / ppcm
    lines["dress_length"] = [[cx, hps_y], [cx, hem_y]]
    bust_r = armpit_y + 2.5 * ppcm
    run, w = width_at(bust_r)
    cm["bust"] = 2 * w
    lines["bust"] = [[run[0], bust_r], [run[1], bust_r]]
    torso = np.arange(int(bust_r + 3 * ppcm), int(min(hem_y - 2 * ppcm, armpit_y + 35 * ppcm)))
    widths = np.array([width_at(r)[1] for r in torso])
    waist_r = int(torso[np.argmin(_smooth(widths, ppcm))])
    run, w = width_at(waist_r)
    cm["waist"] = 2 * w
    lines["waist"] = [[run[0], waist_r], [run[1], waist_r]]
    cm["waist_position"] = (waist_r - hps_y) / ppcm
    hip_r = waist_r + 20 * ppcm
    if hip_r < hem_y - 2 * ppcm:
        run, w = width_at(hip_r)
        cm["hip"] = 2 * w
        cm["hip_position"] = (hip_r - hps_y) / ppcm
        lines["hip"] = [[run[0], hip_r], [run[1], hip_r]]
        notes.append({"code": "hip_below_waist", "cm": 20})
    cm["hem"] = 2 * hem_flat
    lines["hem"] = hem_pts[:: max(1, len(hem_pts) // 80)].tolist()
    shoulder_y = (shoulder["left"][1] + shoulder["right"][1]) / 2
    cm["length_from_shoulder"] = (hem_y - shoulder_y) / ppcm  # shoulder seam to hem (flat_dress.py's length)
    lines["length_from_shoulder"] = [[shoulder["right"][0], shoulder_y], [shoulder["right"][0], hem_y]]
    cm["shoulder_width"] = float(np.linalg.norm(shoulder["right"] - shoulder["left"])) / ppcm
    lines["shoulder_width"] = [shoulder["left"].tolist(), shoulder["right"].tolist()]
    cm["neck_width"] = float(abs(hps["right"][0] - hps["left"][0])) / ppcm
    lines["neck_width"] = [hps["left"].tolist(), hps["right"].tolist()]
    cm["armhole_depth"] = (armpit_y - hps_y) / ppcm
    lines["armhole_depth"] = [[armpit["left"][0], hps_y], [armpit["left"][0], armpit_y]]

    if sleeves:
        lengths, uppers, cuffs = [], [], []
        for side in ("left", "right"):
            sl = measure_sleeve(mask, cx, shoulder[side], armpit[side], side)
            lengths.append(sleeve_length(sl) / ppcm)
            widths = sl.hi - sl.lo
            uppers.append(2 * float(np.interp(6 * ppcm, sl.s, widths)) / ppcm)
            cuffs.append(2 * float(np.median(widths[-max(2, len(widths) // 15):])) / ppcm)
            lines[f"sleeve_length_{side}"] = [shoulder[side].tolist(), (sl.origin + sl.length * sl.axis).tolist()]
        if abs(lengths[0] - lengths[1]) > AGREE_CM:
            notes.append({"code": "sleeves_differ", "values": [round(v, 1) for v in lengths]})
        cm["sleeve_length"] = float(np.mean(lengths))
        cm["upper_arm"] = float(np.mean(uppers))
        cm["cuff"] = float(np.mean(cuffs))
    style = {"sleeve": "none" if not sleeves else ("short" if cm["sleeve_length"] < 35 else "long")}
    flare = cm["hem"] / max(cm.get("hip", cm["waist"]), 1)
    style["silhouette"] = "straight" if flare < 1.15 else ("a-line" if cm["waist"] > 0.85 * cm["bust"] else "fit-and-flare")
    out = Measured(kind, ppcm, {k: round(float(v), 1) for k, v in cm.items()}, lines, estimated, notes)
    out.style = style
    return out


def measure_skirt(mask, ppcm):
    top, last, left, right = _bounds(mask)
    cx = (left + right) / 2
    waist_r = int(top + 1.5 * ppcm)
    run = _run_through(mask, waist_r, int(cx))
    cx = (run[0] + run[1]) / 2
    hem_y = float(np.flatnonzero(mask[:, int(cx)])[-1])
    cm, lines = {}, {}
    cm["waist"] = 2 * (run[1] - run[0]) / ppcm
    lines["waist"] = [[run[0], waist_r], [run[1], waist_r]]
    hip_r = int(top + 20 * ppcm)
    if hip_r < hem_y - 2 * ppcm:
        r2 = _run_through(mask, hip_r, int(cx))
        cm["hip"] = 2 * (r2[1] - r2[0]) / ppcm
        lines["hip"] = [[r2[0], hip_r], [r2[1], hip_r]]
    cm["length"] = (hem_y - top) / ppcm
    lines["length"] = [[cx, top], [cx, hem_y]]
    low = int(hem_y - 0.12 * (hem_y - top))
    xs_low = np.flatnonzero(mask[low:int(hem_y) + 1].any(0))
    flat, pts = _hem_sweep(mask, xs_low[0], xs_low[-1], ppcm)
    cm["hem"] = 2 * flat
    lines["hem"] = pts[:: max(1, len(pts) // 80)].tolist()
    return Measured("skirt", ppcm, {k: round(float(v), 1) for k, v in cm.items()}, lines,
                    notes=[{"code": "hip_below_waist", "cm": 20}])


def measure_pants(mask, ppcm):
    top, last, left, right = _bounds(mask)
    cx = (left + right) / 2
    cm, lines = {}, {}
    waist_r = int(top + 1.5 * ppcm)
    run = _run_through(mask, waist_r, int(cx))
    if run is None:
        raise PhotoError("Could not find the waistband; lay the trousers flat, legs straight.", "waistband")
    cx = (run[0] + run[1]) / 2
    cm["waist"] = 2 * (run[1] - run[0]) / ppcm
    lines["waist"] = [[run[0], waist_r], [run[1], waist_r]]
    # Crotch: first row where the centre is no longer garment (the legs part).
    crotch = next((r for r in range(waist_r, last) if not mask[r, int(cx)]), None)
    if crotch is None:
        raise PhotoError("Could not find where the legs part; lay the legs a little apart.", "crotch")
    cm["rise"] = (crotch - top) / ppcm
    lines["rise"] = [[cx, top], [cx, crotch]]
    hip_r = int(crotch - 0.25 * (crotch - top))
    r2 = _run_through(mask, hip_r, int(cx))
    cm["hip"] = 2 * (r2[1] - r2[0]) / ppcm
    lines["hip"] = [[r2[0], hip_r], [r2[1], hip_r]]
    legs = {}
    for side, pick in (("left", lambda runs: runs[0]), ("right", lambda runs: runs[-1])):
        thigh_r = int(crotch + 2.5 * ppcm)
        runs = row_runs(mask[thigh_r], 3)
        a, b = pick(runs)
        legs[side] = {"thigh": 2 * (b - a) / ppcm, "thigh_line": [[a, thigh_r], [b, thigh_r]]}
        # The leg: the piece of garment below the crotch that holds its thigh run.
        below = mask.copy()
        below[:crotch + 1] = False
        lab, _ = ndimage.label(below)
        leg = lab == lab[thigh_r, (a + b) // 2]
        hem_r = int(np.flatnonzero(leg.any(1))[-1])
        hem_run = pick(row_runs(leg[max(hem_r - int(3 * ppcm), crotch)], 3))  # above a slanted hem edge
        legs[side]["opening"] = 2 * (hem_run[1] - hem_run[0]) / ppcm
        # Seams along their slant: inner from the crotch, outer from the waistband's corner.
        inner_x, outer_x = (hem_run[1], hem_run[0]) if side == "left" else (hem_run[0], hem_run[1])
        legs[side]["inseam"] = float(np.hypot(hem_r - crotch, inner_x - cx)) / ppcm
        waist_outer = run[0] if side == "left" else run[1]
        legs[side]["outseam"] = float(np.hypot(hem_r - top, outer_x - waist_outer)) / ppcm
        legs[side]["inseam_line"] = [[cx, crotch], [inner_x, hem_r]]
    for key in ("thigh", "opening", "inseam", "outseam"):
        cm[key] = float(np.mean([legs[s][key] for s in legs]))
    lines["thigh"] = legs["left"]["thigh_line"]
    lines["inseam"] = legs["left"]["inseam_line"]
    return Measured("pants", ppcm, {k: round(float(v), 1) for k, v in cm.items()}, lines)


def measure_scarf(mask, ppcm):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    box = cv2.boxPoints(rect)
    a, b = np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[2] - box[1])
    long_pair = [box[0], box[1]] if a >= b else [box[1], box[2]]
    short_pair = [box[1], box[2]] if a >= b else [box[0], box[1]]
    return Measured("scarf", ppcm, {"length": round(float(max(a, b)) / ppcm, 1), "width": round(float(min(a, b)) / ppcm, 1)},
                    {"length": [[float(v) for v in p] for p in long_pair], "width": [[float(v) for v in p] for p in short_pair]},
                    notes=[{"code": "scarf_open"}])


def measure_accessory(mask, ppcm):
    """A bag, belt, hat...: its outline's width and height as it lies (the
    page sizes a ready-made shape of its kind with them)."""
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    w, h = (cols[-1] - cols[0]) / ppcm, (rows[-1] - rows[0]) / ppcm
    return Measured("accessory", ppcm, {"width": round(float(w), 1), "height": round(float(h), 1)},
                    {"width": [[float(cols[0]), float(rows[-1])], [float(cols[-1]), float(rows[-1])]],
                     "height": [[float(cols[-1]), float(rows[0])], [float(cols[-1]), float(rows[-1])]]})


def measure_mask(mask, ppcm, kind=None):
    kind = kind or guess_kind(mask, ppcm)
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    try:
        if kind in ("dress", "top"):
            return measure_bodice(mask, ppcm, kind)
        if kind == "accessory":
            return measure_accessory(mask, ppcm)
        return {"skirt": measure_skirt, "pants": measure_pants, "scarf": measure_scarf}[kind](mask, ppcm)
    except PhotoError:
        raise
    except (ValueError, IndexError, TypeError) as error:
        raise PhotoError(
            f"Could not read this {kind}'s shape. Lay it flat and straight, sleeves/legs a little away "
            "from the body, and shoot from straight above."
        ) from error


# Scale. A first, rough one (a typical size for the kind) only to find the
# garment's parts; then the size prior on its steadiest dimension.
ROUGH_HEIGHT_CM = {"dress": 120, "top": 65, "skirt": 70, "pants": 100}
ROUGH_LONG_SIDE_CM = {"scarf": 180, "accessory": 30}
# (measurement, typical cm, relative uncertainty) per kind.
SIZE_PRIOR = {
    "dress": ("shoulder_width", 40.0, 0.07),
    "top": ("shoulder_width", 40.0, 0.07),
    "skirt": ("waist", 74.0, 0.12),
    "pants": ("waist", 78.0, 0.12),
    "scarf": ("length", 180.0, 0.25),
    "accessory": ("width", 30.0, 0.35),
}
SLEEVELESS_PRIOR = ("bust", 92.0, 0.10)


def rough_scale(mask, kind):
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    h, w = rows[-1] - rows[0], cols[-1] - cols[0]
    if kind in ROUGH_LONG_SIDE_CM:
        return max(h, w) / ROUGH_LONG_SIDE_CM[kind]
    return h / ROUGH_HEIGHT_CM.get(kind or "dress", 100)


def measure_photo(image, session, kind=None):
    """(Measured, image, mask) for one photo, its centimetres estimated from
    the size prior (result.scale_source "estimate", result.uncertainty)."""
    mask = clean_mask(garment_mask(image, session))
    ppcm = rough_scale(mask, kind)
    result = measure_mask(mask, ppcm, kind)
    kind = kind or result.kind
    name, typical, uncertainty = SIZE_PRIOR[kind]
    if name not in result.cm:  # a sleeveless dress or top has no shoulder seams to go by
        name, typical, uncertainty = SLEEVELESS_PRIOR
    ppcm *= result.cm[name] / typical
    result = measure_mask(mask, ppcm, kind)  # again: the parts are found at the right scale
    result.guess = guess_kind(mask, ppcm)
    result.scale_source, result.uncertainty, result.scale_by = "estimate", uncertainty, name
    return result, image, mask


LENGTHS = ("dress_length", "length_from_shoulder", "length", "outseam", "inseam", "width", "height")


def rescale(cm, known):
    """The factor that makes the measurements exact, from one measurement
    {name: cm} taken with a tape."""
    (name, value), = known.items()
    if not cm.get(name):
        raise ValueError(f"{name} was not measured in this photo.")
    return float(value) / cm[name]


def combine(results):
    """One set of measurements from several photos of the same garment: the
    mean, with a note wherever they disagree by more than AGREE_CM."""
    names = sorted(set().union(*(r.cm for r in results)))
    cm, notes = {}, []
    for name in names:
        values = [r.cm[name] for r in results if name in r.cm]
        cm[name] = round(float(np.mean(values)), 1)
        if max(values) - min(values) > AGREE_CM:
            notes.append({"code": "photos_differ", "measurement": name, "values": values})
    return cm, notes


def draw(image, result, path):
    img = image.copy()
    d = ImageDraw.Draw(img)
    for name, pts in result.lines.items():
        d.line([tuple(p) for p in pts], fill="red", width=4)
        d.text(tuple(pts[0]), f"{name} {result.cm.get(name.rsplit('_', 1)[0] if name.startswith('sleeve') else name, '')}",
               fill="blue")
    img.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo")
    parser.add_argument("--kind", choices=KINDS)
    parser.add_argument("--debug")
    args = parser.parse_args()

    from rembg import new_session

    image = load_photo(args.photo)
    result, image, _ = measure_photo(image, new_session("u2net"), args.kind)
    if args.debug:
        draw(image, result, args.debug)
    print(json.dumps({"kind": result.kind, "guess": result.guess, "pxPerCm": round(result.px_per_cm, 2),
                      "cm": result.cm, "scale": [result.scale_source, result.scale_by, result.uncertainty],
                      "notes": result.notes, "style": getattr(result, "style", None)}, indent=2, default=float))


if __name__ == "__main__":
    main()
