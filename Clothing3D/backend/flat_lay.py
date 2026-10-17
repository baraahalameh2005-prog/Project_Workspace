"""Read a garment photographed laid flat (from above, on a plain cloth):
outline, collar, shoulders, armholes, sleeves, waist and hem, in centimetres.

A flat-laid dress shows its real flat widths: around the body, the fabric at
any level measures twice the width laid flat. The skirt is laid out as a fan,
so its "levels" are curves from one side seam through the centre to the other
(at the hem, the hem itself), not image rows.

Example (writes a debug image next to the output):
  python backend/flat_lay.py samples/flat_front.jpg --length-cm 130 --shoulder-cm 42
"""

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage

from photo_to_garment import garment_mask, row_runs


MAX_SIDE_PX = 2048  # bigger phone photos are scaled down (time and memory)
MASK_OPEN_PX = 4  # thin things attached to the dress (a hanger's hook, threads) are cut off
# A believable garment mask covers this share of the photo...
MASK_AREA = (0.08, 0.85)
# ...and touches at most this share of the photo's border (it is all in the picture).
MASK_BORDER = 0.15


class PhotoError(ValueError):
    """The photo cannot be used; the message says why and what to change.
    `code` names the problem for pages that word it in their own language."""

    def __init__(self, message, code="shape"):
        super().__init__(message)
        self.code = code


def load_photo(path_or_file):
    """Open a photo upright (phones store the rotation separately) and not
    larger than MAX_SIDE_PX."""
    image = ImageOps.exif_transpose(Image.open(path_or_file)).convert("RGB")
    image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
    return image


def normalize_light(image):
    """Even out shop lighting before segmentation: grey-world white balance,
    then CLAHE (local contrast) on the lightness only, so dark tweed keeps
    its edge against the cloth under it. Used only for finding the dress
    (see find_garment), never for its texture."""
    from skimage import color, exposure

    rgb = np.asarray(image).astype(float) / 255
    rgb = np.clip(rgb * (rgb.mean() / np.maximum(rgb.mean(axis=(0, 1)), 1e-3)), 0, 1)
    lab = color.rgb2lab(rgb)
    lab[..., 0] = exposure.equalize_adapthist(lab[..., 0] / 100, clip_limit=0.01) * 100
    return Image.fromarray((np.clip(color.lab2rgb(lab), 0, 1) * 255).astype(np.uint8))


def clean_mask(mask):
    """Open away thin attachments, keep the largest piece, fill its holes,
    then check that it looks like a whole garment on a plain background."""
    opened = ndimage.binary_opening(mask, structure=np.ones((3, 3)), iterations=MASK_OPEN_PX)
    labels, count = ndimage.label(opened)
    if count == 0:
        raise PhotoError("No garment found in the photo.", "no_garment")
    sizes = ndimage.sum(opened, labels, range(1, count + 1))
    keep = labels == (int(np.argmax(sizes)) + 1)
    # Give back the edge the opening took, but only next to the kept piece.
    keep = ndimage.binary_dilation(keep, iterations=MASK_OPEN_PX) & mask
    keep = ndimage.binary_fill_holes(ndimage.binary_closing(keep, iterations=2))

    area = keep.mean()
    if not MASK_AREA[0] <= area <= MASK_AREA[1]:
        raise PhotoError(
            f"The garment covers {area:.0%} of the photo: either the background was taken for "
            "the garment or the garment was missed. Lay it on a plain sheet and fill most of the picture.",
            "garment_area",
        )
    border = np.concatenate([keep[0], keep[-1], keep[:, 0], keep[:, -1]]).mean()
    if border > MASK_BORDER:
        raise PhotoError(
            f"The garment touches {border:.0%} of the photo's edge: step back so all of it, "
            "with a little space around it, is in the picture.",
            "garment_border",
        )
    return keep


def find_garment(image, session):
    """Garment mask from the photo as shot; only if that fails the checks,
    once more from the light-normalized photo (it helps with dim or tinted
    light, but on a well-lit photo it can make the cutter miss a sleeve)."""
    try:
        return clean_mask(garment_mask(image, session))
    except PhotoError as first:
        try:
            return clean_mask(garment_mask(normalize_light(image), session))
        except PhotoError:
            raise first


@dataclass
class FlatSleeve:
    seam: tuple  # (shoulder point, armpit point), image (x, y)
    axis: np.ndarray  # unit vector from the seam towards the cuff, image (x, y)
    across: np.ndarray  # unit vector across the sleeve
    origin: np.ndarray  # where the axis crosses the seam line
    s: np.ndarray  # distances along the axis, pixels
    lo: np.ndarray  # extent across the sleeve at each s (min, max), pixels
    hi: np.ndarray

    @property
    def length(self):
        return float(self.s[-1])

    def edges(self, s):
        """The two edge points of the sleeve at distance s along it."""
        lo = np.interp(s, self.s, self.lo)
        hi = np.interp(s, self.s, self.hi)
        base = self.origin + s * self.axis
        return base + lo * self.across, base + hi * self.across


@dataclass
class FlatLayout:
    center_x: float
    top: int  # collar top row
    shoulder: dict  # side -> shoulder point (x, y)
    armpit: dict  # side -> armpit point (x, y)
    hem_center: float  # row of the hem at the centre
    left_edge: np.ndarray  # body outline x per row (NaN where unknown)
    right_edge: np.ndarray
    hem_y: np.ndarray  # lowest garment row per column (NaN outside)
    sleeves: dict = field(default_factory=dict)
    px_per_cm: float = 0.0


def outline(mask):
    """First/last garment column per row and lowest garment row per column."""
    h, w = mask.shape
    rows = np.arange(h)
    any_row = mask.any(axis=1)
    left = np.where(any_row, mask.argmax(axis=1), np.nan)
    right = np.where(any_row, w - 1 - mask[:, ::-1].argmax(axis=1), np.nan)
    any_col = mask.any(axis=0)
    bottom = np.where(any_col, h - 1 - mask[::-1].argmax(axis=0), np.nan)
    return rows, left, right, bottom


def find_armpit(mask, center_x, side, start, stop, min_run):
    """Topmost row (from `start`, below the collar) where a sleeve lies clear
    of the body on this side: the armpit is where the gap between them
    begins. Returns (x, y)."""
    for r in range(start, stop):
        runs = row_runs(mask[r], min_len=3)
        body = [run for run in runs if run[0] <= center_x <= run[1]]
        if not body:
            continue
        b0, b1 = body[0]
        outside = [run for run in runs
                   if (run[1] < b0 if side == "left" else run[0] > b1) and run[1] - run[0] >= min_run]
        if outside:
            gap = (outside[-1][1], b0) if side == "left" else (b1, outside[0][0])
            return np.array([(gap[0] + gap[1]) / 2, float(r)])
    raise PhotoError(f"Could not find the {side} armpit: lay the sleeves a little away from the body.", "armpit")


def measure_sleeve(mask, center_x, shoulder, armpit, side):
    """The sleeve: garment pixels beyond the armhole line (shoulder point to
    armpit point), with its axis and its extent across, along its length."""
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], axis=1).astype(float)
    seam_dir = (armpit - shoulder) / np.linalg.norm(armpit - shoulder)
    normal = np.array([seam_dir[1], -seam_dir[0]])
    if (normal[0] > 0) != (side == "right"):
        normal = -normal  # pointing away from the body
    beyond = (pts - shoulder) @ normal > 0
    # Below the armpit the body is the run through the centre; the sleeve is
    # what lies clear of it.
    body_img = np.zeros(mask.shape, bool)
    for r in range(int(armpit[1]) + 1, mask.shape[0]):
        body = [run for run in row_runs(mask[r], 3) if run[0] <= center_x <= run[1]]
        if body:
            body_img[r, body[0][0]:body[0][1] + 1] = True
    in_body = body_img[pts[:, 1].astype(int), pts[:, 0].astype(int)]
    sleeve = pts[beyond & ~in_body]
    labels = np.zeros(mask.shape, bool)
    labels[sleeve[:, 1].astype(int), sleeve[:, 0].astype(int)] = True
    lab, n = ndimage.label(labels)
    if n == 0:
        raise PhotoError(f"No {side} sleeve found.", "sleeve")
    sizes = ndimage.sum(labels, lab, range(1, n + 1))
    keep = lab == (int(np.argmax(sizes)) + 1)
    ys, xs = np.nonzero(keep)
    sleeve = np.stack([xs, ys], axis=1).astype(float)

    # Axis: principal direction of the sleeve pixels, pointing away from the seam.
    mean = sleeve.mean(axis=0)
    _, _, vt = np.linalg.svd(sleeve - mean, full_matrices=False)
    axis = vt[0] if vt[0] @ normal > 0 else -vt[0]
    across = np.array([-axis[1], axis[0]])
    # Origin: where the axis through the sleeve's middle meets the seam line.
    seam_mid = (shoulder + armpit) / 2
    t = ((seam_mid - mean) @ normal) / (axis @ normal)
    origin = mean + t * axis
    along = (sleeve - origin) @ axis
    side_coord = (sleeve - origin) @ across
    step = 4.0
    s_bins = np.arange(0.0, along.max() + step, step)
    s, lo, hi = [], [], []
    for a in s_bins:
        sel = np.abs(along - a) < step / 2
        if sel.sum() < 3:
            continue
        s.append(a)
        lo.append(np.percentile(side_coord[sel], 1))
        hi.append(np.percentile(side_coord[sel], 99))
    return FlatSleeve((tuple(shoulder), tuple(armpit)), axis, across, origin,
                      np.array(s), ndimage.median_filter(np.array(lo), 5), ndimage.median_filter(np.array(hi), 5))


def analyze_flat(image, session, length_cm, shoulder_cm, mask=None):
    """Layout and mask of a flat-laid garment, or PhotoError saying what to
    change in the photo. mask: the garment's mask when it is already known."""
    mask = clean_mask(mask) if mask is not None else find_garment(image, session)
    try:
        return _analyze_mask(mask, length_cm, shoulder_cm)
    except PhotoError:
        raise
    except (ValueError, IndexError) as error:
        raise PhotoError(
            "Could not read the garment's shape (collar, shoulders, sleeves, hem). Lay it flat, front "
            "or back up, sleeves a little away from the body, and shoot from straight above."
        ) from error


def _analyze_mask(mask, length_cm, shoulder_cm):
    rows, left, right, bottom = outline(mask)
    h, w = mask.shape

    # Centre: the middle of the skirt (its widest, most symmetric part).
    filled = np.flatnonzero(~np.isnan(left))
    top, last = int(filled[0]), int(filled[-1])
    skirt_rows = np.arange(top + int(0.55 * (last - top)), top + int(0.8 * (last - top)))
    center_x = float(np.median((left[skirt_rows] + right[skirt_rows]) / 2))
    # The collar's top is the first row reaching the centre.
    top = int(np.flatnonzero(mask[:, int(center_x)])[0])
    hem_center = float(bottom[int(center_x) - 3:int(center_x) + 4].max())

    # Scale from the length (shoulder seam to hem) once the shoulder line is known;
    # the shoulder line is about where the outline is half the measured
    # shoulder width out from the centre. The skirt may lie a little to one
    # side, so the centre is then taken again from the bodice (between the
    # armpits and the waist). A few passes: the first guesses the scale.
    px_per_cm = (hem_center - top) / (length_cm + 5)
    for _ in range(3):
        shoulder = {}
        for side, sign in (("left", -1), ("right", 1)):
            x = center_x + sign * shoulder_cm / 2 * px_per_cm
            col = mask[:, int(round(x))]
            shoulder[side] = np.array([x, float(np.flatnonzero(col)[0])])
        shoulder_y = (shoulder["left"][1] + shoulder["right"][1]) / 2
        px_per_cm = (hem_center - shoulder_y) / length_cm
        # Armpits: well below the shoulder line (not the gaps around the
        # collar), with a sleeve at least 3 cm wide beside the body.
        armpit = {side: find_armpit(mask, center_x, side, int(shoulder_y + 8 * px_per_cm), last, 3 * px_per_cm)
                  for side in ("left", "right")}
        bodice = range(int(max(a[1] for a in armpit.values())) + 5, int(shoulder_y + 30 * px_per_cm))
        mids = [np.mean([run for run in row_runs(mask[r], 3) if run[0] <= center_x <= run[1]][0]) for r in bodice]
        center_x = float(np.median(mids))
    top = int(np.flatnonzero(mask[:, int(center_x)])[0])
    hem_center = float(bottom[int(center_x) - 3:int(center_x) + 4].max())

    # Body outline between the armholes and the hem (sleeves excluded): the
    # run containing the centre, below the armpits; above them, the armhole lines.
    left_edge = np.full(h, np.nan)
    right_edge = np.full(h, np.nan)
    below = int(max(armpit["left"][1], armpit["right"][1]))
    for r in range(below, last + 1):
        body = [run for run in row_runs(mask[r], 3) if run[0] <= center_x <= run[1]]
        if body:
            left_edge[r], right_edge[r] = body[0]
    for side, edge in (("left", left_edge), ("right", right_edge)):
        (sx, sy), (ax, ay) = shoulder[side], armpit[side]
        for r in range(int(sy), int(ay) + 1):
            edge[r] = sx + (ax - sx) * (r - sy) / max(ay - sy, 1)
    for r in range(top, int(min(shoulder["left"][1], shoulder["right"][1]))):
        edge_l, edge_r = left[r], right[r]
        left_edge[r], right_edge[r] = max(edge_l, shoulder["left"][0]), min(edge_r, shoulder["right"][0])
    # Fill the small gaps between the armhole lines and the body runs.
    for edge in (left_edge, right_edge):
        known = np.flatnonzero(~np.isnan(edge[top:last + 1])) + top
        edge[top:last + 1] = np.interp(np.arange(top, last + 1), known, edge[known])

    sleeves = {side: measure_sleeve(mask, center_x, shoulder[side], armpit[side], side) for side in ("left", "right")}
    layout = FlatLayout(center_x, top, shoulder, armpit, hem_center, left_edge, right_edge, bottom, sleeves, px_per_cm)
    return layout, mask


def draw_flat(image, layout, path):
    img = image.copy()
    d = ImageDraw.Draw(img)
    for r in range(layout.top, int(layout.hem_center), 6):
        if not np.isnan(layout.left_edge[r]):
            d.point((layout.left_edge[r], r), fill="lime")
            d.point((layout.right_edge[r], r), fill="lime")
    for side in ("left", "right"):
        sh, ap = layout.shoulder[side], layout.armpit[side]
        d.line([tuple(sh), tuple(ap)], fill="orange", width=4)
        d.ellipse([sh[0] - 8, sh[1] - 8, sh[0] + 8, sh[1] + 8], outline="yellow", width=3)
        d.ellipse([ap[0] - 8, ap[1] - 8, ap[0] + 8, ap[1] + 8], outline="cyan", width=3)
        sl = layout.sleeves[side]
        d.line([tuple(sl.origin), tuple(sl.origin + sl.length * sl.axis)], fill="red", width=3)
        for s in np.linspace(0, sl.length, 12):
            a, b = sl.edges(s)
            d.line([tuple(a), tuple(b)], fill="cyan", width=2)
    d.line([(layout.center_x, layout.top), (layout.center_x, layout.hem_center)], fill="magenta", width=2)
    img.save(path)


def sleeve_length(sleeve):
    """From the shoulder point (top of the armhole seam) to the end of the cuff."""
    shoulder = np.array(sleeve.seam[0])
    return sleeve.length - float((shoulder - sleeve.origin) @ sleeve.axis)


def summary(layout):
    cm = 1 / layout.px_per_cm
    body_w = (layout.right_edge - layout.left_edge) * cm
    shoulder_y = (layout.shoulder["left"][1] + layout.shoulder["right"][1]) / 2
    armpit_y = max(layout.armpit["left"][1], layout.armpit["right"][1])
    torso = np.arange(int(armpit_y), int(armpit_y + 0.5 * (layout.hem_center - armpit_y)))
    waist_row = int(torso[np.nanargmin(body_w[torso])])
    return {
        "pxPerCm": round(layout.px_per_cm, 2),
        "armpitBelowShoulderCm": round((armpit_y - shoulder_y) * cm, 1),
        "chestFlatCm": round(float(body_w[int(armpit_y) + 5]), 1),
        "waistFlatCm": round(float(body_w[waist_row]), 1),
        "waistBelowShoulderCm": round((waist_row - shoulder_y) * cm, 1),
        "sleeveLengthCm": {s: round(sleeve_length(sl) * cm, 1) for s, sl in layout.sleeves.items()},
        "sleeveFlatWidthCm": {s: [round(float(np.interp(f * sl.length, sl.s, sl.hi - sl.lo)) * cm, 1)
                                  for f in (0.1, 0.5, 0.85)] for s, sl in layout.sleeves.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photo")
    parser.add_argument("--length-cm", type=float, required=True)
    parser.add_argument("--shoulder-cm", type=float, required=True)
    parser.add_argument("--debug", help="debug image path (default: output/<photo>_flat.png)")
    args = parser.parse_args()

    from rembg import new_session

    image = load_photo(args.photo)
    layout, _ = analyze_flat(image, new_session("u2net"), args.length_cm, args.shoulder_cm)
    debug = args.debug or f"output/{Path(args.photo).stem}_flat.png"
    draw_flat(image, layout, debug)
    print(json.dumps(summary(layout), indent=2))


if __name__ == "__main__":
    main()
