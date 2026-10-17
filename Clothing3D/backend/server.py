"""Local server for the clothing module: serves the try-on page and puts a
dress on the user's avatar, two ways.

  python backend/server.py            then open http://127.0.0.1:8010/viewer/
  (the avatar page links here with ?avatar=<avatar_id>)

Routes:
  GET  /viewer/, /output/..., /jobs/<id>/...   static files
  GET  /avatars/<avatarId>.glb   an avatar made by the avatar API (same origin for the viewer)
  POST /api/jobs   JSON, one of:
         {method: "flat", front, back: data URLs (image/jpeg|png) or frontId, backId,
          lengthCm, shoulderCm, hemCm?, sleeveCm?, sleeveFrom?, collarRuffle?, cuffRuffle?, avatarId?}
           -> flat_dress.py: dress read from flat-lay photos, cloth-simulated
              on the avatar (minutes)
         {method: "measurements", measurements: {dress_length, bust, ...},
          style?: {silhouette, sleeve, neckline}, units?, front?, back?, avatarId?}
           -> dress_on_avatar.py: Dress3D dress from the garment's
              measurements (photos optional, for fabric), put on the avatar
              (seconds to a minute)
       -> {id}; the job is queued
  GET  /api/jobs/<id> -> {status: queued|running|done|failed, error?, glb?, dressGlb?, info?}
  GET  /api/avatar-frame/<avatarId>  where accessories go on that avatar (metres)
  POST /api/jobs {method: "garment", kind, frontId, backId?, measurements, avatarId?,
                  collarRuffle?, cuffRuffle?}: any kind from measured photos
  POST /api/scale {photoId, known: {name: cm} | factor}: exact sizes from one tape
       measurement (otherwise they are estimated from the garment's proportions)
  POST /api/measure  JSON {photo: data URL, kind?: dress|top|skirt|pants|scarf|accessory, measure?: false}
       -> the garment measured from the photo (measure_photo.py): {photoId,
          kind, guess, pxPerCm, cm, lines, scaleSource, uncertainty, notes,
          style, trims, colour, cutoutUrl, flatUrl, size}; or 422 {error, code} saying
          what to change in the photo. measure: false only stores the photo
          (a detail shot) and returns {photoId, flatUrl}.
       A job may then name photos by id instead of sending them again:
       {..., frontId?, backId?}.

avatarId is the id of an avatar made by the avatar API (its GLB is read from
the avatar module's var/generated/); without it the test avatar
output/avatar.glb is used. Every job has its own folder jobs/<uuid>/, so
requests never share a file. Jobs run one at a time: they need most of the
machine's memory.
"""

import base64
import binascii
import json
import re
import subprocess
import sys
import threading
import uuid
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]  # the clothing folder
JOBS = ROOT / "jobs"
TEST_AVATAR = ROOT / "output" / "avatar.glb"
GENERATED_AVATARS = ROOT.parent / "var" / "generated"  # written by the avatar API (backend/api.py)
PYTHON = Path(sys.executable)
MAX_BODY_BYTES = 40 * 1024 * 1024
DATA_URL = re.compile(r"^data:image/(jpeg|png);base64,(.+)$", re.S)
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
AVATAR_URL = re.compile(r"^/avatars/([0-9a-f]{32})\.glb(?:\?.*)?$")
# Flat-lay method: accepted measurement ranges, cm.
RANGES = {"lengthCm": (40, 200), "shoulderCm": (20, 80), "hemCm": (30, 400), "sleeveCm": (10, 100)}
# Measurement method: Dress3D fields (dress3d/schema.py checks them in detail).
DRESS3D_FIELDS = ("dress_length", "bust", "waist", "hip", "hem", "shoulder_width", "neck_width",
                  "front_neck_drop", "back_neck_drop", "armhole_depth", "waist_position", "hip_position",
                  "sleeve_length", "upper_arm", "cuff")
STYLE_VALUES = {"silhouette": ("straight", "a-line", "fit-and-flare"), "sleeve": ("none", "short", "long"),
                "neckline": ("round", "v", "square", "boat")}

UPLOADS = ROOT / "uploads"
KINDS = ("dress", "top", "skirt", "pants", "scarf", "accessory")
GARMENT_KINDS = KINDS[:5]

jobs = {}  # id -> state dict
jobs_lock = threading.Lock()
run_lock = threading.Lock()  # one job at a time
measure_lock = threading.Lock()  # one rembg session, used by one request at a time
_session = None
_job_session = None  # jobs have their own, so measuring never waits for a dress
_frames = {}  # avatar id -> avatar_frame()


def rembg_session():
    """Loaded once, on first use (about a second; then ~4 s per photo)."""
    global _session
    if _session is None:
        from rembg import new_session

        _session = new_session("u2net")
    return _session


class BadRequest(ValueError):
    pass


def decode_photo(data_url):
    match = DATA_URL.match(data_url or "") if isinstance(data_url, str) else None
    if not match:
        raise BadRequest("Each photo must be a JPEG or PNG data URL.")
    try:
        raw = base64.b64decode(match.group(2), validate=True)
        Image.open(BytesIO(raw)).verify()
    except (binascii.Error, OSError, SyntaxError):
        raise BadRequest("A photo could not be read as an image.")
    return raw


def number(body, key, required):
    value = body.get(key)
    if value in (None, ""):
        if required:
            raise BadRequest(f"{key} is required.")
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise BadRequest(f"{key} must be a number.")
    lo, hi = RANGES[key]
    if not lo <= value <= hi:
        raise BadRequest(f"{key} must be between {lo} and {hi} cm.")
    return value


def avatar_path(body):
    """The avatar GLB for this job: the user's avatar from the avatar API, or the test avatar."""
    avatar_id = body.get("avatarId")
    if avatar_id in (None, ""):
        return TEST_AVATAR
    if not isinstance(avatar_id, str) or not JOB_ID.match(avatar_id):
        raise BadRequest("avatarId must be the 32-character id of an avatar.")
    path = GENERATED_AVATARS / f"{avatar_id}.glb"
    if not path.exists():
        raise BadRequest("That avatar was not found; make the avatar again on the avatar page.")
    return path


def flat_job(body, folder):
    measures = {k: number(body, k, k in ("lengthCm", "shoulderCm")) for k in RANGES}
    sleeve_from = body.get("sleeveFrom") or "shoulder"
    if sleeve_from not in ("shoulder", "neck"):
        raise BadRequest("sleeveFrom must be shoulder or neck.")
    photos = {name: uploaded_photo(body[f"{name}Id"]) if body.get(f"{name}Id") else decode_photo(body.get(name))
              for name in ("front", "back")}
    avatar = avatar_path(body)

    def write():
        for name, raw in photos.items():
            (folder / f"{name}.jpg").write_bytes(raw)

    args = ["flat_dress.py", "--avatar", str(avatar), "--front", str(folder / "front.jpg"),
            "--back", str(folder / "back.jpg"), "--length-cm", str(measures["lengthCm"]),
            "--shoulder-cm", str(measures["shoulderCm"]), "--out", str(folder), "--name", "dress"]
    if measures["hemCm"]:
        args += ["--hem-cm", str(measures["hemCm"])]
    if measures["sleeveCm"]:
        args += ["--sleeve-cm", str(measures["sleeveCm"]), "--sleeve-from", sleeve_from]
    for key, flag in (("collarRuffle", "--collar-ruffle"), ("cuffRuffle", "--cuff-ruffle")):
        if body.get(key) is True:
            args.append(flag)
    return write, args


def mask_img(mask):
    import numpy as np

    return (np.asarray(mask) * 255).astype("uint8")


def cutout(image, mask, max_side=768):
    """The garment alone on a transparent background, cropped to it."""
    import numpy as np

    mask = np.asarray(mask)
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    box = (int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1)
    rgba = image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask_img(mask)))
    out = rgba.crop(box)
    out.thumbnail((max_side, max_side), Image.LANCZOS)
    return out


TRIM_COLOUR_DISTANCE = 90  # RGB distance from the garment's colour that reads as a different trim


def trims(image, mask, result):
    """Whether the collar and the cuffs are a different colour from the
    garment (a white ruffle collar on a black dress...): the page ticks the
    ruffle options for them. Dresses and tops only."""
    import numpy as np

    if result.kind not in ("dress", "top") or "dress_length" not in result.lines:
        return {"collar": False, "cuff": False}
    rgb = np.asarray(image).astype(float)
    mask = np.asarray(mask)
    body = np.median(rgb[mask], axis=0)
    ppcm = result.px_per_cm

    from scipy import ndimage

    # Pixels far from the garment's colour, keeping only patches at least
    # ~1 cm across: a trim is a solid band, a print or tweed flecks are not.
    # The cut-out can leave out a pale trim on a pale background (a white
    # ruffle on a grey bed cover), so the 3 cm just outside it count too when
    # they differ from both the garment and the background.
    dist = np.linalg.norm(rgb - body, axis=2)
    background = np.median(rgb[~mask], axis=0) if (~mask).any() else body
    near = ndimage.binary_dilation(mask, iterations=max(1, int(3 * ppcm)))
    trim_px = (dist > TRIM_COLOUR_DISTANCE) & (mask | (near & (np.linalg.norm(rgb - background, axis=2) > 60)))
    solid = ndimage.binary_opening(trim_px, iterations=max(1, int(0.4 * ppcm)))
    mask = near  # the regions below are judged over the garment and that margin

    def differs(x0, x1, y0, y1, share=0.35):
        h, w = mask.shape
        x0, x1, y0, y1 = (int(np.clip(v, 0, lim)) for v, lim in ((x0, w), (x1, w), (y0, h), (y1, h)))
        region = mask[y0:y1, x0:x1]
        if region.sum() < 20:
            return False
        return bool(solid[y0:y1, x0:x1][region].mean() > share)

    cx = result.lines["dress_length"][0][0]
    # The top 10 cm of the garment round its middle: a collar trim is a ring
    # round the (often same-colour) collar stand there.
    top = float(np.flatnonzero(mask[:, int(cx)])[0]) if mask[:, int(cx)].any() else 0.0
    collar = differs(cx - 13 * ppcm, cx + 13 * ppcm, top - 2 * ppcm, top + 10 * ppcm, share=0.05)
    cuffs = []
    for side in ("left", "right"):
        line = result.lines.get(f"sleeve_length_{side}")
        if line:
            (sx, sy), (ex, ey) = line
            d = np.array([ex - sx, ey - sy]) / max(np.hypot(ex - sx, ey - sy), 1)
            cx2, cy2 = np.array([ex, ey]) - d * 4 * ppcm  # 4 cm up the sleeve from its end
            cuffs.append(differs(cx2 - 4 * ppcm, cx2 + 4 * ppcm, cy2 - 4 * ppcm, cy2 + 4 * ppcm))
    return {"collar": collar, "cuff": bool(cuffs) and all(cuffs)}


def main_colour(image, mask):
    """The garment's typical colour, #rrggbb (median of its pixels)."""
    import numpy as np

    rgb = np.asarray(image)[np.asarray(mask)]
    r, g, b = (int(v) for v in np.median(rgb, axis=0))
    return f"#{r:02x}{g:02x}{b:02x}"


def upload(photo_id):
    """(folder, measured.json) of a photo measured by /api/measure."""
    if not isinstance(photo_id, str) or not JOB_ID.match(photo_id):
        raise BadRequest("A photo id must be 32 hex characters.")
    folder = UPLOADS / photo_id
    meta = folder / "measured.json"
    if not meta.exists() or not (folder / "mask.png").exists():
        raise BadRequest("That photo was not measured; add it again.")
    return folder, json.loads(meta.read_text(encoding="utf-8"))


def garment_job(body, folder):
    """method "garment": any kind, from photos measured by /api/measure.

    dress/top with sleeves -> flat_dress.py (photo texture, cloth simulation);
    sleeveless dress/top    -> Dress3D (dress_on_avatar.make);
    skirt/pants/scarf       -> garment_kinds.make."""
    kind = body.get("kind")
    if kind not in GARMENT_KINDS:
        raise BadRequest(f"kind must be one of {GARMENT_KINDS}.")
    raw = body.get("measurements") or {}
    if not isinstance(raw, dict):
        raise BadRequest("measurements must be an object.")
    cm = {}
    for key, value in raw.items():
        if value in (None, ""):
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise BadRequest(f"{key} must be a number.")
        if not 1 <= value <= 500:
            raise BadRequest(f"{key} must be between 1 and 500 cm.")
        cm[key] = value
    front_dir, front_meta = upload(body.get("frontId"))
    back_dir = upload(body["backId"])[0] if body.get("backId") else None
    avatar = avatar_path(body)

    if kind in ("dress", "top"):
        sleeves = (front_meta.get("style") or {}).get("sleeve", "none") != "none" and cm.get("sleeve_length")
        if sleeves:
            for key in ("length_from_shoulder", "shoulder_width"):
                if key not in cm:
                    raise BadRequest(f"{key} is required.")

            # flat_dress.run in this process, with the masks /api/measure made.
            task = {"type": "flat", "avatar": avatar, "front": front_dir, "back": back_dir or front_dir,
                    "length_cm": cm["length_from_shoulder"], "shoulder_cm": cm["shoulder_width"],
                    "sleeve_cm": cm["sleeve_length"], "hem_cm": cm["hem"] / 2 if cm.get("hem") else None,
                    "collar_ruffle": body.get("collarRuffle") is True, "cuff_ruffle": body.get("cuffRuffle") is True}
            return (lambda: None), task
        measurements = {k: v for k, v in cm.items() if k in DRESS3D_FIELDS}
        if "dress_length" not in measurements or "bust" not in measurements:
            raise BadRequest("dress_length and bust are required.")
        style = dict(front_meta.get("style") or {})
        task = {"type": "dress3d", "avatar": avatar, "measurements": measurements, "style": style, "units": "cm",
                "photos": {"front": front_dir / "garment.jpg", **({"back": back_dir / "garment.jpg"} if back_dir else {})}}
        return (lambda: None), task

    task = {"type": "kind", "kind": kind, "avatar": avatar, "cm": cm,
            "photos": {"front": front_dir, **({"back": back_dir} if back_dir else {})}}
    return (lambda: None), task


def avatar_frame(avatar_id):
    """Where things go on this avatar (metres, glTF axes: +Y up, +Z front),
    for the page's accessories. Cached per avatar."""
    if avatar_id in _frames:
        return _frames[avatar_id]
    import numpy as np
    import trimesh

    import garment_kinds as gk
    from photo_to_garment import arm_centerline, measure_avatar
    from template_dress import neck_points, ring_dirs

    path = avatar_path({"avatarId": avatar_id})
    mesh = trimesh.load(path, force="scene").to_geometry()
    frame = measure_avatar(mesh)
    dirs = ring_dirs()
    angles = np.arctan2(dirs[:, 1], dirs[:, 0])
    waist_y = gk.natural_waist_y(mesh, frame, angles)
    c, r = gk.torso_ring(mesh, frame, waist_y, angles)
    waist = c + (r + 0.008)[:, None] * dirs
    neck_y = frame.shoulder_y + 0.02
    neck = neck_points(mesh, frame, neck_y)
    nc, nr = gk.hull_ring(neck, angles) if neck is not None else (np.zeros(2), np.full(len(angles), 0.06))
    neck_ring = nc + (nr + 0.01)[:, None] * dirs
    top = float(mesh.bounds[1][1])
    head = [l for l in gk.section_loops(mesh, top - 0.10) if l[:, 0].min() < 0 < l[:, 0].max()]
    hp = np.concatenate(head) if head else np.zeros((1, 3))
    from dress_on_avatar import body_girths

    chest = gk.torso_ring(mesh, frame, frame.armpit_y - 0.04, angles)
    cc, cr = chest
    chest_pts = cc + cr[:, None] * dirs

    girths = {k: (round(v, 1) if v else None) for k, v in body_girths(mesh, frame).items()}
    hands = {}
    for side, name in ((-1, "right"), (1, "left")):
        line = arm_centerline(mesh, frame, side)
        hands[name] = [round(float(v), 4) for v in line[-1]]
    out = {
        "height": round(top, 4), "shoulderY": round(frame.shoulder_y, 4), "armpitY": round(frame.armpit_y, 4),
        "waistY": round(waist_y, 4), "waistRing": [[round(float(x), 4), round(float(z), 4)] for x, z in waist],
        "neckY": round(neck_y, 4), "neckRing": [[round(float(x), 4), round(float(z), 4)] for x, z in neck_ring],
        "head": {"topY": round(top, 4), "centerX": round(float(hp[:, 0].mean()), 4),
                 "centerZ": round(float(hp[:, 2].mean()), 4),
                 "radius": round(float(np.ptp(hp[:, 0]) / 2), 4), "frontZ": round(float(hp[:, 2].max()), 4)},
        "hands": hands,
        "bodyGirthsCm": girths,
        "chest": {"y": round(frame.armpit_y - 0.04, 4), "frontZ": round(float(chest_pts[:, 1].max()), 4),
                  "backZ": round(float(chest_pts[:, 1].min()), 4), "halfX": round(float(np.abs(chest_pts[:, 0]).max()), 4)},
    }
    _frames[avatar_id] = out
    return out


def uploaded_photo(photo_id):
    """The photo stored by /api/measure."""
    if not isinstance(photo_id, str) or not JOB_ID.match(photo_id):
        raise BadRequest("A photo id must be 32 hex characters.")
    path = UPLOADS / photo_id / "garment.jpg"
    if not path.exists():
        raise BadRequest("That photo was not found; add it again.")
    return path.read_bytes()


def measure(body):
    """POST /api/measure: (status, payload)."""
    import measure_photo
    from flat_lay import PhotoError, load_photo

    raw = decode_photo(body.get("photo"))
    kind = body.get("kind") or None
    if kind is not None and kind not in KINDS:
        raise BadRequest(f"kind must be one of {KINDS}.")
    photo_id = uuid.uuid4().hex
    folder = UPLOADS / photo_id
    folder.mkdir(parents=True)
    (folder / "photo.jpg").write_bytes(raw)
    image = load_photo(BytesIO(raw))
    if body.get("measure") is False:
        image.save(folder / "flat.jpg", quality=92)
        image.save(folder / "garment.jpg", quality=92)
        return 200, {"photoId": photo_id, "flatUrl": f"/uploads/{photo_id}/flat.jpg", "size": list(image.size)}
    try:
        with measure_lock:
            result, flat, _mask = measure_photo.measure_photo(image, rembg_session(), kind)
    except PhotoError as error:
        return 422, {"error": str(error), "code": error.code}
    flat.save(folder / "flat.jpg", quality=92)
    Image.fromarray(mask_img(_mask)).save(folder / "mask.png")
    cutout(flat, _mask).save(folder / "cutout.png")
    flat.save(folder / "garment.jpg", quality=92)
    payload = {
        "photoId": photo_id, "kind": result.kind, "guess": result.guess,
        "pxPerCm": round(result.px_per_cm, 3), "cm": result.cm,
        "lines": {k: [[round(float(x), 1), round(float(y), 1)] for x, y in pts] for k, pts in result.lines.items()},
        "scaleSource": result.scale_source, "scaleBy": result.scale_by, "uncertainty": result.uncertainty,
        "notes": result.notes, "estimated": result.estimated, "style": getattr(result, "style", None),
        "flatUrl": f"/uploads/{photo_id}/flat.jpg", "size": list(flat.size),
        "colour": main_colour(flat, _mask),
        "trims": trims(flat, _mask, result),
        "cutoutUrl": f"/uploads/{photo_id}/cutout.png",
    }
    payload = json.loads(json.dumps(payload, default=float))  # numpy numbers -> plain
    (folder / "measured.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 200, payload


def scale(body):
    """POST /api/scale {photoId, known: {name: cm} | factor}: exact sizes for a
    photo from one tape measurement (its sizes are otherwise estimated)."""
    import measure_photo

    folder, meta = upload(body.get("photoId"))
    rough = meta["cm"]
    try:
        if body.get("factor"):
            factor = float(body["factor"])
            source = "tape"  # matched to a photo scaled by a tape measurement
        else:
            known = body.get("known")
            if not isinstance(known, dict) or len(known) != 1:
                raise BadRequest("known must be one {name: cm}.")
            factor = measure_photo.rescale(rough, known)
            source = "tape"
    except (TypeError, ValueError) as error:
        raise BadRequest(str(error))
    if not 0.05 < factor < 20:
        raise BadRequest("That measurement does not fit this photo.")
    meta["cm"] = {k: round(v * factor, 1) for k, v in rough.items()}
    meta["pxPerCm"] = round(meta["pxPerCm"] / factor, 4)
    meta["scaleSource"], meta["uncertainty"] = source, 0.0
    (folder / "measured.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return 200, meta


def measurement_job(body, folder):
    raw = body.get("measurements")
    if not isinstance(raw, dict):
        raise BadRequest("measurements must be an object.")
    measurements = {}
    for key, value in raw.items():
        if key not in DRESS3D_FIELDS:
            raise BadRequest(f"Unknown measurement {key}.")
        if value in (None, ""):
            continue
        try:
            measurements[key] = float(value)
        except (TypeError, ValueError):
            raise BadRequest(f"{key} must be a number.")
    for key in ("dress_length", "bust"):
        if key not in measurements:
            raise BadRequest(f"{key} is required.")
    style = {}
    for key, value in (body.get("style") or {}).items():
        if value in (None, "", "auto"):
            continue
        if key not in STYLE_VALUES or value not in STYLE_VALUES[key]:
            raise BadRequest(f"style {key} must be one of {STYLE_VALUES.get(key, ())}.")
        style[key] = value
    units = body.get("units") or "cm"
    if units not in ("cm", "in"):
        raise BadRequest("units must be cm or in.")
    photos = {}
    for name in ("front", "back"):
        if body.get(f"{name}Id"):
            photos[name] = uploaded_photo(body[f"{name}Id"])
        elif body.get(name):
            photos[name] = decode_photo(body[name])
    if "back" in photos and "front" not in photos:
        raise BadRequest("A back photo needs a front photo too.")
    avatar = avatar_path(body)

    def write():
        for name, data in photos.items():
            (folder / f"{name}.jpg").write_bytes(data)

    # Runs inside this process (dress_on_avatar.make): its libraries and the
    # rembg model stay loaded between jobs, which saves ~25 s per dress.
    task = {"avatar": avatar, "measurements": measurements, "style": style, "units": units,
            "photos": {name: folder / f"{name}.jpg" for name in photos}}
    return write, task


def run_inline(job_id, task):
    """A measurement job, in this process."""
    import traceback

    import dress_on_avatar

    global _job_session
    folder = JOBS / job_id
    try:
        if task.get("photos") and _job_session is None:
            from rembg import new_session

            _job_session = new_session("u2net")
        if task.get("type") == "kind":
            run_kind(task, folder)
        elif task.get("type") == "flat":
            run_flat(task, folder)
        else:
            dress_on_avatar.make(task["avatar"], task["measurements"], task["style"], folder, "dress", task["units"],
                                 task["photos"].get("front"), task["photos"].get("back"), _job_session)
        return {"status": "done", "glb": f"/jobs/{job_id}/dress_on_avatar.glb", "dressGlb": f"/jobs/{job_id}/dress.glb",
                "info": json.loads((folder / "dress.json").read_text(encoding="utf-8"))}
    except (dress_on_avatar.InputError, ValueError) as error:
        return {"status": "failed", "error": str(error), "code": getattr(error, "code", None),
                "photoProblem": hasattr(error, "code")}
    except Exception as error:  # a bug: keep the server up, keep the trace
        (folder / "log.txt").write_text(traceback.format_exc(), encoding="utf-8")
        return {"status": "failed", "error": f"internal error ({type(error).__name__}); see log.txt", "photoProblem": False}


def run_flat(task, folder):
    """A dress or top with sleeves (flat_dress.run), from the stored photos."""
    import numpy as np

    import flat_dress

    global _job_session
    if _job_session is None:
        from rembg import new_session

        _job_session = new_session("u2net")
    masks = {half: np.asarray(Image.open(task[half] / "mask.png")) > 127 for half in ("front", "back")}
    args = flat_dress.options(
        avatar=str(task["avatar"]), front=str(task["front"] / "garment.jpg"), back=str(task["back"] / "garment.jpg"),
        length_cm=task["length_cm"], shoulder_cm=task["shoulder_cm"], sleeve_cm=task["sleeve_cm"],
        hem_cm=task["hem_cm"], collar_ruffle=task["collar_ruffle"], cuff_ruffle=task["cuff_ruffle"],
        out=str(folder), name="dress")
    flat_dress.run(args, _job_session, masks)


def run_kind(task, folder):
    """A skirt, trousers or scarf (garment_kinds.make), from the stored photos."""
    import numpy as np

    import garment_kinds

    photos = {}
    for half, src in task["photos"].items():
        meta = json.loads((src / "measured.json").read_text(encoding="utf-8"))
        mask = np.asarray(Image.open(src / "mask.png")) > 127
        photos[half] = garment_kinds.FlatPhoto(Image.open(src / "flat.jpg").convert("RGB"), mask, meta["pxPerCm"])
    garment_kinds.make(task["kind"], task["avatar"], photos, task["cm"], folder, "dress")


def run_job(job_id, args):
    folder = JOBS / job_id
    with run_lock:
        with jobs_lock:
            jobs[job_id]["status"] = "running"
        if isinstance(args, dict):
            state = run_inline(job_id, args)
            with jobs_lock:
                jobs[job_id].update(state)
            return
        result = subprocess.run(
            [str(PYTHON), str(ROOT / "backend" / args[0]), *args[1:]],
            cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
    (folder / "log.txt").write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    state = {"status": "failed"}
    if result.returncode == 0 and (folder / "dress_on_avatar.glb").exists():
        state = {"status": "done", "glb": f"/jobs/{job_id}/dress_on_avatar.glb",
                 "dressGlb": f"/jobs/{job_id}/dress.glb",
                 "info": json.loads((folder / "dress.json").read_text(encoding="utf-8"))}
    else:
        lines = [l for l in (result.stderr + result.stdout).splitlines() if l.strip()]
        known = [l for l in lines if l.startswith(("PHOTO_ERROR", "INPUT_ERROR"))]
        state["error"] = known[-1].split(" ", 1)[1] if known else (lines[-1] if lines else "unknown error")
        state["photoProblem"] = bool(known) and known[-1].startswith("PHOTO_ERROR")
    with jobs_lock:
        jobs[job_id].update(state)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def send_json(self, code, payload):
        data = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path not in ("/api/jobs", "/api/measure", "/api/scale"):
            return self.send_json(404, {"error": "not found"})
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            return self.send_json(413, {"error": "Photos too large (40 MB at most)."})
        job_id = uuid.uuid4().hex
        folder = JOBS / job_id
        try:
            body = json.loads(self.rfile.read(length))
            if not isinstance(body, dict):
                raise BadRequest("The request must be a JSON object.")
            if self.path == "/api/measure":
                return self.send_json(*measure(body))
            if self.path == "/api/scale":
                return self.send_json(*scale(body))
            method = body.get("method") or "flat"
            if method == "flat":
                write, args = flat_job(body, folder)
            elif method == "measurements":
                write, args = measurement_job(body, folder)
            elif method == "garment":
                write, args = garment_job(body, folder)
            else:
                raise BadRequest("method must be flat, measurements or garment.")
        except json.JSONDecodeError:
            return self.send_json(400, {"error": "The request is not JSON."})
        except BadRequest as error:
            return self.send_json(400, {"error": str(error)})
        except Exception as error:  # a bug: answer instead of dropping the connection
            import traceback

            traceback.print_exc()
            return self.send_json(500, {"error": f"internal error ({type(error).__name__})", "code": "internal"})

        folder.mkdir(parents=True)
        write()
        with jobs_lock:
            jobs[job_id] = {"status": "queued", "method": method}
        threading.Thread(target=run_job, args=(job_id, args), daemon=True).start()
        self.send_json(202, {"id": job_id})

    def do_GET(self):
        if self.path.startswith("/api/jobs/"):
            job_id = self.path.rsplit("/", 1)[-1]
            with jobs_lock:
                state = dict(jobs.get(job_id, {})) if JOB_ID.match(job_id) else {}
            return self.send_json(200 if state else 404, state or {"error": "no such job"})
        if self.path.startswith("/api/avatar-frame/"):
            avatar_id = self.path.rsplit("/", 1)[-1]
            try:
                return self.send_json(200, avatar_frame(avatar_id))
            except BadRequest as error:
                return self.send_json(404, {"error": str(error)})
        match = AVATAR_URL.match(self.path)
        if match:
            path = GENERATED_AVATARS / f"{match.group(1)}.glb"
            if not path.exists():
                return self.send_json(404, {"error": "no such avatar"})
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "model/gltf-binary")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        return super().do_GET()


def main():
    JOBS.mkdir(exist_ok=True)
    UPLOADS.mkdir(exist_ok=True)
    server = ThreadingHTTPServer(("127.0.0.1", 8010), Handler)
    print("Serving on http://127.0.0.1:8010/viewer/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
