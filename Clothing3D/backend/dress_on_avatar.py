"""
Dress3D dress (from garment measurements, optionally with a front/back photo
for style and fabric) put on an avatar.

Dress3D (clothing/dress3d, from github.com/baraahalameh2005-prog/clothes)
builds the dress from sewing-style panels so that its girths and lengths match
the measurements; it is a free-standing garment (cm, y up, hem at y = 0, +z
front). This script makes it wearable:

  1. measures the avatar (shoulders, armpits, arm angle) and builds the dress
     with its sleeves at the avatar's arm angle;
  2. cuts the dress out of the photo with rembg (more reliable than Dress3D's
     own segmentation on real photos) and passes the mask to Dress3D;
  3. places it on the avatar: shoulder points on the avatar's shoulders, the
     dress centred on the torso;
  4. pushes every dress vertex that would be inside the body out to a small
     ease above the skin, then smooths the push so no dents appear;
  5. exports the dress and the dressed avatar as GLB (metres, Y up), plus the
     Dress3D report (each value tagged user / photo / default).

The dress keeps Dress3D's own skirt folds; it is not cloth-simulated (that
takes minutes; the flat-lay method in flat_dress.py does it).

Example:
  python backend/dress_on_avatar.py --avatar output/avatar.glb --name d3 ^
      --measurements "{\\"dress_length\\": 130, \\"bust\\": 84, \\"waist\\": 68, \\"hem\\": 236}" ^
      --style "{\\"silhouette\\": \\"fit-and-flare\\", \\"sleeve\\": \\"long\\"}"
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # the clothing folder, for the dress3d package

from dress3d import export as d3export  # noqa: E402
from dress3d.generator import DressGenerator  # noqa: E402
from dress3d.hybrid import reconstruct  # noqa: E402
from dress3d.measure import compare, measure_dress  # noqa: E402
from dress3d.schema import MEASUREMENT_DEFS, parse_spec  # noqa: E402
from photo_to_garment import arm_centerline, fill_background, garment_mask, measure_avatar  # noqa: E402

EASE_M = 0.02  # dress surface at least this far outside the skin (the dress mesh is coarser than the body)
PUSH_ITERATIONS = 3
SMOOTH_ITERATIONS = 6
PLAIN_FABRIC = (150, 40, 70)
PHOTO_MAX_SIDE = 1024  # photos are analysed at this size (enough for Dress3D, much faster)


def arm_angle_from_horizontal(avatar, frame):
    """Mean angle of the arms below horizontal, degrees (Dress3D's sleeve_angle_deg)."""
    angles = []
    for side in (-1, 1):
        line = arm_centerline(avatar, frame, side)
        d = line[-1] - line[1]
        angles.append(np.degrees(np.arctan2(-d[1], abs(d[0]))))
    return float(np.clip(np.mean(angles), 15, 85))


def build_dress(measurements, style, units, sleeve_angle, front=None, back=None, session=None):
    """Dress3D mesh, texture, uv and report. front/back: PIL images or None."""
    options = {"sleeve_angle_deg": sleeve_angle}
    if front is None:
        spec = parse_spec({"units": units, "style": {"silhouette": "a-line", **style},
                           "measurements": measurements, "options": options})
        dm = DressGenerator(spec).build()
        texture = d3export.make_print_texture(dm.atlas_size_cm, kind="plain", base=PLAIN_FABRIC)
        report = {"style": spec.style, "measurements_cm": spec.m, "notes": spec.notes,
                  "source": {k: ("user" if k in spec.provided else "default") for k in spec.m}}
        return dm, texture, None, report

    # Dress3D samples a little outside the garment's outline at the sides; with
    # the background painted in the nearest fabric colour (and the cut-out from
    # rembg) the wall or floor never shows up on the dress.
    def prepared(image):
        image = image.copy()
        image.thumbnail((PHOTO_MAX_SIDE, PHOTO_MAX_SIDE), Image.LANCZOS)
        mask = garment_mask(image, session)
        return fill_background(np.asarray(image), mask), mask

    front_rgb, mask = prepared(front)
    back_rgb, back_mask = prepared(back) if back is not None else (None, None)
    # options fixes the sleeve angle to the avatar's arms (the photo's cue is not used).
    dm, texture, uv, report, _ = reconstruct(front_rgb, measurements, units=units, style_override=style,
                                             back_img=back_rgb, options=options, mask=mask, back_mask=back_mask)
    return dm, texture, uv, report


def place_on_avatar(vertices_cm, landmarks, avatar, frame):
    """Dress vertices (cm, Dress3D frame) -> avatar frame (m): shoulder points
    on the avatar's shoulders, centred on the torso at chest height."""
    v = np.asarray(vertices_cm, float) / 100.0
    sp_y = (landmarks["SP_L"][1] + landmarks["SP_R"][1]) / 200.0
    chest_y = (frame.armpit_y + frame.shoulder_y) / 2
    near = avatar.vertices[np.abs(avatar.vertices[:, 1] - chest_y) < 0.02]
    torso = near[np.abs(near[:, 0]) < frame.shoulder_half * 0.8]
    center_x = float(np.median(torso[:, 0])) if len(torso) else 0.0
    center_z = float((torso[:, 2].min() + torso[:, 2].max()) / 2) if len(torso) else 0.0
    offset = np.array([center_x, frame.shoulder_y - sp_y, center_z])
    return v + offset, offset


def body_girths(avatar, frame):
    """The avatar's chest, waist and hip girths (cm): convex outline of the
    torso (arms left out) at those heights, as a tape would measure them."""
    from scipy.spatial import ConvexHull

    from photo_to_garment import body_points, section_loops

    def girth(y):
        points = body_points(section_loops(avatar, y), frame, y)
        if points is None or len(points) < 6:
            return None
        hull = points[ConvexHull(points).vertices]
        return float(np.linalg.norm(np.diff(np.vstack([hull, hull[:1]]), axis=0), axis=1).sum() * 100)

    ys = np.arange(frame.armpit_y - 0.40, frame.armpit_y - 0.02, 0.01)
    values = [(y, girth(y)) for y in ys]
    values = [(y, g) for y, g in values if g]
    chest = girth(frame.armpit_y - 0.05)
    torso = [(y, g) for y, g in values if frame.armpit_y - 0.35 < y < frame.armpit_y - 0.12]
    waist = min(torso, key=lambda v: v[1]) if torso else None
    hips = [(y, g) for y, g in values if waist and y < waist[0]]
    return {"bust": chest, "waist": waist[1] if waist else None, "hip": max(g for _, g in hips) if hips else None}


def fit_notes(dress_cm, body_cm):
    """Where the dress is smaller than the body (it would not close / is tight)."""
    notes = []
    for name in ("bust", "waist", "hip"):
        d, b = dress_cm.get(name), body_cm.get(name)
        if d and b and d < b:
            notes.append(f"{name}: the dress ({d:.0f} cm) is {b - d:.0f} cm smaller than this body ({b:.0f} cm), "
                         "so it is too tight; the preview stretches it over the body.")
    return notes


def rotation_between(a, b):
    """Rotation matrix turning unit vector a onto unit vector b."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1 + c)


def follow_arms(vertices, panel, landmarks_m, avatar, frame):
    """Turn each sleeve onto the avatar's arm: the sleeve's axis (upper-arm
    plane centre -> cuff centre) is rotated and moved onto the arm's centre
    line, fully from 12 cm down the sleeve, blending to no change at the
    armhole so the sleeve stays sewn to the bodice."""
    v = vertices.copy()
    for panel_id, side, suffix in ((2, 1, "R"), (3, -1, "L")):
        sel = panel == panel_id
        if not sel.any():
            continue
        a0 = np.asarray(landmarks_m[f"upper_arm_plane_{suffix}"]["origin"])
        a1 = np.asarray(landmarks_m[f"cuff_center_{suffix}"])
        line = arm_centerline(avatar, frame, side)
        # Arm point nearest the sleeve's start, and the arm's direction from there.
        k = int(np.argmin(np.linalg.norm(line - a0, axis=1)))
        b0 = line[k]
        b_dir = line[-1] - b0
        rot = rotation_between(a1 - a0, b_dir)
        rel = v[sel] - a0
        t = rel @ ((a1 - a0) / np.linalg.norm(a1 - a0))
        weight = np.clip(t / 0.12, 0, 1)
        weight = weight * weight * (3 - 2 * weight)
        moved = b0 + rel @ rot.T
        v[sel] = v[sel] + (moved - v[sel]) * weight[:, None]
    return v


def push_out(vertices, faces, avatar, ease=EASE_M):
    """Move dress vertices that are inside (or too close to) the body out
    along the body's normal, then smooth the displacement over the dress."""
    query = trimesh.proximity.ProximityQuery(avatar)
    face_normals = np.asarray(avatar.face_normals)

    def push(points):
        """Points closer than `ease` to the skin (or inside) -> `ease` outside,
        along the normal of the nearest body triangle."""
        closest, _, triangle = query.on_surface(points)
        n = face_normals[triangle]
        depth = np.einsum("ij,ij->i", points - closest, n)
        moved = depth < ease
        points = points.copy()
        points[moved] = closest[moved] + n[moved] * ease
        return points

    original = vertices.copy()
    v = vertices.copy()
    # Only points near the body can touch it; skip the rest (most of a skirt).
    near = cKDTree(np.asarray(avatar.vertices)).query(v, distance_upper_bound=0.08)[0] < np.inf
    for _ in range(PUSH_ITERATIONS):
        v[near] = push(v[near])

    # Smooth the push (not the dress), so pushed areas blend into the rest;
    # the final push below undoes any smoothing that went back inside.
    from scipy.sparse import coo_matrix

    edges = trimesh.geometry.faces_to_edges(faces)
    n = len(v)
    adjacency = coo_matrix((np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(n, n)).tocsr()
    degree = np.maximum(np.asarray(adjacency.sum(axis=1)).ravel(), 1)
    disp = v - original
    for _ in range(SMOOTH_ITERATIONS):
        disp = 0.5 * disp + 0.5 * (adjacency @ disp) / degree[:, None]
    v = original + disp
    # Never end inside after smoothing.
    near = cKDTree(np.asarray(avatar.vertices)).query(v, distance_upper_bound=0.08)[0] < np.inf
    v[near] = push(v[near])
    return v


class InputError(ValueError):
    """The measurements or style cannot make a dress; the message says why."""


def make(avatar_path, measurements, style, out, name, units="cm", front=None, back=None, session=None):
    """Build the dress, put it on the avatar and write <name>.glb,
    <name>_on_avatar.glb, <name>_free.glb, <name>.landmarks.json and
    <name>.json in `out`. front/back: photo paths or PIL images (optional).
    Returns the info dict (also written as <name>.json). Raises InputError."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    style = {k: v for k, v in (style or {}).items() if v and v != "auto"}

    avatar = trimesh.load(avatar_path, force="scene").to_geometry()
    frame = measure_avatar(avatar)
    sleeve_angle = arm_angle_from_horizontal(avatar, frame)

    if front is not None:
        from flat_lay import load_photo

        if session is None:
            from rembg import new_session

            session = new_session("u2net")
        front = front if isinstance(front, Image.Image) else load_photo(front)
        if back is not None:
            back = back if isinstance(back, Image.Image) else load_photo(back)
    try:
        dm, texture, uv, report = build_dress(measurements, style, units, sleeve_angle, front, back, session)
    except ValueError as error:
        raise InputError(str(error)) from error

    # Verify the free-standing dress against the measurements (Dress3D's check).
    free = out / f"{name}_free.glb"
    d3export.export_glb(d3export.to_trimesh(dm, texture, uv=uv), free)
    d3export.save_landmarks(dm, out / f"{name}.landmarks.json")
    spec_m = report["measurements_cm"]
    measured, how = measure_dress(str(free), str(out / f"{name}.landmarks.json"),
                                  hip_position=spec_m["hip_position"])
    rows = compare(measured, spec_m, [k for k in MEASUREMENT_DEFS if k in spec_m and k in measured])
    for row in rows:
        row["source"] = report["source"].get(row["measurement"], "")

    body = body_girths(avatar, frame)

    # Put it on the avatar.
    placed, offset = place_on_avatar(dm.render_vertices, dm.landmarks, avatar, frame)
    landmarks_m = {}
    for key in ("upper_arm_plane_R", "upper_arm_plane_L"):
        if key in dm.landmarks:
            landmarks_m[key] = {"origin": np.asarray(dm.landmarks[key]["origin"]) / 100 + offset}
    for key in ("cuff_center_R", "cuff_center_L"):
        if key in dm.landmarks:
            landmarks_m[key] = np.asarray(dm.landmarks[key]) / 100 + offset
    if len(landmarks_m) == 4:
        placed = follow_arms(placed, dm.panel, landmarks_m, avatar, frame)
    welded = trimesh.Trimesh(placed, dm.render_faces, process=False)
    welded.merge_vertices()  # render vertices are split at UV seams; push them together
    pushed = push_out(np.asarray(welded.vertices), np.asarray(welded.faces), avatar)
    # Map back to the render vertices (each render vertex -> its welded vertex).
    _, render_to_welded = cKDTree(welded.vertices).query(placed)
    final = pushed[render_to_welded]

    dress = d3export.to_trimesh(dm, texture, uv=uv)
    dress.vertices = final
    d3export.export_glb(dress, out / f"{name}.glb")
    scene = trimesh.load(avatar_path, force="scene")
    scene.add_geometry(dress, geom_name=name)
    scene.export(out / f"{name}_on_avatar.glb")

    info = {
        "name": name,
        "modelSource": "measurement_driven_pattern",
        "generationTool": "dress_on_avatar.py (Dress3D panels from measurements"
                          + (" and photo" if front is not None else "") + ", placed on the avatar)",
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "avatar": str(avatar_path),
        "sleeveAngleDeg": round(sleeve_angle, 1),
        "placementOffsetM": [round(float(x), 4) for x in offset],
        "style": report["style"],
        "measurementsCm": spec_m,
        "source": report["source"],
        "verification": rows,
        "bodyGirthsCm": {k: (round(v, 1) if v else None) for k, v in body.items()},
        "fitNotes": fit_notes(spec_m, body),
    }
    (out / f"{name}.json").write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avatar", required=True, help="avatar GLB (Anny, Y up, feet at y = 0)")
    parser.add_argument("--measurements", required=True, help="JSON object of Dress3D measurements")
    parser.add_argument("--style", default="{}", help='JSON, e.g. {"silhouette": "a-line", "sleeve": "long"}')
    parser.add_argument("--units", default="cm", choices=("cm", "in"))
    parser.add_argument("--front", help="front photo (optional: style and fabric)")
    parser.add_argument("--back", help="back photo (optional)")
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()
    try:
        measurements = json.loads(args.measurements)
        style = json.loads(args.style)
    except json.JSONDecodeError as error:
        raise SystemExit(f"INPUT_ERROR: measurements/style must be JSON ({error}).")
    try:
        info = make(args.avatar, measurements, style, args.out, args.name, args.units, args.front, args.back)
    except InputError as error:
        raise SystemExit(f"INPUT_ERROR: {error}")
    print(json.dumps({k: info[k] for k in ("name", "sleeveAngleDeg", "style", "bodyGirthsCm", "fitNotes")},
                     default=str))


if __name__ == "__main__":
    main()
