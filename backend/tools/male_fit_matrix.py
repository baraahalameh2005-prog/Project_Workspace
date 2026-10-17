"""
Male avatar test matrix, verified on the exported GLB.

Every case goes through the same path as POST /generate-avatar (fit_body,
then the body is built again with the face preset and the fitted body
controls, then export_appearance_avatar). The GLB is read back from disk and
measured with measurements.BodyMeasurements; those mesh values, not the
optimizer's, are compared with the request. The cases are SYNTHETIC test
data (plausible adult men), except "reported_*", which are the inputs
reconstructed from the avatar a user reported (551c0fd9..., 2026-10-17).

Run from the backend folder (loads the Anny face model, about 1 GB):
    ..\\.venv\\Scripts\\python.exe -B tools\\male_fit_matrix.py LABEL [case,case,...]

Writes var/male_qa/LABEL/: <case>.glb, results.json, results.md.
Exit code 1 when a case misses the criteria without reporting the miss
(a silent miss); reported out-of-range misses are listed, not hidden.
"""

import json
import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import trimesh  # noqa: E402

import fit_to_measurements as ftm  # noqa: E402  (applies the module cache settings)
from appearance import FaceModel, appearance_for, export_appearance_avatar  # noqa: E402
from generate_avatar import build  # noqa: E402
from measurements import BodyMeasurements  # noqa: E402

OUT = BACKEND.parent / "var" / "male_qa"

# Acceptance criteria (cm, kg), from the task: height 1 cm, circumferences
# and shoulder 2 cm. Weight is an estimate (mesh volume x body density).
CRITERIA = {"height": 1.0, "weight": 1.0, "chest": 2.0, "waist": 2.0, "hip": 2.0, "shoulder": 2.0}
# One-at-a-time cases: a measurement that was not changed may move this much.
UNCHANGED_LIMIT_CM = 1.5

AVG = dict(height=175, weight=75, chest=98, waist=85, hip=97, shoulder=40)
CASES = {
    "short_slim": dict(height=163, weight=55, chest=86, waist=72, hip=87, shoulder=37),
    "short_heavy": dict(height=165, weight=85, chest=106, waist=98, hip=104, shoulder=40),
    "average": AVG,
    "average_no_shoulder": {**AVG, "shoulder": None},
    "tall_slim": dict(height=188, weight=70, chest=92, waist=78, hip=93, shoulder=41),
    "tall_heavy": dict(height=190, weight=105, chest=114, waist=104, hip=110, shoulder=44),
    "muscular": dict(height=180, weight=88, chest=110, waist=82, hip=98, shoulder=44),
    "big_waist_small_chest": dict(height=172, weight=82, chest=98, waist=100, hip=102, shoulder=39),
    "big_chest_small_waist": dict(height=178, weight=80, chest=108, waist=80, hip=96, shoulder=43),
    "narrow_hips": dict(height=176, weight=72, chest=96, waist=84, hip=90, shoulder=40),
    "wide_hips": dict(height=174, weight=80, chest=98, waist=88, hip=106, shoulder=39),
    "obese": dict(height=170, weight=115, chest=118, waist=118, hip=116, shoulder=42),
    "near_limit_tall": dict(height=200, weight=95, chest=108, waist=92, hip=104, shoulder=46),
    "near_limit_short": dict(height=155, weight=52, chest=84, waist=72, hip=86, shoulder=35),
    "out_of_capability": dict(height=175, weight=70, chest=130, waist=70, hip=90, shoulder=55),
    "reported_user_inputs": dict(height=170, weight=78, chest=103, waist=94, hip=102, shoulder=52),
    "reported_no_shoulder": dict(height=170, weight=78, chest=103, waist=94, hip=102, shoulder=None),
    # One measurement changed from "average", everything else equal.
    "avg_chest+6": {**AVG, "chest": 104}, "avg_chest-6": {**AVG, "chest": 92},
    "avg_waist+6": {**AVG, "waist": 91}, "avg_waist-6": {**AVG, "waist": 79},
    "avg_hip+6": {**AVG, "hip": 103}, "avg_hip-6": {**AVG, "hip": 91},
    "avg_shoulder+3": {**AVG, "shoulder": 43}, "avg_shoulder-3": {**AVG, "shoulder": 37},
    "avg_height+10": {**AVG, "height": 185}, "avg_height-10": {**AVG, "height": 165},
}
ONE_AT_A_TIME = {name: name.split("_")[1].rstrip("+-0123456789") for name in CASES if name.startswith("avg_")}

# Mesh values compared with the request: the adult male fit's definitions
# (chest and hip as tape girths, measurements.tape_measurements).
KEYS = {"height": "height_m", "weight": "mass_kg", "chest": "chest_tape_m", "waist": "waist_m",
        "hip": "hip_tape_m", "shoulder": "shoulder_breadth_m"}


def mesh_measure(meas, path, output):
    """Measure the exported GLB (glTF y up) in Anny's rest frame (z up)."""
    vertices = np.asarray(trimesh.load(path, force="mesh", process=False).vertices)
    n = output["rest_vertices"].shape[1]
    v = vertices[:n]  # both primitives share one vertex buffer
    original = output["rest_vertices"][0].detach().cpu().numpy()
    rest = np.stack([v[:, 0], -v[:, 2], v[:, 1] + original[:, 2].min()], axis=1)
    glb_output = {
        "rest_vertices": torch.tensor(rest[None], dtype=output["rest_vertices"].dtype),
        "rest_bone_poses": output["rest_bone_poses"],
    }
    measured = meas.measure(glb_output)
    measured.update(meas.tape_measurements(glb_output))
    measured["mass_kg"] = measured["mass_kg"] * ftm.MALE_BODY_DENSITY / ftm.ANNY_DENSITY
    # Shape descriptors for the visual checks (rest pose, front is -y).
    z = (rest[:, 2] - rest[:, 2].min()) / (rest[:, 2].max() - rest[:, 2].min())
    torso = np.abs(rest[:, 0]) < 0.12
    front = lambda lo, hi: -rest[torso & (z > lo) & (z < hi), 1].min()  # noqa: E731
    measured["belly_lead_cm"] = (front(0.52, 0.62) - front(0.70, 0.76)) * 100
    measured["glb_height_m"] = float(v[:, 1].max() - v[:, 1].min())
    measured["glb_floor_m"] = float(v[:, 1].min())
    return measured


def cm(value, key):
    return None if value is None else round(value if key == "mass_kg" else value * 100, 2)


def run_case(face, meas, name, case, folder):
    start = time.perf_counter()
    fitted = ftm.fit_body(model=None, measurements=meas, gender="male", height_cm=case["height"],
                          weight_kg=case["weight"], waist_cm=case["waist"], chest_cm=case["chest"],
                          hip_cm=case["hip"], shoulder_cm=case["shoulder"], shape_model=face.controls)
    fit_seconds = time.perf_counter() - start
    p = fitted["parameters"]
    app = appearance_for(gender="male", skin_tone="#c8997e", eye_color="brown")
    output = build(model=face.controls({**app.face_controls, **fitted["body_controls"]}), gender=0.0,
                   height=p["height"], weight=p["weight"], age=p["age"], muscle=p["muscle"],
                   proportions=p["proportions"])
    path = folder / f"{name}.glb"
    export_appearance_avatar(output=output, face=face, path=path, appearance=app)
    # Out-of-range list as the API reports it (from the exported body).
    targets = {k: case[k] for k in ("height", "weight", "waist", "chest", "hip")}
    if case["shoulder"] is not None:
        targets["shoulder"] = case["shoulder"]
    _, _, out_of_range = ftm.report_male_body(output, meas, targets, fitted["out_of_range"])
    fitted = dict(fitted, out_of_range=out_of_range)
    mesh = mesh_measure(meas, path, output)

    row = {"request": case, "seconds": round(fit_seconds, 1), "glb": path.name,
           "parameters": {k: round(v, 3) for k, v in p.items() if k != "gender"},
           "controls": {k: round(v, 3) for k, v in fitted["body_controls"].items()},
           "at_bound": sorted(k for k, v in fitted["body_controls"].items()
                              if v <= ftm.MALE_CONTROL_BOUNDS.get(k, ftm.DEFAULT_CONTROL_BOUNDS)[0] + ftm.AT_BOUND
                              or v >= ftm.MALE_CONTROL_BOUNDS.get(k, ftm.DEFAULT_CONTROL_BOUNDS)[1] - ftm.AT_BOUND),
           "out_of_range": fitted["out_of_range"],
           "warnings": [w for w in fitted.get("warnings", [])],
           "mesh": {k: cm(mesh.get(v), v) for k, v in KEYS.items()},
           "reported": {k: cm(fitted["result"].get(v), v) for k, v in KEYS.items()},
           "other_cm": {k: cm(mesh.get(k + "_m"), k + "_m") for k in
                        ("upper_arm", "forearm", "thigh", "calf", "neck", "arm_length", "leg_length",
                         "shoulder_width")},
           "armpit_level_chest_cm": cm(mesh.get("chest_m"), "chest_m"),
           "fixed_plane_hip_cm": cm(mesh.get("hip_m"), "hip_m"),
           "belly_lead_cm": round(mesh["belly_lead_cm"], 2),
           "glb_height_cm": round(mesh["glb_height_m"] * 100, 2),
           "glb_floor_m": round(mesh["glb_floor_m"], 4)}
    row["error"] = {k: (None if case.get(k) is None or row["mesh"][k] is None
                        else round(row["mesh"][k] - case[k], 2)) for k in KEYS}
    row["report_vs_mesh"] = {k: (None if row["mesh"][k] is None or row["reported"][k] is None
                                 else round(row["reported"][k] - row["mesh"][k], 2)) for k in KEYS}
    misses = [k for k, e in row["error"].items() if e is not None and abs(e) > CRITERIA[k]]
    row["status"] = {k: ("Pass" if e is not None and abs(e) <= CRITERIA[k] else
                         "Unsupported" if e is None else
                         "Miss (reported)" if k in fitted["out_of_range"] else "Miss")
                     for k, e in row["error"].items() if case.get(k) is not None}
    row["silent_misses"] = [k for k in misses if k not in fitted["out_of_range"]]
    return row


def markdown(rows):
    head = ("| case | chest | waist | hip | shoulder | height | weight* | belly lead | upper arm | thigh | "
            "controls at limit | status |")
    lines = [head, "|" + "---|" * 12]
    for name, r in rows.items():
        e, m = r["error"], r["mesh"]

        def cell(k):
            return "—" if e[k] is None else f"{m[k]} ({e[k]:+.1f})"

        bad = [f"{k}: {s}" for k, s in r["status"].items() if s != "Pass"]
        lines.append(f"| {name} | {cell('chest')} | {cell('waist')} | {cell('hip')} | {cell('shoulder')} | "
                     f"{cell('height')} | {cell('weight')} | {r['belly_lead_cm']} | {r['other_cm']['upper_arm']} | "
                     f"{r['other_cm']['thigh']} | {', '.join(c.split('-')[1] for c in r['at_bound']) or '—'} | "
                     f"{'; '.join(bad) or 'all pass'} |")
    lines.append("")
    lines.append("Values: mesh measurement of the exported GLB (error vs request). *weight = mesh volume x "
                 f"{ftm.MALE_BODY_DENSITY:.0f} kg/m3, an estimate. Belly lead: abdomen front minus chest front, cm.")
    return "\n".join(lines)


def main():
    label = sys.argv[1]
    names = sys.argv[2].split(",") if len(sys.argv) > 2 else list(CASES)
    folder = OUT / label
    folder.mkdir(parents=True, exist_ok=True)
    face = FaceModel()
    meas = BodyMeasurements(face.model)
    rows = {}
    for name in names:
        rows[name] = run_case(face, meas, name, CASES[name], folder)
        r = rows[name]
        print(f"{name:24s} {r['seconds']:5.1f}s err {json.dumps(r['error'])} oor {r['out_of_range']}"
              f" bound {r['at_bound']} belly {r['belly_lead_cm']}", flush=True)

    # One-at-a-time: measurements that were not changed must stay put.
    if "average" in rows:
        base = rows["average"]["mesh"]
        for name, changed in ONE_AT_A_TIME.items():
            if name in rows:
                moved = {k: round(rows[name]["mesh"][k] - base[k], 2) for k in ("chest", "waist", "hip", "shoulder")
                         if k != changed and base[k] is not None and rows[name]["mesh"][k] is not None}
                rows[name]["unchanged_moved_cm"] = moved
                rows[name]["unchanged_ok"] = all(abs(v) <= UNCHANGED_LIMIT_CM for v in moved.values())

    # Determinism: the same request twice gives the same mesh.
    if "average" in rows:
        again = run_case(face, meas, "average_repeat", CASES["average"], folder)
        a = np.asarray(trimesh.load(folder / "average.glb", force="mesh", process=False).vertices)
        b = np.asarray(trimesh.load(folder / "average_repeat.glb", force="mesh", process=False).vertices)
        rows["average"]["repeat_max_vertex_diff_mm"] = float(np.abs(a - b).max() * 1000)
        print("determinism: max vertex difference %.4f mm" % rows["average"]["repeat_max_vertex_diff_mm"])
        (folder / "average_repeat.glb").unlink()
        del again

    (folder / "results.json").write_text(json.dumps({"criteria": CRITERIA, "rows": rows}, indent=1))
    (folder / "results.md").write_text(markdown(rows))
    silent = {n: r["silent_misses"] for n, r in rows.items() if r["silent_misses"]}
    print(f"\nwrote {folder}\\results.md; silent misses: {silent or 'none'}")
    return 1 if silent else 0


if __name__ == "__main__":
    sys.exit(main())
