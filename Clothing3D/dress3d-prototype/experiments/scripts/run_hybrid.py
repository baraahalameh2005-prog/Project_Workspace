"""Experiment E: hybrid photo + measurements -> procedural garment. Evaluated against synthetic GT.

Usage: python run_hybrid.py [--back]   (--back also supplies the back reference photo)
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys
import time

import numpy as np
import trimesh

sys.path.insert(0, f"{ROOT}")
sys.path.insert(0, os.path.dirname(__file__))
from dress3d.hybrid import reconstruct  # noqa: E402
from dress3d import export as ex  # noqa: E402
from dress3d.measure import measure_dress, compare  # noqa: E402
from dress3d.render import render  # noqa: E402
import eval_lib as E  # noqa: E402

IN = f"{ROOT}/experiments/inputs/synthetic"
USE_BACK = "--back" in sys.argv
FULL = "--full" in sys.argv     # user also measures positions + neckline (complete protocol)
OUT = f"{ROOT}/results/hybrid" + ("_frontback" if USE_BACK else "_front") + ("_fullprotocol" if FULL else "")
USER_KEYS = ["dress_length", "bust", "waist", "hip", "hem", "shoulder_width", "sleeve_length", "upper_arm", "cuff"]
if FULL:
    USER_KEYS += ["hip_position", "waist_position", "armhole_depth", "neck_width", "front_neck_drop", "back_neck_drop"]
os.makedirs(OUT, exist_ok=True)
summary = {}
for case in sorted(os.listdir(IN)):
    cdir = os.path.join(IN, case)
    c, gt_glb = E.load_case(cdir)
    gt = c["ground_truth"]["measurements_resolved_cm"]
    user = {k: gt[k] for k in USER_KEYS if k in c["ground_truth"]["measurements"]}
    t0 = time.time()
    dm, tex, uv, rep, pa = reconstruct(os.path.join(cdir, "photo_front.png"), user,
                                       back_img=os.path.join(cdir, "photo_back.png") if USE_BACK else None)
    dt = time.time() - t0
    od = os.path.join(OUT, case); os.makedirs(od, exist_ok=True)
    glb = os.path.join(od, "hybrid.glb")
    size = ex.export_glb(ex.to_trimesh(dm, tex, uv=uv), glb)
    ex.save_landmarks(dm, os.path.join(od, "hybrid.landmarks.json"))
    tex.save(os.path.join(od, "texture_atlas.jpg"), quality=90)
    from PIL import Image
    Image.fromarray((pa.mask * 255).astype(np.uint8)).save(os.path.join(od, "photo_mask.png"))
    # dimensions (hip measured at the GT hip level, as the user would have measured it)
    meas, how = measure_dress(glb, os.path.join(od, "hybrid.landmarks.json"), hip_position=gt["hip_position"])
    keys = [k for k in ["dress_length", "bust", "waist", "hip", "hem", "shoulder_width", "neck_width",
                        "front_neck_drop", "back_neck_drop", "armhole_depth", "sleeve_length", "upper_arm", "cuff"] if k in gt]
    rows = compare(meas, gt, keys)
    for r in rows:
        r["input"] = "user-provided" if r["measurement"] in user else "inferred (photo/default)"
        r["inferred_source"] = rep["source"].get(r["measurement"], "")
    rows.append({"measurement": "waist_position", "target_cm": gt["waist_position"],
                 "measured_cm": round(meas["waist_position_found"], 2),
                 "abs_err_cm": round(meas["waist_position_found"] - gt["waist_position"], 2),
                 "pct_err": round(100 * (meas["waist_position_found"] - gt["waist_position"]) / gt["waist_position"], 2),
                 "input": "user-provided" if "waist_position" in user else "inferred (photo/default)",
                 "inferred_source": rep["source"].get("waist_position", "")})
    # geometry vs GT
    rec = E.mesh_cm(trimesh.load(glb, force="mesh", process=False))
    gtm = E.mesh_cm(gt_glb)
    cam = c["camera"]
    sg = E.silhouettes(gtm.vertices, gtm.faces, cam)
    sr = E.silhouettes(rec.vertices, rec.faces, cam)
    ious = {k: round(E.iou(sg[k], sr[k]), 4) for k in sg}
    ch = E.chamfer(rec, gtm)
    # appearance: front + back renders vs GT photos
    imgs = {}
    for tag, yaw in (("front", 0), ("back", 180)):
        im = render(dm.render_vertices, dm.render_faces, uv=uv, texture=tex, yaw=yaw, size=tuple(cam["size"]),
                    fov_deg=cam["fov_deg"], elev=cam["elev"], center=cam["center"], extent=cam["extent"])
        im.save(os.path.join(od, f"render_{tag}.png")); imgs[tag] = im
    ph = {t: Image.open(os.path.join(cdir, f"photo_{t}.png")).convert("RGB") for t in ("front", "back")}
    de = {t: round(E.delta_e(imgs[t], ph[t], sg[t] & sr[t]), 2) for t in ("front", "back")}
    dmean = {t: round(E.mean_colour_delta_e(imgs[t], sr[t], ph[t], sg[t]), 2) for t in ("front", "back")}
    sheet = [os.path.join(cdir, "photo_front.png"), imgs["front"], os.path.join(cdir, "photo_back.png"), imgs["back"]]
    for yaw in (90,):
        sheet += [render(gtm.vertices, gtm.faces, yaw=yaw, size=tuple(cam["size"]), fov_deg=cam["fov_deg"], elev=cam["elev"],
                         center=cam["center"], extent=cam["extent"], vcolor=np.tile([150, 150, 160], (len(gtm.vertices), 1))),
                  render(dm.render_vertices, dm.render_faces, uv=uv, texture=tex, yaw=yaw, size=tuple(cam["size"]),
                         fov_deg=cam["fov_deg"], elev=cam["elev"], center=cam["center"], extent=cam["extent"])]
    E.side_by_side(sheet, os.path.join(od, "comparison.png"),
                   ["GT photo front", "hybrid front", "GT photo back", "hybrid back", "GT side (grey)", "hybrid side"])
    tm = trimesh.Trimesh(dm.vertices, dm.faces, process=False)
    integ = {"vertices": int(len(dm.vertices)), "faces": int(len(dm.faces)), "glb_bytes": size,
             "winding_consistent": bool(tm.is_winding_consistent),
             "non_manifold_edges": int(len(trimesh.grouping.group_rows(tm.edges_sorted, require_count=3))),
             "boundary_loops": int(len(trimesh.graph.connected_components(
                 tm.edges[trimesh.grouping.group_rows(tm.edges_sorted, require_count=1)])))}
    summary[case] = {"runtime_s": round(dt, 1), "style_inferred": rep["style"], "style_gt": c["ground_truth"]["style"],
                     "dimensions": rows, "silhouette_iou": ious, "chamfer": ch, "deltaE_pixelwise": de,
                     "deltaE_mean_colour": dmean, "integrity": integ, "report": rep}
    json.dump(summary[case], open(os.path.join(od, "metrics.json"), "w"), indent=2, default=str)
    print(f"== {case}  ({dt:.1f}s)  style {rep['style']} vs GT {c['ground_truth']['style']}")
    for r in rows:
        print(f"   {r['measurement']:16s} gt {r['target_cm']:7.2f}  got {r['measured_cm']:7.2f}  err {r['abs_err_cm']:+6.2f} ({r['pct_err']:+6.2f}%)  [{r['input']}]")
    print("   IoU", ious, "chamfer", {k: round(v, 2) for k, v in ch.items()}, "dE px", de, "dE mean", dmean)
json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2, default=str)
