"""Collect all experiment metrics into results/measurements/*.csv and copy key images."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import csv, json, os, shutil
R = f"{ROOT}/results"
M = os.path.join(R, "measurements"); S = os.path.join(R, "screenshots"); MD = os.path.join(R, "models")
for d in (M, S, MD): os.makedirs(d, exist_ok=True)
dim_rows, geo_rows = [], []
for v in ("hybrid_front_v1", "hybrid_front", "hybrid_frontback", "hybrid_front_fullprotocol"):
    s = json.load(open(os.path.join(R, v, "summary.json")))
    for case, r in s.items():
        for d in r["dimensions"]:
            dim_rows.append({"approach": "E_hybrid", "variant": v, "case": case, **{k: d.get(k) for k in
                             ("measurement", "target_cm", "measured_cm", "abs_err_cm", "pct_err", "input", "inferred_source")}})
        geo_rows.append({"approach": "E_hybrid", "variant": v, "case": case, "runtime_s": r["runtime_s"],
                         **{f"iou_{k}": r["silhouette_iou"][k] for k in ("front", "right", "back")},
                         "chamfer_mean_cm": round(r["chamfer"]["chamfer_mean_cm"], 3),
                         "recon_to_gt_p95_cm": round(r["chamfer"]["recon_to_gt_p95_cm"], 3),
                         "dE_mean_front": r["deltaE_mean_colour"]["front"], "dE_mean_back": r["deltaE_mean_colour"]["back"],
                         "openings": r["integrity"]["boundary_loops"], "glb_kb": r["integrity"]["glb_bytes"] // 1024,
                         "style_correct": all(r["style_inferred"][k] == r["style_gt"][k] for k in r["style_gt"])})
    for case in s:
        for f, tag in (("comparison.png", "comparison"),):
            shutil.copy(os.path.join(R, v, case, f), os.path.join(S, f"{v}__{case}__{tag}.png"))
tl = json.load(open(os.path.join(R, "triposr", "run_log.json")))
t1 = json.load(open(os.path.join(R, "triposr", "run_log_T1.json")))
times = {**t1["cases"], **tl["cases"]}
s = json.load(open(os.path.join(R, "triposr", "summary.json")))
for case, r in s.items():
    for tag in ("bbox_aligned", "icp_aligned"):
        a = r[tag]
        for d in a["dimensions"]:
            dim_rows.append({"approach": "A_triposr", "variant": tag, "case": case, "measurement": d["measurement"],
                             "target_cm": d["target_cm"], "measured_cm": d.get("measured_cm"), "abs_err_cm": d.get("abs_err_cm"),
                             "pct_err": d.get("pct_err"), "input": "n/a (no measurements used; scaled to GT height)",
                             "inferred_source": d.get("note", "")})
        geo_rows.append({"approach": "A_triposr", "variant": tag, "case": case,
                         "runtime_s": round(times[case]["encode_s"] + times[case]["extract_s"], 1),
                         **{f"iou_{k}": a["silhouette_iou"][k] for k in ("front", "right", "back")},
                         "chamfer_mean_cm": round(a["chamfer"]["chamfer_mean_cm"], 3),
                         "recon_to_gt_p95_cm": round(a["chamfer"]["recon_to_gt_p95_cm"], 3),
                         "dE_mean_front": r["deltaE_mean_colour"]["front"], "dE_mean_back": r["deltaE_mean_colour"]["back"],
                         "openings": a["openings"], "glb_kb": os.path.getsize(os.path.join(R, "triposr", case, "triposr_raw.glb")) // 1024,
                         "style_correct": "n/a"})
    shutil.copy(os.path.join(R, "triposr", case, "comparison.png"), os.path.join(S, f"triposr__{case}__comparison.png"))
for case in os.listdir(os.path.join(R, "real_photos")):
    p = os.path.join(R, "real_photos", case, "comparison.png")
    if os.path.exists(p): shutil.copy(p, os.path.join(S, f"real_photo__{case}__colour_refine.png"))
    p = os.path.join(R, "real_photos_v1", case, "comparison.png")
    if os.path.exists(p): shutil.copy(p, os.path.join(S, f"real_photo__{case}__no_refine.png"))
for fn, rows in (("dimension_errors.csv", dim_rows), ("geometry_metrics.csv", geo_rows)):
    with open(os.path.join(M, fn), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
# representative models
for v, case in (("hybrid_front", c) for c in s):
    shutil.copy(os.path.join(R, v, case, "hybrid.glb"), os.path.join(MD, f"hybrid__{case}.glb"))
    shutil.copy(os.path.join(R, "triposr", case, "triposr_raw.glb"), os.path.join(MD, f"triposr_raw__{case}.glb"))
print(len(dim_rows), "dimension rows,", len(geo_rows), "geometry rows")
