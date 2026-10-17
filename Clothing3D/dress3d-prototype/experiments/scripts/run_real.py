"""Qualitative test on REAL CC0 museum photos. No ground truth exists: measurements are ASSUMED,
so only robustness/appearance is assessed (photo cues, segmentation, texture), not accuracy."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json, os, sys
import numpy as np
from PIL import Image
sys.path.insert(0, f"{ROOT}"); sys.path.insert(0, os.path.dirname(__file__))
from dress3d.hybrid import reconstruct
from dress3d import export as ex
from dress3d.render import render
import eval_lib as E

IN = f"{ROOT}/experiments/inputs/real"
OUT = f"{ROOT}/results/real_photos"
ASSUMED = {
  "R1_mccardell_plaid": {"dress_length": 105, "bust": 88, "waist": 66, "hip": 100, "hem": 220, "shoulder_width": 36,
                         "sleeve_length": 45, "upper_arm": 28, "cuff": 24},
  "R2_rijksmuseum_blue": {"dress_length": 140, "bust": 84, "waist": 80, "hip": 100, "hem": 200, "shoulder_width": 30,
                          "sleeve_length": 12, "upper_arm": 30, "cuff": 30},
}
for case, meas in ASSUMED.items():
    od = os.path.join(OUT, case); os.makedirs(od, exist_ok=True)
    json.dump({"assumed_measurements_cm": meas, "warning": "ASSUMED values - no published measurements"},
              open(os.path.join(IN, case, "case.json"), "w"), indent=2)
    img = os.path.join(IN, case, "photo_front.jpg")
    try:
        dm, tex, uv, rep, pa = reconstruct(img, meas, colour_refine=True)
    except Exception as e:
        print(case, "FAILED:", type(e).__name__, e); json.dump({"error": str(e)}, open(os.path.join(od, "report.json"), "w")); continue
    Image.fromarray((pa.mask * 255).astype(np.uint8)).save(os.path.join(od, "photo_mask.png"))
    ex.export_glb(ex.to_trimesh(dm, tex, uv=uv), os.path.join(od, "hybrid.glb"))
    views = [render(dm.render_vertices, dm.render_faces, uv=uv, texture=tex, yaw=y, size=(500, 750), fov_deg=25, elev=3)
             for y in (0, 45, 90, 180)]
    overlay = np.asarray(Image.open(img).convert("RGB")).copy()
    overlay[pa.mask] = (0.5 * overlay[pa.mask] + 0.5 * np.array([255, 0, 120])).astype(np.uint8)
    E.side_by_side([img, Image.fromarray(overlay)] + views, os.path.join(od, "comparison.png"),
                   ["real photo (CC0)", "segmentation", "hybrid front", "hybrid 45deg", "hybrid side", "hybrid back"])
    rep_s = {k: rep[k] for k in ("style", "source", "photo_features", "confidence", "options")}
    json.dump(rep_s, open(os.path.join(od, "report.json"), "w"), indent=2, default=str)
    print(case, rep["style"], {k: v for k, v in rep["source"].items() if v != "user"})
