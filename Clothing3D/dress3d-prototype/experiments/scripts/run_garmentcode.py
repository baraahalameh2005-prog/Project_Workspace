"""Experiment C: GarmentCode (pygarment, MIT) parametric sewing pattern for a dress.

Run with: .venv-garmentcode\\Scripts\\python.exe experiments\\scripts\\run_garmentcode.py
Generates FittedShirt + SkirtCircle on the 'neutral' body (authors' own body model), then
measures the 2D panels: waist seam length, hem length, bodice width — to see how garment
dimensions relate to the (body-relative) design parameters. No simulation (the Warp-based
simulator is a separate, non-commercially licensed build requiring CUDA/MSVC — not installed).
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

GC = f"{ROOT}/tools/GarmentCode"
sys.path.insert(0, GC)
from assets.garment_programs.meta_garment import MetaGarment  # noqa: E402
from assets.bodies.body_params import BodyParameters  # noqa: E402

OUT = Path(f"{ROOT}/results/garmentcode")
OUT.mkdir(parents=True, exist_ok=True)
body = BodyParameters(os.path.join(GC, "assets", "bodies", "mean_all.yaml"))
design = yaml.safe_load(open(os.path.join(GC, "assets", "design_params", "default.yaml")))["design"]
design["meta"]["upper"]["v"] = "FittedShirt"
design["meta"]["bottom"]["v"] = "SkirtCircle"
design["meta"]["wb"]["v"] = None
results = {}
for suns in (0.5, 0.75):
    design["flare-skirt"]["suns"]["v"] = suns
    g = MetaGarment(f"dress_suns{suns}", body, design)
    pattern = g.assembly()
    folder = pattern.serialize(OUT, tag="", to_subfolder=True, with_3d=False, with_text=False,
                               view_ids=False, with_printable=False)
    spec = json.load(open(next(Path(folder).glob("*specification.json"))))
    panels = spec["pattern"]["panels"]
    info = {}
    for name, p in panels.items():
        v = np.array(p["vertices"], float)
        info[name] = {"n_vertices": len(v), "bbox_w_cm": float(np.ptp(v[:, 0])), "bbox_h_cm": float(np.ptp(v[:, 1])),
                      "translation": p.get("translation"), "rotation": p.get("rotation")}
    results[f"suns_{suns}"] = {"folder": str(folder), "panels": info,
                               "self_intersecting": bool(g.is_self_intersecting())}
    print(f"suns={suns}: {len(panels)} panels ->", folder)
    for n, i in info.items():
        print(f"   {n:28s} w {i['bbox_w_cm']:6.1f}  h {i['bbox_h_cm']:6.1f}")
results["body_used"] = {k: float(v) for k, v in body.items() if isinstance(v, (int, float))} if hasattr(body, "items") else str(body)
json.dump(results, open(OUT / "garmentcode_panels.json", "w"), indent=2, default=str)
