import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, json, numpy as np
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
from dress3d import export as ex
from dress3d.measure import measure_dress, compare
import os; os.makedirs(f"{ROOT}/cache/smoke", exist_ok=True)
for sleeve in ("none", "long"):
    spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": sleeve},
                       "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150},
                       "options": {"fold_count": 8, "fold_amplitude": 1.5}})
    dm = DressGenerator(spec).build()
    tex = ex.make_print_texture(dm.atlas_size_cm)
    p = f"{ROOT}/cache/smoke/{sleeve}.glb"
    size = ex.export_glb(ex.to_trimesh(dm, tex), p)
    ex.save_landmarks(dm, p.replace(".glb", ".landmarks.json"))
    res, how = measure_dress(p, p.replace(".glb", ".landmarks.json"), hip_position=spec.m["hip_position"])
    print(sleeve, "glb bytes", size)
    for r in compare(res, spec.m): print("  ", r)
    print("  waist found at", round(res["waist_position_found"],1), "target", spec.m["waist_position"])
