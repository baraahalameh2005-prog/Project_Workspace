import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, time, numpy as np, trimesh
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
for sleeve in ("none", "short"):
    spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": sleeve, "neckline": "round"},
                       "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150},
                       "options": {"fold_count": 8, "fold_amplitude": 1.5}})
    t = time.time(); dm = DressGenerator(spec).build(); dt = time.time() - t
    tm = trimesh.Trimesh(dm.vertices, dm.faces, process=False)
    loops = trimesh.graph.connected_components(tm.edges[trimesh.grouping.group_rows(tm.edges_sorted, require_count=1)])
    print(sleeve, f"{dt:.1f}s", "V", len(tm.vertices), "F", len(tm.faces), "bbox", tm.bounds.round(1).tolist(),
          "boundary loops", len(loops), [len(l) for l in loops], "winding_consistent", tm.is_winding_consistent,
          "front_len", round(dm.params["front_length_model"], 2))
