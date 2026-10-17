import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, numpy as np, trimesh
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": "long"}, "measurements": {"dress_length": 95, "bust": 92}})
dm = DressGenerator(spec).build()
tm = trimesh.Trimesh(dm.vertices, dm.faces, process=False)
c = tm.triangles_center; n = tm.face_normals
for pid, name in ((0, "front"), (1, "back")):
    sel = dm.face_panel == pid
    radial = c[sel] * [1, 0, 1]
    print(name, "fraction outward:", round(float(np.mean(np.einsum("ij,ij->i", n[sel], radial) > 0)), 3))
