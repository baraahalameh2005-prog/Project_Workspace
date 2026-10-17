import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, numpy as np, trimesh
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": "short"},
                   "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150}})
dm = DressGenerator(spec).build()
tm = trimesh.Trimesh(dm.vertices, dm.faces, process=False)
fp = dm.face_panel
for pid in range(4):
    sub = trimesh.Trimesh(dm.vertices, dm.faces[fp==pid], process=False)
    print(pid, len(sub.faces), "consistent", sub.is_winding_consistent)
# find adjacent face pairs with inconsistent winding
adj = tm.face_adjacency; edges = tm.face_adjacency_edges
bad = 0; where = []
for (f1, f2), (a, b) in zip(adj, edges):
    def dirn(f):
        t = list(tm.faces[f]); i = t.index(a); return t[(i+1)%3] == b
    if dirn(f1) == dirn(f2): bad += 1; where.append((fp[f1], fp[f2]))
from collections import Counter
print("inconsistent pairs", bad, Counter(map(tuple, where)))
