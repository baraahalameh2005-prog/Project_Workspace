import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import sys, numpy as np, trimesh
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
spec = parse_spec({"style": {"silhouette": "a-line", "sleeve": "none"},
                   "measurements": {"dress_length": 95, "bust": 92, "waist": 78, "hip": 100, "hem": 150}})
g = DressGenerator(spec); dm = g.build()
tm = trimesh.Trimesh(dm.vertices, dm.faces, process=False)
be = tm.edges[trimesh.grouping.group_rows(tm.edges_sorted, require_count=1)]
for comp in trimesh.graph.connected_components(be):
    p = tm.vertices[comp]
    print(len(comp), "x", p[:,0].min().round(1), p[:,0].max().round(1), "y", p[:,1].min().round(1), p[:,1].max().round(1),
          "z", p[:,2].min().round(1), p[:,2].max().round(1), "panels", np.unique(dm.panel[comp]))
# per-panel edges used >2 times
ec = trimesh.grouping.group_rows(tm.edges_sorted, require_count=3)
print("edges used 3x:", len(ec))
