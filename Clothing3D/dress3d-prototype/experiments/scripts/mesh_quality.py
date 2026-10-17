"""Mesh-quality / simulation-readiness metrics for hybrid (procedural) vs TripoSR meshes.

- triangle quality: min angle, aspect ratio (longest edge / shortest altitude), edge-length CV
- manifoldness, boundary loops (garment openings), connected components
- pattern strain (procedural only): 3D edge length / 2D pattern edge length - 1  (0 = isometric,
  i.e. the 3D panel could be sewn from the flat pattern without stretching)
- sleeve/torso interpenetration (procedural only): sleeve vertices lying inside the torso
  cross-section polygon at their height
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys

import numpy as np
import trimesh
from matplotlib.path import Path as MPath

sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec  # noqa: E402
from dress3d.generator import DressGenerator  # noqa: E402
from dress3d.measure import section_polylines  # noqa: E402

OUT = f"{ROOT}/results/measurements"
os.makedirs(OUT, exist_ok=True)


def tri_quality(V, F):
    t = V[F]
    e = np.stack([np.linalg.norm(t[:, 1] - t[:, 0], axis=1), np.linalg.norm(t[:, 2] - t[:, 1], axis=1),
                  np.linalg.norm(t[:, 0] - t[:, 2], axis=1)], 1)
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    alt = 2 * area / np.maximum(e.max(1), 1e-12)
    aspect = e.max(1) / np.maximum(alt, 1e-12)
    a, b, c = e[:, 0], e[:, 1], e[:, 2]
    ang = np.degrees(np.arccos(np.clip(np.stack([(b**2 + c**2 - a**2) / (2*b*c), (a**2 + c**2 - b**2) / (2*a*c),
                                                 (a**2 + b**2 - c**2) / (2*a*b)], 1), -1, 1)))
    el = e.ravel()
    return {"min_angle_p1_deg": float(np.percentile(ang.min(1), 1)), "min_angle_median_deg": float(np.median(ang.min(1))),
            "aspect_p99": float(np.percentile(aspect, 99)), "edge_mean_cm": float(el.mean()),
            "edge_cv": float(el.std() / el.mean()), "degenerate_faces": int((area < 1e-8).sum())}


def topo(V, F):
    m = trimesh.Trimesh(V, F, process=False)
    nm = len(trimesh.grouping.group_rows(m.edges_sorted, require_count=3))
    be = m.edges[trimesh.grouping.group_rows(m.edges_sorted, require_count=1)]
    loops = len(trimesh.graph.connected_components(be)) if len(be) else 0
    return {"vertices": len(V), "faces": len(F), "non_manifold_edges": int(nm), "boundary_loops": int(loops),
            "components": int(len(m.split(only_watertight=False))), "winding_consistent": bool(m.is_winding_consistent)}


def pattern_strain(dm):
    # render mesh has per-panel vertices with 2D pattern coords (cm) -> compare edge lengths
    F = dm.render_faces
    res = {}
    for pid, name in ((0, "front"), (1, "back")):
        sel = F[dm.face_panel == pid]
        E = np.concatenate([sel[:, [0, 1]], sel[:, [1, 2]], sel[:, [2, 0]]])
        l3 = np.linalg.norm(dm.render_vertices[E[:, 0]] - dm.render_vertices[E[:, 1]], axis=1)
        l2 = np.linalg.norm(dm.pattern_uv[E[:, 0]] - dm.pattern_uv[E[:, 1]], axis=1)
        s = l3 / np.maximum(l2, 1e-9) - 1
        res[name] = {"strain_median_pct": float(100 * np.median(np.abs(s))), "strain_p95_pct": float(100 * np.percentile(np.abs(s), 95)),
                     "strain_max_pct": float(100 * np.abs(s).max())}
    return res


def sleeve_penetration(dm):
    V, F, FP = dm.vertices, dm.faces, dm.face_panel
    sv = np.unique(F[FP >= 2])
    if len(sv) == 0:
        return None
    torso = trimesh.Trimesh(V, F[FP < 2], process=False)
    inside = 0
    for y in np.unique(np.round(V[sv, 1], 0)):
        pts = V[sv][np.abs(V[sv, 1] - y) < 0.5]
        for poly in section_polylines(torso, [0, y, 0], [0, 1, 0]):
            if len(poly) > 8 and np.linalg.norm(poly[0] - poly[-1]) < 1e-3:
                inside += int(MPath(poly[:, [0, 2]]).contains_points(pts[:, [0, 2]]).sum())
    return {"sleeve_vertices": int(len(sv)), "inside_torso": int(inside)}


rows = {}
for v in ("hybrid_front", "hybrid_front_fullprotocol"):
    for case in sorted(os.listdir(f"{ROOT}/results/{v}")):
        mj = f"{ROOT}/results/{v}/{case}/metrics.json"
        if not os.path.exists(mj):
            continue
        rep = json.load(open(mj))["report"]
        spec = parse_spec({"style": rep["style"], "measurements": rep["measurements_cm"], "options": rep["options"]})
        dm = DressGenerator(spec).build()
        rows[f"{v}/{case}"] = {"topology": topo(dm.vertices, dm.faces), "quality": tri_quality(dm.vertices, dm.faces),
                               "pattern_strain": pattern_strain(dm), "sleeve_penetration": sleeve_penetration(dm)}
for case in sorted(os.listdir(f"{ROOT}/results/triposr")):
    p = f"{ROOT}/results/triposr/{case}/triposr_aligned_cm.glb"
    if os.path.exists(p):
        m = trimesh.load(p, force="mesh", process=False)
        rows[f"triposr/{case}"] = {"topology": topo(np.asarray(m.vertices), m.faces), "quality": tri_quality(np.asarray(m.vertices), m.faces)}
json.dump(rows, open(os.path.join(OUT, "mesh_quality.json"), "w"), indent=2)
for k, r in rows.items():
    t, q = r["topology"], r["quality"]
    ps = r.get("pattern_strain"); sp = r.get("sleeve_penetration")
    print(f"{k:55s} V{t['vertices']:6d} loops {t['boundary_loops']} nm {t['non_manifold_edges']} comp {t['components']} "
          f"minang p1 {q['min_angle_p1_deg']:.1f} med {q['min_angle_median_deg']:.1f} aspect99 {q['aspect_p99']:.1f} edgeCV {q['edge_cv']:.2f}"
          + (f" strain med/p95 F {ps['front']['strain_median_pct']:.1f}/{ps['front']['strain_p95_pct']:.1f}%" if ps else "")
          + (f" sleeve-in-torso {sp['inside_torso']}/{sp['sleeve_vertices']}" if sp else ""))
