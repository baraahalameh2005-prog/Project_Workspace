"""Evaluate TripoSR outputs against the synthetic GT with the same metrics as the hybrid.

TripoSR has no metric scale. We give it a best-case alignment:
  1. fixed axis permutation from TripoSR's canonical frame to ours (y-up, +z front) — determined
     once by inspection (see AXES below) and applied identically to every case;
  2. uniform scale so its vertical extent equals the GT's vertical extent (uses GT knowledge!);
  3. translation to the GT bbox centre, then rigid ICP refinement (no scaling) against GT.
Girths are measured at the GT's measurement levels. Openings (neck/armholes/hem) cannot be
measured on a closed solid, which is itself a finding.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, f"{ROOT}")
sys.path.insert(0, os.path.dirname(__file__))
from dress3d.measure import horizontal_girth, boundary_loops  # noqa: E402
from dress3d.render import render  # noqa: E402
import eval_lib as E  # noqa: E402

IN = f"{ROOT}/experiments/inputs/synthetic"
OUT = f"{ROOT}/results/triposr"
# TripoSR canonical frame -> ours. Rows: new x, y, z expressed in old coordinates.
AXES = np.array(json.loads(os.environ.get("TRIPOSR_AXES", "[[0,1,0],[0,0,1],[1,0,0]]")), float)


def align(src: trimesh.Trimesh, gt: trimesh.Trimesh):
    v = src.vertices @ AXES.T
    ext_s = v.max(0) - v.min(0); ext_g = gt.vertices.max(0) - gt.vertices.min(0)
    s = ext_g[1] / ext_s[1]
    v = v * s
    v = v - (v.max(0) + v.min(0)) / 2 + (gt.vertices.max(0) + gt.vertices.min(0)) / 2
    m = trimesh.Trimesh(v, src.faces, vertex_colors=src.visual.vertex_colors if src.visual.kind == "vertex" else None,
                        process=False)
    pts_g, _ = trimesh.sample.sample_surface(gt, 8000, seed=3)
    T, _, cost = trimesh.registration.icp(m.vertices[np.random.default_rng(0).choice(len(m.vertices), min(8000, len(m.vertices)), replace=False)],
                                          pts_g, max_iterations=40, reflection=False, scale=False)
    m_icp = m.copy(); m_icp.apply_transform(T)
    return m, m_icp, float(s), float(cost)


def run(case):
    cdir = os.path.join(IN, case)
    c, gt_glb = E.load_case(cdir)
    gt = E.mesh_cm(gt_glb)
    gtm = c["ground_truth"]["measurements_resolved_cm"]
    lm = json.load(open(os.path.join(cdir, "gt.landmarks.json")))["landmarks"]
    raw = trimesh.load(os.path.join(OUT, case, "triposr_raw.obj"), force="mesh", process=False)
    m_bbox, m_icp, s, cost = align(raw, gt)
    res = {"scale_applied": s, "icp_cost": cost}
    cam = c["camera"]
    for tag, m in (("bbox_aligned", m_bbox), ("icp_aligned", m_icp)):
        r = {}
        sg = E.silhouettes(gt.vertices, gt.faces, cam); sr = E.silhouettes(m.vertices, m.faces, cam)
        r["silhouette_iou"] = {k: round(E.iou(sg[k], sr[k]), 4) for k in sg}
        r["chamfer"] = E.chamfer(m, gt)
        dims = []
        for key, y in (("bust", lm["y_bust"]), ("waist", lm["y_waist"]), ("hip", lm["y_hip"]), ("hem", 1.0)):
            g = horizontal_girth(m, y)
            target = gtm[key]
            if g is None:
                dims.append({"measurement": key, "target_cm": target, "measured_cm": None, "note": "no section"})
                continue
            dims.append({"measurement": key, "target_cm": round(target, 2), "measured_cm": round(g["length"], 2),
                         "abs_err_cm": round(g["length"] - target, 2), "pct_err": round(100 * (g["length"] - target) / target, 2),
                         "hull_cm": round(g["hull"], 2), "encloses_axis": g["encloses_center"],
                         "note": "hem measured 1 cm above lowest point (closed solid has no hem edge)" if key == "hem" else ""})
        r["dimensions"] = dims
        r["dress_length"] = "NOT MEASURABLE: closed solid, no neck/HPS landmark"
        r["openings"] = len(boundary_loops(m))
        r["components"] = int(len(m.split(only_watertight=False)))
        r["watertight"] = bool(m.is_watertight)
        res[tag] = r
    # renders
    vc = np.asarray(m_icp.visual.vertex_colors)[:, :3] if m_icp.visual.kind == "vertex" else None
    od = os.path.join(OUT, case)
    imgs = []
    for yaw in (0, 180, 90):
        imgs.append(render(m_icp.vertices, m_icp.faces, vcolor=vc, yaw=yaw, size=tuple(cam["size"]), fov_deg=cam["fov_deg"],
                           elev=cam["elev"], center=cam["center"], extent=cam["extent"], interior_darkening=None))
    for i, t in enumerate(("front", "back", "side")):
        imgs[i].save(os.path.join(od, f"render_{t}.png"))
    side_gt = render(gt.vertices, gt.faces, yaw=90, size=tuple(cam["size"]), fov_deg=cam["fov_deg"], elev=cam["elev"],
                     center=cam["center"], extent=cam["extent"], vcolor=np.tile([150, 150, 160], (len(gt.vertices), 1)))
    E.side_by_side([os.path.join(cdir, "photo_front.png"), imgs[0], os.path.join(cdir, "photo_back.png"), imgs[1], side_gt, imgs[2]],
                   os.path.join(od, "comparison.png"),
                   ["GT photo front", "TripoSR front", "GT photo back", "TripoSR back", "GT side (grey)", "TripoSR side"])
    ph = {t: Image.open(os.path.join(cdir, f"photo_{t}.png")).convert("RGB") for t in ("front", "back")}
    sg = E.silhouettes(gt.vertices, gt.faces, cam); sr = E.silhouettes(m_icp.vertices, m_icp.faces, cam)
    res["deltaE_pixelwise"] = {"front": round(E.delta_e(imgs[0], ph["front"], sg["front"] & sr["front"]), 2),
                               "back": round(E.delta_e(imgs[1], ph["back"], sg["back"] & sr["back"]), 2)}
    res["deltaE_mean_colour"] = {"front": round(E.mean_colour_delta_e(imgs[0], sr["front"], ph["front"], sg["front"]), 2),
                                 "back": round(E.mean_colour_delta_e(imgs[1], sr["back"], ph["back"], sg["back"]), 2)}
    m_icp.export(os.path.join(od, "triposr_aligned_cm.glb"))
    json.dump(res, open(os.path.join(od, "metrics.json"), "w"), indent=2)
    return res


if __name__ == "__main__":
    cases = [a for a in sys.argv[1:] if a.startswith("T")] or sorted(os.listdir(OUT))
    allres = {}
    for case in cases:
        if not os.path.exists(os.path.join(OUT, case, "triposr_raw.obj")):
            continue
        r = run(case)
        allres[case] = r
        print("==", case, "scale", round(r["scale_applied"], 2))
        for tag in ("bbox_aligned", "icp_aligned"):
            print("  ", tag, "IoU", r[tag]["silhouette_iou"], "chamfer", round(r[tag]["chamfer"]["chamfer_mean_cm"], 2),
                  "openings", r[tag]["openings"], "watertight", r[tag]["watertight"])
            for d in r[tag]["dimensions"]:
                print("      ", d)
        print("   dE", r["deltaE_pixelwise"], r["deltaE_mean_colour"])
    if not any(a.startswith("T") for a in sys.argv[1:]):
        json.dump(allres, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
