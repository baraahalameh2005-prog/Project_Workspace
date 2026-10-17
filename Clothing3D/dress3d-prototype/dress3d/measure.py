"""Measure garment dimensions from a mesh file, independently of the generator's parameters.

What is measured purely from geometry (no generator knowledge):
  hem, cuff (boundary-loop edge lengths), neck width / drops (neck boundary loop),
  armhole depth + shoulder width for sleeveless dresses (armhole boundary loops),
  dress length (vertical plane section through HPS), girths (horizontal plane sections),
  waist (minimum girth search).
What uses landmark help (from the sidecar JSON, snapped to the nearest mesh vertex):
  shoulder points & underarm for sleeved dresses, sleeve length end-points, upper-arm plane.
Measurement *levels* (bust = armhole bottom - 2.5 cm, hip = hip_position) follow the schema
definitions in schema.MEASUREMENT_DEFS.
"""
from __future__ import annotations

import json

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra


def load_cm(path) -> trimesh.Trimesh:
    m = trimesh.load(path, force="mesh", process=False)
    m = trimesh.Trimesh(m.vertices * 100.0, m.faces, process=False)  # m -> cm
    m.merge_vertices(merge_tex=True, merge_norm=True)
    return m


def boundary_loops(mesh):
    """Ordered boundary loops -> list of (n,3) arrays (closed implicitly)."""
    be = mesh.edges[trimesh.grouping.group_rows(mesh.edges_sorted, require_count=1)]
    if len(be) == 0:
        return []
    adj = {}
    for a, b in be:
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen, loops = set(), []
    for start in adj:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            nxt = [n for n in adj[cur] if n != prev and (n not in seen or (n == start and len(loop) > 2))]
            if not nxt:
                break
            n = nxt[0]
            if n == start:
                break
            loop.append(n); seen.add(n); prev, cur = cur, n
        loops.append(mesh.vertices[loop])
    return loops


def loop_length(p):
    return float(np.sum(np.linalg.norm(np.roll(p, -1, 0) - p, axis=1)))


def _hull_perimeter_2d(pts2):
    from scipy.spatial import ConvexHull
    h = ConvexHull(pts2)
    v = pts2[h.vertices]
    return float(np.sum(np.linalg.norm(np.roll(v, -1, 0) - v, axis=1)))


def section_polylines(mesh, origin, normal, tol=1e-6):
    """Plane section as a list of chained polylines (open or closed), from raw segments."""
    segs = trimesh.intersections.mesh_plane(mesh, plane_normal=normal, plane_origin=origin)
    if len(segs) == 0:
        return []
    pts = segs.reshape(-1, 3)
    key = np.round(pts / 1e-4).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1, 2)
    nodes = np.zeros((inv.max() + 1, 3)); nodes[inv.ravel()] = pts
    adj = {}
    for a, b in inv:
        if a == b:
            continue
        adj.setdefault(a, []).append(b); adj.setdefault(b, []).append(a)
    seen, out = set(), []
    starts = [n for n in adj if len(adj[n]) == 1] + list(adj)   # open-chain ends first
    for s0 in starts:
        if s0 in seen:
            continue
        chain, prev, cur = [s0], None, s0
        seen.add(s0)
        while True:
            nxt = [n for n in adj[cur] if n != prev and n not in seen]
            if not nxt:
                if len(chain) > 2 and s0 in adj[cur] and prev is not None:
                    chain.append(s0)  # closed
                break
            prev, cur = cur, nxt[0]
            chain.append(cur); seen.add(cur)
        out.append(nodes[chain])
    return out


def girth_at(mesh, origin, normal, center=None, pick="contains"):
    """Perimeter of the plane section loop around `center` (default: origin).

    Returns dict(length, hull, n_loops) or None if no suitable loop is found.
    """
    polys = section_polylines(mesh, origin, normal)
    if not polys:
        return None
    center = np.asarray(origin if center is None else center, float)
    normal = np.asarray(normal, float) / np.linalg.norm(normal)
    # in-plane 2D basis
    a = np.cross(normal, [0, 0, 1.0]) if abs(normal[2]) < 0.9 else np.cross(normal, [1.0, 0, 0])
    a /= np.linalg.norm(a); b = np.cross(normal, a)
    best = None
    for poly in polys:
        if len(poly) < 4:
            continue
        p2 = np.stack([(poly - center) @ a, (poly - center) @ b], 1)
        closed = np.linalg.norm(poly[0] - poly[-1]) < 1e-3
        L = float(np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1)))
        from matplotlib.path import Path as MPath
        contains = MPath(p2).contains_point((0, 0))
        score = (contains, closed, L)
        if best is None or score > best[0]:
            best = (score, L, p2, closed, contains)
    if best is None:
        return None
    _, L, p2, closed, contains = best
    return {"length": L, "hull": _hull_perimeter_2d(p2), "closed": bool(closed), "encloses_center": bool(contains),
            "n_loops": len(polys)}


def horizontal_girth(mesh, y, axis_xz=(0.0, 0.0)):
    return girth_at(mesh, [axis_xz[0], y, axis_xz[1]], [0, 1, 0])


def surface_path_length(mesh, a, b):
    """Shortest path along mesh edges between the vertices nearest to points a and b (upper bound
    on the geodesic)."""
    e = mesh.edges_unique
    w = mesh.edges_unique_length
    n = len(mesh.vertices)
    g = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(n, n)).tocsr()
    ia = int(np.argmin(np.linalg.norm(mesh.vertices - a, axis=1)))
    ib = int(np.argmin(np.linalg.norm(mesh.vertices - b, axis=1)))
    dist = dijkstra(g, indices=ia)
    return float(dist[ib]), float(np.linalg.norm(mesh.vertices[ia] - a)), float(np.linalg.norm(mesh.vertices[ib] - b))


def measure_dress(mesh_path, landmarks_path=None, hip_position=None, waist_search=True):
    mesh = load_cm(mesh_path)
    lm = {}
    if landmarks_path:
        with open(landmarks_path, encoding="utf-8") as f:
            lm = json.load(f)["landmarks"]
    res, how = {}, {}
    loops = boundary_loops(mesh)
    info = sorted([(lp[:, 1].mean(), lp) for lp in loops], key=lambda t: t[0])
    hem = info[0][1]
    res["hem"] = loop_length(hem); how["hem"] = "geometry: lowest boundary loop edge length"
    # neck = boundary loop whose centroid is closest to the vertical axis among upper loops
    upper = [lp for _, lp in info[1:]]
    neck = min(upper, key=lambda lp: abs(lp[:, 0].mean()) + 0.01 * abs(lp[:, 2].mean()))
    others = [lp for lp in upper if lp is not neck]
    r_side, l_side = neck[neck[:, 0] > 0], neck[neck[:, 0] < 0]
    hps_r = r_side[np.argmax(r_side[:, 1])]; hps_l = l_side[np.argmax(l_side[:, 1])]
    y_hps = (hps_r[1] + hps_l[1]) / 2
    res["neck_width"] = float(np.linalg.norm(hps_r - hps_l)); how["neck_width"] = "geometry: neck-loop top points"
    res["front_neck_drop"] = float(y_hps - neck[neck[:, 2] > 0][:, 1].min())
    res["back_neck_drop"] = float(y_hps - neck[neck[:, 2] < 0][:, 1].min())
    how["front_neck_drop"] = how["back_neck_drop"] = "geometry: neck loop"
    sleeved = any(lp[:, 1].mean() < y_hps - 30 for lp in others) or "upper_arm_plane_R" in lm
    if not sleeved and len(others) >= 2:
        arm_r = max(others, key=lambda lp: lp[:, 0].mean()); arm_l = min(others, key=lambda lp: lp[:, 0].mean())
        sp_r = arm_r[np.argmax(arm_r[:, 1])]; sp_l = arm_l[np.argmax(arm_l[:, 1])]
        res["shoulder_width"] = float(np.linalg.norm(sp_r - sp_l))
        res["armhole_depth"] = float(y_hps - (arm_r[:, 1].min() + arm_l[:, 1].min()) / 2)
        how["shoulder_width"] = how["armhole_depth"] = "geometry: armhole loop extrema"
    elif lm:
        def snap(p):
            i = np.argmin(np.linalg.norm(mesh.vertices - p, axis=1)); return mesh.vertices[i]
        sp_r, sp_l = snap(lm["SP_R"]), snap(lm["SP_L"])
        res["shoulder_width"] = float(np.linalg.norm(sp_r - sp_l))
        res["armhole_depth"] = float(y_hps - (snap(lm["underarm_R"])[1] + snap(lm["underarm_L"])[1]) / 2)
        how["shoulder_width"] = how["armhole_depth"] = "landmark-assisted (snapped to mesh vertices)"
        cuffs = sorted(others, key=lambda lp: lp[:, 0].mean())
        res["cuff"] = float(np.mean([loop_length(c) for c in cuffs])); how["cuff"] = "geometry: cuff loops (mean L/R)"
        sl = []
        for tag, sp, cuff in (("R", sp_r, cuffs[-1]), ("L", sp_l, cuffs[0])):
            top = cuff[np.argmax(cuff[:, 1])]
            sl.append(surface_path_length(mesh, sp, top)[0])
            pl = lm[f"upper_arm_plane_{tag}"]
            g = girth_at(mesh, pl["origin"], pl["normal"])
            res.setdefault("_ua", []).append(g["length"] if g else np.nan)
        res["sleeve_length"] = float(np.mean(sl)); how["sleeve_length"] = "mesh-edge shortest path SP->cuff top (mean L/R)"
        res["upper_arm"] = float(np.nanmean(res.pop("_ua"))); how["upper_arm"] = "plane section at landmark plane"
    # dress length: vertical section through HPS, front half
    best = None
    for poly in section_polylines(mesh, [hps_r[0] + 0.3, 0, 0], [1, 0, 0]):
        k = int(np.argmax(poly[:, 1]))
        for part in (poly[:k + 1], poly[k:]):
            if len(part) > 2 and part[:, 2].mean() > 0:
                L = float(np.sum(np.linalg.norm(np.diff(part, axis=0), axis=1)))
                if best is None or L > best:
                    best = L
    res["dress_length"] = best; how["dress_length"] = "geometry: plane x=HPS+0.3cm, front half of section"
    # girths
    bust_y = y_hps - res["armhole_depth"] - 2.5
    g = horizontal_girth(mesh, bust_y)
    res["bust"] = g["length"]; res["bust_hull"] = g["hull"]; how["bust"] = "plane section at armhole bottom - 2.5 cm"
    if hip_position is not None:
        g = horizontal_girth(mesh, y_hps - hip_position)
        if g:
            res["hip"] = g["length"]; res["hip_hull"] = g["hull"]; how["hip"] = "plane section at hip_position"
    if waist_search:
        lo = y_hps - (hip_position if hip_position else res["armhole_depth"] + 35)
        ys = np.arange(bust_y - 4, lo, -0.5)
        gs = [(horizontal_girth(mesh, y) or {"length": np.inf})["length"] for y in ys]
        i = int(np.argmin(gs))
        res["waist"] = float(gs[i]); res["waist_position_found"] = float(y_hps - ys[i])
        how["waist"] = "minimum plane-section girth between bust and hip (0.5 cm steps)"
    res["_y_hps"] = float(y_hps)
    return res, how


def compare(measured: dict, target: dict, keys=None):
    rows = []
    for k in keys or target:
        if k in measured and measured[k] is not None and k in target:
            t, m = target[k], measured[k]
            rows.append({"measurement": k, "target_cm": round(t, 2), "measured_cm": round(m, 2),
                         "abs_err_cm": round(m - t, 2), "pct_err": round(100 * (m - t) / t, 2)})
    return rows
