"""Measurement-driven, pattern-based procedural dress generator.

Construction (all units cm, internal frame: x = wearer's side, d = depth below HPS, z = front):

1. A closed torso cross-section curve is defined for every depth d. Below the armhole its
   perimeter is forced to equal the circumference profile C(d), a monotone PCHIP spline through
   the measured anchors (bust, waist, hip, hem). Shape: superellipse with depth ratio r(d),
   optional skirt folds (cosine offset along the normal), then rescaled so perimeter == C(d).
2. Above the armhole (the "yoke"), the curve's depth is rounded to zero at the shoulder line so
   front and back meet at a shoulder seam.
3. Front and back panels are 2D pattern polygons in (s, d) space, where s = arc length around the
   cross-section measured from centre front (or centre back). Outlines: neckline, shoulder seam,
   armhole, side seam, hem. Panels are triangulated in pattern space (Delaunay + clip) and wrapped
   onto the cross-section curves. Shared seam samples make front/back weld exactly.
4. Sleeves (optional) are ring-tubes lofted from the actual armhole loop to circles whose polygon
   perimeters equal the upper-arm and cuff measurements.
5. The hem depth is solved iteratively so the front surface path HPS -> hem equals dress_length.

This is deterministic geometry, NOT learned reconstruction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial import Delaunay, cKDTree
import shapely

from .schema import DressSpec

M = 721  # samples along the half cross-section (CF -> side -> CB)
P_EXP = 2.3  # superellipse exponent


def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _resample_polyline(pts, h, include_end=True):
    """Resample polyline pts (n,2) at ~uniform spacing h. Keeps first (and last) point exactly."""
    pts = np.asarray(pts, float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    n = max(1, int(np.ceil(cum[-1] / h)))
    t = np.linspace(0, cum[-1], n + 1)
    if not include_end:
        t = t[:-1]
    return np.stack([np.interp(t, cum, pts[:, 0]), np.interp(t, cum, pts[:, 1])], axis=1)


@dataclass
class DressMesh:
    vertices: np.ndarray            # (N,3) cm, y-up, hem at y=0
    faces: np.ndarray               # (F,3)
    face_panel: np.ndarray          # (F,) panel id per face (same order for faces & render_faces)
    render_vertices: np.ndarray     # (R,3) unwelded copy: seam vertices duplicated per panel
    render_faces: np.ndarray        # (F,3) indices into render_vertices
    pattern_uv: np.ndarray          # (R,2) cm in pattern atlas (per render vertex)
    atlas_size_cm: tuple
    panel: np.ndarray               # (R,) 0 front, 1 back, 2 sleeve +x, 3 sleeve -x
    pattern_st: np.ndarray          # (R,2) (s/S_half(d) in [-1,1], d)
    landmarks: dict
    params: dict
    outlines: dict = field(default_factory=dict)   # 2D pattern outlines (cm) for SVG


class DressGenerator:
    def __init__(self, spec: DressSpec):
        self.spec = spec
        m, o = spec.m, spec.options
        self.h = float(o["mesh_resolution_cm"])
        self.r0 = float(o["depth_ratio"])
        self.nf = int(o["fold_count"])
        self.famp = float(o["fold_amplitude"])
        self.nw = m["neck_width"] / 2
        self.x_sp = m["shoulder_width"] / 2
        self.d_sp = 0.11 * (self.x_sp - self.nw) + 2.0      # shoulder slope drop (~4 cm)
        self.d_arm = m["armhole_depth"]
        if m["dress_length"] < m["waist_position"] + 10:
            raise ValueError("dress_length must be at least waist_position + 10 cm")
        self._theta = np.linspace(0, np.pi, M)
        self._ux = np.abs(np.sin(self._theta)) ** (2 / P_EXP)
        self._uz = np.sign(np.cos(self._theta)) * np.abs(np.cos(self._theta)) ** (2 / P_EXP)

    # ---------------------------------------------------------------- profile
    def _build_profile(self, d_hem):
        m = self.spec.m
        d_bust = self.d_arm + 2.5
        anchors = [(self.d_arm, m["bust"]), (d_bust, m["bust"]), (m["waist_position"], m["waist"])]
        if m["hip_position"] < d_hem - 2:
            anchors.append((m["hip_position"], m["hip"]))
        anchors.append((d_hem, m["hem"]))
        # optional shape anchors (e.g. from a photo's width profile), never near measured anchors
        for d, c in self.spec.options.get("profile_anchors", []) or []:
            if d_bust + 2 < d < d_hem - 2 and all(abs(d - a[0]) > 2.5 for a in anchors):
                anchors.append((float(d), float(c)))
        anchors.sort()
        ds = np.array([a[0] for a in anchors]); cs = np.array([a[1] for a in anchors])
        keep = np.concatenate([[True], np.diff(ds) > 0.5])
        self.C = PchipInterpolator(ds[keep], cs[keep], extrapolate=True)
        self.d_hem = d_hem
        # half-width/depth at armhole row (no folds there)
        r = self.r0
        per = self._half_perimeter(self._ux, self._uz * r) * 2
        self.a_arm = m["bust"] / per
        self.b_arm = self.a_arm * r
        self.a_top = max(self.x_sp + 1.0, self.a_arm)
        self.L_round = 0.8 * self.b_arm   # shoulder rounding height

    @staticmethod
    def _half_perimeter(x, z):
        return np.sum(np.hypot(np.diff(x, axis=-1), np.diff(z, axis=-1)), axis=-1)

    def _depth_ratio(self, d, C):
        m = self.spec.m
        flare = np.clip((C - m["waist"]) / m["waist"], 0, 1) * (d > m["waist_position"])
        return self.r0 + (0.93 - self.r0) * flare

    def _fold_amp(self, d):
        if self.nf <= 0 or self.famp <= 0:
            return np.zeros_like(d)
        start = self.spec.m["waist_position"]
        t = np.clip((d - start) / max(self.d_hem - start, 1e-6), 0, 1)
        return self.famp * t ** 1.5

    def shoulder_line(self, x):
        """Depth of the shoulder seam below HPS at |x| (0 inside the neck width)."""
        ax = np.abs(x)
        return np.where(ax <= self.nw, 0.0, (ax - self.nw) / (self.x_sp - self.nw) * self.d_sp)

    def neck_line(self, x, back=False):
        m, st = self.spec.m, self.spec.style["neckline"]
        drop = m["back_neck_drop"] if back else m["front_neck_drop"]
        t = np.clip(np.abs(x) / self.nw, 0, 1)
        if back or st == "round":
            return drop * np.sqrt(np.clip(1 - t ** 2, 0, 1))
        if st == "v":
            return drop * (1 - t)
        if st == "square":
            return np.where(t < 0.85, drop, drop * np.sqrt(np.clip(1 - ((t - 0.85) / 0.15) ** 2, 0, 1)))
        if st == "boat":
            return drop * np.sqrt(np.clip(1 - t ** 4, 0, 1))
        raise ValueError(st)

    def armhole_x(self, tau):
        """x of the armhole edge, tau in [0,1] from shoulder point to underarm."""
        scoop = 0.12 * (self.d_arm - self.d_sp)
        return self.x_sp + (self.a_arm - self.x_sp) * _smoothstep(tau) - scoop * np.sin(np.pi * tau) ** 1.2

    # ------------------------------------------------------- cross-sections
    def half_curves(self, d):
        """Half cross-section for each depth in d (N,). Returns x,z,arc arrays (N,M)."""
        d = np.atleast_1d(np.asarray(d, float))
        ux, uz = self._ux[None, :], self._uz[None, :]
        dd = np.maximum(d, self.d_arm)
        C = self.C(dd)
        r = self._depth_ratio(dd, C)[:, None]
        bx, bz = np.repeat(ux, len(d), 0), uz * r
        a0 = (C / (2 * self._half_perimeter(bx, bz)))[:, None]
        x, z = a0 * bx, a0 * bz
        amp = self._fold_amp(dd)[:, None]
        if np.any(amp > 0):
            tx, tz = np.gradient(x, axis=1), np.gradient(z, axis=1)
            nrm = np.hypot(tx, tz) + 1e-12
            nx, nz = -tz / nrm, tx / nrm
            sgn = np.sign(np.sum(nx * x + nz * z, axis=1, keepdims=True))
            off = amp * np.cos(self.nf * self._theta)[None, :] * sgn
            x, z = x + off * nx, z + off * nz
            for _ in range(2):
                k = (C / (2 * self._half_perimeter(x, z)))[:, None]
                x, z = x * k, z * k
        # yoke rows: widen to shoulder and round depth to the shoulder line
        yoke = d < self.d_arm
        if np.any(yoke):
            dy = d[yoke][:, None]
            w = _smoothstep((self.d_arm - dy) / max(self.d_arm - self.d_sp, 1e-6))
            a = self.a_arm + (self.a_top - self.a_arm) * w
            xy = a * ux
            zy = self.b_arm * uz
            t = np.clip((dy - self.shoulder_line(xy)) / self.L_round, 0, 1)
            g = np.sqrt(np.clip(t * (2 - t), 0, 1))
            x[yoke], z[yoke] = xy, zy * g
        arc = np.concatenate([np.zeros((len(d), 1)),
                              np.cumsum(np.hypot(np.diff(x, axis=1), np.diff(z, axis=1)), axis=1)], axis=1)
        return x, z, arc

    def s_of_x(self, xq, d):
        """Arc length from CF to the front point with |x| = xq on the row at depth d."""
        xq, d = np.atleast_1d(xq).astype(float), np.atleast_1d(d).astype(float)
        x, z, arc = self.half_curves(d)
        out = np.empty(len(d))
        for i in range(len(d)):
            k = int(np.argmax(x[i]))
            xi, ai = np.maximum.accumulate(x[i, :k + 1]), arc[i, :k + 1]
            out[i] = np.interp(abs(xq[i]), xi, ai)
        return out

    def map_points(self, s, d, back=False, chunk=1500):
        """Pattern (s,d) -> 3D (x, d, z). s signed: + is the +x side."""
        s, d = np.asarray(s, float), np.asarray(d, float)
        out = np.empty((len(s), 3))
        for i0 in range(0, len(s), chunk):
            sl = slice(i0, i0 + chunk)
            x, z, arc = self.half_curves(d[sl])
            ss = np.abs(s[sl])
            if back:
                ss = arc[:, -1] - ss
            px = np.array([np.interp(ss[i], arc[i], x[i]) for i in range(len(ss))])
            pz = np.array([np.interp(ss[i], arc[i], z[i]) for i in range(len(ss))])
            out[sl, 0] = px * np.sign(s[sl] + 1e-12)
            out[sl, 1] = d[sl]
            out[sl, 2] = pz
        return out

    def side_s(self, d):
        _, _, arc = self.half_curves(d)
        return arc[:, -1] / 2

    # ---------------------------------------------------------- pattern panels
    def _panel_segments(self, back):
        h = self.h / 2.5   # dense boundary => boundary edges are Delaunay edges (conforming)
        nx = np.linspace(0, self.nw, 40)
        neck = np.stack([nx, self.neck_line(nx, back)], 1)
        sx = np.linspace(self.nw, self.x_sp, 20)
        shoulder = np.stack([sx, self.shoulder_line(sx)], 1)
        tau = np.linspace(0, 1, 40)
        arm = np.stack([self.armhole_x(tau), self.d_sp + tau * (self.d_arm - self.d_sp)], 1)
        # convert x -> s (front/back symmetric above the waist)
        segs = {}
        for name, xy in (("neck", neck), ("shoulder", shoulder), ("armhole", arm)):
            s = self.s_of_x(xy[:, 0], xy[:, 1])
            segs[name] = _resample_polyline(np.stack([s, xy[:, 1]], 1), h)
        segs["armhole"][-1, 0] = self.side_s(np.array([self.d_arm]))[0]
        sd = np.linspace(self.d_arm, self.d_hem, 200)
        side = np.stack([self.side_s(sd), sd], 1)
        segs["side"] = _resample_polyline(side, h)
        segs["side"][:, 0] = self.side_s(segs["side"][:, 1])   # exact quarter-perimeter => welds
        hs = segs["side"][-1, 0]
        segs["hem"] = _resample_polyline(np.array([[hs, self.d_hem], [0.0, self.d_hem]]), h)
        return segs

    def _build_panel(self, back):
        segs = self._panel_segments(back)
        # right half outline: CF neck -> HPS -> SP -> underarm -> side -> hem -> CF hem
        right = np.concatenate([segs["neck"], segs["shoulder"][1:], segs["armhole"][1:],
                                segs["side"][1:], segs["hem"][1:]])
        left = right[::-1].copy(); left[:, 0] *= -1
        outline = np.concatenate([right, left[1:-1]])
        poly = shapely.Polygon(outline)
        # interior grid (denser in d near the shoulder line where the surface curves over)
        h = self.h
        smax = outline[:, 0].max()
        sg = np.arange(-smax, smax + h, h)
        dg = np.arange(0, self.d_hem + h, h)
        dg_fine = np.arange(0, self.d_sp + self.L_round, h / 3)
        S1, D1 = np.meshgrid(sg, dg)
        S2, D2 = np.meshgrid(sg, dg_fine)
        pts = np.concatenate([np.stack([S1.ravel(), D1.ravel()], 1), np.stack([S2.ravel(), D2.ravel()], 1)])
        # keep fine points only in the band just under the shoulder line
        xs_guess = np.abs(pts[:, 0])
        band = pts[:, 1] - self.shoulder_line(xs_guess)
        is_fine = np.arange(len(pts)) >= S1.size
        pts = pts[~is_fine | ((band >= 0) & (band < self.L_round))]
        inside = shapely.contains_xy(poly, pts[:, 0], pts[:, 1])
        pts = pts[inside]
        bdist = shapely.distance(shapely.points(pts), poly.exterior)
        dmin = np.where(pts[:, 1] < self.d_sp + self.L_round, 0.3 * h, 0.5 * h)
        pts = pts[bdist > dmin]
        allp = np.concatenate([outline, pts])
        tri = Delaunay(allp)
        tp = allp[tri.simplices]
        cen = tp.mean(1)
        keep = shapely.contains_xy(poly, cen[:, 0], cen[:, 1])
        grown = poly.buffer(1e-6)
        for a, b in ((0, 1), (1, 2), (2, 0)):
            mid = (tp[:, a] + tp[:, b]) / 2
            keep &= shapely.contains_xy(grown, mid[:, 0], mid[:, 1])
        # drop slivers spanned by consecutive outline points (they create seam short-cuts)
        on_out = (tri.simplices < len(outline)).all(1)
        e = np.stack([np.linalg.norm(tp[:, 1] - tp[:, 0], axis=1), np.linalg.norm(tp[:, 2] - tp[:, 1], axis=1),
                      np.linalg.norm(tp[:, 0] - tp[:, 2], axis=1)], 1)
        u_, v_ = tp[:, 1] - tp[:, 0], tp[:, 2] - tp[:, 0]
        area = 0.5 * np.abs(u_[:, 0] * v_[:, 1] - u_[:, 1] * v_[:, 0])
        keep &= ~(on_out & (area < 0.08 * e.max(1) ** 2))
        faces = tri.simplices[keep]
        # drop unused points
        used = np.unique(faces)
        remap = -np.ones(len(allp), int); remap[used] = np.arange(len(used))
        on_sh = np.zeros(len(remap), bool)
        n0 = len(segs["neck"]) - 1
        ns = len(segs["shoulder"])
        on_sh[n0:n0 + ns] = True                                   # right shoulder
        nr = len(right)
        on_sh[nr + (nr - (n0 + ns)) - 1: nr + (nr - n0) - 1] = True  # mirrored left shoulder
        on_sh = on_sh[used]
        allp, faces = allp[used], remap[faces]
        xyz = self.map_points(allp[:, 0], allp[:, 1], back=back)
        xyz[on_sh] = self.shoulder_point(np.sign(allp[on_sh, 0]), allp[on_sh, 1])
        return xyz, faces, allp, segs, outline

    def shoulder_point(self, sign, d):
        """Exact 3D point of the shoulder seam at depth d (z = 0)."""
        x = self.nw + np.clip(d / self.d_sp, 0, 1) * (self.x_sp - self.nw)
        return np.stack([x * sign, d, np.zeros_like(d)], 1)

    # ------------------------------------------------------------------ sleeves
    def _build_sleeve(self, loop, side_sign):
        m, o = self.spec.m, self.spec.options
        K = len(loop)
        seg = np.linalg.norm(np.roll(loop, -1, 0) - loop, axis=1)
        u = np.concatenate([[0], np.cumsum(seg)[:-1]]) / seg.sum()
        phi = 2 * np.pi * u
        alpha = np.radians(o["sleeve_angle_deg"])
        # internal frame: d is "down"; axis goes outward (+x*side) and down (+d)
        e = np.array([side_sign * np.cos(alpha), np.sin(alpha), 0.0])
        e1 = np.array([side_sign * np.sin(alpha), -np.cos(alpha), 0.0])   # "up" ⟂ axis
        e2 = np.array([0.0, 0.0, 1.0])
        O = loop.mean(0)
        T_cap = 6.0
        sp = loop[0]

        def circle(center, circ):
            unit = np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2
            per = np.sum(np.linalg.norm(np.roll(unit, -1, 0) - unit, axis=1))
            return center + unit * (circ / per)

        def rings_for(T):
            n = max(3, int(np.ceil(T / self.h)))
            ts = np.linspace(0, T, n + 1)[1:]
            # circumference along the sleeve: upper_arm at T_cap -> cuff at T (linear)
            rings = []
            for t in ts:
                tc = min(t, T_cap)
                circ = np.interp(t, [T_cap, T], [m["upper_arm"], m["cuff"]]) if T > T_cap else m["cuff"]
                c = circle(O + e * t, circ)
                w = _smoothstep(t / T_cap)
                rings.append((1 - w) * (loop + e * t) + w * c)
            return ts, rings

        T = m["sleeve_length"]
        for _ in range(6):   # solve T so that the top line SP->cuff equals sleeve_length
            ts, rings = rings_for(T)
            top = np.vstack([sp] + [r[0] for r in rings])
            L = np.sum(np.linalg.norm(np.diff(top, axis=0), axis=1))
            T *= m["sleeve_length"] / L
        ts, rings = rings_for(T)
        V = np.vstack(rings)
        faces = []
        base = -1  # ring 0 indices handled by caller offset
        nR = len(rings)
        for k in range(nR):
            for j in range(K):
                j2 = (j + 1) % K
                if k == 0:
                    a, b = ("L", j), ("L", j2)
                else:
                    a, b = ("S", (k - 1) * K + j), ("S", (k - 1) * K + j2)
                c, dd = ("S", k * K + j), ("S", k * K + j2)
                faces.append((a, b, dd)); faces.append((a, dd, c))
        # sleeve pattern coordinates: (u * ring perimeter, t)
        per0 = m["upper_arm"]
        uv = np.concatenate([np.stack([u * per0, np.full(K, t)], 1) for t in ts])
        lm = {
            "upper_arm_plane": (O + e * T_cap, e),
            "cuff_center": O + e * T,
            "sleeve_axis": e,
            "T": T,
        }
        return V, faces, uv, lm, u

    # ------------------------------------------------------------------ build
    def front_length(self):
        """HPS -> hem length along the front surface at x = neck half width."""
        d = np.linspace(0.0, self.d_hem, 600)
        s = self.s_of_x(np.full_like(d, self.nw + 1e-3), d)
        p = self.map_points(s, d)
        return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

    def build(self) -> DressMesh:
        m, style = self.spec.m, self.spec.style
        d_hem = m["dress_length"]
        for _ in range(4):
            self._build_profile(d_hem)
            L = self.front_length()
            d_hem += m["dress_length"] - L
        self._build_profile(d_hem)

        Vf, Ff, Pf, segf, outf = self._build_panel(back=False)
        Vb, Fb, Pb, segb, outb = self._build_panel(back=True)
        verts = [Vf, Vb]
        faces = [Ff, Fb + len(Vf)]
        fpanel = [np.zeros(len(Ff), int), np.ones(len(Fb), int)]
        panel = [np.zeros(len(Vf), int), np.ones(len(Vb), int)]
        smax = max(np.abs(outf[:, 0]).max(), np.abs(outb[:, 0]).max())
        col_w = 2 * smax + 4
        pat = [Pf + [smax + 2, 0], Pb + [col_w + smax + 2, 0]]
        st = [np.stack([Pf[:, 0] / np.maximum(self.side_s(Pf[:, 1]), 1e-6), Pf[:, 1]], 1),
              np.stack([Pb[:, 0] / np.maximum(self.side_s(Pb[:, 1]), 1e-6), Pb[:, 1]], 1)]
        outlines = {"front": outf, "back": outb}
        landmarks, sleeve_info = {}, {}
        atlas_h = self.d_hem + 4
        atlas_w = 2 * col_w

        if style["sleeve"] != "none":
            arm = segf["armhole"]
            for side in (1, -1):
                f3 = self.map_points(arm[:, 0] * side, arm[:, 1], back=False)
                b3 = self.map_points(arm[:, 0] * side, arm[:, 1], back=True)
                loop = np.concatenate([f3, b3[::-1][1:-1]])  # SP -> front -> underarm -> back -> (SP)
                loop[0] = self.shoulder_point(np.array([side]), np.array([self.d_sp]))[0]
                V, F, uv, lm, u = self._build_sleeve(loop, side)
                offs = sum(len(v) for v in verts)
                loop_idx = np.arange(len(loop)) + offs
                sl_idx = np.arange(len(V)) + offs + len(loop)
                FF = np.array([[loop_idx[i] if t == "L" else sl_idx[i] for (t, i) in tri] for tri in F])
                if side == -1:
                    FF = FF[:, ::-1]
                verts += [loop, V]
                faces.append(FF)
                pid = 2 if side == 1 else 3
                fpanel.append(np.full(len(FF), pid))
                panel += [np.full(len(loop), pid), np.full(len(V), pid)]
                uv0 = np.stack([u * m["upper_arm"], np.zeros(len(loop))], 1)
                y0 = 2 if side == 1 else 2 + lm["T"] + 4
                pat += [uv0 + [2 * col_w + 2, y0], uv + [2 * col_w + 2, y0]]
                st += [np.stack([u, np.zeros(len(loop))], 1),
                       np.stack([np.tile(u, len(V) // len(u)), uv[:, 1] / lm["T"]], 1)]
                sleeve_info[side] = lm
                atlas_w = 2 * col_w + max(m["upper_arm"], m["cuff"]) + 4
                atlas_h = max(atlas_h, 2 * lm["T"] + 12)

        V = np.vstack(verts); F = np.vstack(faces); P = np.vstack(pat); FP = np.concatenate(fpanel)
        PN = np.concatenate(panel); ST = np.vstack(st)
        # weld coincident seam vertices (front/back shoulder + side seams, sleeve/armhole loop)
        tree = cKDTree(V)
        pairs = tree.query_pairs(1e-4, output_type="ndarray")
        rep = np.arange(len(V))
        for a, b in sorted(map(tuple, pairs), key=lambda p: -p[1]):
            ra, rb = rep[a], rep[b]
            while rep[ra] != ra: ra = rep[ra]
            while rep[rb] != rb: rb = rep[rb]
            if ra != rb:
                rep[max(ra, rb)] = min(ra, rb)
        for i in range(len(rep)):
            r = i
            while rep[r] != r: r = rep[r]
            rep[i] = r
        F0 = F.copy()
        F = rep[F]
        ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
        F, F0, FP = F[ok], F0[ok], FP[ok]

        # orient faces outward (front/back: away from the vertical axis; sleeves: away from axis)
        tri = V[F]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        c = tri.mean(1)
        radial = np.stack([c[:, 0], np.zeros(len(c)), c[:, 2]], 1)
        flip = np.zeros(len(F), bool)
        for pid in (0, 1):
            sel = FP == pid
            if np.sum(np.einsum("ij,ij->i", n[sel], radial[sel])) < 0:
                flip |= sel
        for side, pid in ((1, 2), (-1, 3)):
            sel = FP == pid
            if not np.any(sel):
                continue
            ax = sleeve_info[side]["sleeve_axis"]
            o = sleeve_info[side]["upper_arm_plane"][0]
            rel = c[sel] - o
            rel = rel - np.outer(rel @ ax, ax)
            if np.sum(np.einsum("ij,ij->i", n[sel], rel)) < 0:
                flip |= sel
        F[flip] = F[flip][:, ::-1]
        F0[flip] = F0[flip][:, ::-1]
        # the internal frame (x, d-down, z) is left-handed relative to the exported y-up frame:
        # mirroring d -> y flips every winding, so reverse all faces to keep normals outward.
        F, F0 = F[:, ::-1].copy(), F0[:, ::-1].copy()

        def compact(Vx, Fx):
            used = np.unique(Fx)
            remap = -np.ones(len(Vx), int); remap[used] = np.arange(len(used))
            return used, remap[Fx]

        # welded mesh (analysis / simulation) and unwelded mesh (per-panel UVs for export/render)
        uw, Fw = compact(V, F)
        ur, Fr = compact(V, F0)
        to_yup = lambda X: np.stack([X[:, 0], self.d_hem - X[:, 1], X[:, 2]], 1)
        Vy, Vr = to_yup(V[uw]), to_yup(V[ur])
        F = Fw
        P, PN, ST = P[ur], PN[ur], ST[ur]

        def to_y(p):
            p = np.asarray(p, float)
            return [float(p[0]), float(self.d_hem - p[1]), float(p[2])]

        landmarks.update({
            "HPS_R": to_y([self.nw, 0, 0]), "HPS_L": to_y([-self.nw, 0, 0]),
            "SP_R": to_y([self.x_sp, self.d_sp, 0]), "SP_L": to_y([-self.x_sp, self.d_sp, 0]),
            "underarm_R": to_y([self.a_arm, self.d_arm, 0]), "underarm_L": to_y([-self.a_arm, self.d_arm, 0]),
            "y_hps": float(self.d_hem), "y_bust": float(self.d_hem - self.d_arm - 2.5),
            "y_waist": float(self.d_hem - m["waist_position"]), "y_hip": float(self.d_hem - m["hip_position"]),
            "y_hem": 0.0,
        })
        for side, tag in ((1, "R"), (-1, "L")):
            if side in sleeve_info:
                o, ax = sleeve_info[side]["upper_arm_plane"]
                landmarks[f"upper_arm_plane_{tag}"] = {"origin": to_y(o), "normal": [float(ax[0]), float(-ax[1]), float(ax[2])]}
                landmarks[f"cuff_center_{tag}"] = to_y(sleeve_info[side]["cuff_center"])
        params = {"d_hem_vertical": float(self.d_hem), "a_arm": float(self.a_arm), "b_arm": float(self.b_arm),
                  "shoulder_drop": float(self.d_sp), "front_length_model": float(self.front_length())}
        return DressMesh(vertices=Vy, faces=F, face_panel=FP, render_vertices=Vr, render_faces=Fr,
                         pattern_uv=P, atlas_size_cm=(atlas_w, atlas_h), panel=PN,
                         pattern_st=ST, landmarks=landmarks, params=params, outlines=outlines)
