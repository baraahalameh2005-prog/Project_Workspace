"""Hybrid reconstruction: photo (shape cues + appearance) + measurements (dimensions).

Pipeline
  1. photo.analyse(): segmentation, width profile, sleeve/neckline/waist cues (classical CV).
  2. Style + unmeasured positions inferred from the photo; user measurements always win.
  3. Photo width profile converted to extra circumference anchors *between* measured girths
     (ratio circumference/visible-width interpolated between measured levels).
  4. Measurement-driven generator builds the garment (dimensions controlled by construction).
  5. Front photo (and back photo if supplied) projected onto the panels as the texture.

Every inferred value is reported with its source: "user", "photo", or "default".
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from .generator import DressGenerator
from .photo import analyse
from .schema import parse_spec, INCH

FOLD_DEFAULTS = {"straight": (0, 0.0), "a-line": (8, 1.5), "fit-and-flare": (12, 2.5)}


def _to_cm(meas: dict, units: str):
    out = {}
    for k, v in meas.items():
        if v is None or v == "":
            continue
        method = "circumference"
        if isinstance(v, dict):
            method, v = v.get("method", "circumference"), v["value"]
        v = float(v) * (INCH if units == "in" else 1.0)
        if method == "flat":
            v *= 2
        out[k] = v
    return out


def reconstruct(front_img, measurements: dict, units="cm", style_override=None, back_img=None,
                options=None, iterations=3, colour_refine=False, mask=None):
    """mask: optional user-supplied garment mask (bool HxW, same size as the photo) — overrides
    automatic segmentation; the most reliable option for real photos with mannequins/hangers."""
    img = np.asarray(Image.open(front_img).convert("RGB") if isinstance(front_img, str) else front_img)
    pa = analyse(img, mask_hint=mask, colour_refine=colour_refine and mask is None)
    f = pa.features
    user = _to_cm(measurements, units)
    style_override = style_override or {}
    src = {k: "user" for k in user}

    style = {"silhouette": f.get("silhouette", "a-line"),
             "neckline": f.get("neckline", "round"),
             "sleeve": "none"}
    if f.get("sleeve_detected") or "sleeve_length" in user:
        if "sleeve_length" in user:
            style["sleeve"] = "short" if user["sleeve_length"] < 35 else "long"
        else:
            style["sleeve"] = "long" if f.get("sleeve_end_rel", 0) > 0.42 else "short"
    style_src = {k: "photo" for k in style}
    for k, v in style_override.items():
        if v:
            style[k] = v; style_src[k] = "user"

    nf, famp = FOLD_DEFAULTS[style["silhouette"]]
    opts = {"fold_count": nf, "fold_amplitude": famp}
    opts.update(options or {})
    Hpx = pa.bottom - pa.top
    d_hem = 0.93 * user["dress_length"]
    gen = spec = None
    for it in range(iterations):
        scale = d_hem / Hpx            # cm per pixel (vertical)
        m = dict(user)

        def infer(key, val, why, lo=None, hi=None):
            if key in user or val is None or not np.isfinite(val):
                return
            if (lo is not None and val < lo) or (hi is not None and val > hi):
                src[key] = f"default (photo cue {val:.1f} cm rejected as implausible)"
                return
            m[key] = float(val); src[key] = f"photo ({why})"

        L = user["dress_length"]

        infer("armhole_depth", f["underarm_rel"] * d_hem, "underarm row", lo=12, hi=min(32, 0.45 * L))
        if f.get("waist_contrast", 0) >= 0.12:   # weak minima are unreliable -> default
            arm = m.get("armhole_depth", 0.115 * user["bust"] + 9.5)
            infer("waist_position", f["waist_rel"] * d_hem, "min torso width row", lo=arm + 8, hi=min(60, L - 10))
        infer("neck_width", f["neck_width_px"] * scale, "HPS peaks", lo=10, hi=m.get("shoulder_width", 60) - 6)
        if "front_neck_drop_rel" in f:
            infer("front_neck_drop", f["front_neck_drop_rel"] * d_hem, "shaded neck opening", lo=1, hi=30)
        infer("back_neck_drop", max(0.5, f["back_neck_drop_rel"] * d_hem), "top contour dip", lo=0.5, hi=15)
        # sleeve angle from the detected far end of the sleeve
        if style["sleeve"] != "none" and "sleeve_angle_deg" in f:
            opts["sleeve_angle_deg"] = float(np.clip(f["sleeve_angle_deg"], 15, 85))
            src["sleeve_angle_deg"] = "photo (line fit through sleeve run centres)"
        # profile anchors from the photo width profile (only below the bust)
        w = f["width_profile_px"]
        spec0 = parse_spec({"style": style, "measurements": m, "options": opts})
        mm = spec0.m
        meas_lv = [(mm["armhole_depth"] + 2.5, mm["bust"]), (mm["waist_position"], mm["waist"])]
        if mm["hip_position"] < d_hem - 3:
            meas_lv.append((mm["hip_position"], mm["hip"]))
        meas_lv.append((d_hem - 0.5, mm["hem"]))
        row = lambda d: int(np.clip(pa.top + d / scale, pa.top, pa.bottom))
        ks = [(d, c / w[row(d)]) for d, c in meas_lv if np.isfinite(w[row(d)]) and w[row(d)] > 0]
        anchors = []
        if len(ks) >= 2:
            kd, kv = np.array([k[0] for k in ks]), np.array([k[1] for k in ks])
            for d in np.arange(mm["armhole_depth"] + 6, d_hem - 3, 4.0):
                wd = w[row(d)]
                if np.isfinite(wd):
                    anchors.append((float(d), float(wd * np.interp(d, kd, kv))))
        opts["profile_anchors"] = anchors
        spec = parse_spec({"style": style, "measurements": m, "options": opts})
        gen = DressGenerator(spec)
        dm = gen.build()
        d_hem = dm.params["d_hem_vertical"]

    for k in spec.m:
        src.setdefault(k, "default")
    tex, uv = project_texture(pa, dm, gen, scale, back_img)
    report = {"style": style, "style_source": style_src, "measurements_cm": spec.m, "source": src,
              "photo_features": {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                                 for k, v in f.items() if k != "width_profile_px" and not k.startswith("_")},
              "confidence": pa.confidence, "notes": spec.notes, "options": {k: v for k, v in opts.items() if k != "profile_anchors"},
              "n_profile_anchors": len(opts.get("profile_anchors", []))}
    return dm, tex, uv, report, pa


def project_texture(pa, dm, gen: DressGenerator, scale, back_img=None):
    """Front/back photo projection into a 2-wide atlas. Returns (PIL atlas, uv per render vertex)."""
    H, W = pa.mask.shape
    front = pa.fill
    if back_img is not None:
        bimg = np.asarray(Image.open(back_img).convert("RGB") if isinstance(back_img, str) else back_img)
        pb = analyse(bimg)
        back, bcx, btop, bscale = pb.fill, pb.cx, pb.top, (gen.d_hem / (pb.bottom - pb.top))
        bleft, bright = pb.left, pb.right
        if back.shape != front.shape:
            back = np.asarray(Image.fromarray(back).resize((W, H)))
    else:
        back, bcx, btop, bscale, bleft, bright = front, pa.cx, pa.top, scale, pa.left, pa.right
    atlas = np.concatenate([front, back], axis=1)
    V = dm.render_vertices
    y_hps = dm.landmarks["y_hps"]
    d = y_hps - V[:, 1]
    # per-row horizontal scale: visible torso half-width (px) / mesh half-width (cm)
    x, z, arc = gen.half_curves(np.clip(d, 0, gen.d_hem))
    half_cm = x.max(1)
    uv = np.zeros((len(V), 2))
    for is_back, sel in ((False, dm.panel != 1), (True, dm.panel == 1)):
        cx, top, sc, L, R = (bcx, btop, bscale, bleft, bright) if is_back else (pa.cx, pa.top, scale, pa.left, pa.right)
        rows = np.clip(top + d[sel] / sc, 0, H - 1)
        ri = rows.astype(int)
        half_px = (R[ri] - L[ri] + 1) / 2
        sx = np.where(np.isfinite(half_px) & (d[sel] > gen.d_arm + 2) & (dm.panel[sel] < 2),
                      half_px / np.maximum(half_cm[sel], 1e-3), 1 / sc)
        xs = V[sel, 0] * (-1 if is_back else 1)
        if is_back and back_img is None:
            # no back photo: back assumed to look like the front; avoid sampling the dark front
            # neck opening by clamping rows inside the neck region to just below the neckline
            m = gen.spec.m
            dd = d[sel]
            in_neck = (np.abs(V[sel, 0]) < m["neck_width"] / 2 + 1.0) & (dd < m["front_neck_drop"] + 1.5)
            rows = np.where(in_neck, np.clip(top + (m["front_neck_drop"] + 1.5) / sc, 0, H - 1), rows)
            ri = rows.astype(int)
        cols = np.clip(cx + xs * sx, 0, W - 1)
        u = (cols + (W if is_back else 0)) / (2 * W)
        uv[sel] = np.stack([u, 1 - rows / (H - 1)], 1)
    return Image.fromarray(atlas), uv
