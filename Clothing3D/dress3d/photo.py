"""Classical (non-learned) analysis of a front dress photograph.

Assumptions (documented limitations):
  * garment roughly upright, front-facing, centred, on a plain/contrasting background;
  * photo taken from a distance (weak perspective);
  * garment on a dress form/hanger or laid flat, sleeves not folded over the body.

Everything here uses OpenCV / NumPy / SciPy only — no model downloads, nothing leaves the machine.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage


@dataclass
class PhotoAnalysis:
    mask: np.ndarray                    # bool HxW
    image: np.ndarray                   # uint8 HxWx3 RGB
    fill: np.ndarray                    # image with background filled by nearest garment colour
    top: int                            # HPS row
    bottom: int                         # hem row
    cx: float                           # centre column
    left: np.ndarray                    # per-row torso run left edge (nan outside)
    right: np.ndarray
    full_left: np.ndarray               # per-row outermost garment pixel
    full_right: np.ndarray
    features: dict = field(default_factory=dict)
    confidence: dict = field(default_factory=dict)


def segment(img_rgb: np.ndarray, mask_hint: np.ndarray | None = None) -> np.ndarray:
    """Background estimated from the image border; refined with GrabCut."""
    if mask_hint is not None:
        return mask_hint.astype(bool)
    H, W = img_rgb.shape[:2]
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(3, min(H, W) // 40)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    bd = np.linalg.norm(border - bg, axis=1)
    inl = bd < max(20.0, np.median(bd) * 3)          # ignore garment pixels touching the frame
    bg = np.median(border[inl], axis=0)
    spread = np.percentile(np.linalg.norm(border[inl] - bg, axis=1), 95)
    dist = np.linalg.norm(lab - bg, axis=2)
    thr = float(np.clip(2.5 * spread, 12.0, 40.0))
    gc = np.full((H, W), cv2.GC_PR_BGD, np.uint8)
    gc[dist > thr] = cv2.GC_PR_FGD
    gc[dist > 2.5 * thr] = cv2.GC_FGD
    frame = np.zeros((H, W), bool)
    frame[:b], frame[-b:], frame[:, :b], frame[:, -b:] = True, True, True, True
    gc[frame & (dist < thr)] = cv2.GC_BGD
    bgm, fgm = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), gc, None, bgm, fgm, 4, cv2.GC_INIT_WITH_MASK)
        m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
    except cv2.error:
        m = dist > thr
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    lbl, n = ndimage.label(m)
    if n == 0:
        raise ValueError("no garment found")
    sizes = ndimage.sum(m, lbl, range(1, n + 1))
    m = lbl == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(m)


def _fit_gmm(x, k, iters=40, seed=0):
    """Full-covariance Gaussian mixture by EM (k-means init). Returns (weights, means, covs)."""
    from scipy.cluster.vq import kmeans2
    x = x.astype(np.float64)
    _, lab_ = kmeans2(x, k, seed=seed, minit="++")
    w = np.array([max(np.mean(lab_ == j), 1e-3) for j in range(k)]); w /= w.sum()
    mu = np.array([x[lab_ == j].mean(0) if np.any(lab_ == j) else x[np.random.default_rng(j).integers(len(x))] for j in range(k)])
    cov = np.array([np.cov(x.T) + np.eye(3) for _ in range(k)])
    for _ in range(iters):
        lp = _gmm_component_logp((w, mu, cov), x)
        lse = np.logaddexp.reduce(lp, axis=1, keepdims=True)
        r = np.exp(lp - lse)
        nk = r.sum(0) + 1e-9
        w = nk / nk.sum()
        mu = (r.T @ x) / nk[:, None]
        for j in range(k):
            d = x - mu[j]
            cov[j] = (r[:, j, None] * d).T @ d / nk[j] + np.eye(3) * 1.0
    return w, mu, cov


def _gmm_component_logp(g, x):
    w, mu, cov = g
    out = np.empty((len(x), len(w)))
    for j in range(len(w)):
        inv = np.linalg.inv(cov[j]); _, logdet = np.linalg.slogdet(cov[j])
        d = x - mu[j]
        out[:, j] = np.log(w[j]) - 0.5 * (np.einsum("ij,jk,ik->i", d, inv, d) + logdet + 3 * np.log(2 * np.pi))
    return out


def _gmm_loglik(g, x):
    return np.logaddexp.reduce(_gmm_component_logp(g, x.astype(np.float64)), axis=1)


def refine_garment_mask(img_rgb: np.ndarray, mask: np.ndarray, k: int = 4, keep_pct: float = 2.0):
    """Remove non-garment foreground (mannequin, stand, hanger) by colour.

    A k-component Gaussian mixture (OpenCV EM) is fitted to Lab colours sampled from the central
    skirt region (30-75 % of the mask height, central half of its width), where the garment is
    almost certainly visible. Pixels whose log-likelihood is below the `keep_pct` percentile of the
    training samples are dropped; the largest component is kept and holes are filled.
    Assumes the mannequin/stand colours differ from the fabric colours.
    """
    rows = np.flatnonzero(mask.any(1))
    top, bot = rows[0], rows[-1]
    H = bot - top
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    r0, r1 = top + int(0.30 * H), top + int(0.75 * H)
    sample = []
    for y in range(r0, r1, 2):
        xs = np.flatnonzero(mask[y])
        if len(xs) < 4:
            continue
        a, b = xs[0], xs[-1]
        q = (b - a) // 4
        sample.append(lab[y, a + q:b - q:2])
    sample = np.concatenate(sample).reshape(-1, 3)
    if len(sample) < 200:
        return mask
    gmm = _fit_gmm(sample[np.random.default_rng(0).choice(len(sample), min(len(sample), 6000), replace=False)], k)
    loglik = lambda x: _gmm_loglik(gmm, x)
    thr = np.percentile(loglik(sample[np.random.default_rng(1).choice(len(sample), min(len(sample), 1500), replace=False)]), keep_pct)
    ys, xs = np.nonzero(mask)
    # evaluate on a coarse grid for speed, then upsample
    step = 2
    sub = mask[::step, ::step]
    sy, sx = np.nonzero(sub)
    ll = loglik(lab[sy * step, sx * step])
    keep_sub = np.zeros_like(sub)
    keep_sub[sy, sx] = ll >= thr
    keep = cv2.resize(keep_sub.astype(np.uint8), (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool) & mask
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)).astype(bool)
    lbl, n = ndimage.label(keep)
    if n == 0:
        return mask
    sizes = ndimage.sum(keep, lbl, range(1, n + 1))
    keep = lbl == (1 + int(np.argmax(sizes)))
    return ndimage.binary_fill_holes(keep)


def _runs(row):
    idx = np.flatnonzero(row)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[cuts + 1]]
    ends = np.r_[idx[cuts], idx[-1]]
    return list(zip(starts, ends))


def analyse(img_rgb: np.ndarray, mask_hint=None, colour_refine=False) -> PhotoAnalysis:
    mask = segment(img_rgb, mask_hint)
    if colour_refine:
        mask = refine_garment_mask(img_rgb, mask)
    H, W = mask.shape
    rows = np.flatnonzero(mask.any(1))
    top, bottom = int(rows[0]), int(rows[-1])
    # centre column: median of mask column-centroids over the lower half (skirt is symmetric)
    lower = mask[(top + bottom) // 2: bottom + 1]
    cols = np.arange(W)
    cx = float(np.median([(cols * r).sum() / r.sum() for r in lower if r.any()]))
    left = np.full(H, np.nan); right = np.full(H, np.nan)
    fl = np.full(H, np.nan); fr = np.full(H, np.nan)
    nruns = np.zeros(H, int)
    side_runs = {}
    for y in range(top, bottom + 1):
        rr = _runs(mask[y])
        if not rr:
            continue
        nruns[y] = len(rr)
        fl[y], fr[y] = rr[0][0], rr[-1][1]
        c = min(rr, key=lambda r: 0 if r[0] <= cx <= r[1] else min(abs(r[0] - cx), abs(r[1] - cx)))
        left[y], right[y] = c
        others = [r for r in rr if r != c and (r[1] - r[0]) > 1]
        if others:
            side_runs[y] = others
    # background fill for texture sampling (nearest garment pixel)
    _, (iy, ix) = ndimage.distance_transform_edt(~mask, return_indices=True)
    fill = img_rgb[iy, ix]
    pa = PhotoAnalysis(mask=mask, image=img_rgb, fill=fill, top=top, bottom=bottom, cx=cx,
                       left=left, right=right, full_left=fl, full_right=fr)
    _features(pa, nruns, side_runs)
    return pa


def _smooth(a, k=5):
    a = a.copy(); ok = ~np.isnan(a)
    if ok.sum() < k:
        return a
    a[ok] = ndimage.uniform_filter1d(a[ok], k)
    return a


def _features(pa: PhotoAnalysis, nruns, side_runs):
    """Infer style + relative positions (fractions of the HPS->hem pixel height)."""
    f, conf = pa.features, pa.confidence
    top, bot = pa.top, pa.bottom
    Hpx = bot - top
    w = _smooth(pa.right - pa.left + 1, 7)
    wf = pa.full_right - pa.full_left + 1
    rel = lambda y: (y - top) / Hpx
    # --- sleeves: separate side runs in the upper 60 % of the garment
    sr = [y for y in side_runs if 0.06 < rel(y) < 0.6]
    if len(sr) > 0.02 * Hpx:
        y_sep = min(sr); y_end = max(sr)
        f["sleeve_detected"] = True
        f["underarm_rel"] = rel(y_sep)
        # sleeve centre-line per row (right sleeve) -> angle below horizontal
        pts = []
        for y in sr:
            for r in side_runs[y]:
                if (r[0] + r[1]) / 2 > pa.cx:
                    pts.append(((r[0] + r[1]) / 2, y))
        # sleeve extent: farthest garment pixel from the centre in the upper part
        ext = np.nanmax(np.abs(np.r_[pa.full_right[top:y_end + 1] - pa.cx, pa.cx - pa.full_left[top:y_end + 1]]))
        f["sleeve_reach_px"] = float(ext)
        f["sleeve_end_rel"] = rel(y_end)
        if len(pts) > 5:
            p = np.array(pts)
            far = p[np.argmax(p[:, 0])]
            f["_sleeve_far_px"] = far.tolist()
            # sleeve axis: horizontal run centres of a slanted tube lie on its axis projection
            a, _ = np.polyfit(p[:, 1], p[:, 0], 1)          # x = a*y + b  (image coords, y down)
            f["sleeve_angle_deg"] = float(np.degrees(np.arctan2(1.0, a)))
        conf["sleeve"] = "detected from separated sleeve silhouettes"
    else:
        # sleeves that do not separate from the body: look for a width bump at the top
        up = slice(top, top + int(0.3 * Hpx))
        bump = np.nanmax(wf[up]) / np.nanmedian(wf[top + int(0.3 * Hpx): top + int(0.4 * Hpx)])
        f["sleeve_detected"] = bool(bump > 1.25)
        conf["sleeve"] = f"no separated sleeve; top/bust width ratio {bump:.2f}"
    # --- neckline from the top contour: HPS = highest garment pixels left/right of centre
    topc = np.full(pa.mask.shape[1], np.nan)
    colsany = pa.mask.any(0)
    topc[colsany] = np.argmax(pa.mask[:, colsany], axis=0)
    xs = np.arange(len(topc))
    band = (np.abs(xs - pa.cx) < 0.25 * Hpx)
    lc = np.where(band & (xs < pa.cx), topc, np.nan); rc = np.where(band & (xs > pa.cx), topc, np.nan)
    if np.all(np.isnan(lc)) or np.all(np.isnan(rc)):
        raise ValueError("cannot locate shoulders")
    hl, hr = int(np.nanargmin(lc)), int(np.nanargmin(rc))
    # the HPS is the *innermost* of the highest pixels (shoulder slopes down outwards)
    hl = int(np.max(np.flatnonzero(lc <= np.nanmin(lc) + 1)))
    hr = int(np.min(np.flatnonzero(rc <= np.nanmin(rc) + 1)))
    f["neck_width_px"] = float(hr - hl)
    centre_top = np.nanmax(topc[int(pa.cx) - 2: int(pa.cx) + 3])
    f["back_neck_drop_rel"] = rel(centre_top) - rel(min(lc[hl], rc[hr]))
    # front neckline: through the neck opening one sees the shaded inside of the back panel.
    # Compare, row by row, the centre strip with fabric just outside the HPS (same row => same
    # print stripe), and take the last row (from the top) where the centre is clearly darker.
    lum = cv2.cvtColor(pa.image, cv2.COLOR_RGB2GRAY).astype(float)
    nwp = hr - hl
    y_hps = min(lc[hl], rc[hr])
    a0 = int(centre_top) + 1; a1 = top + int(0.30 * Hpx)

    def dark_depth(off_frac):
        c0 = int(pa.cx + off_frac * nwp)
        cen = np.array([np.median(lum[y, c0 - 2:c0 + 3]) for y in range(a0, a1)])
        ref = np.array([np.median(np.r_[lum[y, int(pa.cx - 0.8 * nwp):int(pa.cx - 0.6 * nwp)],
                                        lum[y, int(pa.cx + 0.6 * nwp):int(pa.cx + 0.8 * nwp)]]) for y in range(a0, a1)])
        ratio = cen / np.maximum(ref, 1)
        dark = ratio < 0.8
        if not dark[:6].any():
            return None, ratio
        k0 = int(np.argmax(dark))
        # deepest dark row such that >= 65 % of rows from the first dark row are dark
        # (prints seen inside the opening interrupt the run; fabric below is never dark)
        frac = np.cumsum(dark[k0:]) / np.arange(1, len(dark) - k0 + 1)
        ok = np.flatnonzero(dark[k0:] & (frac >= 0.65))
        k = k0 + (int(ok[-1]) + 1 if len(ok) else 0)
        return a0 + k, ratio

    yfn, ratio = dark_depth(0.0)
    if yfn is not None:
        f["front_neck_drop_rel"] = rel(yfn) - rel(y_hps)
        conf["front_neck_drop"] = f"shaded neck opening, min centre/fabric ratio {np.nanmin(ratio):.2f}"
        yh, _ = dark_depth(0.25)
        d_full = rel(yfn) - rel(y_hps)
        if d_full < 0.05:
            f["neckline"] = "boat"
        elif yh is not None:
            r = (rel(yh) - rel(y_hps)) / max(d_full, 1e-6)
            f["neckline_ratio"] = float(r)
            # expected depth at 50 % of the neck half-width: v 0.5, round 0.87, square 1.0
            f["neckline"] = "v" if r < 0.68 else ("square" if r > 0.94 else "round")
    else:
        conf["front_neck_drop"] = "neck opening not detected; default used"
    # --- torso widths below the underarm
    ua = f.get("underarm_rel")
    if ua is None:
        # sleeveless: armhole bottom = first row below the armhole scoop where the outer contour
        # stops moving outwards quickly (curvature of the side contour)
        # sleeveless: below the armhole scoop (width minimum) the contour moves outwards until the
        # side seam starts; armhole bottom = first row reaching 98 % of the local maximum width
        up0, up1 = top + int(0.06 * Hpx), top + int(0.30 * Hpx)
        ww = _smooth(w, 9)[up0:up1]
        k = int(np.nanargmin(ww))
        win = _smooth(w, 9)[up0 + k: up0 + k + int(0.15 * Hpx)]
        j = int(np.argmax(win >= 0.98 * np.nanmax(win)))
        f["underarm_rel"] = rel(up0 + k + j)
        conf["armhole_depth"] = "estimated: 98 % width-recovery row below the armhole scoop (sleeveless)"
    ua = f["underarm_rel"]
    y_ua = top + int(ua * Hpx)
    lo = y_ua + int(0.04 * Hpx); hi = top + int(0.70 * Hpx)
    seg = w[lo:hi]
    if len(seg) > 5:
        k = int(np.nanargmin(seg)); y_w = lo + k
        wb = np.nanmedian(w[y_ua + int(0.02 * Hpx): y_ua + int(0.06 * Hpx)])
        f["waist_rel"] = rel(y_w)
        f["waist_contrast"] = float(1 - w[y_w] / max(wb, np.nanmax(w[y_w:hi + 1])))
        conf["waist_position"] = f"min torso width, contrast {f['waist_contrast']:.2f}"
    f["width_profile_px"] = w
    hem_w = np.nanmax(w[bot - int(0.08 * Hpx): bot + 1])   # widest hem row (hem curves in perspective)
    bust_w = np.nanmedian(w[y_ua + int(0.02 * Hpx): y_ua + int(0.06 * Hpx)])
    f["hem_to_bust_width"] = float(hem_w / bust_w)
    wc = f.get("waist_contrast", 0)
    if f["hem_to_bust_width"] < 1.15:
        f["silhouette"] = "straight"
    elif wc > 0.12 and f["hem_to_bust_width"] > 1.6:
        f["silhouette"] = "fit-and-flare"
    else:
        f["silhouette"] = "a-line"
