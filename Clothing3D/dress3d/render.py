"""Tiny CPU software rasterizer (numpy). Used for synthetic reference 'photos' and for screenshots.

Perspective pinhole camera orbiting the vertical axis. Two-sided Lambert shading + ambient,
bilinear texture sampling. Not physically based; adequate for silhouettes and texture checks.
"""
from __future__ import annotations

import numpy as np
from PIL import Image


def _rot_y(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def _rot_x(deg):
    a = np.radians(deg)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def render(V, F, uv=None, texture=None, vcolor=None, yaw=0.0, elev=5.0, size=(512, 768), fov_deg=30.0,
           center=None, extent=None, bg=(232, 230, 226), light=(0.35, 0.45, 0.82), return_buffers=False,
           interior_darkening=0.55):
    """V (N,3) any unit, y-up, front = +z. yaw rotates the object (0 = front view).

    extent: scene height to frame (same units as V); default = bbox max dimension * 1.15.
    """
    V = np.asarray(V, float); F = np.asarray(F)
    W, H = size
    center = V.min(0) * 0.5 + V.max(0) * 0.5 if center is None else np.asarray(center, float)
    extent = (V.max(0) - V.min(0)).max() * 1.15 if extent is None else extent
    R = _rot_x(elev) @ _rot_y(-yaw)
    P = (V - center) @ R.T
    f = 1.0 / np.tan(np.radians(fov_deg) / 2)
    dist = extent / 2 * f + 0.0                       # camera distance so `extent` fills the height
    zc = dist - P[:, 2]                               # depth (camera looks toward -z from +z)
    px = (P[:, 0] * f / zc) * (H / 2) + W / 2
    py = H / 2 - (P[:, 1] * f / zc) * (H / 2)
    zbuf = np.full((H, W), np.inf); fid = np.full((H, W), -1, np.int64)
    b1 = np.zeros((H, W)); b2 = np.zeros((H, W))
    tri_x, tri_y, tri_z = px[F], py[F], zc[F]
    for i in range(len(F)):
        x0, x1, x2 = tri_x[i]; y0, y1, y2 = tri_y[i]
        xmin, xmax = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), W - 1))
        ymin, ymax = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), H - 1))
        if xmin > xmax or ymin > ymax:
            continue
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        xs, ys = np.meshgrid(np.arange(xmin, xmax + 1) + 0.5, np.arange(ymin, ymax + 1) + 0.5)
        l0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / den
        l1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / den
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any():
            continue
        z = l0 * tri_z[i, 0] + l1 * tri_z[i, 1] + l2 * tri_z[i, 2]
        sub = zbuf[ymin:ymax + 1, xmin:xmax + 1]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
        fid[ymin:ymax + 1, xmin:xmax + 1][upd] = i
        b1[ymin:ymax + 1, xmin:xmax + 1][upd] = l0[upd]
        b2[ymin:ymax + 1, xmin:xmax + 1][upd] = l1[upd]

    # background: soft vertical gradient
    g = np.linspace(1.03, 0.92, H)[:, None, None]
    img = np.clip(np.array(bg, float)[None, None, :] * g, 0, 255) * np.ones((H, W, 3))
    mask = fid >= 0
    if mask.any():
        fi = fid[mask]; l0 = b1[mask]; l1 = b2[mask]; l2 = 1 - l0 - l1
        tri = F[fi]
        # smooth vertex normals in camera space
        Pn = P
        fn = np.cross(Pn[F[:, 1]] - Pn[F[:, 0]], Pn[F[:, 2]] - Pn[F[:, 0]])
        vn = np.zeros_like(Pn)
        for k in range(3):
            np.add.at(vn, F[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        n = vn[tri[:, 0]] * l0[:, None] + vn[tri[:, 1]] * l1[:, None] + vn[tri[:, 2]] * l2[:, None]
        n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        # view vector ~ +z; two-sided
        facing = np.sign(np.einsum("ij,ij->i", fn[fi], np.tile([0, 0, 1.0], (len(fi), 1))))
        n = n * np.where(facing == 0, 1, facing)[:, None]
        Ld = np.array(light) / np.linalg.norm(light)
        lam = np.clip(n @ Ld, 0, 1)
        back = np.clip(n @ np.array([-0.4, 0.2, 0.6]) / np.linalg.norm([-0.4, 0.2, 0.6]), 0, 1)
        shade = 0.30 + 0.62 * lam + 0.12 * back
        # surfaces seen from their inner side (inside of the garment) are self-occluded in reality:
        # approximate with a constant darkening (documented approximation, not a GI simulation)
        if interior_darkening is not None:
            shade = np.where(facing < 0, shade * interior_darkening, shade)
        if texture is not None and uv is not None:
            tex = np.asarray(texture.convert("RGB"), float)
            th, tw = tex.shape[:2]
            u = uv[tri[:, 0], 0] * l0 + uv[tri[:, 1], 0] * l1 + uv[tri[:, 2], 0] * l2
            v = uv[tri[:, 0], 1] * l0 + uv[tri[:, 1], 1] * l1 + uv[tri[:, 2], 1] * l2
            tx = np.clip(u * (tw - 1), 0, tw - 1); ty = np.clip((1 - v) * (th - 1), 0, th - 1)
            x0 = np.floor(tx).astype(int); y0 = np.floor(ty).astype(int)
            x1 = np.minimum(x0 + 1, tw - 1); y1 = np.minimum(y0 + 1, th - 1)
            fx = (tx - x0)[:, None]; fy = (ty - y0)[:, None]
            col = (tex[y0, x0] * (1 - fx) * (1 - fy) + tex[y0, x1] * fx * (1 - fy)
                   + tex[y1, x0] * (1 - fx) * fy + tex[y1, x1] * fx * fy)
        elif vcolor is not None:
            vc = np.asarray(vcolor, float)[:, :3]
            col = vc[tri[:, 0]] * l0[:, None] + vc[tri[:, 1]] * l1[:, None] + vc[tri[:, 2]] * l2[:, None]
        else:
            col = np.tile([176.0, 38, 62], (len(fi), 1))
        img[mask] = np.clip(col * shade[:, None], 0, 255)
    out = Image.fromarray(img.astype(np.uint8))
    if return_buffers:
        return out, mask
    return out


def turntable(V, F, yaws=(0, 90, 180, 270), **kw):
    """Concatenate several views horizontally."""
    ims = [render(V, F, yaw=y, **kw) for y in yaws]
    W, H = ims[0].size
    sheet = Image.new("RGB", (W * len(ims), H))
    for i, im in enumerate(ims):
        sheet.paste(im, (i * W, 0))
    return sheet
