"""Shared evaluation metrics for all approaches (same cameras, same metrics)."""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json
import os
import sys

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
import cv2

sys.path.insert(0, f"{ROOT}")
from dress3d.render import render  # noqa: E402

YAWS = {"front": 0, "right": 90, "back": 180, "left": 270}


def load_case(case_dir):
    c = json.load(open(os.path.join(case_dir, "case.json")))
    gt = trimesh.load(os.path.join(case_dir, "gt.glb"), force="mesh", process=False)
    return c, gt


def mesh_cm(m):
    return trimesh.Trimesh(np.asarray(m.vertices) * 100, m.faces, process=False)


def silhouettes(V, F, cam):
    out = {}
    for k, y in YAWS.items():
        _, mask = render(V, F, yaw=y, size=tuple(cam["size"]), fov_deg=cam["fov_deg"], elev=cam["elev"],
                         center=cam["center"], extent=cam["extent"], return_buffers=True)
        out[k] = mask
    return out


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def chamfer(m1, m2, n=20000, seed=0):
    p1, _ = trimesh.sample.sample_surface(m1, n, seed=seed)
    p2, _ = trimesh.sample.sample_surface(m2, n, seed=seed + 1)
    d12, _ = cKDTree(p2).query(p1); d21, _ = cKDTree(p1).query(p2)
    return {"chamfer_mean_cm": float((d12.mean() + d21.mean()) / 2),
            "recon_to_gt_p95_cm": float(np.percentile(d12, 95)),
            "gt_to_recon_p95_cm": float(np.percentile(d21, 95)),
            "hausdorff_cm": float(max(d12.max(), d21.max()))}


def delta_e(img_a, img_b, mask):
    la = cv2.cvtColor(np.asarray(img_a), cv2.COLOR_RGB2LAB).astype(float)
    lb = cv2.cvtColor(np.asarray(img_b), cv2.COLOR_RGB2LAB).astype(float)
    # OpenCV 8-bit Lab: L*255/100, a+128, b+128
    la[..., 0] *= 100 / 255; lb[..., 0] *= 100 / 255
    d = np.linalg.norm(la - lb, axis=2)[mask]
    return float(d.mean()) if d.size else float("nan")


def mean_colour_delta_e(img_a, mask_a, img_b, mask_b):
    """Difference of mean garment colour (robust to small misregistration)."""
    def mean_lab(img, m):
        lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(float)[m]
        lab[:, 0] *= 100 / 255
        return lab.mean(0)
    return float(np.linalg.norm(mean_lab(img_a, mask_a) - mean_lab(img_b, mask_b)))


def side_by_side(paths_or_imgs, out, labels=None, h=450):
    ims = [Image.open(p) if isinstance(p, str) else p for p in paths_or_imgs]
    ims = [im.convert("RGB").resize((int(im.width * h / im.height), h)) for im in ims]
    W = sum(i.width for i in ims)
    s = Image.new("RGB", (W, h + (22 if labels else 0)), (255, 255, 255))
    x = 0
    from PIL import ImageDraw
    dr = ImageDraw.Draw(s)
    for i, im in enumerate(ims):
        s.paste(im, (x, 22 if labels else 0))
        if labels:
            dr.text((x + 4, 4), labels[i], fill=(0, 0, 0))
        x += im.width
    s.save(out)
