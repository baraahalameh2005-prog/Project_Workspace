"""Textures, GLB export (metres, glTF convention), landmark sidecar and pattern SVG."""
from __future__ import annotations

import json

import numpy as np
import trimesh
from PIL import Image, ImageDraw

from .generator import DressMesh


def make_print_texture(atlas_size_cm, ppcm=8, base=(176, 38, 62), accent=(245, 232, 210),
                       kind="dots", spacing=4.0, radius=0.7):
    """Fabric print drawn in physical pattern space (cm), so print scale is real-world."""
    W, H = int(atlas_size_cm[0] * ppcm), int(atlas_size_cm[1] * ppcm)
    img = Image.new("RGB", (W, H), base)
    dr = ImageDraw.Draw(img)
    if kind == "dots":
        for j, y in enumerate(np.arange(0, atlas_size_cm[1], spacing)):
            off = spacing / 2 if j % 2 else 0
            for x in np.arange(off, atlas_size_cm[0], spacing):
                r = radius * ppcm
                dr.ellipse([x * ppcm - r, y * ppcm - r, x * ppcm + r, y * ppcm + r], fill=accent)
    elif kind == "stripes":
        for y in np.arange(0, atlas_size_cm[1], spacing):
            dr.rectangle([0, y * ppcm, W, (y + spacing * 0.35) * ppcm], fill=accent)
    elif kind == "plain":
        pass
    # mild weave noise so flat colours do not look synthetic
    arr = np.asarray(img).astype(np.int16)
    rng = np.random.default_rng(0)
    arr = np.clip(arr + rng.normal(0, 4, arr.shape[:2])[..., None], 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def pattern_uv(dm: DressMesh):
    W, H = dm.atlas_size_cm
    return np.stack([dm.pattern_uv[:, 0] / W, 1 - dm.pattern_uv[:, 1] / H], 1)


def to_trimesh(dm: DressMesh, texture: Image.Image | None = None, uv=None, color=(176, 38, 62)):
    v_m = dm.render_vertices / 100.0  # cm -> m
    if texture is None:
        mesh = trimesh.Trimesh(v_m, dm.render_faces, process=False)
        mesh.visual.face_colors = list(color) + [255]
        return mesh
    uv = pattern_uv(dm) if uv is None else uv
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=texture, metallicFactor=0.0,
                                              roughnessFactor=0.92, doubleSided=True)
    vis = trimesh.visual.TextureVisuals(uv=uv, material=mat)
    return trimesh.Trimesh(v_m, dm.render_faces, visual=vis, process=False)


def export_glb(mesh: trimesh.Trimesh, path):
    data = mesh.export(file_type="glb")
    with open(path, "wb") as f:
        f.write(data)
    return len(data)


def save_landmarks(dm: DressMesh, path, extra=None):
    out = {"units": "cm", "frame": "y-up, hem at y=0, +z = front, +x = garment's right side as viewed from front-left",
           "landmarks": dm.landmarks, "generator_params": dm.params}
    if extra:
        out.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def pattern_svg(dm: DressMesh, path):
    """Write the flattened front/back panel outlines (cm). Approximate: see README limitations."""
    outs = dm.outlines
    pad = 4
    smax = max(np.abs(o[:, 0]).max() for o in outs.values())
    hmax = max(o[:, 1].max() for o in outs.values())
    W = 2 * (2 * smax + pad) + pad
    H = hmax + 2 * pad
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}cm" height="{H}cm" viewBox="0 0 {W:.2f} {H:.2f}">']
    for i, (name, o) in enumerate(outs.items()):
        ox = pad + smax + i * (2 * smax + pad)
        pts = " ".join(f"{ox + x:.2f},{pad + y:.2f}" for x, y in o)
        parts.append(f'<polygon points="{pts}" fill="none" stroke="black" stroke-width="0.15"/>')
        parts.append(f'<text x="{ox:.1f}" y="{pad + hmax / 2:.1f}" font-size="3" text-anchor="middle">{name} (cut 1)</text>')
    parts.append(f'<text x="{pad}" y="{H - 1:.1f}" font-size="2">Units: cm. No seam allowance. Sleeve pattern not flattened.</text>')
    parts.append("</svg>")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
