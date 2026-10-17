"""Render a GLB from the front, side and back into one PNG (no GPU needed).

Simple painter's-algorithm renderer: orthographic camera, one colour per
triangle sampled from its texture, flat shading. Good enough to check a
garment's shape and print, and for screenshots in the report.

Example:
  python backend/render_preview.py output/dress01_on_avatar.glb output/dress01_preview.png
"""

import argparse

import numpy as np
import trimesh
from PIL import Image, ImageDraw

VIEW_SIZE = (520, 900)
BACKGROUND = (236, 236, 240)
# Camera direction (towards the viewer) and screen-right axis per view.
VIEWS = {
    "front": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
    "side": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])),
    "back": (np.array([0.0, 0.0, -1.0]), np.array([-1.0, 0.0, 0.0])),
}


def face_colours(mesh):
    """RGB per face: texture sampled at the face centre, else the base colour."""
    visual = mesh.visual
    material = getattr(visual, "material", None)
    image = None
    if material is not None:
        image = getattr(material, "baseColorTexture", None) or getattr(material, "image", None)
    uv = getattr(visual, "uv", None)
    if image is not None and uv is not None and len(uv) == len(mesh.vertices):
        texture = np.asarray(image.convert("RGB"))
        h, w = texture.shape[:2]
        # Average the centre and the three corners so fine prints don't speckle.
        corners = uv[mesh.faces]
        samples = np.concatenate([corners, corners.mean(axis=1, keepdims=True)], axis=1) % 1.0
        x = np.clip((samples[..., 0] * w).astype(int), 0, w - 1)
        y = np.clip(((1.0 - samples[..., 1]) * h).astype(int), 0, h - 1)
        return texture[y, x].astype(float).mean(axis=1)
    colour = np.array([200.0, 170.0, 150.0])
    factor = getattr(material, "baseColorFactor", None) if material is not None else None
    if factor is not None:
        colour = np.asarray(factor[:3], float)
        if colour.max() <= 1.0:
            colour = colour * 255
    elif getattr(visual, "kind", None) == "face":
        return visual.face_colors[:, :3].astype(float)
    return np.tile(colour, (len(mesh.faces), 1))


def scene_meshes(path):
    scene = trimesh.load(path, force="scene")
    meshes = []
    for node in scene.graph.nodes_geometry:
        transform, name = scene.graph[node]
        mesh = scene.geometry[name]
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh = mesh.copy()
        mesh.apply_transform(transform)
        meshes.append(mesh)
    return meshes


def render(meshes, toward_viewer, right):
    up = np.array([0.0, 1.0, 0.0])
    triangles = np.concatenate([m.triangles for m in meshes])
    normals = np.concatenate([m.face_normals for m in meshes])
    colours = np.concatenate([face_colours(m) for m in meshes])

    lo = np.min([m.bounds[0] for m in meshes], axis=0)
    hi = np.max([m.bounds[1] for m in meshes], axis=0)
    width, height = VIEW_SIZE
    scale = 0.92 * min(height / (hi[1] - lo[1]), width / np.linalg.norm(hi[[0, 2]] - lo[[0, 2]]))
    centre = (lo + hi) / 2

    screen_x = (triangles - centre) @ right * scale + width / 2
    screen_y = height / 2 - (triangles - centre) @ up * scale
    depth = (triangles @ toward_viewer).mean(axis=1)

    light = toward_viewer + 0.6 * up + 0.3 * right
    light /= np.linalg.norm(light)
    facing = normals @ toward_viewer
    shade_normals = np.where(facing[:, None] < 0, -normals, normals)
    shade = 0.45 + 0.55 * np.clip(shade_normals @ light, 0, 1)
    shade = np.where(facing < 0, shade * 0.6, shade)  # inside of the garment
    shaded = np.clip(colours * shade[:, None], 0, 255).astype(int)

    image = Image.new("RGB", VIEW_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(image)
    for i in np.argsort(depth):
        pts = list(zip(screen_x[i], screen_y[i]))
        colour = tuple(shaded[i])
        draw.polygon(pts, fill=colour, outline=colour)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("glb", help="GLB file to render")
    parser.add_argument("png", help="output image")
    args = parser.parse_args()

    meshes = scene_meshes(args.glb)
    views = [render(meshes, *VIEWS[name]) for name in ("front", "side", "back")]
    sheet = Image.new("RGB", (VIEW_SIZE[0] * len(views), VIEW_SIZE[1]), BACKGROUND)
    for i, view in enumerate(views):
        sheet.paste(view, (i * VIEW_SIZE[0], 0))
    sheet.save(args.png)
    print(f"Saved {args.png}")


if __name__ == "__main__":
    main()
