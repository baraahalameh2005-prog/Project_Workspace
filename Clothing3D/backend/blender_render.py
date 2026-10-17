"""Render a GLB with Blender (Eevee) from the front, a three-quarter angle and
the back into one PNG, for checking a garment and for the report.

  blender -b --factory-startup --python backend/blender_render.py -- in.glb out.png
"""

import math
import sys

import bpy
from mathutils import Vector

VIEWS_DEG = (0, 35, 90, 180)  # camera angle around the model; 0 = front
SIZE = (520, 900)


def main():
    glb, png = sys.argv[sys.argv.index("--") + 1:][:2]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    bpy.ops.import_scene.gltf(filepath=glb)
    meshes = [o for o in scene.objects if o.type == "MESH"]
    zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
    bottom, top = min(zs), max(zs)
    height = top - bottom
    target = Vector((0.0, 0.0, bottom + height / 2))

    world = bpy.data.worlds.new("world")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.9, 0.9, 0.93, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.5
    scene.world = world
    for name, rot, energy in (("key", (50, 0, 30), 2.2), ("fill", (60, 0, -120), 0.8), ("rim", (70, 0, 180), 1.2)):
        light = bpy.data.lights.new(name, "SUN")
        light.energy = energy
        obj = bpy.data.objects.new(name, light)
        obj.rotation_euler = [math.radians(a) for a in rot]
        scene.collection.objects.link(obj)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = height * 1.08
    cam = bpy.data.objects.new("cam", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.engine = "BLENDER_EEVEE_NEXT"
    scene.render.resolution_x, scene.render.resolution_y = SIZE
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "Standard"

    paths = []
    for deg in VIEWS_DEG:
        a = math.radians(deg)
        # The model faces Blender's -Y.
        cam.location = target + Vector((3.0 * math.sin(a), -3.0 * math.cos(a), 0.0))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        path = png.replace(".png", f"_{deg}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        paths.append(path)

    # Side by side into one image.
    images = [bpy.data.images.load(p) for p in paths]
    w, h = SIZE
    sheet = bpy.data.images.new("sheet", w * len(images), h)
    pixels = [0.0] * (w * len(images) * h * 4)
    for k, image in enumerate(images):
        src = list(image.pixels)
        for row in range(h):
            start = (row * w * len(images) + k * w) * 4
            pixels[start:start + w * 4] = src[row * w * 4:(row + 1) * w * 4]
    sheet.pixels = pixels
    sheet.filepath_raw = png
    sheet.file_format = "PNG"
    sheet.save()
    print("saved", png, flush=True)


main()
