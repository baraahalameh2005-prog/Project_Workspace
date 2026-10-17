"""Blender side of template_dress.py: drape the garment's starting shape on the
avatar with cloth simulation and export GLB. Run inside Blender:

  blender -b --factory-startup --python backend/blender_drape.py -- --avatar a.glb ^
      --rest dress_rest.npz --fabric texture.jpg --trim 0.9,0.9,0.9 ^
      --frames 60 --out dress.glb --out-with-avatar dress_on_avatar.glb

The starting shape is in the avatar's glTF frame (Y up); Blender is Z up, so
points are converted as (x, y, z) -> (x, -z, y) and back on export.
"""

import argparse
import sys

import bpy
import numpy as np

BUTTON_RADIUS_M = 0.009
# Medium-weight wool tweed: heavier and stiffer than Blender's cotton preset,
# a little softer than its denim one. Mass is per vertex, in kg.
TWEED = dict(quality=6, mass=0.5, air_damping=1.0, tension_stiffness=30.0, compression_stiffness=30.0,
             shear_stiffness=15.0, bending_stiffness=4.0)


def args_after_dashes():
    parser = argparse.ArgumentParser()
    for name in ("--avatar", "--rest", "--fabric", "--trim", "--out", "--out-with-avatar"):
        parser.add_argument(name, required=True)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--buttons-m", default="", help="button positions, metres below the shoulder line")
    parser.add_argument("--shoulder-y", type=float, default=0.0)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1:])


def to_blender(points):
    return np.stack([points[..., 0], -points[..., 2], points[..., 1]], axis=-1)


def make_materials(fabric_path, trim_rgb):
    fabric = bpy.data.materials.new("fabric")
    fabric.use_nodes = True
    bsdf = fabric.node_tree.nodes["Principled BSDF"]
    image = fabric.node_tree.nodes.new("ShaderNodeTexImage")
    image.image = bpy.data.images.load(fabric_path)
    fabric.node_tree.links.new(image.outputs["Color"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.85

    trim = bpy.data.materials.new("trim")
    trim.use_nodes = True
    bsdf = trim.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*trim_rgb, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.7
    button = bpy.data.materials.new("button")
    button.use_nodes = True
    bsdf = button.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (0.93, 0.92, 0.9, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.25
    for material in (fabric, trim):
        material.use_backface_culling = False
    return fabric, trim, button


def add_buttons(dress, depths, shoulder_y, material):
    """Small pearl buttons down the front centre, sitting on the draped surface."""
    buttons = []
    for depth in depths:
        z = shoulder_y - depth
        # Blender's -Y is the avatar's front: shoot a ray from in front of it.
        hit, location, normal, _ = dress.ray_cast((0.0, -2.0, z), (0.0, 1.0, 0.0))
        if not hit:
            continue
        bpy.ops.mesh.primitive_uv_sphere_add(radius=BUTTON_RADIUS_M, segments=16, ring_count=8,
                                             location=location + normal * 0.003)
        b = bpy.context.active_object
        b.scale = (1.0, 0.45, 1.0)
        b.data.materials.append(material)
        buttons.append(b)
    return buttons


def build_part(name, rings, uv_front, uv_back, trim, pin, closed=True):
    """A closed tube from rings (n, around, 3) as one mesh, with material 1 on
    the trim rings and a "pin" vertex group. uv_front/uv_back (n, around + 1, 2)
    give the photo texture on the front half (ring positions around/4 to
    3*around/4) and the back half; without them the part is all trim.
    closed=False: an open sheet (a scarf), rows of points instead of rings,
    textured all over from uv_front."""
    n, around = rings.shape[:2]
    verts = to_blender(rings).reshape(-1, 3)
    faces, face_trim = [], []
    for i in range(n - 1):
        for j in range(around if closed else around - 1):
            k = (j + 1) % around
            faces.append((i * around + j, (i + 1) * around + j, (i + 1) * around + k, i * around + k))
            face_trim.append(bool(trim[i] and trim[i + 1]))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], faces)

    uv = mesh.uv_layers.new(name="UVMap")
    q = around // 4
    per_row = around if closed else around - 1
    for f, poly in enumerate(mesh.polygons):
        i, j = divmod(f, per_row)
        poly.material_index = 1 if face_trim[f] or uv_front is None else 0
        if uv_front is None:
            continue
        table = uv_front if not closed or q <= j < 3 * q else uv_back
        corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
        for loop_index, (ri, cj) in zip(poly.loop_indices, corners):
            uv.data[loop_index].uv = tuple(table[ri, cj])

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    group = obj.vertex_groups.new(name="pin")
    if pin.ndim == 2:  # a weight per vertex (n, around)
        for i in range(n):
            for j in range(around):
                if pin[i, j] > 0:
                    group.add([i * around + j], float(pin[i, j]), "REPLACE")
        return obj
    for i in range(n):
        if pin[i] > 0:
            group.add(list(range(i * around, (i + 1) * around)), float(pin[i]), "REPLACE")
    return obj


def main():
    args = args_after_dashes()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    bpy.ops.import_scene.gltf(filepath=args.avatar)
    avatar = [o for o in scene.objects if o.type == "MESH"]
    for obj in avatar:
        obj.modifiers.new("collision", "COLLISION")
        obj.collision.thickness_outer = 0.006
        obj.collision.cloth_friction = 5.0
    bpy.ops.mesh.primitive_plane_add(size=4.0, location=(0, 0, 0))
    floor = bpy.context.active_object
    floor.modifiers.new("collision", "COLLISION")

    rest = np.load(args.rest)
    trim_rgb = [float(v) for v in args.trim.split(",")]
    fabric, trim, button = make_materials(args.fabric, trim_rgb)
    parts = []
    for name in [str(n) for n in rest["parts"]]:
        textured = name + "_uvf" in rest
        closed = bool(rest[name + "_closed"]) if name + "_closed" in rest else True
        obj = build_part(name, rest[name], rest[name + "_uvf"] if textured else None,
                         rest[name + "_uvb"] if textured else None, rest[name + "_trim"], rest[name + "_pin"], closed)
        obj.data.materials.append(fabric)
        obj.data.materials.append(trim)
        parts.append(obj)

    # One object, one cloth simulation (the pieces stay separate meshes).
    bpy.ops.object.select_all(action="DESELECT")
    for obj in parts:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    dress = bpy.context.active_object
    dress.name = "dress"

    # Blender's cloth solver runs on the CPU (the graphics card is not used),
    # so its cost is time and RAM: kept low with a coarse mesh, quality 6 and
    # collision quality 3 (Blender 4.2 has one quality for both collisions).
    cloth = dress.modifiers.new("cloth", "CLOTH")
    s = cloth.settings
    for name, value in TWEED.items():
        setattr(s, name, value)
    s.vertex_group_mass = "pin"  # bodice above the armpits and sleeves hold their shape
    c = cloth.collision_settings
    c.distance_min = 0.005
    c.collision_quality = 3
    c.use_self_collision = True  # skirt folds do not pass through each other
    c.self_distance_min = 0.003
    cloth.point_cache.frame_start = 1
    cloth.point_cache.frame_end = args.frames

    for f in range(1, args.frames + 1):
        scene.frame_set(f)
        if f % 10 == 0:
            print(f"cloth frame {f}/{args.frames}", flush=True)

    # Freeze the last frame into the mesh.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    draped = bpy.data.meshes.new_from_object(dress.evaluated_get(depsgraph))
    bpy.ops.ptcache.free_bake_all()  # nothing of the simulation cache stays in the file
    dress.modifiers.clear()
    dress.data = draped
    pin_index = dress.vertex_groups["pin"].index
    free = dress.vertex_groups.new(name="free")
    free.add([v.index for v in dress.data.vertices
              if all(g.group != pin_index or g.weight < 0.5 for g in v.groups)], 1.0, "REPLACE")
    smooth = dress.modifiers.new("smooth", "LAPLACIANSMOOTH")
    smooth.lambda_factor = 0.4
    smooth.iterations = 3
    smooth.use_volume_preserve = True
    smooth.vertex_group = "free"
    bpy.context.view_layer.objects.active = dress
    bpy.ops.object.modifier_apply(modifier="smooth")
    for poly in dress.data.polygons:
        poly.use_smooth = True
    bpy.data.objects.remove(floor)
    depths = [float(v) for v in args.buttons_m.split(",") if v]
    buttons = add_buttons(dress, depths, args.shoulder_y, button)

    bpy.ops.object.select_all(action="DESELECT")
    dress.select_set(True)
    for b in buttons:
        b.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.out, use_selection=True, export_format="GLB", export_yup=True)
    for obj in avatar:
        obj.modifiers.clear()
        obj.select_set(True)
    bpy.ops.export_scene.gltf(filepath=args.out_with_avatar, use_selection=True, export_format="GLB", export_yup=True)
    print("exported", args.out, flush=True)


main()
