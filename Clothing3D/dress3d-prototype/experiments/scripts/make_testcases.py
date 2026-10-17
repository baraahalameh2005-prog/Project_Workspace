"""Create synthetic ground-truth dresses + reference 'photos' (SYNTHETIC — not real photographs).

Ground truth uses construction options the hybrid pipeline does NOT know (depth ratio, sleeve angle,
fold count/amplitude, waist/hip/armhole positions, neckline dims), so the hybrid has to infer them.
"""
import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
import json, os, sys
sys.path.insert(0, f"{ROOT}")
from dress3d.schema import parse_spec
from dress3d.generator import DressGenerator
from dress3d import export as ex
from dress3d.render import render

OUT = f"{ROOT}/experiments/inputs/synthetic"
CASES = {
 "T1_straight_sleeveless": {
   "style": {"silhouette": "straight", "sleeve": "none", "neckline": "round"},
   "measurements": {"dress_length": 90, "bust": 96, "waist": 92, "hip": 100, "hem": 102, "shoulder_width": 37,
                    "neck_width": 18, "front_neck_drop": 8, "back_neck_drop": 2, "armhole_depth": 21,
                    "waist_position": 40, "hip_position": 60},
   "options": {"depth_ratio": 0.68, "fold_count": 0, "fold_amplitude": 0},
   "print": {"kind": "stripes", "base": (30, 45, 90), "accent": (235, 235, 240), "spacing": 3.0}},
 "T2_aline_vneck": {
   "style": {"silhouette": "a-line", "sleeve": "none", "neckline": "v"},
   "measurements": {"dress_length": 100, "bust": 90, "waist": 76, "hip": 98, "hem": 170, "shoulder_width": 36,
                    "neck_width": 16, "front_neck_drop": 15, "back_neck_drop": 3, "armhole_depth": 20,
                    "waist_position": 38, "hip_position": 58},
   "options": {"depth_ratio": 0.66, "fold_count": 9, "fold_amplitude": 1.8},
   "print": {"kind": "dots", "base": (176, 38, 62), "accent": (245, 232, 210), "spacing": 4.0}},
 "T3_fitflare_shortsleeve": {
   "style": {"silhouette": "fit-and-flare", "sleeve": "short", "neckline": "round"},
   "measurements": {"dress_length": 92, "bust": 88, "waist": 70, "hip": 120, "hem": 210, "shoulder_width": 37,
                    "neck_width": 17, "front_neck_drop": 10, "back_neck_drop": 2.5, "armhole_depth": 19,
                    "waist_position": 37, "hip_position": 55, "sleeve_length": 20, "upper_arm": 32, "cuff": 31},
   "options": {"depth_ratio": 0.72, "fold_count": 13, "fold_amplitude": 2.6, "sleeve_angle_deg": 38},
   "print": {"kind": "dots", "base": (20, 110, 110), "accent": (250, 210, 80), "spacing": 3.0, "radius": 0.45}},
 "T4_straight_longsleeve_square": {
   "style": {"silhouette": "straight", "sleeve": "long", "neckline": "square"},
   "measurements": {"dress_length": 105, "bust": 100, "waist": 94, "hip": 104, "hem": 112, "shoulder_width": 39,
                    "neck_width": 20, "front_neck_drop": 11, "back_neck_drop": 2.5, "armhole_depth": 22,
                    "waist_position": 42, "hip_position": 62, "sleeve_length": 60, "upper_arm": 34, "cuff": 24},
   "options": {"depth_ratio": 0.70, "fold_count": 0, "fold_amplitude": 0, "sleeve_angle_deg": 58},
   "print": {"kind": "stripes", "base": (70, 70, 72), "accent": (200, 60, 50), "spacing": 5.0}},
}
CAM = dict(size=(600, 900), fov_deg=25, elev=3)
manifest = {}
for name, c in CASES.items():
    d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    pr = c.pop("print")
    spec = parse_spec(c)
    dm = DressGenerator(spec).build()
    tex = ex.make_print_texture(dm.atlas_size_cm, base=pr["base"], accent=pr["accent"], kind=pr["kind"],
                                spacing=pr["spacing"], radius=pr.get("radius", 0.7))
    tex.save(os.path.join(d, "gt_texture.png"))
    ex.export_glb(ex.to_trimesh(dm, tex), os.path.join(d, "gt.glb"))
    ex.save_landmarks(dm, os.path.join(d, "gt.landmarks.json"))
    # camera framing shared by all evaluations of this case
    ext3 = dm.vertices.max(0) - dm.vertices.min(0)
    extent = max(ext3[1], ext3[0] * CAM["size"][1] / CAM["size"][0]) * 1.22
    center = (dm.vertices.max(0) + dm.vertices.min(0)) / 2
    cam = dict(CAM, center=center.tolist(), extent=float(extent))
    uv = ex.pattern_uv(dm)
    for yaw, tag in ((0, "front"), (180, "back")):
        render(dm.render_vertices, dm.render_faces, uv=uv, texture=tex, yaw=yaw, **cam).save(os.path.join(d, f"photo_{tag}.png"))
    full = dict(c); full["measurements_resolved_cm"] = spec.m
    json.dump({"case": name, "ground_truth": full, "camera": cam,
               "label": "SYNTHETIC reference image rendered from a procedural ground-truth mesh"},
              open(os.path.join(d, "case.json"), "w"), indent=2)
    print(name, "V", len(dm.vertices), "F", len(dm.faces), "d_hem", round(dm.params["d_hem_vertical"], 1))
