"""
Avatar appearance: face shape presets, eye colour, skin tone, female lip tint and
eyebrows, applied after the (unchanged) body fit.

Pipeline (see api.py):
  1. fit_to_measurements.fit_body fits the body with the plain Anny model (frozen).
  2. The fitted phenotype is evaluated once more with the face model: Anny with the
     cached 63-label local-change list (data/face_model_labels.json). With every
     control at 0 it reproduces the plain model exactly; the gender face preset only
     moves head vertices.
  3. export_appearance_avatar writes one static GLB mesh with:
       - primitive 0: the whole body except the brow band, material "skin", base colour
         texture (TEXCOORD_0) = skin colour + female lip strip + procedural eye;
       - primitive 1: the brow band, material "skin_brows" (same PBR values), base colour
         texture (TEXCOORD_1) = skin colour with painted brow hairs.
     Both primitives share the same POSITION / NORMAL / TEXCOORD accessors, so the
     vertices, normals and their order are those of the plain export.

Prototype history: tools/eye_texture_check.py, lip_texture_check.py, lip_shape_check.py,
brow_texture_check.py (refined brows visually approved 2026-10-12).
"""

import functools
import io
import json
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app.config import apply_environment, settings

apply_environment(settings)

import trimesh
from PIL import Image, ImageDraw, ImageFilter

from generate_avatar import _replace_file, parse_skin_tone

BASE_DIR = Path(__file__).resolve().parent
FACE_MODEL_LABELS = json.loads((BASE_DIR / "data" / "face_model_labels.json").read_text())["labels"]

# ------------------------------------------------------------------ options

# name -> (iris sRGB, darker limbal ring sRGB)
EYE_COLORS = {
    "brown": ((92, 58, 34), (46, 30, 20)),
    "hazel": ((122, 92, 52), (62, 46, 28)),
    "green": ((86, 116, 70), (40, 60, 36)),
    "blue": ((70, 110, 152), (34, 56, 88)),
    "gray": ((112, 122, 130), (58, 64, 72)),
}
DEFAULT_EYE_COLOR = "brown"

# Gender face presets (Anny local changes, -1..1; negative = decrease). Only head
# vertices move (face-control sweep 2026-10-11: body-stable controls).
FACE_PRESETS = {
    "female": {
        "mouth-scale-horiz-incr": -0.35,      # narrower mouth (reviewed in the browser)
        "cheek-volume-incr": 0.25,            # softer, fuller cheeks
        "chin-width-incr": -0.2,              # narrower chin
        "nose-scale-horiz-incr": -0.15,       # slightly narrower nose
    },
    "male": {
        "chin-width-incr": 0.4,               # broader jaw / chin
        "chin-bones-incr": 0.35,              # stronger jaw bone
        "chin-prominent-incr": 0.25,          # more projected chin
        "nose-scale-horiz-incr": 0.15,        # slightly wider nose
    },
}

# ------------------------------------------------------------------ texture layout

TEX_W, TEX_H = 256, 128                 # left half: skin -> lip strip, right half: eye
# Strip columns for lip weight 0 and 1, kept away from the texture edge and from the eye
# half so that mipmaps / filtering do not blend in the sclera (browser halo, 2026-10-12).
STRIP_FIRST, STRIP_LAST = 8, 112
EYE_UV_CENTRE = (0.75, 0.5)
EYE_UV_HALF = (0.25, 0.5)
IRIS_RADIUS = 0.45
PUPIL_RADIUS = 0.33
SCLERA = (236, 232, 226)
PUPIL = (14, 12, 12)

# lip tint (female only; approved colour direction, lip_texture_check "female" settings)
LIP_FACTOR = (0.80, 0.58, 0.60)
LIP_STRENGTH = 1.0
LIP_EDGE = (0.35, 0.75)
LIP_MASK = "mpfb_lips.jpg"

# brow band and brow design (refined version, brow_texture_check 2026-10-12, run 20261012_013909)
REFERENCE_EYE_HALF_DISTANCE_M = 0.0295  # F1's eye centres are at x = +-29.5 mm; designs are in that scale
PATCH_HALF_W = 0.075
PATCH_Y = (0.004, 0.046)
PATCH_MIN_NORMAL_Z = 0.15
BROW_TILE_PX = (1024, 288)
BROW_SUPERSAMPLE = 3


@dataclass(frozen=True)
class BrowDesign:
    inner: tuple
    peak: tuple
    tail: tuple
    thickness: tuple                    # inner, peak, tail (mm)
    hairs: int
    hair_len: tuple
    rgb: tuple
    seed: int = 7
    clip_scale: float = 0.55
    clip_feather_mm: float = 0.35
    base_scale: float = 0.42
    base_fill: int = 80
    base_blur_mm: float = 0.5
    head_t: float = 0.18
    head_angle_deg: tuple = (25.0, 12.0)
    head_len_factor: float = 0.6


BROWS = {
    # visually approved refined female design
    "female": BrowDesign(inner=(12.0, 15.0), peak=(36.0, 20.0), tail=(53.0, 15.5), thickness=(4.2, 3.4, 1.0),
                         hairs=1000, hair_len=(2.2, 3.6), rgb=(68, 47, 35)),
    # male: same method; straighter, lower arch, thicker, slightly darker
    "male": BrowDesign(inner=(11.0, 14.5), peak=(35.0, 17.5), tail=(53.0, 15.0), thickness=(5.4, 4.6, 1.6),
                       hairs=1400, hair_len=(2.4, 3.8), rgb=(58, 40, 30), head_angle_deg=(22.0, 10.0)),
}


@dataclass
class Appearance:
    gender: str
    skin_tone: str
    eye_color: str
    face_controls: dict = field(default_factory=dict)

    def summary(self):
        features = ["eye_texture", "eyebrows", "face_preset"]
        if self.gender == "female":
            features.append("lip_tint")
        return {"gender_style": self.gender, "skin_tone": self.skin_tone, "eye_color": self.eye_color,
                "face_controls": dict(self.face_controls), "features": features,
                "not_available": ["hair"]}


def normalize_eye_color(value):
    """Return a known eye colour name (case-insensitive); raise ValueError otherwise."""
    if isinstance(value, str) and value.strip().lower() in EYE_COLORS:
        return value.strip().lower()
    raise ValueError(f"Unknown eye colour {value!r}; expected one of {sorted(EYE_COLORS)}.")


def appearance_for(gender, skin_tone, eye_color):
    parse_skin_tone(skin_tone)
    return Appearance(gender=gender, skin_tone=skin_tone, eye_color=normalize_eye_color(eye_color),
                      face_controls=dict(FACE_PRESETS[gender]))


# ------------------------------------------------------------------ face model

class _Controlled:
    """Anny model proxy that passes a fixed set of local changes."""

    def __init__(self, model, controls):
        self._model = model
        self._controls = controls

    def __call__(self, phenotype_kwargs=None, **kwargs):
        return self._model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=dict(self._controls), **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class FaceModel:
    """The 63-label Anny model plus topology data that does not depend on the body."""

    def __init__(self):
        import anny
        from anny.face_segmentation import get_face_segmentation_mask

        # phenotypes="all" also exposes cup size and firmness (used by the
        # athletic body style); at their default 0.5 the body is identical
        # to the default phenotypes (checked: max vertex difference 0.0).
        self.model = anny.Anny(local_changes=FACE_MODEL_LABELS, phenotypes="all")
        self.labels = list(self.model.local_change_labels)
        self.faces = self.model.faces.detach().cpu().numpy()
        n = int(self.model.template_vertices.shape[-2])
        self.eyes = _eyeballs(self.faces, n)
        self.eye_index = np.concatenate(self.eyes)
        self.eye_front = {s: get_face_segmentation_mask(self.model, [f"eye_front.{s}"]).cpu().numpy() for s in ("L", "R")}
        mask_path = Path(anny.__file__).resolve().parent / "data" / "mpfb2" / "textures" / LIP_MASK
        self.lip_raw = _lip_weights(self.model, self.faces, n, mask_path)
        self.lip_raw[self.eye_index] = 0.0

    def controls(self, face_controls, body_controls=None, phenotypes=None):
        """The model with the face controls (names without l-/r-), plus exact
        body local changes and phenotype values (body styles)."""
        values = {label: 0.0 for label in self.labels}
        for label, value in (body_controls or {}).items():
            if label not in values:
                raise KeyError(f"body control {label} is not in the face model")
            values[label] = float(value)
        for name, value in face_controls.items():
            hits = [label for label in self.labels if label == name or label in (f"l-{name}", f"r-{name}")]
            if not hits:
                raise KeyError(f"face control {name} is not in the face model")
            for label in hits:
                values[label] = float(value)
        if phenotypes:
            from body_style import StyledModel

            return StyledModel(self.model, values, phenotypes)
        return _Controlled(self.model, values)


def _eyeballs(faces, n_vertices):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    graph = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n_vertices, n_vertices))
    _, labels = connected_components(graph, directed=False)
    sizes = np.bincount(labels)
    pieces = [np.where(labels == k)[0] for k in np.where(sizes == 72)[0]]
    if len(pieces) != 2:
        raise RuntimeError(f"expected two 72-vertex eyeballs, found {len(pieces)}")
    return pieces


def _lip_weights(model, faces, n_vertices, mask_path):
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=float) / 255.0
    tc = model.texture_coordinates.detach().cpu().numpy()
    fti = model.face_texture_coordinate_indices.detach().cpu().numpy()
    h, w = mask.shape
    weights = np.zeros(n_vertices)
    for k in range(3):
        uv = tc[fti[:, k]]
        col = np.clip(np.round(uv[:, 0] * w).astype(int), 0, w - 1)
        row = np.clip(np.round((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
        np.maximum.at(weights, faces[:, k], mask[row, col])
    return weights


# ------------------------------------------------------------------ geometry helpers

def export_frame(vertices):
    """Same axis change and floor shift as generate_avatar.export_avatar."""
    v = np.stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]], axis=1)
    v[:, 1] -= v[:, 1].min()
    return v


def _eye_uv(vertices, faces, eyes, eye_front):
    uv = np.zeros((len(vertices), 2))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    for idx in eyes:
        side = "L" if vertices[idx, 0].mean() > 0 else "R"
        front = eye_front[side]
        axis = (mesh.face_normals[front] * mesh.area_faces[front, None]).sum(axis=0)
        axis /= np.linalg.norm(axis)
        centre = (vertices[idx].min(axis=0) + vertices[idx].max(axis=0)) / 2
        radius = float(np.linalg.norm(vertices[idx] - centre, axis=1).max())
        right = np.cross([0.0, 1.0, 0.0], axis)
        right /= np.linalg.norm(right)
        up = np.cross(axis, right)
        d = vertices[idx] - centre
        local = np.stack([d @ right, d @ up], axis=1) / radius
        local[d @ axis < 0] = [0.95, 0.0]           # back hemisphere: sclera
        uv[idx, 0] = EYE_UV_CENTRE[0] + EYE_UV_HALF[0] * local[:, 0]
        uv[idx, 1] = EYE_UV_CENTRE[1] + EYE_UV_HALF[1] * local[:, 1]
    return uv, mesh


def _smoothstep(w, lo, hi):
    t = np.clip((w - lo) / (hi - lo), 0, 1)
    return t * t * (3 - 2 * t)


# ------------------------------------------------------------------ textures

@functools.lru_cache(maxsize=64)
def base_texture(skin_tone, eye_color, female):
    """256 x 128 RGB: left half skin -> lip strip (flat skin for male), right half the eye."""
    skin = np.array(parse_skin_tone(skin_tone)[:3], dtype=float)
    img = np.zeros((TEX_H, TEX_W, 3), dtype=np.uint8)
    half = TEX_W // 2
    lip = skin * np.array(LIP_FACTOR)
    strength = LIP_STRENGTH if female else 0.0
    x = np.arange(half)
    t = np.clip((x - STRIP_FIRST) / (STRIP_LAST - STRIP_FIRST), 0, 1)[:, None] * strength
    img[:, :half] = (skin * (1 - t) + lip * t).round().astype(np.uint8)[None]

    iris, edge = EYE_COLORS[eye_color]
    yy, xx = np.mgrid[0:TEX_H, 0:half]
    r = np.hypot((xx + 0.5 - half / 2) / (half / 2), (yy + 0.5 - TEX_H / 2) / (TEX_H / 2))
    eye = np.empty((TEX_H, half, 3))
    eye[:] = SCLERA
    k = np.clip((r - IRIS_RADIUS * 0.82) / (IRIS_RADIUS * 0.18), 0, 1)[..., None]
    ring = (1 - k) * np.array(iris) + k * np.array(edge)
    eye[r <= IRIS_RADIUS] = ring[r <= IRIS_RADIUS]
    eye[r <= IRIS_RADIUS * PUPIL_RADIUS] = PUPIL
    img[:, half:] = eye.round().astype(np.uint8)
    return Image.fromarray(img, "RGB")


def _brow_curve(design, t):
    p0, p1, p2 = (np.array(design.inner), np.array(design.peak), np.array(design.tail))
    c = 2 * p1 - 0.5 * (p0 + p2)
    t = np.asarray(t)[..., None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p2
    tan = 2 * (1 - t) * (c - p0) + 2 * t * (p2 - c)
    return pts, tan / np.linalg.norm(tan, axis=-1, keepdims=True)


def _brow_thickness(design, t):
    t = np.asarray(t)
    a, b, c = design.thickness
    return np.where(t < 0.5, a + (b - a) * (t / 0.5), b + (c - b) * (np.clip(t - 0.5, 0, None) / 0.5) ** 1.3)


def brow_color(design, skin):
    """Design colour, darkened for dark skin so the brows stay visible."""
    return np.minimum(np.array(design.rgb, dtype=float), skin * 0.55)


@functools.lru_cache(maxsize=32)
def brow_texture(skin_tone, sex):
    """Brow tile (patch box at reference scale): exact skin colour with clipped hair strokes."""
    design = BROWS[sex]
    skin = np.array(parse_skin_tone(skin_tone)[:3], dtype=float)
    w, h = BROW_TILE_PX[0] * BROW_SUPERSAMPLE, BROW_TILE_PX[1] * BROW_SUPERSAMPLE
    x0, x1 = -PATCH_HALF_W * 1000, PATCH_HALF_W * 1000
    y0, y1 = PATCH_Y[0] * 1000, PATCH_Y[1] * 1000
    sx, sy = w / (x1 - x0), h / (y1 - y0)

    def to_px(p):
        return ((p[..., 0] - x0) * sx, (y1 - p[..., 1]) * sy)

    def outline(scale):
        ts = np.linspace(0, 1, 200)
        pts, tan = _brow_curve(design, ts)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        half = _brow_thickness(design, ts)[:, None] * scale
        phi = np.linspace(0, np.pi, 24)[:, None]
        cap = pts[0] + half[0] * (-nrm[0] * np.cos(phi) - tan[0] * np.sin(phi))
        return np.concatenate([pts + nrm * half, (pts - nrm * half)[::-1], cap])

    def fill(poly, side, value, blur_mm):
        img = Image.new("L", (w, h), 0)
        px, py = to_px(poly * [side, 1])
        ImageDraw.Draw(img).polygon(list(zip(px, py)), fill=value)
        return np.asarray(img.filter(ImageFilter.GaussianBlur(blur_mm * sx)), dtype=float)

    base_cov = np.zeros((h, w))
    clip = np.zeros((h, w))
    hairs = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(hairs)
    rng = np.random.default_rng(design.seed)
    n = design.hairs
    for side in (1, -1):
        base_cov = np.maximum(base_cov, fill(outline(design.base_scale), side, design.base_fill, design.base_blur_mm))
        clip = np.maximum(clip, fill(outline(design.clip_scale), side, 255, design.clip_feather_mm))
        t = rng.beta(1.3, 1.6, n)
        pts, tan = _brow_curve(design, t)
        nrm = np.stack([-tan[:, 1], tan[:, 0]], 1)
        start = pts + nrm * ((rng.random(n) - 0.5) * _brow_thickness(design, t) * 0.8)[:, None]
        head = t < design.head_t
        a0, a1 = design.head_angle_deg
        lift = np.radians(np.where(head, a0 - (a0 - a1) * t / design.head_t, 7.0))
        ang = np.arctan2(tan[:, 1], tan[:, 0]) + lift + rng.normal(0, 0.10, n)
        length = rng.uniform(*design.hair_len, n) * np.where(head, design.head_len_factor, np.where(t > 0.8, 0.75, 1.0))
        end = start + np.stack([np.cos(ang), np.sin(ang)], 1) * length[:, None]
        sx0, sy0 = to_px(start * [side, 1])
        ex0, ey0 = to_px(end * [side, 1])
        alpha = rng.uniform(150, 235, n)
        width = max(1, int(round(0.14 * sx)))
        for i in range(n):
            draw.line([(sx0[i], sy0[i]), (ex0[i], ey0[i])], fill=int(alpha[i]), width=width)
    cov = np.maximum(base_cov, np.asarray(hairs, dtype=float) * clip / 255.0)
    cov = Image.fromarray(np.clip(cov, 0, 255).astype(np.uint8)).resize(BROW_TILE_PX, Image.LANCZOS)
    cov = np.asarray(cov, dtype=float)[..., None] / 255.0
    img = skin * (1 - cov) + brow_color(design, skin) * cov
    return Image.fromarray(np.clip(img.round(), 0, 255).astype(np.uint8), "RGB")


# ------------------------------------------------------------------ UVs and brow band

@dataclass
class Layout:
    vertices: np.ndarray            # export frame (Y up, metres, feet at 0)
    uv0: np.ndarray                 # our convention: v up
    uv1: np.ndarray
    brow_faces: np.ndarray          # bool per face
    scale: float


def layout(face, rest_vertices, female):
    """Texture coordinates and brow band for one body (rest_vertices: Anny frame, (V, 3))."""
    v = export_frame(rest_vertices)
    faces = face.faces
    uv0, mesh = _eye_uv(v, faces, face.eyes, face.eye_front)
    non_eye = np.setdiff1d(np.arange(len(v)), face.eye_index)
    weights = _smoothstep(face.lip_raw, *LIP_EDGE) if female else np.zeros(len(v))
    uv0[non_eye, 0] = (STRIP_FIRST + weights[non_eye] * (STRIP_LAST - STRIP_FIRST) + 0.5) / TEX_W
    uv0[non_eye, 1] = 0.5

    eye_c = v[face.eye_index].mean(axis=0)
    half_dist = float(np.mean([abs(v[e, 0].mean()) for e in face.eyes]))
    scale = half_dist / REFERENCE_EYE_HALF_DISTANCE_M
    rel = (v - [0.0, eye_c[1], 0.0]) / scale                 # reference-scale coordinates
    in_box = (np.abs(rel[:, 0]) <= PATCH_HALF_W) & (rel[:, 1] >= PATCH_Y[0]) & (rel[:, 1] <= PATCH_Y[1])
    in_box[face.eye_index] = False
    in_box &= face.lip_raw == 0
    brow_faces = in_box[faces].all(axis=1) & (mesh.face_normals[:, 2] > PATCH_MIN_NORMAL_Z)
    uv1 = np.stack([(rel[:, 0] + PATCH_HALF_W) / (2 * PATCH_HALF_W),
                    (rel[:, 1] - PATCH_Y[0]) / (PATCH_Y[1] - PATCH_Y[0])], axis=1)
    return Layout(vertices=v, uv0=uv0, uv1=np.clip(uv1, 0, 1), brow_faces=brow_faces, scale=scale)


# ------------------------------------------------------------------ GLB export

def _pad(b, fill=b"\x00"):
    return b + fill * ((4 - len(b) % 4) % 4)


def _png(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _add_brow_primitive(glb, faces, brow_faces, uv1, png_bytes):
    """Split the triangle list into body and brow primitives sharing all vertex accessors."""
    n = struct.unpack("<I", glb[12:16])[0]
    gltf = json.loads(glb[20:20 + n])
    bin_len = struct.unpack("<I", glb[20 + n:24 + n])[0]
    binary = bytearray(glb[20 + n + 8:20 + n + 8 + bin_len])
    prim = gltf["meshes"][0]["primitives"][0]

    acc = gltf["accessors"][prim["indices"]]
    view = gltf["bufferViews"][acc["bufferView"]]
    start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    dtype = {5125: np.uint32, 5123: np.uint16}[acc["componentType"]]
    existing = np.frombuffer(bytes(binary[start:start + acc["count"] * np.dtype(dtype).itemsize]), dtype=dtype)
    if not np.array_equal(existing.reshape(-1, 3), faces):
        raise RuntimeError("exported index buffer differs from the model faces")

    def add_view(blob, target=None):
        nonlocal binary
        binary = bytearray(_pad(bytes(binary)))
        view = {"buffer": 0, "byteOffset": len(binary), "byteLength": len(blob)}
        binary += blob
        if target:
            view["target"] = target
        gltf["bufferViews"].append(view)
        return len(gltf["bufferViews"]) - 1

    uv = np.stack([uv1[:, 0], 1 - uv1[:, 1]], axis=1).astype(np.float32)   # glTF: v down
    gltf["accessors"].append({"bufferView": add_view(uv.tobytes(), 34962), "componentType": 5126,
                              "count": len(uv), "type": "VEC2", "min": uv.min(0).tolist(), "max": uv.max(0).tolist()})
    uv_accessor = len(gltf["accessors"]) - 1
    index_accessors = []
    for part in (faces[~brow_faces], faces[brow_faces]):
        flat = part.astype(np.uint32).ravel()
        gltf["accessors"].append({"bufferView": add_view(flat.tobytes(), 34963), "componentType": 5125,
                                  "count": len(flat), "type": "SCALAR",
                                  "min": [int(flat.min())], "max": [int(flat.max())]})
        index_accessors.append(len(gltf["accessors"]) - 1)
    gltf.setdefault("images", []).append({"bufferView": add_view(png_bytes), "mimeType": "image/png", "name": "brows"})
    gltf.setdefault("textures", []).append({"source": len(gltf["images"]) - 1})
    # Clamp at the texture edges (glTF's default is repeat) for both textures.
    gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
    for tex in gltf["textures"]:
        tex["sampler"] = 0
    material = json.loads(json.dumps(gltf["materials"][prim["material"]]))
    material["name"] = "skin_brows"
    material["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gltf["textures"]) - 1, "texCoord": 1}
    gltf["materials"].append(material)

    attributes = dict(prim["attributes"], TEXCOORD_1=uv_accessor)
    gltf["meshes"][0]["primitives"] = [
        dict(prim, attributes=attributes, indices=index_accessors[0]),
        {"attributes": attributes, "indices": index_accessors[1], "material": len(gltf["materials"]) - 1,
         "mode": prim.get("mode", 4)},
    ]
    gltf["buffers"][0]["byteLength"] = len(_pad(bytes(binary)))
    js = _pad(json.dumps(gltf, separators=(",", ":")).encode(), b" ")
    bn = _pad(bytes(binary))
    return (b"glTF" + struct.pack("<II", 2, 12 + 8 + len(js) + 8 + len(bn))
            + struct.pack("<I", len(js)) + b"JSON" + js + struct.pack("<I", len(bn)) + b"BIN\x00" + bn)


def export_appearance_avatar(output, face, path, appearance):
    """
    Write the avatar GLB with textures for skin, eyes, female lips and brows.
    output: Anny output of the face model for the fitted body. The file is written to a
    temporary name first and then swapped in.
    """
    female = appearance.gender == "female"
    rest = output["rest_vertices"][0].detach().cpu().numpy()
    lay = layout(face, rest, female)
    base = base_texture(appearance.skin_tone, appearance.eye_color, female)
    brows = brow_texture(appearance.skin_tone, appearance.gender)

    material = trimesh.visual.material.PBRMaterial(name="skin", baseColorTexture=base,
                                                   baseColorFactor=[255, 255, 255, 255],
                                                   metallicFactor=0.0, roughnessFactor=0.65)
    mesh = trimesh.Trimesh(vertices=lay.vertices, faces=face.faces,
                           visual=trimesh.visual.TextureVisuals(uv=lay.uv0, material=material), process=False)
    mesh.vertex_normals                                     # smooth shading, as export_avatar
    glb = mesh.export(file_type="glb", include_normals=True)
    data = _add_brow_primitive(glb, face.faces, lay.brow_faces, lay.uv1, _png(brows))

    path = Path(path)
    temp_path = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        temp_path.write_bytes(data)
        _replace_file(temp_path, path)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return lay
