"""Flexible garment-measurement schema.

Internal unit: centimetres. GLB export converts to metres (glTF convention).

Input JSON shape (every measurement except dress_length and bust is optional):

{
  "units": "cm" | "in",
  "style": {"silhouette": "straight|a-line|fit-and-flare",
            "sleeve": "none|short|long",
            "neckline": "round|v|square|boat"},
  "measurements": {
     "bust": 92                         # plain number = full circumference
     "waist": {"value": 39, "method": "flat"}   # flat = half-width laid flat -> x2
     ...
  }
}

Each measurement can carry `"method": "circumference"` (default for girths) or `"flat"`
(garment laid flat, edge-to-edge; converted with x2). Lengths ignore `method`.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

INCH = 2.54

# name -> (kind, description). kind: "girth" (closed loop) or "length".
MEASUREMENT_DEFS = {
    "dress_length":    ("length", "HPS (high point shoulder, at neck) straight down the front surface to hem"),
    "bust":            ("girth",  "Chest/bust circumference 2.5 cm below the armhole bottom"),
    "waist":           ("girth",  "Narrowest torso circumference between bust and hip"),
    "hip":             ("girth",  "Circumference at hip_position below HPS"),
    "hem":             ("girth",  "Hem edge length (sweep), measured along the fabric edge"),
    "shoulder_width":  ("length", "Straight distance between left and right shoulder points"),
    "neck_width":      ("length", "Straight distance between the two HPS points (neck opening width)"),
    "front_neck_drop": ("length", "Vertical drop from HPS line to centre-front neckline"),
    "back_neck_drop":  ("length", "Vertical drop from HPS line to centre-back neckline"),
    "armhole_depth":   ("length", "Vertical distance from HPS to armhole bottom (underarm)"),
    "waist_position":  ("length", "Vertical distance from HPS to waist line"),
    "hip_position":    ("length", "Vertical distance from HPS to hip line"),
    "sleeve_length":   ("length", "Shoulder point to cuff edge, along the top of the sleeve"),
    "upper_arm":       ("girth",  "Sleeve circumference perpendicular to the sleeve axis, 6 cm along the axis from the armhole centre"),
    "cuff":            ("girth",  "Sleeve opening edge length"),
}

SILHOUETTES = ("straight", "a-line", "fit-and-flare")
SLEEVES = ("none", "short", "long")
NECKLINES = ("round", "v", "square", "boat")


@dataclass
class DressSpec:
    style: dict
    m: dict                      # resolved measurements in cm (all keys present)
    provided: set = field(default_factory=set)   # keys the user actually supplied
    notes: list = field(default_factory=list)    # unit conversions / defaults applied
    options: dict = field(default_factory=dict)  # non-measurement construction options

    def to_json(self) -> dict:
        return {"style": self.style, "measurements_cm": self.m,
                "provided": sorted(self.provided), "notes": self.notes, "options": self.options}


def _value(raw, kind, units, name, notes):
    method = "circumference"
    if isinstance(raw, dict):
        method = raw.get("method", "circumference")
        raw = raw["value"]
    v = float(raw)
    if units == "in":
        notes.append(f"{name}: {v} in x 2.54 = {v * INCH:.2f} cm")
        v *= INCH
    if kind == "girth" and method == "flat":
        notes.append(f"{name}: flat {v:.2f} cm x 2 = {2 * v:.2f} cm circumference")
        v *= 2
    if v <= 0:
        raise ValueError(f"{name} must be positive")
    return v


def parse_spec(data: dict | str) -> DressSpec:
    if isinstance(data, str):
        data = json.loads(data)
    data = copy.deepcopy(data)
    units = data.get("units", "cm")
    if units not in ("cm", "in"):
        raise ValueError("units must be 'cm' or 'in'")
    style = {"silhouette": "a-line", "sleeve": "none", "neckline": "round"}
    style.update(data.get("style", {}))
    if style["silhouette"] not in SILHOUETTES:
        raise ValueError(f"silhouette must be one of {SILHOUETTES}")
    if style["sleeve"] not in SLEEVES:
        raise ValueError(f"sleeve must be one of {SLEEVES}")
    if style["neckline"] not in NECKLINES:
        raise ValueError(f"neckline must be one of {NECKLINES}")

    notes, m = [], {}
    raw = data.get("measurements", {})
    unknown = set(raw) - set(MEASUREMENT_DEFS)
    if unknown:
        raise ValueError(f"unknown measurements: {sorted(unknown)}")
    for k, v in raw.items():
        if v is None or v == "":
            continue
        m[k] = _value(v, MEASUREMENT_DEFS[k][0], units, k, notes)
    provided = set(m)
    for req in ("dress_length", "bust"):
        if req not in m:
            raise ValueError(f"'{req}' is required")

    def default(k, val, why):
        if k not in m:
            m[k] = float(val)
            notes.append(f"{k}: not provided, assumed {val:.1f} cm ({why})")

    sil, B = style["silhouette"], m["bust"]
    default("waist", {"straight": 0.95 * B, "a-line": 0.84 * B, "fit-and-flare": 0.80 * B}[sil],
            f"typical {sil} ratio to bust")
    default("hip", {"straight": max(B, m["waist"]) * 1.02, "a-line": 1.08 * B,
                    "fit-and-flare": 1.25 * B}[sil], f"typical {sil} ratio to bust")
    default("shoulder_width", 0.40 * B + 1.5, "ratio to bust")
    default("neck_width", 17.0 if style["neckline"] != "boat" else 26.0, "typical neck opening")
    default("front_neck_drop", {"round": 9.0, "v": 16.0, "square": 11.0, "boat": 4.0}[style["neckline"]],
            f"typical {style['neckline']} neckline")
    default("back_neck_drop", 2.5, "typical back neck")
    default("armhole_depth", 0.115 * B + 9.5, "ratio to bust")
    default("waist_position", max(m["armhole_depth"] + 12, 0.43 * B), "ratio to bust")
    default("hip_position", m["waist_position"] + 20.0, "waist + 20 cm")
    default("hem", {"straight": m["hip"] * 1.0, "a-line": m["hip"] * 1.45,
                    "fit-and-flare": m["hip"] * 2.0}[sil], f"typical {sil} sweep")
    if style["sleeve"] != "none":
        default("sleeve_length", 22.0 if style["sleeve"] == "short" else 58.0, f"typical {style['sleeve']} sleeve")
        default("upper_arm", 0.36 * B, "ratio to bust")
        default("cuff", m["upper_arm"] * (0.95 if style["sleeve"] == "short" else 0.65), "ratio to upper arm")

    # consistency checks (geometric feasibility, not fit)
    if m["hip_position"] >= m["dress_length"] - 2:
        notes.append("hip_position is at/below the hem; hip anchor ignored")
    if m["waist_position"] <= m["armhole_depth"] + 4:
        raise ValueError("waist_position must be at least 4 cm below armhole_depth")
    if m["neck_width"] >= m["shoulder_width"] - 4:
        raise ValueError("neck_width must be at least 4 cm narrower than shoulder_width")

    options = {k: data.get("options", {}).get(k, v) for k, v in {
        "fold_count": 0, "fold_amplitude": 0.0, "sleeve_angle_deg": 50.0,
        "mesh_resolution_cm": 1.5, "depth_ratio": 0.70}.items()}
    return DressSpec(style=style, m=m, provided=provided, notes=notes, options=options)
