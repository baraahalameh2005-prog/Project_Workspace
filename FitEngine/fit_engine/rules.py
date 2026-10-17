"""Sizing rules: which measurements matter for each garment type, and the
ease allowance (garment minus body, in cm) considered comfortable.

All numbers here are DEFAULT ASSUMPTIONS drawn from common pattern-making
practice, not values measured on this project's real garments. They are kept
in one place so the team can tune them after real fitting tests, and the
clothing system can override them per product via `ease_overrides`.
"""

SUPPORTED_CATEGORIES = ("t_shirt", "shirt", "jacket", "dress", "pants", "skirt")
SUPPORTED_FIT_TYPES = ("slim", "regular", "loose", "oversized")
SUPPORTED_STRETCH = ("none", "low", "medium", "high")

CATEGORY_NAMES_AR = {
    "t_shirt": "تيشيرت",
    "shirt": "قميص",
    "jacket": "جاكيت",
    "dress": "فستان",
    "pants": "بنطلون",
    "skirt": "تنورة",
}

# Measurement importance per category. Keys are GARMENT keys.
# primary   -> decides whether a size can be worn at all.
# secondary -> refines the choice and produces warnings.
# Weights are relative importance in the score.
CATEGORY_RULES = {
    "t_shirt": {
        "primary": {"chest": 1.0, "shoulder_width": 0.7},
        "secondary": {"waist": 0.4, "hip": 0.3},
    },
    "shirt": {
        "primary": {"chest": 1.0, "shoulder_width": 0.8},
        "secondary": {"waist": 0.5, "sleeve_length": 0.5, "hip": 0.3},
    },
    "jacket": {
        "primary": {"chest": 1.0, "shoulder_width": 0.9},
        "secondary": {"sleeve_length": 0.6, "waist": 0.4, "hip": 0.3},
    },
    "dress": {
        "primary": {"chest": 1.0, "waist": 0.9, "hip": 0.9},
        "secondary": {"shoulder_width": 0.5},
    },
    "pants": {
        "primary": {"waist": 1.0, "hip": 0.9},
        "secondary": {"thigh": 0.6, "inseam": 0.5},
    },
    "skirt": {
        "primary": {"waist": 1.0, "hip": 0.9},
        "secondary": {},
    },
}

# Target ease (garment - body) in cm for a REGULAR fit, NON-stretch fabric.
BASE_EASE = {
    "t_shirt": {"chest": (6, 16), "waist": (4, 24), "hip": (2, 24), "shoulder_width": (-1, 3)},
    "shirt": {"chest": (8, 18), "waist": (6, 22), "hip": (4, 22), "shoulder_width": (-1, 3),
              "sleeve_length": (-2, 4)},
    "jacket": {"chest": (10, 22), "waist": (8, 26), "hip": (4, 24), "shoulder_width": (0, 4),
               "sleeve_length": (-1, 5)},
    "dress": {"chest": (4, 12), "waist": (3, 12), "hip": (4, 14), "shoulder_width": (-1, 2)},
    "pants": {"waist": (0, 5), "hip": (3, 12), "thigh": (3, 12), "inseam": (-3, 3)},
    "skirt": {"waist": (0, 5), "hip": (3, 14)},
}

# How much the fit type shifts circumference ease: (shift_min, shift_max) in cm.
FIT_SHIFT = {
    "slim": (-4, -6),
    "regular": (0, 0),
    "loose": (6, 10),
    "oversized": (14, 24),
}

# Shoulder width shift per fit type (dropped shoulders on oversized pieces).
SHOULDER_SHIFT = {
    "slim": (-1, -1),
    "regular": (0, 0),
    "loose": (0, 3),
    "oversized": (2, 10),
}

# Fraction of FIT_SHIFT applied per category/key. Waistbands of pants and
# skirts sit on the body regardless of the cut, so their waist is not shifted.
FIT_SENSITIVITY = {
    ("pants", "waist"): 0.0,
    ("skirt", "waist"): 0.0,
    ("pants", "hip"): 0.5,
    ("pants", "thigh"): 0.6,
    ("skirt", "hip"): 0.7,
    ("dress", "waist"): 0.7,
}

# Stretch: the garment may be smaller than the body by this fraction of the
# body circumference (negative ease), and the comfortable maximum shrinks.
STRETCH_NEGATIVE_PCT = {"none": 0.0, "low": 0.03, "medium": 0.07, "high": 0.12}
STRETCH_MAX_REDUCTION = {"none": 0.0, "low": 0.0, "medium": 2.0, "high": 4.0}

# Distance outside the target range still labelled "slightly".
TOLERANCE = {"circumference": 3.0, "shoulder_width": 1.5, "length": 2.0}

# Tight is penalised more than loose (a loose garment can still be worn).
TIGHT_PENALTY_FACTOR = 2.0

# Inside the range a tiny preference for the centre breaks ties between sizes.
CENTER_PREFERENCE = 0.05

# An adjacent size whose score is within this margin is offered as an alternative.
ALTERNATIVE_MARGIN = 0.75

CIRCUMFERENCE_KEYS = {"chest", "waist", "hip", "thigh"}


def tolerance_for(key):
    if key in CIRCUMFERENCE_KEYS:
        return TOLERANCE["circumference"]
    if key == "shoulder_width":
        return TOLERANCE["shoulder_width"]
    return TOLERANCE["length"]


def target_ease(category, key, fit_type, stretch, body_value_cm, overrides=None):
    """Return (min_ease, max_ease, source) in cm for one garment measurement.

    source is "override" when the product supplied its own range.
    """
    if overrides and key in overrides:
        lo, hi = overrides[key]
        return float(lo), float(hi), "override"

    lo, hi = BASE_EASE[category][key]
    lo, hi = float(lo), float(hi)

    if key in CIRCUMFERENCE_KEYS:
        s = FIT_SENSITIVITY.get((category, key), 1.0)
        dlo, dhi = FIT_SHIFT[fit_type]
        lo += dlo * s
        hi += dhi * s
        lo -= body_value_cm * STRETCH_NEGATIVE_PCT[stretch]
        hi -= STRETCH_MAX_REDUCTION[stretch]
    elif key == "shoulder_width":
        dlo, dhi = SHOULDER_SHIFT[fit_type]
        lo += dlo
        hi += dhi
    # lengths (sleeve_length, inseam) do not depend on cut or stretch

    if hi < lo + 2.0:  # always keep a usable comfort window
        hi = lo + 2.0
    return round(lo, 2), round(hi, 2), "default"
