"""Synthetic test data builders. These are DEMO numbers, not real garment specs."""

import copy


def body(unit="cm", **m):
    return {"unit": unit, "measurements": m}


TSHIRT_SIZES = [
    ("S", 92, 42, 88, 90),
    ("M", 100, 44, 96, 98),
    ("L", 108, 46, 104, 106),
    ("XL", 116, 48, 112, 114),
]


def tshirt(fit_type="regular", stretch="none", **extra):
    g = {
        "garment_id": "TEST-TS",
        "category": "t_shirt",
        "fit_type": fit_type,
        "stretch": stretch,
        "unit": "cm",
        "sizes": [
            {"label": l, "measurements": {"chest": c, "shoulder_width": s, "waist": w, "hip": h,
                                          "length": 70, "sleeve_length": 20}}
            for l, c, s, w, h in TSHIRT_SIZES
        ],
    }
    g.update(extra)
    return g


def dress(stretch="none", fit_type="regular"):
    rows = [("S", 90, 74, 96, 38), ("M", 96, 80, 102, 39), ("L", 102, 86, 108, 40), ("XL", 108, 92, 114, 41)]
    return {
        "garment_id": "TEST-DR",
        "category": "dress",
        "fit_type": fit_type,
        "stretch": stretch,
        "unit": "cm",
        "sizes": [{"label": l, "measurements": {"chest": c, "waist": w, "hip": h, "shoulder_width": s,
                                                "length": 120}}
                  for l, c, w, h, s in rows],
    }


def pants(stretch="none", fit_type="regular"):
    rows = [("28", 72, 90, 52, 78), ("30", 77, 95, 55, 79), ("32", 82, 100, 58, 80), ("34", 87, 105, 61, 81)]
    return {
        "garment_id": "TEST-PT",
        "category": "pants",
        "fit_type": fit_type,
        "stretch": stretch,
        "unit": "cm",
        "sizes": [{"label": l, "measurements": {"waist": w, "hip": h, "thigh": t, "inseam": i}}
                  for l, w, h, t, i in rows],
    }


def flat_version(garment):
    """Return the same garment expressed as flat (half-circumference) measurements."""
    g = copy.deepcopy(garment)
    g["measurement_type"] = "flat"
    for s in g["sizes"]:
        for k in ("chest", "waist", "hip", "thigh", "sleeve_opening"):
            if k in s["measurements"]:
                s["measurements"][k] = s["measurements"][k] / 2
    return g


def comp(result, key):
    """Get the comparison row for one garment measurement."""
    for c in result["comparisons"]:
        if c["measurement"] == key:
            return c
    raise KeyError(key)


def codes(messages):
    return [m["code"] for m in messages]
