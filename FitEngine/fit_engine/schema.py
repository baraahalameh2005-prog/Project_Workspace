"""Canonical measurement names, plausible ranges and body<->garment mapping.

IMPORTANT distinction:
  * Body measurements describe the person (from the avatar system or manual input).
  * Garment measurements describe the finished piece of clothing for one size
    (from the clothing-management system).
They are never the same thing: garment chest = body chest + ease.
"""

# ---- Body measurements (all in cm) -------------------------------------------
# key: (min plausible, max plausible) in cm, adult/teen range.
BODY_RANGES = {
    "height": (50.0, 250.0),
    "chest": (40.0, 200.0),
    "waist": (30.0, 200.0),
    "hip": (40.0, 220.0),
    "shoulder_width": (20.0, 70.0),
    "arm_length": (30.0, 100.0),   # shoulder tip -> wrist
    "inseam": (30.0, 120.0),       # crotch -> ankle
    "thigh": (25.0, 110.0),        # thigh circumference at fullest point
}

# Keys that are accepted but never used to choose a size.
BODY_IGNORED_KEYS = {"weight"}

# ---- Garment measurements (all in cm, full circumference) ---------------------
GARMENT_RANGES = {
    "chest": (30.0, 280.0),
    "waist": (20.0, 280.0),
    "hip": (30.0, 300.0),
    "shoulder_width": (15.0, 90.0),
    "sleeve_length": (5.0, 110.0),   # shoulder seam -> sleeve end
    "sleeve_opening": (5.0, 80.0),   # circumference of sleeve end
    "length": (15.0, 200.0),         # total garment length
    "inseam": (5.0, 130.0),
    "thigh": (20.0, 150.0),
}

# Garment keys that are circumferences. If a garment is measured "flat"
# (laid on a table, half circumference), these are multiplied by 2.
GARMENT_CIRCUMFERENCE_KEYS = {"chest", "waist", "hip", "sleeve_opening", "thigh"}

# Which body measurement a garment measurement is compared against.
# Garment keys missing here (sleeve_opening, length) are informational only.
GARMENT_TO_BODY = {
    "chest": "chest",
    "waist": "waist",
    "hip": "hip",
    "shoulder_width": "shoulder_width",
    "sleeve_length": "arm_length",
    "inseam": "inseam",
    "thigh": "thigh",
}

# Length-type comparisons use short/long labels instead of tight/loose.
LENGTH_KEYS = {"sleeve_length", "inseam"}

# Accepted alternative names -> canonical key (helps future integration).
BODY_ALIASES = {
    "bust": "chest",
    "chest_circumference": "chest",
    "waist_circumference": "waist",
    "hips": "hip",
    "hip_circumference": "hip",
    "shoulders": "shoulder_width",
    "shoulder": "shoulder_width",
    "arm": "arm_length",
    "sleeve": "arm_length",
    "leg_inseam": "inseam",
    "thigh_circumference": "thigh",
    "stature": "height",
}

GARMENT_ALIASES = {
    "bust": "chest",
    "hips": "hip",
    "shoulders": "shoulder_width",
    "shoulder": "shoulder_width",
    "sleeve": "sleeve_length",
    "cuff": "sleeve_opening",
    "total_length": "length",
    "body_length": "length",
}

NAMES = {
    "height": ("الطول", "height"),
    "chest": ("محيط الصدر", "chest"),
    "waist": ("محيط الخصر", "waist"),
    "hip": ("محيط الورك", "hip"),
    "shoulder_width": ("عرض الكتفين", "shoulder width"),
    "arm_length": ("طول الذراع", "arm length"),
    "inseam": ("طول الساق الداخلي", "inseam"),
    "thigh": ("محيط الفخذ", "thigh"),
    "sleeve_length": ("طول الكم", "sleeve length"),
    "sleeve_opening": ("محيط نهاية الكم", "sleeve opening"),
    "length": ("طول القطعة", "garment length"),
    "weight": ("الوزن", "weight"),
}


def name_ar(key):
    return NAMES.get(key, (key, key))[0]


def name_en(key):
    return NAMES.get(key, (key, key))[1]
