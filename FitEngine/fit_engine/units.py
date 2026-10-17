"""Unit conversion. The canonical unit of the whole system is the centimetre (cm)."""

CANONICAL_UNIT = "cm"

# factor to multiply a value in <unit> to get centimetres
_FACTORS = {
    "cm": 1.0,
    "centimeter": 1.0,
    "centimeters": 1.0,
    "سم": 1.0,
    "mm": 0.1,
    "millimeter": 0.1,
    "millimeters": 0.1,
    "ملم": 0.1,
    "m": 100.0,
    "meter": 100.0,
    "meters": 100.0,
    "م": 100.0,
    "in": 2.54,
    "inch": 2.54,
    "inches": 2.54,
    '"': 2.54,
    "إنش": 2.54,
}

SUPPORTED_UNITS = ("cm", "mm", "m", "in")


def normalize_unit(unit):
    """Return the canonical spelling of a unit, or None if unsupported."""
    if unit is None:
        return None
    if not isinstance(unit, str):
        return None
    u = unit.strip().lower()
    if u not in _FACTORS:
        return None
    f = _FACTORS[u]
    for canon in SUPPORTED_UNITS:
        if _FACTORS[canon] == f:
            return canon
    return None


def to_cm(value, unit):
    """Convert a numeric value from *unit* to centimetres. Unit must be valid."""
    canon = normalize_unit(unit)
    if canon is None:
        raise ValueError(f"Unsupported unit: {unit!r}")
    return float(value) * _FACTORS[canon]
