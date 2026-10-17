"""Input validation and normalisation to canonical keys in centimetres."""

import math

from . import schema
from .messages import msg, fmt
from .rules import SUPPORTED_CATEGORIES, SUPPORTED_FIT_TYPES, SUPPORTED_STRETCH
from .units import normalize_unit, to_cm


class FitInputError(Exception):
    """Raised when the input cannot be used at all. Carries structured errors."""

    def __init__(self, errors, warnings=None):
        super().__init__("; ".join(e["message_en"] for e in errors))
        self.errors = errors
        self.warnings = warnings or []


def _err(code, field, **kw):
    m = msg(code, **kw)
    m["field"] = field
    return m


def _range_hint(value, unit_canon, lo, hi, is_flat_candidate=False):
    """Suggest a likely unit mistake when a value is out of range."""
    if unit_canon == "cm":
        if lo <= value * 2.54 <= hi:
            return " هل القياس بالإنش؟", " Was it measured in inches?"
        if lo <= value / 10 <= hi:
            return " هل القياس بالملم؟", " Was it measured in millimetres?"
        if is_flat_candidate and lo <= value * 2 <= hi:
            return (" هل هذا نصف المحيط؟ استخدم measurement_type = flat.",
                    " Is this a half (flat) measurement? Use measurement_type = flat.")
    if unit_canon == "in" and lo <= value <= hi:
        return " هل القياس بالسنتيمتر وليس بالإنش؟", " Was it in cm rather than inches?"
    return "", ""


def _parse_value(raw, default_unit, where, errors):
    """Accept 96, "96", or {"value": 38, "unit": "in"}. Return (value, unit) or None."""
    unit = default_unit
    if isinstance(raw, dict):
        if "unit" in raw:
            unit = raw.get("unit")
        raw = raw.get("value")
    if isinstance(raw, bool) or raw is None:
        errors.append(_err("invalid_number", where, value=raw, where=where))
        return None
    if isinstance(raw, str):
        try:
            raw = float(raw.strip().replace(",", "."))
        except ValueError:
            errors.append(_err("invalid_number", where, value=raw, where=where))
            return None
    if not isinstance(raw, (int, float)) or math.isnan(raw) or math.isinf(raw):
        errors.append(_err("invalid_number", where, value=raw, where=where))
        return None
    if raw <= 0:
        errors.append(_err("non_positive", where, value=raw, where=where))
        return None
    canon = normalize_unit(unit)
    if canon is None:
        errors.append(_err("unknown_unit", where, unit=unit, where=where))
        return None
    return float(raw), canon


def _normalize_measurements(raw_map, default_unit, aliases, ranges, where_prefix,
                            errors, warnings, flat_keys=frozenset(), ignored=frozenset()):
    out = {}
    if not isinstance(raw_map, dict):
        errors.append(_err("invalid_document", where_prefix, where=where_prefix))
        return out
    for raw_key, raw_val in raw_map.items():
        key = str(raw_key).strip().lower()
        if key in ignored:
            if key == "weight":
                warnings.append(msg("weight_ignored"))
            continue
        if key in aliases:
            canon = aliases[key]
            warnings.append(msg("alias_used", alias=raw_key, key=canon, where=where_prefix))
            key = canon
        where = f"{where_prefix}.{key}"
        if key not in ranges:
            warnings.append(msg("unknown_measurement", key=raw_key, where=where_prefix))
            continue
        if key in out:
            warnings.append(msg("duplicate_measurement", key=key, where=where_prefix))
            continue
        parsed = _parse_value(raw_val, default_unit, where, errors)
        if parsed is None:
            continue
        value, unit = parsed
        cm = to_cm(value, unit)
        if key in flat_keys:
            cm *= 2.0
        lo, hi = ranges[key]
        if not (lo <= cm <= hi):
            h_ar, h_en = _range_hint(value, unit, lo, hi,
                                     is_flat_candidate=(not flat_keys and key in schema.GARMENT_CIRCUMFERENCE_KEYS
                                                        and ranges is schema.GARMENT_RANGES))
            errors.append(_err("out_of_range", where, value_cm=fmt(cm), where=where,
                               lo=fmt(lo), hi=fmt(hi), hint_ar=h_ar, hint_en=h_en))
            continue
        out[key] = round(cm, 2)
    return out


def _doc_unit(doc, where, errors, warnings):
    unit = doc.get("unit")
    if unit is None:
        warnings.append(msg("missing_unit", where=where))
        return "cm"
    canon = normalize_unit(unit)
    if canon is None:
        errors.append(_err("unknown_unit", f"{where}.unit", unit=unit, where=where))
        return None
    return canon


def normalize_body(body):
    """Validate a body document. Returns (measurements_cm, warnings, meta).

    Accepted shapes:
      {"unit": "cm", "measurements": {"chest": 96, ...}, "source": "manual"}
      {"chest": 96, "waist": 80, ...}   (flat, cm assumed)
    """
    errors, warnings = [], []
    if not isinstance(body, dict):
        raise FitInputError([_err("invalid_document", "body", where="body")])
    if "measurements" in body:
        unit = _doc_unit(body, "body", errors, warnings)
        raw = body.get("measurements")
    else:
        unit = _doc_unit(body, "body", errors, warnings) if "unit" in body else "cm"
        raw = {k: v for k, v in body.items() if k not in ("unit", "source", "user_id")}
    if unit is None:
        raise FitInputError(errors, warnings)

    m = _normalize_measurements(raw, unit, schema.BODY_ALIASES, schema.BODY_RANGES,
                                "body.measurements", errors, warnings,
                                ignored=schema.BODY_IGNORED_KEYS)
    if errors:
        raise FitInputError(errors, warnings)
    if not m:
        raise FitInputError([_err("no_body_measurements", "body.measurements")], warnings)

    # soft consistency checks (warnings only)
    details = []
    if "chest" in m and "waist" in m and m["waist"] > m["chest"] * 1.6:
        details.append("waist >> chest")
    if "waist" in m and "hip" in m and m["hip"] < m["waist"] * 0.6:
        details.append("hip << waist")
    if "height" in m and "inseam" in m and m["inseam"] > m["height"] * 0.6:
        details.append("inseam > 60% height")
    if details:
        warnings.append(msg("unusual_proportions", detail=", ".join(details)))

    meta = {"source": body.get("source", "unspecified"), "user_id": body.get("user_id")}
    return m, warnings, meta


def normalize_garment(garment):
    """Validate a garment document. Returns (garment_dict, warnings).

    garment_dict = {
        "garment_id", "name", "category", "fit_type", "stretch",
        "sizes": [{"label": "M", "measurements": {...cm, full circumference}}],
        "ease_overrides": {key: (lo, hi)}
    }
    """
    errors, warnings = [], []
    if not isinstance(garment, dict):
        raise FitInputError([_err("invalid_document", "garment", where="garment")])

    supported = ", ".join(SUPPORTED_CATEGORIES)
    category = garment.get("category")
    if category is None:
        errors.append(_err("missing_category", "garment.category", supported=supported))
    else:
        category = str(category).strip().lower()
        if category not in SUPPORTED_CATEGORIES:
            errors.append(_err("unknown_category", "garment.category", value=category,
                               supported=supported))

    fit_type = garment.get("fit_type")
    if fit_type is None:
        warnings.append(msg("fit_type_default"))
        fit_type = "regular"
    else:
        fit_type = str(fit_type).strip().lower()
        if fit_type not in SUPPORTED_FIT_TYPES:
            warnings.append(msg("unknown_fit_type", value=fit_type))
            fit_type = "regular"

    stretch = garment.get("stretch")
    if stretch is None:
        warnings.append(msg("stretch_default"))
        stretch = "none"
    else:
        stretch = str(stretch).strip().lower()
        if stretch not in SUPPORTED_STRETCH:
            warnings.append(msg("unknown_stretch", value=stretch))
            stretch = "none"

    mtype = garment.get("measurement_type")
    mtype = "circumference" if mtype is None else str(mtype).strip().lower()
    if mtype not in ("circumference", "flat"):
        errors.append(_err("unknown_measurement_type", "garment.measurement_type", value=mtype))
        mtype = "circumference"
    flat_keys = schema.GARMENT_CIRCUMFERENCE_KEYS if mtype == "flat" else frozenset()

    unit = _doc_unit(garment, "garment", errors, warnings)

    overrides = {}
    raw_overrides = garment.get("ease_overrides")
    if raw_overrides is None:
        raw_overrides = {}
    if not isinstance(raw_overrides, dict):
        errors.append(_err("invalid_document", "garment.ease_overrides",
                           where="garment.ease_overrides"))
        raw_overrides = {}
    for k, v in raw_overrides.items():
        k2 = schema.GARMENT_ALIASES.get(str(k).lower(), str(k).lower())
        ok = (isinstance(v, (list, tuple)) and len(v) == 2
              and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                      and math.isfinite(x) for x in v)
              and v[0] < v[1])
        if not ok:
            errors.append(_err("invalid_ease_override", f"garment.ease_overrides.{k}", key=k))
        elif k2 not in schema.GARMENT_TO_BODY:
            warnings.append(msg("unknown_ease_override", key=k))
        else:
            overrides[k2] = (float(v[0]), float(v[1]))

    sizes_raw = garment.get("sizes")
    sizes = []
    if not isinstance(sizes_raw, list) or not sizes_raw:
        errors.append(_err("no_sizes", "garment.sizes"))
    elif unit is not None:
        seen = set()
        for i, s in enumerate(sizes_raw):
            raw_label = s.get("label") if isinstance(s, dict) else None
            if (not isinstance(raw_label, (str, int, float)) or isinstance(raw_label, bool)
                    or not str(raw_label).strip()):
                errors.append(_err("size_without_label", f"garment.sizes[{i}]", index=i + 1))
                continue
            label = str(s["label"]).strip()
            if label in seen:
                errors.append(_err("duplicate_size", f"garment.sizes[{i}]", label=label))
                continue
            seen.add(label)
            s_unit = unit
            if "unit" in s:
                s_unit = normalize_unit(s["unit"])
                if s_unit is None:
                    errors.append(_err("unknown_unit", f"garment.sizes[{i}].unit",
                                       unit=s["unit"], where=f"size {label}"))
                    continue
            m = _normalize_measurements(s.get("measurements", {}), s_unit,
                                        schema.GARMENT_ALIASES, schema.GARMENT_RANGES,
                                        f"garment.sizes[{label}]", errors, warnings,
                                        flat_keys=flat_keys)
            if not m:
                errors.append(_err("size_no_measurements", f"garment.sizes[{i}]", label=label))
                continue
            sizes.append({"label": label, "measurements": m})

    if errors:
        raise FitInputError(errors, warnings)

    return {
        "garment_id": garment.get("garment_id"),
        "name": garment.get("name"),
        "category": category,
        "fit_type": fit_type,
        "stretch": stretch,
        "measurement_type": mtype,
        "sizes": sizes,
        "ease_overrides": overrides,
    }, warnings
