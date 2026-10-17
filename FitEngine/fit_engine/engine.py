"""Core fit-evaluation and size-recommendation logic.

Public entry point: recommend_size(body, garment) -> dict (JSON-serialisable).
The function never raises on bad input; it returns status "invalid_input".
"""

from . import rules, schema
from .messages import OVERALL_LABELS, STATUS_LABELS, fmt, fmt_signed, msg
from .validation import FitInputError, normalize_body, normalize_garment

ENGINE_VERSION = "0.2.0"

WIDTH_TIGHT = {"slightly_tight", "too_tight"}
WIDTH_LOOSE = {"slightly_loose", "too_loose"}
ASSUMPTION_CODES = {"stretch_default", "unknown_stretch", "fit_type_default", "unknown_fit_type"}


# ---------------------------------------------------------------------------
# Per-size evaluation
# ---------------------------------------------------------------------------
def _classify(key, ease, lo, hi):
    """Return (status, distance_outside_range_cm, direction)."""
    tol = rules.tolerance_for(key)
    is_len = key in schema.LENGTH_KEYS
    if ease < lo:
        d = lo - ease
        if is_len:
            return ("slightly_short" if d <= tol else "too_short"), d, "under"
        return ("slightly_tight" if d <= tol else "too_tight"), d, "under"
    if ease > hi:
        d = ease - hi
        if is_len:
            return ("slightly_long" if d <= tol else "too_long"), d, "over"
        return ("slightly_loose" if d <= tol else "too_loose"), d, "over"
    return "good", 0.0, None


def evaluate_size(body_cm, garment, size):
    """Compare one garment size against the body. Returns a dict."""
    cat = garment["category"]
    rule = rules.CATEGORY_RULES[cat]
    weights = {**{k: (w, "primary") for k, w in rule["primary"].items()},
               **{k: (w, "secondary") for k, w in rule["secondary"].items()}}

    comparisons, info = [], []
    total_w = total_p = 0.0
    for key, (w, importance) in weights.items():
        bkey = schema.GARMENT_TO_BODY[key]
        bval = body_cm.get(bkey)
        gval = size["measurements"].get(key)
        row = {
            "measurement": key,
            "body_measurement": bkey,
            "name_ar": schema.name_ar(key),
            "name_en": schema.name_en(key),
            "importance": importance,
            "body_cm": bval,
            "garment_cm": gval,
        }
        if bval is None or gval is None:
            status = "missing_body" if bval is None else "missing_garment"
            row.update({"status": status, "ease_cm": None, "target_ease_cm": None})
        else:
            lo, hi, src = rules.target_ease(cat, key, garment["fit_type"], garment["stretch"],
                                            bval, garment["ease_overrides"])
            ease = round(gval - bval, 2)
            status, dist, direction = _classify(key, ease, lo, hi)
            tol = rules.tolerance_for(key)
            if status == "good":
                center, half = (lo + hi) / 2, max((hi - lo) / 2, 1e-6)
                p = rules.CENTER_PREFERENCE * ((ease - center) / half) ** 2
            else:
                p = (dist / tol) ** 2
                if direction == "under":
                    p *= rules.TIGHT_PENALTY_FACTOR
            total_w += w
            total_p += w * p
            row.update({"status": status, "ease_cm": ease, "target_ease_cm": [lo, hi],
                        "ease_source": src, "penalty": round(p, 4)})
        row["label_ar"], row["label_en"] = STATUS_LABELS[row["status"]]
        comparisons.append(row)

    # garment measurements present but not used for sizing this category
    for key, gval in size["measurements"].items():
        if key in weights:
            continue
        code = "not_relevant" if key in schema.GARMENT_TO_BODY else "not_compared"
        info.append({"measurement": key, "garment_cm": gval,
                     **msg(code, name_ar=schema.name_ar(key), name_en=schema.name_en(key))})

    compared = [c for c in comparisons if c["ease_cm"] is not None]
    primary_compared = [c for c in compared if c["importance"] == "primary"]
    width = [c for c in compared if c["measurement"] not in schema.LENGTH_KEYS]

    tight_block = [c["measurement"] for c in width if c["status"] == "too_tight"]
    loose_block = [c["measurement"] for c in width
                   if c["status"] == "too_loose" and c["importance"] == "primary"]
    # primary measurements the body has but this size lacks (see _mark_incomplete)
    missing_g = [c["measurement"] for c in comparisons
                 if c["importance"] == "primary" and c["status"] == "missing_garment"]
    fits_compared = bool(primary_compared) and not tight_block and not loose_block

    has_tight = any(c["status"] in WIDTH_TIGHT for c in width)
    has_loose = any(c["status"] in WIDTH_LOOSE for c in width)
    if not primary_compared:
        overall = "insufficient_data"
    elif has_tight and has_loose:
        overall = "mixed"
    elif has_tight:
        overall = "tight"
    elif has_loose:
        overall = "loose"
    else:
        overall = "good"

    return {
        "size": size["label"],
        "score": round(total_p / total_w, 4) if total_w else None,
        "acceptable": fits_compared and not missing_g,
        "fits_compared": fits_compared,
        "missing_garment_primary": missing_g,
        "incomplete_primary": missing_g,
        "overall_fit": overall,
        "tight_blocking": tight_block,
        "loose_blocking": loose_block,
        "comparisons": comparisons,
        "not_compared": info,
        "n_primary": len(rule["primary"]),
        "n_primary_compared": len(primary_compared),
        "n_secondary": len(rule["secondary"]),
        "n_secondary_compared": len(compared) - len(primary_compared),
    }


# ---------------------------------------------------------------------------
# Helpers for messages
# ---------------------------------------------------------------------------
def _join_ar(keys):
    return "، ".join(schema.name_ar(k) for k in keys)


def _join_en(keys):
    return ", ".join(schema.name_en(k) for k in keys)


def _issues(ev):
    out = []
    for c in ev["comparisons"]:
        if c["ease_cm"] is None or c["status"] == "good":
            continue
        lo, hi = c["target_ease_cm"]
        code = "issue_length" if c["measurement"] in schema.LENGTH_KEYS else "issue_width"
        m = msg(code, name_ar=c["name_ar"], name_en=c["name_en"], ease=fmt_signed(c["ease_cm"]),
                label_ar=c["label_ar"], label_en=c["label_en"],
                lo=fmt_signed(lo), hi=fmt_signed(hi))
        m.update({"measurement": c["measurement"], "status": c["status"],
                  "importance": c["importance"]})
        out.append(m)
    return out


def _dedupe(messages):
    seen, out = set(), []
    for m in messages:
        k = (m["code"], m["message_en"])
        if k not in seen:
            seen.add(k)
            out.append(m)
    return out


def _confidence(ev, assumed):
    if ev["n_primary_compared"] == 0:
        return "none"
    if ev["n_primary_compared"] < ev["n_primary"]:
        return "low"
    sec_ok = ev["n_secondary"] == 0 or ev["n_secondary_compared"] * 2 >= ev["n_secondary"]
    conf = "high" if sec_ok else "medium"
    if conf == "high" and assumed:
        conf = "medium"
    return conf


def _size_summary(ev):
    return {
        "size": ev["size"],
        "score": ev["score"],
        "acceptable": ev["acceptable"],
        "incomplete_primary": ev["incomplete_primary"],
        "overall_fit": ev["overall_fit"],
        "statuses": {c["measurement"]: c["status"] for c in ev["comparisons"]},
    }


def _base_result(garment, body_meta):
    return {
        "engine_version": ENGINE_VERSION,
        "garment_id": garment.get("garment_id") if garment else None,
        "garment_name": garment.get("name") if garment else None,
        "category": garment.get("category") if garment else None,
        "fit_type": garment.get("fit_type") if garment else None,
        "stretch": garment.get("stretch") if garment else None,
        "body_source": body_meta.get("source") if body_meta else None,
        "user_id": body_meta.get("user_id") if body_meta else None,
    }


def _overall(code):
    ar, en = OVERALL_LABELS[code]
    return {"overall_fit": code, "overall_fit_ar": ar, "overall_fit_en": en}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def recommend_size(body, garment):
    """Recommend the best size of `garment` for `body`.

    body:    {"unit": "cm", "measurements": {"chest": 96, ...}}
    garment: {"category": "t_shirt", "fit_type": "regular", "stretch": "none",
              "unit": "cm", "sizes": [{"label": "M", "measurements": {...}}]}
    """
    warnings = []
    try:
        body_cm, w1, body_meta = normalize_body(body)
        warnings += w1
    except FitInputError as e:
        body_cm, body_meta, body_err = None, {}, e
    else:
        body_err = None
    try:
        g, w2 = normalize_garment(garment)
        warnings += w2
    except FitInputError as e:
        g, g_err = None, e
    else:
        g_err = None

    if body_err or g_err:
        errors = (body_err.errors if body_err else []) + (g_err.errors if g_err else [])
        warns = (body_err.warnings if body_err else []) + (g_err.warnings if g_err else []) + warnings
        res = _base_result(garment if isinstance(garment, dict) else None, body_meta)
        res.update({"status": "invalid_input", "recommended_size": None, "closest_size": None,
                    **_overall("insufficient_data"), "confidence": "none",
                    "errors": errors, "warnings": _dedupe(warns)})
        return res

    # sanity check: a garment circumference far below the body usually means
    # half (flat) measurements were sent without measurement_type = "flat".
    if g["measurement_type"] == "circumference":
        for s in g["sizes"]:
            for k in ("chest", "waist", "hip"):
                gv, bv = s["measurements"].get(k), body_cm.get(k)
                if gv is not None and bv is not None and gv < 0.65 * bv:
                    warnings.append(msg("possible_flat_measurement", name_ar=schema.name_ar(k),
                                        name_en=schema.name_en(k), size=s["label"],
                                        garment=fmt(gv), body=fmt(bv)))

    # fit type or stretch was assumed rather than supplied -> lower confidence
    assumed = any(w["code"] in ASSUMPTION_CODES for w in warnings)
    evals = [evaluate_size(body_cm, g, s) for s in g["sizes"]]
    _mark_incomplete(evals)
    order = {ev["size"]: i for i, ev in enumerate(evals)}
    rule = rules.CATEGORY_RULES[g["category"]]
    primary_keys = list(rule["primary"])
    secondary_keys = list(rule["secondary"])

    res = _base_result(g, body_meta)
    res["disclaimer"] = msg("disclaimer")
    res["all_sizes"] = [_size_summary(ev) for ev in evals]

    missing_body = [schema.GARMENT_TO_BODY[k] for k in primary_keys + secondary_keys
                    if schema.GARMENT_TO_BODY[k] not in body_cm]
    missing_garment = sorted({k for k in primary_keys + secondary_keys
                              for s in g["sizes"] if k not in s["measurements"]})
    res["missing_measurements"] = {"body": missing_body, "garment": missing_garment}

    candidates = [ev for ev in evals if ev["n_primary_compared"] > 0]
    reasons = []

    # ---- 1. not enough data -------------------------------------------------
    if not candidates:
        res.update({"status": "insufficient_data", "recommended_size": None,
                    "closest_size": None, **_overall("insufficient_data"),
                    "confidence": "none", "alternative_size": None,
                    "comparisons": evals[0]["comparisons"] if evals else [],
                    "not_compared": evals[0]["not_compared"] if evals else [],
                    "issues": [],
                    "reasons": [msg("insufficient_primary", keys_ar=_join_ar(primary_keys),
                                    keys_en=_join_en(primary_keys))],
                    "warnings": _dedupe(warnings)})
        return res

    # sizes evaluated on fewer key measurements than the body allows rank last
    ranked = sorted(candidates, key=lambda ev: (bool(ev["incomplete_primary"]), ev["score"],
                                                order[ev["size"]]))
    acceptable = [ev for ev in ranked if ev["acceptable"]]
    for ev in candidates:
        if ev["incomplete_primary"]:
            warnings.append(msg("incomplete_size", size=ev["size"],
                                keys_ar=_join_ar(ev["incomplete_primary"]),
                                keys_en=_join_en(ev["incomplete_primary"])))

    # ---- 2a. only sizes with missing key garment data might fit ---------------
    maybe = [ev for ev in ranked if ev["fits_compared"] and ev["incomplete_primary"]]
    if not acceptable and maybe:
        sizes = ", ".join(ev["size"] for ev in sorted(maybe, key=lambda ev: order[ev["size"]]))
        res.update({"status": "insufficient_data", "situation": "incomplete_garment_data",
                    "recommended_size": None, "closest_size": None,
                    **_overall("insufficient_data"), "confidence": "none",
                    "alternative_size": None,
                    "comparisons": maybe[0]["comparisons"],
                    "not_compared": maybe[0]["not_compared"],
                    "issues": [],
                    "reasons": [msg("incomplete_garment_data", sizes=sizes)],
                    "warnings": _dedupe(warnings)})
        return res

    # ---- 2b. no wearable size ------------------------------------------------
    if not acceptable:
        closest = ranked[0]
        all_tight = all(ev["tight_blocking"] and not ev["loose_blocking"] for ev in candidates)
        all_loose = all(ev["loose_blocking"] and not ev["tight_blocking"] for ev in candidates)
        if all_tight:
            reasons.append(msg("all_too_small", size=closest["size"],
                               keys_ar=_join_ar(closest["tight_blocking"]),
                               keys_en=_join_en(closest["tight_blocking"])))
            situation = "all_sizes_too_small"
        elif all_loose:
            reasons.append(msg("all_too_large", size=closest["size"],
                               keys_ar=_join_ar(closest["loose_blocking"]),
                               keys_en=_join_en(closest["loose_blocking"])))
            situation = "all_sizes_too_large"
        else:
            parts_ar, parts_en = [], []
            for k in primary_keys:
                ok = [ev["size"] for ev in evals
                      for c in ev["comparisons"]
                      if c["measurement"] == k and c["ease_cm"] is not None
                      and c["status"] not in ("too_tight", "too_loose")]
                if any(c["measurement"] == k and c["ease_cm"] is not None
                       for ev in evals for c in ev["comparisons"]):
                    parts_ar.append(f"{schema.name_ar(k)} يناسبه: {'، '.join(ok) if ok else 'لا يوجد'}")
                    parts_en.append(f"{schema.name_en(k)} fits: {', '.join(ok) if ok else 'none'}")
            reasons.append(msg("no_single_size", detail_ar="؛ ".join(parts_ar) + ".",
                               detail_en="; ".join(parts_en) + "."))
            situation = "no_single_size_fits"
        res.update({"status": "no_suitable_size", "situation": situation,
                    "recommended_size": None, "closest_size": closest["size"],
                    **_overall(closest["overall_fit"]),
                    "confidence": _confidence(closest, assumed),
                    "alternative_size": None,
                    "comparisons": closest["comparisons"],
                    "not_compared": closest["not_compared"],
                    "issues": _issues(closest), "reasons": reasons})
        _add_coverage_notes(res, closest, primary_keys, secondary_keys, g, warnings)
        res["warnings"] = _dedupe(warnings)
        return res

    # ---- 3. normal recommendation -------------------------------------------
    rec = acceptable[0]
    if rec["overall_fit"] == "good" and all(
            c["status"] == "good" for c in rec["comparisons"] if c["ease_cm"] is not None):
        reasons.append(msg("reason_all_good", size=rec["size"]))
    else:
        reasons.append(msg("reason_best_acceptable", size=rec["size"]))
        reasons.append(msg("reason_primary_ok", size=rec["size"],
                           keys_ar=_join_ar(primary_keys), keys_en=_join_en(primary_keys)))

    alternative = _pick_alternative(rec, acceptable, evals, order)
    if alternative and rec["overall_fit"] != "good":
        a, b = sorted([rec["size"], alternative["size"]], key=lambda s: order[s])
        reasons.append(msg("between_sizes", a=a, b=b))

    res.update({"status": "ok", "recommended_size": rec["size"], "closest_size": rec["size"],
                **_overall(rec["overall_fit"]),
                "confidence": _confidence(rec, assumed),
                "alternative_size": alternative,
                "comparisons": rec["comparisons"], "not_compared": rec["not_compared"],
                "issues": _issues(rec), "reasons": reasons})
    _add_coverage_notes(res, rec, primary_keys, secondary_keys, g, warnings)
    res["warnings"] = _dedupe(warnings)
    return res


def _mark_incomplete(evals):
    """A size is incomplete when it lacks a primary measurement that the body has
    and that another size of the same garment does have. Such a size was compared
    on fewer keys than its neighbours, so its score is not comparable and it
    cannot be recommended. If every size lacks the key, all are compared on the
    same keys and the gap only lowers confidence (like a missing body value).
    """
    measured = {c["measurement"] for ev in evals for c in ev["comparisons"]
                if c["ease_cm"] is not None}
    for ev in evals:
        ev["incomplete_primary"] = [k for k in ev["missing_garment_primary"] if k in measured]
        ev["acceptable"] = ev["fits_compared"] and not ev["incomplete_primary"]


def _pick_alternative(rec, acceptable, evals, order):
    idx = order[rec["size"]]
    acc = {ev["size"]: ev for ev in acceptable}
    larger = next((ev for ev in evals if order[ev["size"]] == idx + 1 and ev["size"] in acc), None)
    smaller = next((ev for ev in evals if order[ev["size"]] == idx - 1 and ev["size"] in acc), None)

    tight_keys = [c["measurement"] for c in rec["comparisons"]
                  if c["status"] == "slightly_tight"]
    if tight_keys and larger:
        return {"size": larger["size"], "direction": "larger",
                **msg("alt_fixes_tight", size=larger["size"],
                      keys_ar=_join_ar(tight_keys), keys_en=_join_en(tight_keys))}

    best = None
    for ev, direction in ((larger, "larger"), (smaller, "smaller")):
        if ev and ev["score"] <= rec["score"] + rules.ALTERNATIVE_MARGIN:
            if best is None or ev["score"] < best[0]["score"]:
                best = (ev, direction)
    if best:
        ev, direction = best
        code = "alt_larger" if direction == "larger" else "alt_smaller"
        return {"size": ev["size"], "direction": direction, **msg(code, size=ev["size"])}
    return None


def _add_coverage_notes(res, ev, primary_keys, secondary_keys, g, warnings):
    missing_p = [c["measurement"] for c in ev["comparisons"]
                 if c["importance"] == "primary" and c["ease_cm"] is None]
    missing_s = [c["measurement"] for c in ev["comparisons"]
                 if c["importance"] == "secondary" and c["ease_cm"] is None]
    if missing_p:
        warnings.append(msg("missing_primary_partial", keys_ar=_join_ar(missing_p),
                            keys_en=_join_en(missing_p)))
    if missing_s:
        warnings.append(msg("missing_secondary", keys_ar=_join_ar(missing_s),
                            keys_en=_join_en(missing_s)))
    if g["stretch"] != "none":
        pct = int(round(rules.STRETCH_NEGATIVE_PCT[g["stretch"]] * 100))
        res["reasons"].append(msg("stretch_note", stretch=g["stretch"], pct=pct))
    overridden = [c["measurement"] for c in ev["comparisons"] if c.get("ease_source") == "override"]
    if overridden:
        res["reasons"].append(msg("ease_override_used", keys_ar=_join_ar(overridden),
                                  keys_en=_join_en(overridden)))
