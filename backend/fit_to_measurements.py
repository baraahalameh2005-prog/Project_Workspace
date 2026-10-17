import argparse
import json
import math
import time

import numpy as np

from app.config import apply_environment, settings

# PyTorch and Anny read their cache folders when they are imported.
apply_environment(settings)

import torch
import anny


DT = torch.float64

PARAM_MIN = 0.001
PARAM_MAX = 0.999

# Anny 0.6.1's age anchors (anny/models/model_data.py) are newborn 0.0,
# baby 0.25, child 0.5, young 0.75 and old 1.0; by Anny's own calibration
# 0.5 is about 13 years. This bound is kept for the general (female) fit;
# male bodies use ADULT_MALE_AGE (see fit_adult_male).
AGE_MIN = 0.5
AGE_MAX = PARAM_MAX

# Inner height / weight solves stop at these tolerances.
HEIGHT_TOLERANCE_M = 1e-4
WEIGHT_TOLERANCE_KG = 1e-2

# The shape fit stops early once every target (circumferences,
# height, weight) is within this relative error.
GOOD_ENOUGH_RELATIVE_ERROR = 0.002

# Finite-difference step for the shape parameters. A step this size
# keeps the Jacobian well above the noise left by the inner solves.
DIFF_STEP = 1e-2

GENDER_VALUES = {
    "male": 0.0,
    "female": 1.0,
}

# requested name -> (result key, scale to the requested unit, unit)
TARGET_MEASUREMENTS = {
    "height": ("height_m", 100.0, "cm"),
    "weight": ("mass_kg", 1.0, "kg"),
    "waist": ("waist_m", 100.0, "cm"),
    "chest": ("chest_m", 100.0, "cm"),
    "hip": ("hip_m", 100.0, "cm"),
    # Male bodies only: shoulder tip to shoulder tip (measurements.shoulder_breadth).
    "shoulder": ("shoulder_breadth_m", 100.0, "cm"),
}

# How each requested measurement is reached by the fit.
TARGET_CONTROL = {
    "height": "solved directly (height parameter), shape assists if out of range",
    "weight": "solved directly (weight parameter), shape assists if out of range",
    "waist": "fitted indirectly (age, muscle, proportions)",
    "chest": "fitted indirectly (age, muscle, proportions)",
    "hip": "fitted indirectly (age, muscle, proportions)",
}

# Adult male fit: chest and hip are tape girths (measurements.tape_measurements:
# nipple height / greatest buttock prominence, arms left out, hollows bridged).
MALE_TARGET_MEASUREMENTS = {
    **TARGET_MEASUREMENTS,
    "chest": ("chest_tape_m", 100.0, "cm"),
    "hip": ("hip_tape_m", 100.0, "cm"),
}

# The same for the adult male fit (fit_adult_male).
ADULT_MALE_CONTROL = {
    "height": "solved directly (height parameter)",
    "weight": "solved directly (weight parameter, muscle; body density 1040 kg/m3)",
    "waist": "fitted with Anny's waist control",
    "chest": "tape at nipple height, arms left out; fitted with Anny's bust control",
    "hip": "tape at the most prominent point of the buttocks; fitted with Anny's hips control",
    "shoulder": "fitted with Anny's torso V-shape control",
}


def normalize_gender(gender):
    """Return "male" or "female"; raise ValueError for anything else."""

    if isinstance(gender, str):
        value = gender.strip().lower()

        if value in GENDER_VALUES:
            return value

    raise ValueError(
        f"Invalid gender {gender!r}; expected 'male' or 'female'."
    )


def gender_value(gender):
    return GENDER_VALUES[normalize_gender(gender)]


def make_phenotype(
    model,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = {
        k: torch.full(
            (1,),
            0.5,
            dtype=DT,
        )
        for k in model.phenotype_labels
    }

    ph["gender"] = torch.full(
        (1,),
        gender,
        dtype=DT,
    )

    ph["age"] = torch.full(
        (1,),
        age,
        dtype=DT,
    )

    ph["height"] = torch.full(
        (1,),
        height,
        dtype=DT,
    )

    ph["weight"] = torch.full(
        (1,),
        weight,
        dtype=DT,
    )

    ph["muscle"] = torch.full(
        (1,),
        muscle,
        dtype=DT,
    )

    ph["proportions"] = torch.full(
        (1,),
        proportions,
        dtype=DT,
    )

    return ph


# ============================================================
# FULL EVALUATION
# ============================================================

def evaluate(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    result = measurements.measure(
        output
    )

    return output, result


# ============================================================
# LIGHT EVALUATION
# ============================================================

def evaluate_anthropometry(
    model,
    measurements,
    gender,
    age,
    height,
    weight,
    muscle,
    proportions,
):
    ph = make_phenotype(
        model,
        gender,
        age,
        height,
        weight,
        muscle,
        proportions,
    )

    output = model(
        phenotype_kwargs=ph
    )

    base = measurements.anthropometry(
        output["rest_vertices"]
    )

    height_m = float(
        base["height"].item()
    )

    mass_kg = float(
        base["mass"].item()
    )

    return output, height_m, mass_kg


# ============================================================
# MONOTONIC 1-D SOLVER
# ============================================================

def solve_monotonic(
    func,
    target,
    start,
    tolerance,
    lo=PARAM_MIN,
    hi=PARAM_MAX,
    max_evaluations=12,
):
    """
    Find x in [lo, hi] with func(x) == target for increasing func.

    func(x) returns (value, payload). Uses secant steps from a warm
    start, kept inside a shrinking bracket, with bisection as the
    fallback. Returns (x, value, payload, evaluations) for the best
    point seen.
    """

    a, b = lo, hi
    best = None
    history = []

    def run(x):
        nonlocal a, b, best

        value, payload = func(x)
        history.append((x, value))

        if value < target:
            a = max(a, x)
        else:
            b = min(b, x)

        if best is None or abs(value - target) < abs(best[1] - target):
            best = (x, value, payload)

        return value

    x = min(max(start, lo), hi)
    value = run(x)

    while (
        abs(value - target) > tolerance
        and len(history) < max_evaluations
        and b - a > 1e-9
    ):
        if len(history) == 1:
            step = 0.02 if value < target else -0.02
            x_next = x + step
        else:
            (x0, y0), (x1, y1) = history[-2], history[-1]

            if y1 != y0:
                x_next = x1 + (target - y1) * (x1 - x0) / (y1 - y0)
            else:
                x_next = (a + b) / 2.0

        if not (a < x_next < b):
            tried = {x for x, _ in history}

            if x_next >= b and b >= hi and hi not in tried:
                # Target may lie beyond the upper bound: check it once
                # instead of creeping towards it.
                x_next = hi
            elif x_next <= a and a <= lo and lo not in tried:
                x_next = lo
            else:
                x_next = (a + b) / 2.0

        x = x_next
        value = run(x)

    return best[0], best[1], best[2], len(history)


# ============================================================
# SOLVE HEIGHT
# ============================================================

def solve_height(
    model,
    measurements,
    gender,
    age,
    weight,
    muscle,
    proportions,
    target_height,
    start=0.5,
    details=False,
):
    def func(height):
        output, current_height, current_mass = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            height,
            weight,
            muscle,
            proportions,
        )

        return current_height, (output, current_height, current_mass)

    solved = solve_monotonic(
        func,
        target_height,
        start,
        HEIGHT_TOLERANCE_M,
    )

    return solved if details else solved[0]


# ============================================================
# SOLVE WEIGHT
# ============================================================

def solve_weight(
    model,
    measurements,
    gender,
    age,
    height,
    muscle,
    proportions,
    target_weight,
    start=0.5,
    details=False,
):
    def func(weight):
        output, current_height, current_mass = evaluate_anthropometry(
            model,
            measurements,
            gender,
            age,
            height,
            weight,
            muscle,
            proportions,
        )

        return current_mass, (output, current_height, current_mass)

    solved = solve_monotonic(
        func,
        target_weight,
        start,
        WEIGHT_TOLERANCE_KG,
    )

    return solved if details else solved[0]


# ============================================================
# EVALUATE BODY SHAPE
# ============================================================

def evaluate_shape(
    model,
    measurements,
    gender,
    target_height,
    target_weight,
    age,
    muscle,
    proportions,
    start=None,
    stats=None,
):
    """
    Solve height and weight for the given shape, then measure the body.

    start is an optional (height, weight) warm start, usually the
    solution for the previous, nearby shape. stats, if given, counts
    the Anny evaluations used.
    """

    height, weight = start if start is not None else (0.5, 0.5)

    # Step 1: solve height
    height, _, _, n_height = solve_height(
        model,
        measurements,
        gender,
        age,
        weight,
        muscle,
        proportions,
        target_height,
        start=height,
        details=True,
    )

    # Step 2: solve weight
    weight, _, payload, n_weight = solve_weight(
        model,
        measurements,
        gender,
        age,
        height,
        muscle,
        proportions,
        target_weight,
        start=weight,
        details=True,
    )

    output, current_height, _ = payload
    n_height_again = 0

    # Step 3: weight changes height a little; solve height again
    # only if it moved outside the tolerance.
    if abs(current_height - target_height) > HEIGHT_TOLERANCE_M:
        height, _, payload, n_height_again = solve_height(
            model,
            measurements,
            gender,
            age,
            weight,
            muscle,
            proportions,
            target_height,
            start=height,
            details=True,
        )

        output = payload[0]

    if stats is not None:
        stats["anny_evaluations"] += (
            n_height + n_weight + n_height_again
        )

    # Step 4: measure the body of the last evaluation, which already
    # uses the solved height and weight.
    result = measurements.measure(
        output
    )

    return output, result, height, weight


# ============================================================
# ACCURACY
# ============================================================

def _finite(value):
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def percentage_error(requested, actual):
    """
    abs(actual - requested) / abs(requested) * 100.

    Returns None when either value is missing or not finite, or the
    requested value is zero.
    """

    if not (_finite(requested) and _finite(actual)):
        return None

    requested = float(requested)
    actual = float(actual)

    if requested == 0.0:
        return None

    return abs(actual - requested) / abs(requested) * 100.0


def accuracy_report(targets, result, controls=None, keys=None):
    """
    Compare requested measurements with the measured body.

    targets maps names from TARGET_MEASUREMENTS to requested values
    in their user units (cm or kg). A measurement the pipeline could
    not produce is reported as "measurement_failed", never dropped.
    controls overrides TARGET_CONTROL (how each target was reached) and
    keys overrides TARGET_MEASUREMENTS (which result value is compared).
    """

    controls = controls or TARGET_CONTROL
    keys = keys or TARGET_MEASUREMENTS

    report = {}

    for name, requested in targets.items():
        key, scale, unit = keys[name]

        raw = result.get(key) if result else None

        entry = {
            "unit": unit,
            "requested": (
                float(requested)
                if _finite(requested)
                else None
            ),
            "actual": None,
            "absolute_error": None,
            "percentage_error": None,
            "control": controls[name],
            "status": "measurement_failed",
        }

        if _finite(raw):
            actual = float(raw) * scale

            entry["actual"] = actual
            entry["status"] = "measured"

            if entry["requested"] is not None:
                entry["absolute_error"] = abs(
                    actual - entry["requested"]
                )

                entry["percentage_error"] = percentage_error(
                    entry["requested"],
                    actual,
                )

        report[name] = entry

    return report


# ============================================================
# FIT BODY SHAPE
# ============================================================

class _GoodEnough(Exception):
    pass


def fit(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
    stats=None,
):
    try:
        from scipy.optimize import least_squares
    except ImportError:
        raise RuntimeError(
            "scipy is required. Install it with: "
            "python -m pip install scipy"
        )

    if stats is None:
        stats = {}

    stats.setdefault("anny_evaluations", 0)
    stats.setdefault("shape_evaluations", 0)

    target_height = height_cm / 100.0
    target_weight = weight_kg

    # Height and weight are solved inside every shape evaluation and
    # their residuals are normally ~0. They stay in the objective so
    # that, when a target is out of reach of the height or weight
    # parameter alone, the shape parameters help close the gap
    # instead of the miss going unnoticed.
    targets = np.array(
        [
            waist_cm / 100.0,
            chest_cm / 100.0,
            hip_cm / 100.0,
            target_height,
            target_weight,
        ],
        dtype=np.float64,
    )

    # age, muscle, proportions
    x0 = np.array(
        [
            AGE_MIN,
            0.5,
            0.5,
        ],
        dtype=np.float64,
    )

    lower = np.array(
        [
            AGE_MIN,
            PARAM_MIN,
            PARAM_MIN,
        ],
        dtype=np.float64,
    )

    upper = np.array(
        [
            AGE_MAX,
            PARAM_MAX,
            PARAM_MAX,
        ],
        dtype=np.float64,
    )

    state = {
        "start": (0.5, 0.5),
        "cache": {},
        "best": None,
    }

    def run(params):
        key = tuple(float(x) for x in params)

        if key in state["cache"]:
            return state["cache"][key]

        stats["shape_evaluations"] += 1

        output, result, height, weight = evaluate_shape(
            model,
            measurements,
            gender,
            target_height,
            target_weight,
            params[0],
            params[1],
            params[2],
            start=state["start"],
            stats=stats,
        )

        state["start"] = (height, weight)

        values = [
            result.get("waist_m"),
            result.get("chest_m"),
            result.get("hip_m"),
            result.get("height_m"),
            result.get("mass_kg"),
        ]

        entry = {
            "params": np.array(params, dtype=np.float64),
            "output": output,
            "result": result,
            "height": height,
            "weight": weight,
            "residual": None,
        }

        if all(_finite(v) for v in values):
            entry["residual"] = (
                np.array(values, dtype=np.float64) - targets
            ) / targets

            best = state["best"]

            if best is None or (
                np.linalg.norm(entry["residual"])
                < np.linalg.norm(best["residual"])
            ):
                state["best"] = entry

        state["cache"][key] = entry

        return entry

    def objective(params):
        entry = run(params)
        n = stats["shape_evaluations"]

        if entry["residual"] is None:
            print(
                f"  evaluation {n:>3}"
                " | invalid measurement"
                f" | age {params[0]:.3f}"
                f" | muscle {params[1]:.3f}"
                f" | proportions {params[2]:.3f}"
            )

            # Large residual steers the optimizer away; this point is
            # never chosen as the result.
            return np.full(
                len(targets),
                10.0,
                dtype=np.float64,
            )

        r = entry["residual"]

        if n % 5 == 0:
            print(
                f"  evaluation {n:>3}"
                f" | error {np.linalg.norm(r):.5f}"
                f" | age {params[0]:.3f}"
                f" | muscle {params[1]:.3f}"
                f" | proportions {params[2]:.3f}"
            )

        if np.max(np.abs(r)) <= GOOD_ENOUGH_RELATIVE_ERROR:
            raise _GoodEnough()

        return r

    print("Optimizing body shape...")

    stats["stop_reason"] = "optimizer converged"

    try:
        least_squares(
            objective,
            x0,
            bounds=(lower, upper),
            method="trf",
            diff_step=DIFF_STEP,
            xtol=1e-4,
            ftol=1e-4,
            gtol=1e-6,
            max_nfev=30,
            verbose=0,
        )
    except _GoodEnough:
        stats["stop_reason"] = "all targets within tolerance"

    # Use the best measured point instead of evaluating the
    # optimizer's final point again.
    best = state["best"]

    if best is None:
        stats["stop_reason"] = "no valid measurement found"
        best = run(x0)

    age, muscle, proportions = best["params"]
    height = best["height"]
    weight = best["weight"]

    print()
    print("Solved parameters:")

    print(
        f"  Height      : {height:.4f}"
    )

    print(
        f"  Weight      : {weight:.4f}"
    )

    print(
        f"  Age         : {age:.4f}"
    )

    print(
        f"  Muscle      : {muscle:.4f}"
    )

    print(
        f"  Proportions : {proportions:.4f}"
    )

    print(
        f"  Shape evaluations : {stats['shape_evaluations']}"
        f" | Anny evaluations : {stats['anny_evaluations']}"
        f" | {stats['stop_reason']}"
    )

    return best["output"], best["result"], np.array(
        [
            height,
            weight,
            age,
            muscle,
            proportions,
        ],
        dtype=np.float64,
    )


# ============================================================
# FASTAPI REUSABLE FUNCTION
# ============================================================

def fit_body(
    model,
    measurements,
    gender,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
    shoulder_cm=None,
    shape_model=None,
):
    """
    Reusable function for FastAPI.

    Takes the user's body measurements and returns
    the fitted Anny parameters and the generated body data,
    plus an accuracy report comparing the measured body with
    the requested values.

    Male bodies use fit_adult_male when shape_model (see there) is
    given; shoulder_cm is used for male bodies only. Female bodies
    always use fit(), unchanged.
    """

    gender = normalize_gender(gender)

    # Start measuring the fitting time.
    start_time = time.perf_counter()

    stats = {}
    body_controls = {}
    targets = {
        "height": height_cm,
        "weight": weight_kg,
        "waist": waist_cm,
        "chest": chest_cm,
        "hip": hip_cm,
    }

    if gender == "male" and shape_model is not None:
        fit_method = "adult_male"

        output, result, params, body_controls = fit_adult_male(
            shape_model=shape_model,
            measurements=measurements,
            height_cm=height_cm,
            weight_kg=weight_kg,
            waist_cm=waist_cm,
            chest_cm=chest_cm,
            hip_cm=hip_cm,
            shoulder_cm=shoulder_cm,
            stats=stats,
        )

        if shoulder_cm is not None:
            targets["shoulder"] = shoulder_cm
    else:
        fit_method = "standard"

        output, result, params = fit(
            model=model,
            measurements=measurements,
            gender=GENDER_VALUES[gender],
            height_cm=height_cm,
            weight_kg=weight_kg,
            waist_cm=waist_cm,
            chest_cm=chest_cm,
            hip_cm=hip_cm,
            stats=stats,
        )

    # Calculate total fitting time.
    elapsed = time.perf_counter() - start_time

    print()

    print(
        f"Fitting time: {elapsed:.2f} seconds"
    )

    print(
        f"Fitting time: {elapsed / 60:.2f} minutes"
    )

    accuracy = accuracy_report(
        targets,
        result,
        controls=ADULT_MALE_CONTROL if fit_method == "adult_male" else None,
        keys=MALE_TARGET_MEASUREMENTS if fit_method == "adult_male" else None,
    )

    return {
        "output": output,
        "result": result,
        "accuracy": accuracy,
        "fit_method": fit_method,
        "body_controls": body_controls,
        # Targets the adult male fit could not reach within its control ranges.
        "out_of_range": list(stats.get("out_of_range", [])),
        "fit_info": {
            "seconds": elapsed,
            **stats,
        },
        "parameters": {
            "gender": gender,
            "height": float(params[0]),
            "weight": float(params[1]),
            "age": float(params[2]),
            "muscle": float(params[3]),
            "proportions": float(params[4]),
        },
    }


# ============================================================
# ADULT MALE FIT
# ============================================================

# Anny's own calibration (anny/data/shape_calibration/boys.pth,
# morphological_age_mapping) maps age 0.415 -> 11 years, 0.67 -> 16,
# 0.77 -> 18 and 0.83 -> 64. The general fit above never goes below
# AGE_MIN = 0.5 (about 13 years), and male fits stayed at that bound,
# so male bodies had adolescent proportions (larger head share, shorter
# legs, narrow shoulders). Male bodies use a fixed adult age instead,
# about 25 years.
ADULT_MALE_AGE = 0.78

# Anny's "proportions": 0 = idealproportions, 1 = uncommonproportions.
# 0.2 needed the smallest measurement controls over the male test matrix
# (tools/male_fit_matrix.py) and was kept after visual review.
ADULT_MALE_PROPORTIONS = 0.2

# Anny's mass is mesh volume * 980 kg/m3 (anny/anthropometry.py, "for
# simplicity"). Adult male tissue is about 1045-1060 kg/m3 (15-25 % fat)
# and the closed mesh also holds residual lung air, so the mesh volume
# of a real man weighs about 1030-1045 kg/m3. With 980 the fit had to
# add ~6 % volume to reach the weight, and with chest, waist and hip
# fixed that volume went into bulky arms and legs.
ANNY_DENSITY = 980.0
MALE_BODY_DENSITY = 1040.0
# One kilogram of Anny's mass in real (density-corrected) kilograms.
KG_PER_ANNY_KG = MALE_BODY_DENSITY / ANNY_DENSITY

# Anny local changes (MPFB measurement targets, -1..1) used by the male fit.
# Each requested measurement is reached by exactly one control, and a control
# never moves to compensate another measurement: target -> (control, metres
# per unit). Slopes measured on an adult male (average of the decrease and
# increase sides); the fit refines them with secant steps.
#
# An earlier version solved all controls together (least squares). It then
# used any control for any error, e.g. Anny's pregnancy shape against the
# chest, or the hips control against the pelvis-width control, and gave
# unnatural bodies when a target was out of reach (avatar 551c0fd9...).
MALE_TARGET_CONTROLS = {
    "chest": ("measure-bust-circ-incr", 0.14),
    "waist": ("measure-waist-circ-incr", 0.10),
    "hip": ("measure-hips-circ-incr", 0.145),
    # Only when a shoulder width is requested. Also widens the upper chest;
    # the bust control then corrects the chest.
    "shoulder": ("torso-vshape-incr", 0.053),
}

# Control ranges. Anny's local changes run from -1 to 1; the bust control
# stops at -0.6 because below that the male chest looks flat and sunken
# (front and side renders, 2026-10-17). A measurement that would need more
# than its control's range is reported as out of range, not forced.
MALE_CONTROL_BOUNDS = {
    "measure-bust-circ-incr": (-0.6, 1.0),
}
DEFAULT_CONTROL_BOUNDS = (-1.0, 1.0)

# Muscle stays at Anny's average unless the weight parameter alone cannot
# reach the weight; then muscle carries the rest, within these limits. Above
# 0.7 the arms become bodybuilder-sized (165 cm / 85 kg at muscle 0.87: upper
# arm 44 cm); a weight that needs more is reported, not forced.
MALE_DEFAULT_MUSCLE = 0.5
MALE_MUSCLE_RANGE = (0.3, 0.7)
# Muscle step per kg of missing weight (first step; then secant).
MALE_MUSCLE_PER_KG = 0.02
# Stage A: at most this many muscle steps.
MALE_MUSCLE_STEPS = 8

# Stage C (weight correction after the local controls): passes, tolerance,
# damping of each weight-parameter step and the probe used for its slope.
MALE_WEIGHT_PASSES = 3
MALE_WEIGHT_PASS_TOLERANCE_KG = 1.0
MALE_WEIGHT_DAMPING = 0.7
MALE_WEIGHT_PROBE = 0.05

MALE_MAX_ITERATIONS = 20
# Circumferences and shoulder: done within 2 mm; weight within 0.2 kg.
MALE_TOLERANCE_M = 0.002
MALE_WEIGHT_TOLERANCE_KG = 0.2
# A control is "at its bound" this close to it.
AT_BOUND = 1e-3
# A measurement is reported as out of range when its control is held at a
# bound and it is still missed by more than this.
MALE_OUT_OF_RANGE_M = 0.01


def report_male_body(output, measurements, targets, out_of_range):
    """
    Measurements, accuracy and out-of-range list of the body that is actually
    exported (the fitted body with the face preset), for the adult male fit.
    targets: {"height": cm, "weight": kg, "waist": cm, "chest": cm, "hip": cm,
    optional "shoulder": cm}. out_of_range: the fit's list; "weight" is set
    again from this body.
    """

    result = dict(measurements.measure(output))
    result.update(measurements.tape_measurements(output))
    result = _density_corrected(result)

    accuracy = accuracy_report(targets, result, controls=ADULT_MALE_CONTROL, keys=MALE_TARGET_MEASUREMENTS)

    weight = accuracy.get("weight", {})
    missed = weight.get("absolute_error") is None or weight["absolute_error"] > MALE_WEIGHT_PASS_TOLERANCE_KG
    out_of_range = [name for name in out_of_range if name != "weight"] + (["weight"] if missed else [])

    return result, accuracy, out_of_range


def _density_corrected(result):
    """Anny's mass and BMI recomputed with MALE_BODY_DENSITY."""

    corrected = dict(result)
    factor = MALE_BODY_DENSITY / ANNY_DENSITY

    for key in ("mass_kg", "bmi"):
        if _finite(corrected.get(key)):
            corrected[key] = float(corrected[key]) * factor

    return corrected


def fit_adult_male(
    shape_model,
    measurements,
    height_cm,
    weight_kg,
    waist_cm,
    chest_cm,
    hip_cm,
    shoulder_cm=None,
    stats=None,
):
    """
    Fit an adult male body, in stages, so that the body stays natural:

    A. Phenotype: the adult male body (ADULT_MALE_AGE, ADULT_MALE_PROPORTIONS,
       muscle MALE_DEFAULT_MUSCLE) with Anny's height and weight parameters
       solved for the height and weight. Muscle moves only when the weight
       parameter alone cannot reach the weight.
    B. Local shape: chest, waist, hip and the optional shoulder width, each
       with its own Anny measurement control (MALE_TARGET_CONTROLS), by
       secant steps within MALE_CONTROL_BOUNDS, with the phenotype frozen.
       A control never moves for another measurement.
    C. Weight: the controls add or remove some mass. If the weight is then
       missed by more than MALE_WEIGHT_PASS_TOLERANCE_KG, the weight
       parameter is corrected (damped) and B repeated, at most
       MALE_WEIGHT_PASSES times, and only while that does not push a
       control to its limit; a remaining miss is reported.

    Why frozen: when the weight parameter was solved again after every
    control step, each step that trimmed a girth also removed mass, the
    weight parameter refilled it as body fat everywhere, and the controls
    trimmed harder: average men ended at Anny's maximum body fat with
    every control near its limit.

    Measurements a control cannot reach within its range, and a weight
    missed by more than MALE_WEIGHT_PASS_TOLERANCE_KG, end up in
    stats["out_of_range"]. Chest and hip are tape girths
    (measurements.tape_measurements).

    shape_model(controls) must return an Anny model callable with those
    local changes (every other local change at 0); measurements is
    a BodyMeasurements for the same topology.

    Returns (output, result, params, controls); result has mass_kg and
    bmi for MALE_BODY_DENSITY.
    """

    if stats is None:
        stats = {}

    stats.setdefault("anny_evaluations", 0)
    stats.setdefault("shape_evaluations", 0)

    target_height = height_cm / 100.0
    # Anny's mass * MALE_BODY_DENSITY / ANNY_DENSITY must equal the weight.
    anny_weight = weight_kg * ANNY_DENSITY / MALE_BODY_DENSITY

    requested = {
        "chest": chest_cm / 100.0,
        "waist": waist_cm / 100.0,
        "hip": hip_cm / 100.0,
    }

    if shoulder_cm is not None:
        requested["shoulder"] = shoulder_cm / 100.0

    result_keys = {name: key for name, (key, _, _) in MALE_TARGET_MEASUREMENTS.items()}

    names = list(requested)
    labels = {name: MALE_TARGET_CONTROLS[name][0] for name in names}
    bounds = {name: MALE_CONTROL_BOUNDS.get(labels[name], DEFAULT_CONTROL_BOUNDS) for name in names}

    # ---------------- A. phenotype for height and weight

    def phenotype_body(muscle, start):
        stats["shape_evaluations"] += 1
        return evaluate_shape(
            shape_model({}),
            measurements,
            GENDER_VALUES["male"],
            target_height,
            anny_weight,
            ADULT_MALE_AGE,
            muscle,
            ADULT_MALE_PROPORTIONS,
            start=start,
            stats=stats,
        )

    muscle = MALE_DEFAULT_MUSCLE
    _, result, height_param, weight_param = phenotype_body(muscle, (0.5, 0.5))
    previous = None

    for _ in range(MALE_MUSCLE_STEPS):
        mass_error = float(result["mass_kg"]) - anny_weight

        if abs(mass_error) <= MALE_WEIGHT_TOLERANCE_KG or not (
            (mass_error < 0 and weight_param >= PARAM_MAX - AT_BOUND)
            or (mass_error > 0 and weight_param <= PARAM_MIN + AT_BOUND)
        ):
            break

        per_kg = MALE_MUSCLE_PER_KG

        if previous is not None and abs(muscle - previous[0]) > 1e-4:
            change = (mass_error - previous[1]) / (muscle - previous[0])
            if change > 0:
                per_kg = 1.0 / change

        new_muscle = float(np.clip(muscle - mass_error * per_kg, *MALE_MUSCLE_RANGE))

        if new_muscle == muscle:
            break

        previous = (muscle, mass_error)
        muscle = new_muscle
        _, result, height_param, weight_param = phenotype_body(muscle, (height_param, weight_param))

    stats["phenotype"] = {"muscle": muscle, "weight": weight_param, "height": height_param}

    # ---------------- B. local shape with the phenotype frozen

    def body(controls, weight, body_muscle=None):
        body_muscle = muscle if body_muscle is None else body_muscle
        stats["shape_evaluations"] += 1
        stats["anny_evaluations"] += 1

        output, result = evaluate(
            shape_model(dict(controls)),
            measurements,
            GENDER_VALUES["male"],
            ADULT_MALE_AGE,
            height_param,
            weight,
            body_muscle,
            ADULT_MALE_PROPORTIONS,
        )

        # Chest and hip as a tape measures them (measurements.tape_measurements).
        result = dict(result)
        result.update(measurements.tape_measurements(output))

        errors = {}

        for name in names:
            value = result.get(result_keys[name])
            errors[name] = None if not _finite(value) else float(value) - requested[name]

        mass = result.get("mass_kg")

        return {
            "controls": dict(controls),
            "weight": weight,
            "output": output,
            "result": result,
            "errors": errors,
            "mass_error": None if not _finite(mass) else float(mass) - anny_weight,
        }

    def blocked(name, entry):
        """The control is at the bound that stops it from fixing this error."""
        low, high = bounds[name]
        value = entry["controls"][labels[name]]
        error = entry["errors"][name]
        return error is not None and (
            (error > 0 and value <= low + AT_BOUND) or (error < 0 and value >= high - AT_BOUND)
        )

    def score(entry):
        # Errors of reachable targets only: a target whose control is held
        # at the blocking bound cannot get better, so it must not decide
        # which body is kept.
        values = [
            float("inf") if e is None else abs(e)
            for name, e in entry["errors"].items()
            if not blocked(name, entry)
        ]
        return max(values) if values else 0.0

    def fit_controls(weight, start_controls):
        slopes = {name: MALE_TARGET_CONTROLS[name][1] for name in names}
        current = body(start_controls, weight)
        best = current
        previous = None

        for _ in range(MALE_MAX_ITERATIONS):
            if any(e is None for e in current["errors"].values()):
                stats["stop_reason"] = "a measurement failed"
                break

            if all(abs(e) <= MALE_TOLERANCE_M or blocked(name, current) for name, e in current["errors"].items()):
                stats["stop_reason"] = "all reachable targets within tolerance"
                break

            new_controls = dict(current["controls"])

            for name in names:
                label = labels[name]

                # Secant slope for this control alone, from the last two bodies.
                if previous is not None:
                    step = current["controls"][label] - previous["controls"][label]
                    if abs(step) > 1e-3:
                        measured = (current["errors"][name] - previous["errors"][name]) / step
                        nominal = MALE_TARGET_CONTROLS[name][1]
                        slopes[name] = float(np.clip(measured, 0.3 * nominal, 3.0 * nominal))

                low, high = bounds[name]
                new_controls[label] = float(np.clip(
                    current["controls"][label] - current["errors"][name] / slopes[name],
                    low,
                    high,
                ))

            if new_controls == current["controls"]:
                stats["stop_reason"] = "no control can move further"
                break

            previous = current
            current = body(new_controls, weight)

            if score(current) <= score(best):
                best = current
        else:
            stats["stop_reason"] = "iteration limit"

        return best

    best = fit_controls(weight_param, {labels[name]: 0.0 for name in names})
    stats["weight_passes"] = 0

    # ---------------- C. weight correction (damped), only while it stays natural

    def weight_missed(entry):
        return entry["mass_error"] is None or abs(entry["mass_error"]) * KG_PER_ANNY_KG > MALE_WEIGHT_PASS_TOLERANCE_KG

    # Muscle carries the correction when the weight parameter is at its limit
    # in the needed direction, or once a weight-parameter step was rejected
    # (e.g. a heavy man whose waist control would leave its range: less
    # muscle, not less belly).
    use_muscle = False

    for _ in range(MALE_WEIGHT_PASSES + 1):
        if not weight_missed(best):
            break

        mass_error = best["mass_error"]

        weight_blocked = use_muscle or (mass_error < 0 and best["weight"] >= PARAM_MAX - AT_BOUND) or (
            mass_error > 0 and best["weight"] <= PARAM_MIN + AT_BOUND
        )
        name, value, low, high = (
            ("muscle", muscle, *MALE_MUSCLE_RANGE) if weight_blocked
            else ("weight", best["weight"], PARAM_MIN, PARAM_MAX)
        )

        def phenotype_with(new_value):
            return (new_value, muscle) if name == "weight" else (best["weight"], new_value)

        probe_value = float(np.clip(value + MALE_WEIGHT_PROBE, low, high))

        if probe_value == value:
            probe_value = float(np.clip(value - MALE_WEIGHT_PROBE, low, high))

        if probe_value == value:
            break

        probe = body(best["controls"], *phenotype_with(probe_value))
        slope = (probe["mass_error"] - mass_error) / (probe_value - value)

        if slope <= 0:
            break

        new_value = float(np.clip(value - MALE_WEIGHT_DAMPING * mass_error / slope, low, high))

        if abs(new_value - value) < 1e-4:
            break

        new_weight, new_muscle = phenotype_with(new_value)
        saved = (height_param, muscle)
        muscle = new_muscle
        # The new phenotype changes the height a little: solve it again.
        height_param = solve_height(
            shape_model(best["controls"]), measurements, GENDER_VALUES["male"], ADULT_MALE_AGE, new_weight,
            muscle, ADULT_MALE_PROPORTIONS, target_height, start=height_param,
        )
        candidate = fit_controls(new_weight, best["controls"])
        stats["weight_passes"] += 1

        newly_blocked = [
            name for name in names if blocked(name, candidate) and not blocked(name, best)
        ]

        if newly_blocked or score(candidate) > max(score(best), MALE_TOLERANCE_M):
            stats["weight_stopped_by"] = newly_blocked or "a measurement got worse"
            height_param, muscle = saved

            if name == "weight" and not use_muscle:
                use_muscle = True
                continue

            break

        best = candidate

    stats["out_of_range"] = [
        name
        for name in names
        if blocked(name, best) and abs(best["errors"][name]) > MALE_OUT_OF_RANGE_M
    ]

    if weight_missed(best):
        stats["out_of_range"].append("weight")

    fitted_controls = {label: float(value) for label, value in best["controls"].items()}

    params = np.array(
        [
            height_param,
            best["weight"],
            ADULT_MALE_AGE,
            muscle,
            ADULT_MALE_PROPORTIONS,
        ],
        dtype=np.float64,
    )

    print(
        f"  Adult male fit: {stats['shape_evaluations']} evaluations"
        f" | {stats.get('stop_reason')} | weight passes {stats['weight_passes']}"
        f" | out of range {stats['out_of_range']}"
        f" | muscle {muscle:.3f} | weight parameter {best['weight']:.3f}"
        f" | controls { {k: round(v, 3) for k, v in fitted_controls.items()} }"
    )

    return (
        best["output"],
        _density_corrected(best["result"]),
        params,
        fitted_controls,
    )


# ============================================================
# COMMAND LINE
# ============================================================

def print_accuracy(accuracy):
    print(
        f"{'name':<7} {'requested':>10} {'actual':>10}"
        f" {'abs_err':>9} {'pct_err':>8}  status"
    )

    for name, entry in accuracy.items():
        if entry["actual"] is None:
            print(
                f"{name:<7} {entry['requested']:>10.2f}"
                f" {'-':>10} {'-':>9} {'-':>8}  {entry['status']}"
            )
            continue

        print(
            f"{name:<7} {entry['requested']:>10.2f}"
            f" {entry['actual']:>10.2f}"
            f" {entry['absolute_error']:>9.2f}"
            f" {entry['percentage_error']:>7.2f}%"
            f"  {entry['unit']}"
        )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gender",
        choices=["male", "female"],
        default="female",
    )

    parser.add_argument(
        "--height",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--weight",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--waist",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--chest",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--hip",
        type=float,
        required=True,
    )

    args = parser.parse_args()

    print("Loading Anny...")

    model = anny.Anny()

    from measurements import BodyMeasurements

    measurements = BodyMeasurements(model)

    print()
    print("FITTING BODY SHAPE")
    print("------------------")
    print()

    print("Target:")

    print(
        f"  Height : {args.height:.2f} cm"
    )

    print(
        f"  Weight : {args.weight:.2f} kg"
    )

    print(
        f"  Waist  : {args.waist:.2f} cm"
    )

    print(
        f"  Chest  : {args.chest:.2f} cm"
    )

    print(
        f"  Hip    : {args.hip:.2f} cm"
    )

    print()

    fitted = fit_body(
        model=model,
        measurements=measurements,
        gender=args.gender,
        height_cm=args.height,
        weight_kg=args.weight,
        waist_cm=args.waist,
        chest_cm=args.chest,
        hip_cm=args.hip,
    )

    result = fitted["result"]
    params = fitted["parameters"]

    print()
    print("ACCURACY")
    print("--------")

    print_accuracy(fitted["accuracy"])

    print()
    print("RESULT")
    print("------")

    clean = {
        key: (
            round(float(value), 4)
            if value is not None
            else None
        )
        for key, value in result.items()
        if key != "hip_z"
    }

    print(
        json.dumps(
            clean,
            indent=2,
        )
    )

    print()
    print("PARAMETERS")
    print("----------")

    print(
        json.dumps(
            {
                "height": round(
                    params["height"],
                    4,
                ),
                "weight": round(
                    params["weight"],
                    4,
                ),
                "age": round(
                    params["age"],
                    4,
                ),
                "muscle": round(
                    params["muscle"],
                    4,
                ),
                "proportions": round(
                    params["proportions"],
                    4,
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
