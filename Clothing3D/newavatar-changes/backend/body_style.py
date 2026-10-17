"""
Body fitting for the avatar: one controlled fit, two body styles.

Both styles run on the face model (appearance.FaceModel: Anny with its local
changes and cup size; with every control at 0 its body is identical to the
plain model's) and solve height and weight exactly, like
fit_to_measurements.fit_body, which stays as the earlier production fit.

"standard" (all users) fixes two problems of the earlier fit:
  - adult bodies: age is kept between 0.67 (Anny's "young adult" anchor) and
    0.85; the earlier fit allowed 0.5, half-way to "child", and fitted
    0.50-0.66, so bodies looked teenage;
  - circumferences: Anny's bust, waist and hip circumference controls
    (within +-0.6, as in Baraah's AC67 experiment) fit the three
    circumferences on top of age, muscle and proportions, so they no longer
    pull against each other;
  - women's muscle is capped at 0.8: unbounded, the fit pushed light female
    bodies to 0.94-0.99, which looks male.

"athletic" (women only): age 0.67-0.8, muscle 0.55-0.85, a fixed athletic
shape (narrower shoulders, slimmer hips, thighs and calves, firm buttocks,
flat stomach, natural bust: cup size 0.62, firmness 0.65) with the waist and
hip controls fitted on top. The bust is part of the shape, not fitted, so the
chest can come out larger than requested; the accuracy report shows it.
Tuned visually with tools/athletic_female_preview.py.
"""

import time
from dataclasses import dataclass, field

import numpy as np

import fit_to_measurements as ftm

BODY_STYLES = ("standard", "athletic")
DEFAULT_BODY_STYLE = "standard"

CONTROL_OFFSET = 1.0  # the optimiser works on control + offset (keeps x > 0)

# Beyond this weight miss the athletic body is not used (see api.py).
ATHLETIC_MAX_WEIGHT_ERROR_KG = 1.5


@dataclass(frozen=True)
class FitStyle:
    age: tuple
    muscle: tuple
    fit_controls: tuple
    control_limit: float
    shape: dict = field(default_factory=dict)
    phenotype: dict = field(default_factory=dict)
    proportions: float = None  # fixed value, or None to fit it
    good_enough: float = 0.005  # stop when every target is within this relative error
    max_evaluations: int = 30
    chest_fitted: bool = True


STANDARD = {
    gender: FitStyle(
        age=(0.67, 0.85),
        muscle=(ftm.PARAM_MIN, 0.8 if gender == "female" else ftm.PARAM_MAX),
        fit_controls=("measure-bust-circ-incr", "measure-waist-circ-incr", "measure-hips-circ-incr"),
        control_limit=0.6,
        # 1 % is about 0.7-1 cm on a waist or hip; with the circumference
        # controls the fit gets there in far fewer evaluations than 0.2 %.
        good_enough=0.01,
        max_evaluations=20,
    )
    for gender in ftm.GENDER_VALUES
}

ATHLETIC_SHAPE = {
    "hip-scale-horiz-incr": -0.25,
    "buttocks-volume-incr": 0.15,
    "measure-thigh-circ-incr": -0.45,
    "measure-calf-circ-incr": -0.2,
    "measure-shoulder-dist-incr": -0.35,
    "stomach-pregnant-incr": -0.3,
}
ATHLETIC_PHENOTYPE = {"cupsize": 0.62, "firmness": 0.65}
ATHLETIC = FitStyle(
    age=(0.67, 0.8),
    # Anny cannot reach an average adult weight at low muscle (a 1.60 m woman
    # tops out near 57 kg at muscle 0.6), so heavier bodies may use more muscle.
    muscle=(0.55, 0.85),
    fit_controls=("measure-waist-circ-incr", "measure-hips-circ-incr"),
    control_limit=1.0,
    shape=ATHLETIC_SHAPE,
    phenotype=ATHLETIC_PHENOTYPE,
    proportions=0.5,
    good_enough=0.01,
    max_evaluations=20,
    chest_fitted=False,
)


def normalize_body_style(value):
    """Return a known body style (case-insensitive); raise ValueError otherwise."""
    if isinstance(value, str) and value.strip().lower() in BODY_STYLES:
        return value.strip().lower()
    raise ValueError(f"Unknown body style {value!r}; expected one of {list(BODY_STYLES)}.")


class StyledModel:
    """Anny model proxy that adds fixed local changes and phenotype values to
    every call (the controls go on top of any already set)."""

    def __init__(self, model, controls=None, phenotypes=None):
        self._model = model
        self.controls = dict(controls or {})
        self.phenotypes = dict(phenotypes or {})

    def __call__(self, phenotype_kwargs=None, **kwargs):
        import torch

        phenotype_kwargs = dict(phenotype_kwargs or {})
        for label, value in self.phenotypes.items():
            phenotype_kwargs[label] = torch.full((1,), float(value), dtype=ftm.DT)
        changes = {**kwargs.pop("local_changes_kwargs", {}), **self.controls}
        return self._model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=changes, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class _GoodEnough(Exception):
    pass


class FitError(RuntimeError):
    """The body could not be fitted or measured; the message says why."""


def fit_style(style, model, measurements, gender, height_cm, weight_kg, waist_cm, chest_cm, hip_cm):
    """
    Fit a body of the given FitStyle. model: an Anny model with the style's
    local-change labels and phenotypes="all" (FaceModel.model). Returns the
    keys of fit_to_measurements.fit_body, plus "body_controls" (the local
    changes the body was built with) and "body_phenotype".
    """
    from scipy.optimize import least_squares

    gender = ftm.normalize_gender(gender)
    start_time = time.perf_counter()
    proxy = StyledModel(model, style.shape, style.phenotype)
    targets = np.array([waist_cm / 100, chest_cm / 100, hip_cm / 100, height_cm / 100, weight_kg])
    fit_proportions = style.proportions is None
    n = len(style.fit_controls)

    # Parameters: age, muscle, [proportions], then the controls (+ offset).
    lower = [style.age[0], style.muscle[0]] + ([ftm.PARAM_MIN] if fit_proportions else [])
    upper = [style.age[1], style.muscle[1]] + ([ftm.PARAM_MAX] if fit_proportions else [])
    x0 = [style.age[0] + 0.03, float(np.clip(0.5, *style.muscle))] + ([0.5] if fit_proportions else [])
    first = len(x0)
    lower = np.array(lower + [CONTROL_OFFSET - style.control_limit] * n)
    upper = np.array(upper + [CONTROL_OFFSET + style.control_limit] * n)
    x0 = np.clip(np.array(x0 + [CONTROL_OFFSET] * n), lower, upper)

    stats = {"anny_evaluations": 0, "shape_evaluations": 0}
    state = {"start": (0.5, 0.5), "best": None}

    def run(params):
        stats["shape_evaluations"] += 1
        fitted = {c: float(v) - CONTROL_OFFSET for c, v in zip(style.fit_controls, params[first:])}
        proxy.controls = {**style.shape, **fitted}
        proportions = params[2] if fit_proportions else style.proportions
        output, result, h, w = ftm.evaluate_shape(proxy, measurements, ftm.GENDER_VALUES[gender],
                                                  targets[3], targets[4], params[0], params[1], proportions,
                                                  start=state["start"], stats=stats)
        state["start"] = (h, w)
        values = [result.get(k) for k in ("waist_m", "chest_m", "hip_m", "height_m", "mass_kg")]
        # A chest that is not fitted (athletic bust) only drops out of the error.
        if not style.chest_fitted and not ftm._finite(values[1]):
            values[1] = targets[1]
        if not all(ftm._finite(v) for v in values):
            return None
        residual = (np.array(values, float) - targets) / targets
        if not style.chest_fitted:
            residual[1] = 0.0
        best = state["best"]
        if best is None or np.linalg.norm(residual) < np.linalg.norm(best["residual"]):
            state["best"] = dict(params=np.array(params), proportions=float(proportions),
                                 controls=dict(proxy.controls), output=output, result=result,
                                 height=h, weight=w, residual=residual)
        return residual

    def objective(params):
        r = run(params)
        if r is None:
            return np.full(len(targets), 10.0)
        if np.max(np.abs(r)) <= style.good_enough:
            raise _GoodEnough()
        return r

    stats["stop_reason"] = "optimizer converged"
    try:
        least_squares(objective, x0, bounds=(lower, upper), method="trf", diff_step=ftm.DIFF_STEP,
                      xtol=1e-4, ftol=1e-4, gtol=1e-6, max_nfev=style.max_evaluations)
    except _GoodEnough:
        stats["stop_reason"] = "all targets within tolerance"
    best = state["best"]
    if best is None:
        raise FitError("The body could not be measured for these inputs.")

    accuracy = ftm.accuracy_report({"height": height_cm, "weight": weight_kg, "waist": waist_cm,
                                    "chest": chest_cm, "hip": hip_cm}, best["result"])
    controls = {"waist": "measure-waist-circ-incr", "chest": "measure-bust-circ-incr", "hip": "measure-hips-circ-incr"}
    for name, label in controls.items():
        if label in style.fit_controls:
            accuracy[name]["control"] = "fitted (age, muscle, proportions and its circumference control)"
        else:
            accuracy[name]["control"] = "set by the body style's shape (not fitted)"
    return {
        "output": best["output"],
        "result": best["result"],
        "accuracy": accuracy,
        "fit_info": {"seconds": time.perf_counter() - start_time, **stats},
        "parameters": {"gender": gender, "height": float(best["height"]), "weight": float(best["weight"]),
                       "age": float(best["params"][0]), "muscle": float(best["params"][1]),
                       "proportions": best["proportions"]},
        "body_controls": best["controls"],
        "body_phenotype": dict(style.phenotype),
    }


def fit_body(model, measurements, gender, height_cm, weight_kg, waist_cm, chest_cm, hip_cm):
    """The standard adult body (same signature as fit_to_measurements.fit_body)."""
    return fit_style(STANDARD[ftm.normalize_gender(gender)], model, measurements, gender,
                     height_cm, weight_kg, waist_cm, chest_cm, hip_cm)


def fit_athletic(model, measurements, height_cm, weight_kg, waist_cm, chest_cm, hip_cm):
    """The athletic female body."""
    return fit_style(ATHLETIC, model, measurements, "female", height_cm, weight_kg, waist_cm, chest_cm, hip_cm)
