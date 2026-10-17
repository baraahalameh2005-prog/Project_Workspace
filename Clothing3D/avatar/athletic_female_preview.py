"""
Preview: an adult, athletic female avatar fitted to measurements.

Diagnostic tool; does not change the production fit, measurements, export or
API. Fits one female input three ways and exports each to GLB:

  production   the frozen production fit (age >= 0.5, no local controls)
  adult        Baraah's AC67 adult fit (age >= 0.67, bust/waist/hips
               circumference controls within +-0.6)
  athletic     AC67 plus a muscle floor and a fixed athletic-shape preset
               (narrower waist line, rounder hips and buttocks, slight
               V-shaped torso); the circumference controls still fit the
               measurements around the preset.

Run from the backend folder:
    ..\\.venv\\Scripts\\python.exe -B tools\\athletic_female_preview.py

Writes only to var/athletic_preview/<timestamp>/.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS.parent))

import fit_to_measurements as ftm  # applies the module cache settings first
from app import config

import anny
import torch
from scipy.optimize import least_squares

from generate_avatar import export_avatar
from measurements import BodyMeasurements

REF = json.loads((TOOLS / "stage5_ac67_reference.json").read_text())
FIT_CONTROLS = REF["controls"]  # bust, waist, hips circumference
CONTROL_LIMIT = 1.0  # wider than AC67 (0.6): the preset adds hip that the fit must be able to take back
CONTROL_OFFSET = REF["control_offset"]
ADULT_AGE_MIN = REF["age_min"]  # 0.67

# Fixed athletic shape, applied on top of the fit (Anny local-change units).
# (A V-shaped torso and wide shoulders read as male, so the shoulders are
# narrowed instead.)
ATHLETIC_PRESET = {
    # Slim, toned hips and legs.
    "hip-scale-horiz-incr": -0.25,
    "buttocks-volume-incr": 0.15,
    "measure-thigh-circ-incr": -0.45,
    "measure-calf-circ-incr": -0.2,
    "measure-shoulder-dist-incr": -0.35,
    "stomach-pregnant-incr": -0.3,
}
ATHLETIC_PRESET_BUST = {"measure-bust-circ-incr": 0.0}
# Toned, not bulky: unbounded, the fit pushes muscle to its maximum (0.94-0.99)
# to keep a light body's circumferences small, and that looks male.
ATHLETIC_MUSCLE = (0.55, 0.72)
# Young adult: Anny's 'young' anchor is 2/3 and 'old' is 1, so the fit may not age her past 0.8.
ATHLETIC_AGE_MAX = 0.8
# Fuller, firm bust: Anny's cup size and firmness phenotypes (0..1, default
# 0.5). Anny hides them unless built with phenotypes="all".
ATHLETIC_BREASTS = {"cupsize": 0.62, "firmness": 0.65}
# A big bust is bigger than a slim chest measurement allows, so for the
# athletic body the bust is set (ATHLETIC_PRESET_BUST), not fitted: only waist and hips are.

# A slim, athletic adult woman (cm, kg).
INPUT = {"gender": "female", "height_cm": 170.0, "weight_kg": 58.0, "chest_cm": 86.0, "waist_cm": 64.0, "hip_cm": 92.0}

ONLY = set(sys.argv[1:])  # e.g. "athletic": run only those variants

OUT = config.MODULE_ROOT / "var" / "athletic_preview" / time.strftime("%Y%m%d_%H%M%S")


class ControlledModel:
    """Passes fixed + fitted local changes to Anny on every call."""

    def __init__(self, model, fixed, phenotypes=None):
        self._model = model
        self.fixed = dict(fixed)
        self.phenotypes = dict(phenotypes or {})
        self.controls = {}

    def __call__(self, phenotype_kwargs=None, **kwargs):
        changes = {**self.fixed, **self.controls}
        phenotype_kwargs = dict(phenotype_kwargs or {})
        for label, value in self.phenotypes.items():
            phenotype_kwargs[label] = torch.full((1,), value, dtype=ftm.DT)
        return self._model(phenotype_kwargs=phenotype_kwargs, local_changes_kwargs=changes, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class _GoodEnough(Exception):
    pass


def controlled_fit(model, measurements, inp, age_min, muscle_range, fixed, phenotypes=None, age_max=ftm.AGE_MAX):
    fit_controls = [c for c in FIT_CONTROLS if c not in fixed]
    """AC67-style fit: age, muscle, proportions + the 3 circumference controls."""
    start = time.perf_counter()
    proxy = ControlledModel(model, fixed, phenotypes)
    targets = np.array([inp["waist_cm"] / 100, inp["chest_cm"] / 100, inp["hip_cm"] / 100,
                        inp["height_cm"] / 100, inp["weight_kg"]])
    n = len(fit_controls)
    muscle_min, muscle_max = muscle_range
    x0 = np.array([max(age_min, 0.7), float(np.clip(0.5, muscle_min, muscle_max)), 0.5] + [CONTROL_OFFSET] * n)
    lower = np.array([age_min, muscle_min, ftm.PARAM_MIN] + [CONTROL_OFFSET - CONTROL_LIMIT] * n)
    upper = np.array([age_max, muscle_max, ftm.PARAM_MAX] + [CONTROL_OFFSET + CONTROL_LIMIT] * n)
    stats = {"anny_evaluations": 0, "shape_evaluations": 0}
    state = {"start": (0.5, 0.5), "best": None}

    def run(params):
        stats["shape_evaluations"] += 1
        proxy.controls = {c: float(v) - CONTROL_OFFSET for c, v in zip(fit_controls, params[3:])}
        output, result, h, w = ftm.evaluate_shape(proxy, measurements, ftm.GENDER_VALUES[inp["gender"]],
                                                  targets[3], targets[4], params[0], params[1], params[2],
                                                  start=state["start"], stats=stats)
        state["start"] = (h, w)
        vals = [result.get(k) for k in ("waist_m", "chest_m", "hip_m", "height_m", "mass_kg")]
        if not all(ftm._finite(v) for v in vals):
            return None
        residual = (np.array(vals, float) - targets) / targets
        if state["best"] is None or np.linalg.norm(residual) < np.linalg.norm(state["best"]["residual"]):
            state["best"] = dict(params=np.array(params), controls=dict(proxy.controls), output=output,
                                 result=result, residual=residual)
        return residual

    def objective(params):
        r = run(params)
        if r is None:
            return np.full(len(targets), 10.0)
        if np.max(np.abs(r)) <= ftm.GOOD_ENOUGH_RELATIVE_ERROR:
            raise _GoodEnough()
        return r

    try:
        least_squares(objective, x0, bounds=(lower, upper), method="trf", diff_step=ftm.DIFF_STEP,
                      xtol=1e-4, ftol=1e-4, gtol=1e-6, max_nfev=30)
    except _GoodEnough:
        pass
    best = state["best"]
    best["seconds"] = time.perf_counter() - start
    return best


def report(name, inp, best, seconds):
    acc = ftm.accuracy_report({"height": inp["height_cm"], "weight": inp["weight_kg"], "waist": inp["waist_cm"],
                               "chest": inp["chest_cm"], "hip": inp["hip_cm"]}, best["result"])
    errors = {k: round(v["actual"] - v["requested"], 2) for k, v in acc.items() if v["actual"] is not None}
    params = best.get("params")
    line = {"name": name, "seconds": round(seconds, 1), "errors_cm_or_kg": errors}
    if params is not None:
        line.update(age=round(float(params[0]), 3), muscle=round(float(params[1]), 3),
                    proportions=round(float(params[2]), 3),
                    controls={k.split("-")[1]: round(v, 3) for k, v in best["controls"].items()})
    print(json.dumps(line), flush=True)
    return line


def main():
    OUT.mkdir(parents=True)
    labels = FIT_CONTROLS + list(ATHLETIC_PRESET)
    t = time.perf_counter()
    plain = anny.Anny()
    model = anny.Anny(local_changes=labels, phenotypes="all")
    print(f"models loaded in {time.perf_counter() - t:.1f} s", flush=True)
    m_plain = BodyMeasurements(plain)
    m_ctrl = BodyMeasurements(model)
    results = []

    t = time.perf_counter()
    fitted = None if ONLY and "production" not in ONLY else ftm.fit_body(plain, m_plain, INPUT["gender"], INPUT["height_cm"], INPUT["weight_kg"],
                          INPUT["waist_cm"], INPUT["chest_cm"], INPUT["hip_cm"])
    if fitted:
        p = fitted["parameters"]
        best = {"result": fitted["result"], "output": fitted["output"],
                "params": np.array([p["age"], p["muscle"], p["proportions"]]), "controls": {}}
        results.append(report("production", INPUT, best, time.perf_counter() - t))
        export_avatar(fitted["output"], plain, OUT / "production.glb")

    for name, age_min, muscle_range, fixed, breasts in (
            ("adult", ADULT_AGE_MIN, (ftm.PARAM_MIN, ftm.PARAM_MAX), {}, {}),
            ("athletic", ADULT_AGE_MIN, ATHLETIC_MUSCLE, {**ATHLETIC_PRESET, **ATHLETIC_PRESET_BUST}, ATHLETIC_BREASTS)):
        if ONLY and name not in ONLY:
            continue
        best = controlled_fit(model, m_ctrl, INPUT, age_min, muscle_range, fixed, breasts,
                              ATHLETIC_AGE_MAX if name == "athletic" else ftm.AGE_MAX)
        results.append(report(name, INPUT, best, best["seconds"]))
        export_avatar(best["output"], model, OUT / f"{name}.glb")

    (OUT / "results.json").write_text(json.dumps({"input": INPUT, "preset": ATHLETIC_PRESET,
                                                  "muscle_range": ATHLETIC_MUSCLE, "results": results}, indent=2))
    print("OUT", OUT, flush=True)


if __name__ == "__main__":
    main()
