"""Adapters from teammates' data formats to fit_engine's body document.

Avatar module (Baraah, POST /generate-avatar on :8002):

  * the REQUEST body:   {"height": cm, "weight": kg, "waist": cm, "chest": cm, "hip": cm,
                         "shoulder"?: cm, "gender"?, ...}
  * the RESPONSE body:  {"avatar_id", "accuracy": {name: {"requested", "actual", ...}},
                         "measurements": {"chest_m", ...}, ...}

Decision (2026-10-17): sizing uses the measurements the USER ENTERED (the request,
or accuracy[*].requested in the response), not the ones measured on the generated
mesh, because the avatar can drift from the request (e.g. +15 % waist when the
requested weight is out of the model's reach).

Field mapping, all in cm:

  avatar           fit_engine        why
  chest            chest             chest circumference (tape at nipple height)
  waist            waist             waist circumference
  hip              hip               hip circumference
  height           height            plausibility checks only
  shoulder         shoulder_width    tip to tip straight across the back (male only
                                     in the avatar API; same definition as ours)
  weight           (ignored)         fit_engine never sizes by weight

Not mapped on purpose: measurements.shoulder_width_m, arm_length_m and
leg_length_m are joint-centre distances, not tape measurements (the avatar
handoff says shoulder_width_m is narrower than a tape-measured shoulder), and
the other measurements.* values are mesh measurements, not user input.
"""

AVATAR_TO_BODY = {
    "chest": "chest",
    "waist": "waist",
    "hip": "hip",
    "height": "height",
    "shoulder": "shoulder_width",
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def avatar_to_body(avatar):
    """Build a fit_engine body document from an avatar request or response.

    Returns (body, notes). `body` is always a dict that recommend_size()
    accepts; if nothing usable was found its measurements are empty and the
    engine answers invalid_input / no_body_measurements. `notes` lists, in
    English, what was taken from where (for logs and the UI).
    """
    notes = []
    if not isinstance(avatar, dict):
        return {"unit": "cm", "source": "avatar", "measurements": {}}, ["avatar data is not a JSON object"]

    measurements = {}
    accuracy = avatar.get("accuracy")
    if isinstance(accuracy, dict):
        # generation response: use the values the user asked for
        for key, target in AVATAR_TO_BODY.items():
            entry = accuracy.get(key)
            value = _number(entry.get("requested")) if isinstance(entry, dict) else None
            if value is not None:
                measurements[target] = value
        notes.append("response: used accuracy[*].requested (user input), not generated mesh measurements")
    else:
        # generation request
        for key, target in AVATAR_TO_BODY.items():
            value = _number(avatar.get(key))
            if value is not None:
                measurements[target] = value
        notes.append("request: used the measurements the user entered")

    if "weight" in avatar or (isinstance(accuracy, dict) and "weight" in accuracy):
        notes.append("weight ignored: sizes are chosen from body measurements only")

    body = {"unit": "cm", "source": "avatar", "measurements": measurements}
    if isinstance(avatar.get("avatar_id"), str):
        body["user_id"] = avatar["avatar_id"]
    return body, notes
