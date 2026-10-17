# Avatar API — frontend handoff

Backend: FastAPI app in `backend/api.py`. Base URL in development: `http://127.0.0.1:8002`.

Copied from the production backend (`D:\BarahaAvatar`). Differences in this module are
marked **Module:** below.

## Run

Easiest: double-click `start_avatar.bat` in the repository root. It starts the backend and the
frontend in their own windows, waits until the backend is ready and opens `http://127.0.0.1:5173/`.
Close the two windows to stop them.

By hand, from `backend\`:

```powershell
..\.venv\Scripts\python.exe -B api.py
```

The server loads one Anny model (the face model, with local changes and cup size) at startup:
about 1.3 GB of memory; leave about 2 GB free. The first start on a new machine builds Anny's cache
(several minutes); later starts take seconds.

Environment: `pip install -r requirements.txt -c constraints-production.txt --extra-index-url
https://download.pytorch.org/whl/cpu` with Python 3.12, or `-c constraints-py311.txt` with Python 3.11.

Host and port come from `backend/app/config.py` (127.0.0.1:8002; override with
`AVATAR_HOST` / `AVATAR_PORT`).

Generation requests are processed one at a time; a request takes roughly the fitting time:
about 3-15 s for most bodies, up to about 60 s for bodies the model cannot match exactly
(measured 2026-10-16 on the six baseline cases).

**Module:** generated GLBs are written to `var/generated/`; the Anny, Warp and PyTorch
caches to `var/cache/anny`, `var/cache/warp` and `var/cache/torchinductor`. Each can be
moved with `AVATAR_OUTPUT_DIR`, `ANNY_CACHE_DIR`, `WARP_CACHE_PATH` or
`TORCHINDUCTOR_CACHE_DIR`, but never into the protected projects.

## Tests

From `backend\`:

```powershell
$env:AVATAR_SKIP_MODEL_TESTS = "1"
..\.venv\Scripts\python.exe -B -m unittest tests.test_config tests.test_avatar tests.test_appearance tests.test_body_style
```

`tools\fit_check.py <label> [old]` fits the six baseline inputs and writes the errors, parameters
and times to `var/fit_check/` (with `old`, also the earlier fit, for comparison).

`AVATAR_SKIP_MODEL_TESTS=1` skips the tests that load the real model.
The tests never write `var/generated/` or a shared `generated_avatar.glb`.

## Endpoints

### `POST /generate-avatar`

Request (JSON):

| Field | Type | Required | Range / format |
|---|---|---|---|
| `height` | number | yes | 100-230 cm |
| `weight` | number | yes | 25-250 kg |
| `waist` | number | yes | 40-200 cm |
| `chest` | number | yes | 50-200 cm |
| `hip` | number | yes | 50-200 cm |
| `gender` | string | no (default `"female"`) | `"female"` or `"male"`, case-insensitive |
| `skin_tone` | string | no (default `"#c8997e"`) | `"#RRGGBB"` (sRGB) |
| `eye_color` | string | no (default `"brown"`) | `"brown"`, `"hazel"`, `"green"`, `"blue"`, `"gray"`, case-insensitive |
| `body_style` | string | no (default `"standard"`) | `"standard"` or `"athletic"` (female only), case-insensitive |

Other fields are ignored and do not change the avatar.
Invalid input returns **422** with a `detail` list; so does a body that cannot be fitted at all
(`"type": "fit_failed"`).

**How the body is fitted** (`backend/body_style.py`, 2026-10-16). Height and weight are solved
directly. Waist, chest and hip are fitted with Anny's age, muscle and proportions **plus** its bust,
waist and hip circumference controls (within ±0.6). Age is kept adult (0.67-0.85; the earlier fit
allowed 0.5, half-way to Anny's child shape), and women's muscle is capped at 0.8 (the earlier fit
pushed light female bodies to 0.94-0.99, which looks male). The fit stops when every target is within
1 %. `athletic` (women only) is an adult, toned body: narrower shoulders, slimmer hips and legs, firm
buttocks, flat stomach, natural bust (cup size 0.62); its bust is part of the shape, so the chest can
come out larger than requested. If the athletic body misses the weight by more than 1.5 kg, the
standard body is used and `warnings` says so. The earlier fit stays in `fit_to_measurements.fit_body`.

Response (200):

```jsonc
{
  "message": "Avatar generated successfully",
  "avatar_url": "/models/generated/<avatar_id>.glb",   // unique for this generation
  "avatar_id": "<32 hex chars>",
  "legacy_avatar_url": null,                           // Module: always null (see below)
  "parameters": { "gender": "female", "height": 0.5, "weight": 0.6, "age": 0.70, "muscle": 0.52, "proportions": 0.6 },
  "body_style": "standard",                            // the style actually used
  "body_controls": { "measure-waist-circ-incr": 0.12, "...": "..." },  // Anny local changes of the body
  "body_phenotype": {},                                // athletic: {"cupsize": 0.62, "firmness": 0.65}
  "appearance": { "gender_style": "female", "skin_tone": "#c8997e", "eye_color": "brown",
                  "face_controls": { "mouth-scale-horiz-incr": -0.35, "...": "..." },
                  "features": ["eye_texture", "eyebrows", "face_preset", "lip_tint"], "not_available": ["hair"] },
  "measurements": { "height_m": 1.626, "mass_kg": 49.99, "waist_m": 0.699, "chest_m": 0.865, "hip_m": 0.940, "...": "..." },
  "failed_measurements": [],
  "accuracy": {
    "waist": { "requested": 71.8, "actual": 69.92, "unit": "cm",
               "absolute_error": 1.88, "percentage_error": 2.62,
               "control": "fitted indirectly (age, muscle, proportions)", "status": "measured" }
    // also height, weight, chest, hip
  },
  "warnings": ["waist: 69.92 cm vs requested 71.80 cm (2.62% off), outside the fitting tolerance."],
  "fit_time_seconds": 11.5
}
```

- `measurements` values are in metres (`*_m`) and kilograms (`mass_kg`).
  `shoulder_width_m`, `arm_length_m` and `leg_length_m` are joint-centre estimates,
  not tape measurements. They are reported, not controlled.
- `accuracy` uses `abs(actual - requested) / abs(requested) * 100`.
- `warnings` lists targets that were not reached within the fitting tolerance
  (height 0.01 cm, weight 0.01 kg, circumferences 1 %) or could not be measured, and says when a
  weight is beyond what Anny can reach at that height (for example a 1.60 m woman of 75 kg).
- **Chest and hip are tape-style measurements** (since 2026-10-16): the convex outline of the
  horizontal cross-section with the arms and hands left out (by Anny's skinning weights), as a tape
  lies across the cleavage and under the arms. Chest: at 0.145 × height above the waist. Hip: the
  largest such loop between just below the waist and the hip joints. The earlier perimeters could
  include the arms or hands (the chest was then rejected and reported as failed; the hip read
  1.5-6 cm too large).

### Loading the avatar in three.js

```ts
const data = await response.json();
const url = `${API_URL}${data.avatar_url}`;   // unique file, no cache-buster needed
useGLTF(url) / new GLTFLoader().load(url, ...)
```

Use `avatar_url` so each user gets their own file.

**Module:** the shared `generated_avatar.glb` that production overwrote on every request is
not written, and `legacy_avatar_url` is `null`. The field is kept so existing clients do
not break on a missing key. (`WRITE_LEGACY_AVATAR` in `api.py` can turn the old behaviour
back on; it is off by default.)

The GLB is one static mesh (Y up, feet at y = 0, metres, smooth normals, no skeleton) with **two
primitives that share the same vertex attributes** (13,718 vertices): the body (material `skin`,
base-colour texture via TEXCOORD_0: skin colour, female lip tint, eyes) and the brow band (material
`skin_brows`, texture via TEXCOORD_1: painted eyebrows). Textures use a clamp-to-edge sampler.
The body fit is unchanged; after it, a gender face preset moves head vertices only (a few mm), so
`measurements` / `accuracy` describe the fitted body before the face preset. No hair or clothing.

### `GET /avatar-options`

Machine-readable list of the accepted inputs, ranges and defaults above. Build form
controls from it instead of hard-coding ranges.

### `GET /health`

`{"status": "ok", "model_loaded": true}`. Does not load the model.

### Static files

- `/models/generated/<avatar_id>.glb` — generated avatars (from `var/generated/`).
- `/models/<name>.glb` and `/test/<name>.html` — `.glb` / `.html` files directly in the
  `backend/` folder. **Module:** the backend folder holds no such files (production's
  `generated_avatar.glb`, `shirt_*.glb` and `test_avatar.html` were not copied), so these
  return 404 for now.

Other files (source code, `.git`, subfolders, hidden files) return 404.

## Known limitations

- Generated files in `generated/` are not deleted automatically. Each is about 0.5-1 MB.
- `avatar_id` is random but not secret: anyone who knows a URL can load that avatar.
- Requests are processed one at a time; a second request waits for the first.
- Hair is not available. Face shape follows gender automatically (not adjustable).
- `shoulder_width_m` is the distance between the shoulder joints, not a tape measurement, and is
  not fitted.
- Anny has a weight range per height and age: very heavy or very light bodies for their height
  cannot be reached (see `warnings`).
- The rest pose has the arms about 40° out; the pose for clothing is not agreed yet.
