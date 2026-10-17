# Run and Test Guide — Clothing3D

Audit date: 2026-10-17 · commit `db690ac` · machine: Windows 11, user profile `C:\Users\ACTC`

## 1. Prerequisites and what this machine has

| Requirement | Needed by | On this machine (checked 2026-10-17) |
|---|---|---|
| Python 3 with numpy, scipy, Pillow, trimesh | everything | Python 3.12.6 (`C:\Users\ACTC\AppData\Local\Programs\Python\Python312`) has **none** of them. The avatar module's `.venv` has numpy, scipy, Pillow, trimesh, torch |
| `rembg` (+ onnxruntime) and its `u2net` model (`%USERPROFILE%\.u2net\u2net.onnx`, downloaded on first use) | measuring, every garment job | **Missing** in every environment; model **not downloaded** |
| `opencv-python(-headless)` (`cv2`) | `measure_photo.py`, `dress3d/photo.py` | **Missing** |
| `shapely` | `dress3d/generator.py` | **Missing** |
| `scikit-image` | `backend/flat_lay.py:55` (imported inside a function) | **Missing** |
| A `.venv` in the project (the README's commands use `.venv\Scripts\python.exe`) | startup | **Missing**. The project has **no `requirements.txt`** for the clothing backend |
| Blender **4.2 LTS** at `%USERPROFILE%\tools\blender-4.2.23-windows-x64\blender.exe` | dresses with sleeves, tops, skirts, trousers, scarves | **Not there.** Blender **5.2** is installed at `C:\Program Files\Blender Foundation\Blender 5.2` (compatibility with the scripts unknown; e.g. `blender_render.py` uses `BLENDER_EEVEE_NEXT`) |
| Avatar GLB(s) in `<parent>\var\generated\` or `output\avatar.glb` | any try-on | **Missing** |
| Node.js + npm | Dress3D app frontend | Node 24.20.0, npm 11.19.0 present; `node_modules` and `dist` **missing** |
| Python 3.11 | Dress3D `requirements-py311.txt` | **Not installed** (only 3.12) |
| Internet | three.js and fonts in the fitting room | Assumed available |
| Microsoft Edge + Playwright | `tools/ui_test.py` | Edge present; Playwright **missing** |

> **Update 2026-10-17:** `Clothing3D\.venv` now exists (Python 3.12.6). The Python packages in the table above are
> installed there: pinned set from `dress3d-prototype/requirements.txt`, plus rembg 2.0.85, onnxruntime 1.31.0 and
> scikit-image 0.26.0. Python 3.11 is not needed. Reproduce with:
> ```
> py -3.12 -m venv .venv
> .venv\Scripts\python.exe -m pip install -r dress3d-prototype\requirements.txt
> .venv\Scripts\python.exe -m pip install rembg onnxruntime scikit-image -c dress3d-prototype\requirements.txt
> ```
> Still missing: Blender 4.2 at the expected path, an avatar GLB, the `u2net` model (downloaded on first use),
> Playwright, and the Dress3D frontend build.

## 2. Documented startup commands (from `README.md`)

Clothing backend (run from the project root):
```
.venv\Scripts\python.exe backend\server.py
```
→ prints `Serving on http://127.0.0.1:8010/viewer/`, then open
`http://127.0.0.1:8010/viewer/?avatar=<avatar_id>`.

Dress3D app (from `dress3d-prototype\`):
```
pip install -r requirements-py311.txt          (Python 3.11)
cd prototype\web && npm ci && npm run build
python prototype\server.py                      → http://127.0.0.1:8765/
```

Avatar module (needed for real avatars): run from a NewAvatar working copy. `newavatar-changes/start_avatar.ps1`
shows the intended set: avatar API `:8002`, avatar page `:5173`, clothing `:8010`, Dress3D `:8765`.

Tests (README):
```
.venv\Scripts\python.exe -m unittest tests.test_measure_photo tests.test_dress3d
```

## 3. What was run in this audit

| Action | Result |
|---|---|
| The three unittest suites, with `D:\BarahaAvatarModule\.venv\Scripts\python.exe -B` | All three stopped at import: `No module named 'cv2'` / `'shapely'`. 0 tests executed |
| Starting the clothing backend | **Not started.** The documented interpreter `.venv\Scripts\python.exe` does not exist (see §4) |
| Starting the Dress3D app | **Not started.** Needs `npm ci` + build, and Python packages that are missing |

`git status` was clean after the test attempts (no files written).

## 4. Why the servers were not started

The documented command cannot run as written: there is no `.venv`. Making it run needs at least one of the
following, and each one is an installation or a change that was not approved:

1. Creating a `.venv` and installing packages (`numpy scipy pillow trimesh opencv-python-headless shapely
   scikit-image rembg onnxruntime`). There is no pinned list, so versions would have to be chosen.
2. Starting `backend\server.py` with another project's Python (the avatar module's `.venv`). The server process
   starts with only Pillow, so the page would probably load. But measuring fails (`rembg`/`cv2` missing),
   every garment job fails, and no avatar exists to display.
3. Installing Blender 4.2 at the expected path, or changing `BLENDER` in `template_dress.py`.
4. Placing an avatar GLB at `output\avatar.glb`, or arranging the folder inside a NewAvatar working copy as
   `clothing\` so that `var\generated` is found.

## 5. Suggested test steps once the environment exists (not yet performed)

1. `python -m unittest tests.test_measure_photo tests.test_dress3d -v` (expect 10 tests).
2. `cd dress3d-prototype && python -m unittest discover -s tests -v` (expect 5 tests).
3. Start the avatar API, generate one avatar, and note its `avatar_id`.
4. Start `backend\server.py`, then open `http://127.0.0.1:8010/viewer/?avatar=<avatar_id>`.
5. Upload `samples\flat_front.jpg` / `samples\flat_back.jpg` as a dress. Check the measurements, then "wear it".
   Expect about 30 s according to the README.
6. Try a skirt, then an accessory. Reload the page to see the wardrobe persist.

## 6. Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError: cv2 / shapely / rembg` | Packages missing (§1) |
| Job fails with a Blender error, or "file not found" for `blender.exe` | Blender not at `%USERPROFILE%\tools\blender-4.2.23-windows-x64` |
| "That avatar was not found; make the avatar again" | The GLB is not in `<parent of project>\var\generated\` |
| Page loads but no avatar appears (no `?avatar=`) | `output\avatar.glb` missing |
| First measurement very slow, or fails offline | `rembg` downloads `u2net.onnx` on first use |
| Blank 3D view | three.js CDN unreachable (no internet) |
| Port 8010 in use | The port is fixed in `server.py:676` |
