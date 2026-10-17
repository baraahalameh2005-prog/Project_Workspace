# Implementation Status — Clothing3D

Audit date: 2026-10-17 · commit `db690ac`

Status labels:
- **Implemented (code read):** the code exists and was read; it was **not executed** in this audit.
- **Verified:** executed in this audit and the result checked.
- **Claimed:** stated in the project's own documentation; not re-checked.
- **Blocked:** cannot run on this machine without installing software or changing files.

**Nothing in this project was verified as working in this audit.** No feature could be executed: the required
Python packages and the expected Blender install are missing (see `AUDIT_FINDINGS.md`).

## 1. Fitting room (backend + `viewer/index.html`)

| Feature | Status | Notes / evidence |
|---|---|---|
| Garment kind selection: dress/abaya, top, skirt, trousers, scarf, accessories (bag, belt, hat, glasses, necklace) | Implemented (code read) | `viewer/index.html:287`, `server.py:76` |
| Photo upload (camera, file, drag and drop), tagged front/back/detail | Implemented (code read) | `/api/measure`, `measure: false` for detail photos |
| Automatic measurement from one photo, scale from the least-variable dimension, with uncertainty | Implemented (code read); Claimed accurate | `measure_photo.py`. README states the result is an estimate |
| Exact scale from one tape measurement | Implemented (code read) | `POST /api/scale`, `measure_photo.rescale` |
| Collar/cuff trim (ruffle) detection by colour | Implemented (code read) | `server.py:192` `trims()` |
| Dress/top **with sleeves** → cloth-simulated GLB on avatar | Implemented (code read); **Blocked** | Needs Blender at the hard-coded 4.2 path |
| Sleeveless dress → Dress3D GLB on avatar | Implemented (code read); **Blocked** | Needs `shapely`, `cv2`, `rembg` |
| Skirt / trousers / scarf → cloth-simulated GLB | Implemented (code read); **Blocked** | Needs Blender |
| Accessories | Implemented (code read) | Ready-made shapes sized/coloured in the page; positions from `/api/avatar-frame/<id>` |
| Wardrobe ("خزانتي"), several pieces worn together | Implemented (code read) | Browser `localStorage` only (per browser, per avatar id) |
| Garment alone view ("القطعة لحالها") | Implemented (code read) | — |
| Sleeve angle matched to the avatar's arm angle | Implemented (code read) | `dress_on_avatar.py:58`. Not tested with other avatar poses |
| Speed figures (dress ~30 s, skirt ~11 s, …) | Claimed | README, "8 GB laptop". Not measured here |

## 2. Product / clothing data

| Item | Status |
|---|---|
| Product catalogue, product database, persistent product records | **Not implemented.** There is no product model, database or product API. Garments exist only as per-request job folders (`jobs/<uuid>/`) and photo folders (`uploads/<uuid>/`), plus the browser's `localStorage` wardrobe |
| Size charts / size recommendation | **Not implemented** in this repository |
| Cleanup of `jobs/` and `uploads/` | **Not implemented.** README: they "hold every request's files" |

## 3. Avatar integration

| Item | Status |
|---|---|
| Reads the user's avatar GLB from the avatar module's `var/generated/` | Implemented (code read); **Blocked in the current folder layout** (`ROOT.parent/var/generated`) |
| Default test avatar `output/avatar.glb` | **Missing.** `output/` is ignored by Git and was not committed |
| Avatar page ↔ fitting room buttons | Exist only in `newavatar-changes/` (proposed avatar-module changes), not in the published NewAvatar repository |

## 4. Dress3D app (`dress3d-prototype/`)

| Item | Status |
|---|---|
| Measurement schema, dress mesh from measurements, photo cues on synthetic images, web UI | Claimed "verified" in `dress3d-prototype/HANDOVER.md` §1. Not re-checked |
| Real photos (mannequins, hangers, collars) | Claimed **not reliable** (HANDOVER §1) |
| Accuracy against a real garment measured with a tape | **Never validated** (HANDOVER §1) |
| Web frontend build | **Not built** (`prototype/web/dist` missing; `node_modules` missing) |

## 5. Tests — actually executed in this audit

Run with `D:\BarahaAvatarModule\.venv\Scripts\python.exe -B` (the only Python 3.12 environment with numpy, scipy,
Pillow and trimesh; `-B` so no files were written), from the project root.

| Suite | Test cases in file | Result | Cause |
|---|---|---|---|
| `tests.test_measure_photo` | 5 | **Did not run: import error** | `ModuleNotFoundError: No module named 'cv2'` |
| `tests.test_dress3d` | 5 | **Did not run: import error** | `ModuleNotFoundError: No module named 'shapely'` |
| `dress3d-prototype/tests/test_core.py` | 5 | **Did not run: import error** | `ModuleNotFoundError: No module named 'shapely'` |
| `tools/ui_test.py` (Playwright) | — | Not run | Needs Playwright, a running server and paths under `C:/Users/hp/` |

**No test passed or failed on its own logic.** The README's claims about tests could not be confirmed.

### Update 2026-10-17 — after creating the project `.venv` (Python 3.12.6)

| Suite | Command | Result |
|---|---|---|
| `tests.test_measure_photo` + `tests.test_dress3d` | `.venv\Scripts\python.exe -m unittest tests.test_measure_photo tests.test_dress3d -v` | **Ran 10, OK** |
| `dress3d-prototype/tests/test_core.py` | `..\.venv\Scripts\python.exe -m unittest discover -s tests -v` (in `dress3d-prototype`) | **Ran 5, OK** |

These tests cover measurement from **drawn masks** (no rembg, no real photos) and Dress3D construction on synthetic
data. They do not exercise the server, rembg, Blender, or avatar integration, which are still unverified.

## 6. Unimplemented / incomplete (from the project's own documentation)

- Puffed shoulders are not modelled; the scarf drapes stiffly; garments worn together do not collide (README).
- Sizes without a tape measurement are estimates (README).
- Dress3D: segmentation, sleeve/torso interpenetration, true flat patterns (HANDOVER §6).
- No dependency list (`requirements.txt`) for the clothing backend.
