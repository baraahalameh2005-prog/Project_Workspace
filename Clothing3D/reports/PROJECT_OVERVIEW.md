# Project Overview — Clothing3D

Audit date: 2026-10-17 · Local path: `D:\GraduationProject_Workspace\Clothing3D`
Repository: `https://github.com/Saja-Khalil/-clothing-3d.git` · branch `main` · commit `db690ac` (2026-10-17 01:43 +0300)

All statements below come from reading the code and documentation in this commit. Whether a feature actually
works is recorded separately in `IMPLEMENTATION_STATUS.md`.

## 1. What the project is

The clothing module of the virtual-fitting graduation project. The intended flow:

1. A user photographs a garment laid flat (front, optionally back).
2. The backend removes the background (`rembg`, model `u2net`) and measures the garment from the photo.
3. The garment is rebuilt in 3D from templates sized by those measurements, textured from the photo, and draped
   on the user's avatar with Blender's cloth simulation.
4. A browser page ("غرفة القياس", the fitting room) shows the avatar wearing the garment and keeps a wardrobe.

It depends on a separate **avatar module** (the `NewAvatar` repository), which generates the avatar GLB files.

## 2. Technologies

| Part | Technology |
|---|---|
| Clothing backend | Python, standard-library `http.server` (no framework), numpy, scipy, Pillow, trimesh, OpenCV, shapely, rembg |
| Cloth simulation / rendering | Blender, run headless as a subprocess (`bpy` scripts) |
| Fitting-room page | One static HTML file with inline JavaScript; three.js 0.169.0 loaded from cdn.jsdelivr.net |
| Dress3D app (separate) | Python `http.server` + React 18 / Vite 5 / TypeScript / three.js frontend |
| Avatar module (external) | FastAPI + Anny body model (not in this repository) |

## 3. Folder structure

| Folder / file | Purpose |
|---|---|
| `backend/` | The clothing backend. `server.py` is the entry point (port 8010) |
| `backend/server.py` | HTTP server: static files, `/api/measure`, `/api/scale`, `/api/jobs`, `/api/avatar-frame/<id>`, `/avatars/<id>.glb` |
| `backend/measure_photo.py` | Automatic garment measurement from one flat-lay photo (all kinds) |
| `backend/flat_lay.py` | Reads a flat-laid garment: collar, shoulders, armpits, sleeves, hem |
| `backend/photo_to_garment.py` | Shared helpers: background removal, garment mask, avatar measuring |
| `backend/flat_dress.py` | Dresses/tops with sleeves → draped GLB (Blender). Also a CLI |
| `backend/garment_kinds.py` | Skirts, trousers, scarves → draped GLB (Blender) |
| `backend/dress_on_avatar.py` | Sleeveless dresses via the Dress3D library, placed on the avatar (no Blender) |
| `backend/template_dress.py` | Template parts (sleeves, ruffles, folds); defines the Blender path |
| `backend/blender_drape.py`, `blender_render.py` | Scripts run *inside* Blender: cloth simulation + GLB export; renders |
| `backend/photo_to_hanger.py`, `render_preview.py` | Auxiliary tools (not called by the server) |
| `viewer/index.html` | The fitting-room page (served by the backend at `/viewer/`) |
| `dress3d/` | Dress3D library used by the backend. Copy of `dress3d-prototype/dress3d/` with changes in `hybrid.py` and `measure.py` |
| `dress3d-prototype/` | Separate Dress3D app: a dress from photo + measurements, shown alone in 3D (port 8765), with research, experiments and results |
| `dress3d_examples/` | One synthetic test case (`T2_aline_vneck`) used by `tests/test_dress3d.py` |
| `tests/` | `test_measure_photo.py`, `test_dress3d.py` (unittest) |
| `tools/ui_test.py` | Playwright browser test of the fitting room (hard-coded paths, see audit) |
| `samples/` | Sample garment photos (`flat_front.jpg`, `flat_back.jpg`, `photo1.jpeg`) |
| `results/` | Committed example outputs (dress GLBs, renders, an athletic avatar GLB) |
| `avatar/athletic_female_preview.py` | Avatar experiment; must be copied into the avatar module to run |
| `newavatar-changes/` | Proposed changes to the avatar module (changed files + `newavatar.patch`, start scripts). **Not used by this project at runtime** |

Folders created at runtime and ignored by Git: `jobs/`, `uploads/`, `output/`, `.venv/`.

## 4. Entry points

| What | Command (from the project's README) | URL |
|---|---|---|
| Clothing backend + fitting room | `.venv\Scripts\python.exe backend\server.py` | `http://127.0.0.1:8010/viewer/?avatar=<avatar_id>` |
| Dress3D app | `python prototype/server.py` (inside `dress3d-prototype/`, after building `prototype/web`) | `http://127.0.0.1:8765/` |
| Avatar API (external, NewAvatar) | not in this repository | `http://127.0.0.1:8002` |
| Avatar page (external, NewAvatar frontend) | not in this repository | `http://127.0.0.1:5173` |

## 5. Expected folder layout (important)

The code expects this project to live **inside** an avatar-module working copy, as a folder named `clothing`:

```
NewAvatar\                 (avatar module)
  var\generated\<id>.glb   (written by the avatar API)
  clothing\                (this project)
```

Evidence: `README.md` line 4 (`C:\Users\hp\NewAvatar\clothing`), `backend/server.py:60`
(`GENERATED_AVATARS = ROOT.parent / "var" / "generated"`), `newavatar-changes/frontend/src/App.tsx:76`.
In its current location, `ROOT.parent` is `D:\GraduationProject_Workspace`, which has no `var\generated`.
