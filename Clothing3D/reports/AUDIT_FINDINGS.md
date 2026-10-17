# Audit Findings — Clothing3D

Audit date: 2026-10-17 · commit `db690ac` · Scope: whole repository, read-only except for this `reports/` folder.

## What was done

- **Rename:** `D:\GraduationProject_Workspace\Saja_Clothing3D` → `D:\GraduationProject_Workspace\Clothing3D`.
  - Before renaming, checked: no process command line referenced the folder; no python/node/blender process was
    running; ports 8002/8010/8765/5173 were free; no `.git\index.lock`; the destination did not exist; no file
    refers to the old name.
  - After renaming: `git status` showed `main...origin/main`, clean.
- **Read in full:** `backend/server.py`. The other files were read through targeted searches (imports, paths,
  routes, Blender calls), plus the README, `dress3d-prototype/HANDOVER.md` and `viewer/index.html`.
- **Run:** the three unittest suites (all import errors). Nothing else was executed.
- **Not changed:** source code, configuration, dependencies, Git history. The only new files are this folder.

## Findings

| # | Severity | Finding | Evidence | Risk |
|---|---|---|---|---|
| F1 | **Blocker** | No Python environment and **no dependency list** for the clothing backend. The README assumes `.venv\Scripts\python.exe`; `.venv` is ignored and there is no `requirements.txt` | `ls`; `.gitignore`; only `dress3d-prototype/requirements*.txt` exist | Nobody can reproduce the environment reliably; versions are unknown |
| F2 | **Blocker** | Required packages are missing on this machine: `rembg`, `onnxruntime`, `cv2`, `shapely`, `skimage` | Import checks; test runs: `No module named 'cv2'`, `'shapely'` | Measuring and every garment job fail |
| F3 | **Blocker** | Blender path is hard-coded to `%USERPROFILE%\tools\blender-4.2.23-windows-x64\blender.exe`; this machine has Blender 5.2 in `C:\Program Files\Blender Foundation\Blender 5.2` | `backend/template_dress.py:50`; filesystem check | Sleeved dresses, tops, skirts, trousers and scarves fail. 5.2 compatibility with the `bpy` scripts is unknown |
| F4 | **Blocker** | Avatar lookup depends on folder placement: `ROOT.parent/var/generated`. In the current location that is `D:\GraduationProject_Workspace\var\generated`, which does not exist | `backend/server.py:60`; README line 4 | Every try-on with `?avatar=` fails with "avatar not found" |
| F5 | **Blocker** | The default test avatar `output/avatar.glb` is not in the repository (`output/` is ignored) | `server.py:59`; `viewer/index.html:426` | Without an avatar id, the page has no avatar |
| F6 | High | All three test suites fail at import. **No test logic has been confirmed on this machine** | Test output (see `IMPLEMENTATION_STATUS.md` §5) | README test claims are unverified |
| F7 | Medium | The server serves the **whole project folder** as static files (`directory=ROOT`), including `.git/`, `backend/` source, every `uploads/` photo and every `jobs/` output | `server.py:594` | Bound to `127.0.0.1` only, so the exposure is local. It must change before any network deployment |
| F8 | Medium | No cleanup or retention for `jobs/` and `uploads/`; job state is in memory and is lost on restart | `server.py:79`; README | Disk growth; garment photos kept indefinitely (privacy) |
| F9 | Medium | No product or clothing data model: no catalogue, database, product API or size chart. The wardrobe is browser `localStorage` only | Code search; `viewer/index.html:341` | The "product upload" part of the project is not implemented as a stored product |
| F10 | Medium | Two copies of the Dress3D library: `dress3d/` and `dress3d-prototype/dress3d/` differ in `hybrid.py` and `measure.py` | `git diff --no-index --stat` | Fixes can land in one copy only |
| F11 | Medium | The avatar-side integration (try-on button, start scripts, athletic body style, tape-style measurements) exists only as **proposed** files in `newavatar-changes/`; it is not in the published NewAvatar repository | `newavatar-changes/`; NewAvatar `main` at `47197d0` | The end-to-end flow depends on a patch that has not been merged. It also conflicts with other local avatar work (separate report, 2026-10-17) |
| F12 | Low | `tools/ui_test.py` hard-codes `C:/Users/hp/NewAvatar/...` and a fixed avatar id | `tools/ui_test.py:8–9, 73–74` | Only works on the author's machine |
| F13 | Low | The Dress3D app needs a frontend build (`dist/` is absent). **Correction:** Python 3.11 is *not* required. Dress3D was developed on Python 3.12.6 (`requirements.txt`); `requirements-py311.txt` is an alternative pin set for a 3.11 machine | `dress3d-prototype/README.md:19`, `research/research-log.md:14` | The "🧵 الفستان لحاله 3D" link opens nothing until the frontend is built |
| F14 | Low | The fitting room loads three.js and fonts from CDNs | `viewer/index.html:9, 161–162` | Does not work offline |
| F15 | Info | Accuracy limits stated by the project: sizes without a tape measurement are estimates; Dress3D is "never validated" against real garments; puffed shoulders not modelled; garments do not collide with each other | README; `dress3d-prototype/HANDOVER.md` §1 | Demonstration quality, not measured accuracy |

## Update 2026-10-17: environment created (F1, F2, F6 resolved for this machine)

- `.venv` created with Python 3.12.6 (ignored by Git).
- Installed `dress3d-prototype/requirements.txt` (pinned), then `rembg onnxruntime scikit-image` with that file as a
  constraint (`-c`), so the pinned versions did not change. `pip check`: no broken requirements.
  Added versions: rembg 2.0.85, onnxruntime 1.31.0, scikit-image 0.26.0.
- Tests executed: `tests.test_measure_photo` + `tests.test_dress3d` **10/10 OK**;
  `dress3d-prototype/tests` **5/5 OK**.
- All 10 backend modules import without errors.
- Still open: F3 (Blender), F4/F5 (no avatar). The `u2net` model is not downloaded yet; it downloads on the
  first `/api/measure`. Source code was not changed.

## Recommended next steps (for approval, in order)

1. **Decide the folder layout (F4).** Either put this project inside a NewAvatar working copy as `clothing\`
   (the layout the code expects), or approve a small configurable avatar-folder setting.
2. **Create the environment (F1, F2), with approval.** Make a `.venv` in Clothing3D and install the packages
   listed in `RUN_AND_TEST_GUIDE.md` §4. Ideally Saja provides her exact `pip freeze`, so it can be committed as
   `requirements.txt`.
3. **Resolve Blender (F3).** Install Blender 4.2 LTS at the expected path (no code change), or approve trying
   the installed 5.2 through a configurable path.
4. **Run the tests (F6)**, then the fitting room with one real avatar from the avatar API, following
   `RUN_AND_TEST_GUIDE.md` §5.
5. Only after a working run: address F7–F10 and decide on F11 together with the avatar-module owner.
