# API and Integration — Clothing3D

Audit date: 2026-10-17 · commit `db690ac` · Source of truth: `backend/server.py` (read in full)

Nothing on this page was called against a running server in this audit (the server could not be started; see
`RUN_AND_TEST_GUIDE.md`).

## 1. Servers and ports

| Server | Address | Code |
|---|---|---|
| Clothing backend + fitting room | `127.0.0.1:8010` (fixed, `server.py:676`) | this repository |
| Dress3D app | `127.0.0.1:8765` (`DRESS3D_PORT` env var overrides) | `dress3d-prototype/prototype/server.py` |
| Avatar API | `127.0.0.1:8002` | NewAvatar (external) |
| Avatar page | `127.0.0.1:5173` | NewAvatar frontend (external) |

The fitting room links to `http://127.0.0.1:8765/` and `http://127.0.0.1:5173/` (`viewer/index.html:172–173`).

## 2. Clothing backend endpoints (port 8010)

All POST bodies are JSON, at most 40 MB (`413` otherwise). Errors: `400 {error}` for bad input,
`500 {error, code: "internal"}` for bugs.

### `POST /api/measure`
Request: `{photo: "data:image/jpeg|png;base64,…", kind?: "dress"|"top"|"skirt"|"pants"|"scarf"|"accessory", measure?: false}`

Response `200`:
```
{photoId, kind, guess, pxPerCm, cm: {name: value}, lines: {name: [[x,y],[x,y]]},
 scaleSource, scaleBy, uncertainty, notes, estimated, style, flatUrl, size: [w,h],
 colour: "#rrggbb", trims: {collar, cuff}, cutoutUrl}
```
With `measure: false`, the photo is only stored: `{photoId, flatUrl, size}`.
`422 {error, code}` if the photo cannot be read as a garment (`PhotoError`).
Writes `uploads/<photoId>/` (photo, flat, mask, cutout, garment, `measured.json`).

### `POST /api/scale`
Request: `{photoId, known: {name: cm}}` (exactly one entry) or `{photoId, factor}`.
Response `200`: the updated `measured.json` (sizes rescaled, `scaleSource: "tape"`, `uncertainty: 0`).

### `POST /api/jobs` → `202 {id}`
Queued; one job runs at a time. `method` selects the pipeline:

| method | Body | Pipeline |
|---|---|---|
| `"flat"` (default) | `front`, `back` (data URLs) or `frontId`, `backId`; `lengthCm`, `shoulderCm` (required), `hemCm?`, `sleeveCm?`, `sleeveFrom?` (`shoulder`/`neck`), `collarRuffle?`, `cuffRuffle?`, `avatarId?` | `flat_dress.py` as a subprocess (Blender) |
| `"measurements"` | `measurements: {dress_length, bust, …}` (Dress3D fields), `style?: {silhouette, sleeve, neckline}`, `units?: cm/in`, photos optional, `avatarId?` | `dress_on_avatar.make` in-process |
| `"garment"` (used by the page) | `kind`, `frontId`, `backId?`, `measurements`, `collarRuffle?`, `cuffRuffle?`, `avatarId?` | dress/top with sleeves → `flat_dress.run` (Blender); sleeveless → Dress3D; skirt/pants/scarf → `garment_kinds.make` (Blender) |

Measurement ranges (flat): `lengthCm 40–200`, `shoulderCm 20–80`, `hemCm 30–400`, `sleeveCm 10–100`.
Garment method: every value 1–500 cm.

### `GET /api/jobs/<id>`
`{status: "queued"|"running"|"done"|"failed", method, glb?, dressGlb?, info?, error?, code?, photoProblem?}`.
On `done`: `glb` = `/jobs/<id>/dress_on_avatar.glb`, `dressGlb` = `/jobs/<id>/dress.glb`, `info` = `dress.json`.
Job state is in memory only, so it is lost when the server restarts. Files stay in `jobs/<id>/`.

### `GET /api/avatar-frame/<avatarId>`
Positions on the avatar for accessories (metres, +Y up, +Z front): height, shoulder/armpit/waist/neck heights,
waist and neck rings, head, hands, chest, `bodyGirthsCm`. Cached in memory per avatar.

### `GET /avatars/<avatarId>.glb`
Serves `GENERATED_AVATARS/<avatarId>.glb` (`model/gltf-binary`), so the page loads the avatar from the same origin.

### Static files
Any other `GET` is served from the project root (`SimpleHTTPRequestHandler`, `directory=ROOT`): `/viewer/`,
`/jobs/…`, `/uploads/…`, `/output/…` — and also every other file in the folder (see `AUDIT_FINDINGS.md` F7).

## 3. Dress3D app endpoints (port 8765)

`GET /api/schema`, `GET /api/examples`, `GET /api/examples/<…>`, `POST /api/generate`
(`{front_image?, back_image?, units, measurements, style, options}`), `GET /api/out/<id>/<file>`
(`dress.glb`, `pattern.svg`, `mask.png`, `report.json`). It serves the built frontend from `prototype/web/dist`.

## 4. Avatar dependencies

| Dependency | Where the code expects it | Present now? |
|---|---|---|
| User avatars `<32-hex id>.glb` | `<parent of Clothing3D>\var\generated\` = `D:\GraduationProject_Workspace\var\generated\` | **No** |
| Default test avatar | `Clothing3D\output\avatar.glb` (page: `../output/avatar.glb`; jobs without `avatarId`) | **No** (ignored by Git, never committed) |
| Avatar format assumed | One GLB mesh, Y up, metres, feet at y = 0, facing +Z; the code measures shoulders, armpits, arm angle, girths from the mesh | Not checked against a live avatar |
| Avatar id format | 32 lowercase hex characters (`JOB_ID` regex) | — |

Relevant committed GLBs that are **not** at the expected paths: `results/avatar_athletic.glb`,
`results/dress01_flat_on_avatar.glb`.

## 5. Hard-coded paths

| File | Path | Effect |
|---|---|---|
| `backend/template_dress.py:50` | `%USERPROFILE%\tools\blender-4.2.23-windows-x64\blender.exe` | Every Blender-based garment fails if it is not there |
| `backend/server.py:60` | `ROOT.parent / "var" / "generated"` | Avatars are found only when this folder sits inside the avatar module |
| `tools/ui_test.py:8, 73–74` | `C:/Users/hp/NewAvatar/...` | UI test only works on the author's machine |
| `README.md:4, 75` | `C:\Users\hp\...` | Documentation only |

No file refers to the old local folder name, so the rename to `Clothing3D` changed nothing for the code.

## 6. Front-end connections

- `viewer/index.html` calls only same-origin URLs: `/api/measure`, `/api/scale`, `/api/jobs`, `/api/jobs/<id>`,
  `/api/avatar-frame/<id>`, `/avatars/<id>.glb`.
- It reads `?avatar=<id>` from the URL; without it, it uses the test avatar and the wardrobe key `wardrobe:test`.
- three.js and the Arabic font load from the internet (`cdn.jsdelivr.net`, `fonts.googleapis.com`).
- The avatar page's "👗 جرّبي الملابس" button exists only in `newavatar-changes/frontend/src/App.tsx`, which is not
  part of the published avatar module.
