# Clothing module (Saja)

Part of the virtual-fitting graduation project. On the working machine this folder lives at
`C:\Users\hp\NewAvatar\clothing`, next to Baraah's avatar module.

A photo of a garment → the garment in 3D (GLB) on the user's avatar. Everything runs locally; no photo is sent
to an online service. The shape comes from templates sized by the photo, the look from the photos themselves, and
the drape from Blender's cloth simulation (`modelSource: "template_parametric"`). Accessories are ready-made
shapes sized and coloured from their photo, and the page says so.

## The fitting room (غرفة القياس)

```
.venv\Scripts\python.exe backend\server.py        (or start_avatar.bat, which starts everything)
```
then open `http://127.0.0.1:8010/viewer/?avatar=<avatar_id>`. The avatar page's **👗 جرّبي الملابس** button
opens it with the avatar just made.

1. **The piece:** dress/abaya, top, skirt, trousers, scarf, or an accessory (bag, belt, hat, glasses, necklace).
2. **Photos:** one or more, each tagged front / back / detail (camera, file or drag and drop).
3. **Measurements:** measured automatically from the photo (`backend/measure_photo.py`), drawn on it, and
   editable. Widths are shown laid flat, the way a tape measures a garment on a table.
4. **Wear it:** the garment is built and draped on the avatar. **خزانتي** (wardrobe) keeps every piece,
   and several can be worn together. **القطعة لحالها** hides the avatar to see the piece alone.

### Measuring without a reference

A photo alone has no absolute size: a small dress close to the camera looks the same as a big one further away.
So `measure_photo.py`:
- reads every proportion from the photo;
- takes the scale from the dimension that varies least between sizes: the shoulder width of dresses and tops
  (women's 37–43 cm, about ±7 %), the waist of skirts and trousers, the length of a scarf;
- reports the result as an estimate, with its uncertainty.

One optional tape measurement (`POST /api/scale`) makes every other measurement exact. Collar and cuff ruffles are
detected from their contrasting colour.

### Speed (8 GB laptop)

| Piece | Time |
| --- | --- |
| Dress (from 3.5 min) | ~30 s |
| Skirt | ~11 s |
| Trousers | ~23 s |
| Scarf | ~8 s |
| Accessory | immediate |

Jobs run inside the server, so libraries and models stay loaded. The cloth simulation uses 20 frames.

## Files

| File | What it does |
| --- | --- |
| `backend/server.py` | Local server: `/api/measure`, `/api/scale`, `/api/jobs` (method `garment`), `/api/avatar-frame/<id>`, `/avatars/<id>.glb` |
| `backend/measure_photo.py` | Automatic measurements of any kind from one flat-lay photo |
| `backend/flat_dress.py` | Dresses and tops with sleeves: photos + measurements → draped GLB (`run()` is called by the server) |
| `backend/garment_kinds.py` | Skirts, trousers and scarves: rings sized from the photos, photo texture, cloth simulation |
| `backend/dress_on_avatar.py` | Sleeveless dresses: Dress3D panels from measurements, placed on the avatar |
| `backend/flat_lay.py` | Reads a flat-laid garment: collar, shoulders, armpits, sleeves, hem |
| `backend/template_dress.py` | Template parts: sleeves, ruffles (collar, cuffs), skirt folds |
| `backend/blender_drape.py` | Runs inside Blender: cloth simulation, light smoothing, GLB export |
| `backend/blender_render.py` | Runs inside Blender: front / 35° / side / back renders |
| `backend/photo_to_garment.py` | Background removal (rembg u2net) and avatar measuring, shared |
| `dress3d/` | Baraah's Dress3D library (from her `clothes` repo), with fixes: no matplotlib, back-photo mask, the avatar's sleeve angle, stops once the length settles (62 s → 21 s) |
| `viewer/index.html` | The fitting room page (three.js) |
| `tests/` | `test_measure_photo.py` (each kind on drawn masks, tape rescaling), `test_dress3d.py` |
| `newavatar-changes/` | Changes to the avatar module: every changed or new file as it is now, and `newavatar.patch` (the edits as a diff). Body style (athletic), tape-style chest/hip, the try-on and 3D-dress buttons, start scripts |
| `dress3d-prototype/` | The Dress3D app: a dress from its photo and measurements, alone in 3D (port 8765; the **🧵 شوفي الفستان 3D** button). Run: `pip install -r requirements-py311.txt` (Python 3.11), `npm ci && npm run build` in `prototype/web`, then `python prototype/server.py` |
| `tools/ui_test.py` | Drives the fitting room in Edge (Playwright) like a user: dress, skirt, accessories; screenshots each step |

```
.venv\Scripts\python.exe -m unittest tests.test_measure_photo tests.test_dress3d
```

Needs Blender 4.2 LTS at `C:\Users\hp\tools\blender-4.2.23-windows-x64` (path in `backend/template_dress.py`).
`jobs/`, `uploads/` and `output/` hold every request's files and are not in Git.

### Command line (flat-lay photos with tape measurements)

```
.venv\Scripts\python.exe backend\flat_dress.py --avatar output\avatar.glb ^
    --front samples\flat_front.jpg --back samples\flat_back.jpg ^
    --length-cm 130 --shoulder-cm 42 --hem-cm 118 --sleeve-cm 60 --sleeve-from neck ^
    --collar-ruffle --cuff-ruffle --name dress01_flat
```

## Known limits

- Sizes without a tape measurement are estimates (see above).
- Puffed shoulders are not modelled.
- The scarf drapes stiffly.
- Garments worn together do not collide with each other.

## Results in Git (`results/`)

| File | |
| --- | --- |
| `dress01_flat.glb` | The final dress |
| `dress01_flat_on_avatar.glb` | The dress on the test avatar |
| `dress01_flat_render.png` | Renders from four sides |
| `dress01_flat_compare.png` | The photo next to the 3D dress |
| `avatar_athletic.glb`, `avatar_athletic.png` | Athletic adult female avatar (experiment, see below) |

## Avatar experiment

`avatar/athletic_female_preview.py` fits an adult, athletic female body: age 0.67–0.8, muscle 0.55–0.72,
narrower shoulders, slimmer hips and legs, natural bust (Anny cup size 0.62). It runs inside Baraah's avatar module
(it imports its fit and measurement code), so copy it to `NewAvatar/backend/tools/` and run it from `backend/` with
the avatar `.venv`.

## Reports

- Slides: https://claude.ai/artifact/7FJ8Kv8zsrE33TVTGj7qVD
- Report: https://claude.ai/artifact/R5srUrxveN7xBbNCXHivGA
