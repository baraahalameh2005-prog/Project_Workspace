# FitEngine in this workspace

`FitEngine/` is Nagham's size-recommendation and fit-assessment module (fit_engine v0.2.0), copied here on
2026-10-17 from her own project folder. It is self-contained: Python standard library only, no install, and
it does not import or change any code in `backend/`, `frontend/` or `Clothing3D/`.

## Ports in this workspace

| Service | Port | Owner |
|---|---|---|
| Avatar API / avatar page | 8002 / 5173 | Baraah |
| Fitting room / Dress3D | 8010 / 8765 | Saja |
| **FitEngine UI + API** | **8020** | Nagham |

FitEngine's default port is 8765, which Dress3D uses here, so start it on 8020:

```
cd FitEngine
python -m fit_engine serve --port 8020          -> http://127.0.0.1:8020/
python -m unittest discover -s tests -t . -v    -> Ran 88 tests ... OK
```

## How it connects to the avatar module

- `fit_engine/adapters.py` → `avatar_to_body(avatar)` turns a `POST /generate-avatar` request or response
  into a FitEngine body document, using the measurements the **user entered** (not the generated mesh).
- HTTP: `POST /avatar-to-body {"avatar": {...}}`, then `POST /recommend {"body": ..., "garment": ...}`.
- UI: section «تعبئة القياسات من الأفاتار» on http://127.0.0.1:8020/.

Details, the field mapping and what is still open (a one-click button needs an avatar-module change that
Baraah decides on; there is no size-chart data on the clothing side yet): `docs/INTEGRATION.md` and
`docs/DATA_MODEL.md`.
