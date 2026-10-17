"""Inert stand-in for `rembg` so TripoSR's utils module imports without it.

This project deliberately does not use rembg: its default model (BRIA RMBG-2.0) requires a
commercial licence. Foreground masks come from dress3d.photo (OpenCV GrabCut) instead.
"""


def new_session(*a, **k):
    raise RuntimeError("rembg is intentionally not installed in this workspace (licence policy)")


def remove(*a, **k):
    raise RuntimeError("rembg is intentionally not installed in this workspace (licence policy)")
