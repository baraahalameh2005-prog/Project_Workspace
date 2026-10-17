"""Local CPU stand-in for `torchmcubes.marching_cubes` (whose build needs CUDA/C++ toolchains).

Uses scikit-image (BSD-3). torchmcubes returns vertex coordinates in reversed index order
(TripoSR then re-orders them with [..., [2, 1, 0]]), so we reverse skimage's (i, j, k) output.
"""
import numpy as np
import torch
from skimage.measure import marching_cubes as _mc


def marching_cubes(vol: torch.Tensor, thresh: float):
    v = vol.detach().cpu().numpy().astype(np.float32)
    if not (v.min() < thresh < v.max()):
        return torch.zeros((0, 3)), torch.zeros((0, 3), dtype=torch.long)
    verts, faces, _, _ = _mc(v, level=thresh)
    verts = verts[:, ::-1].copy()
    return torch.from_numpy(verts).float(), torch.from_numpy(faces.astype(np.int64))
