#!/usr/bin/env python3
"""Move a standard 3DGS PLY from COLMAP's world frame into an e2e run's exported frame, so it can be
compared with that run's splat and rendered on the same path by scripts/orbit_vkgs.py.

    uv run python scripts/to_training_frame.py <e2e_run_dir> <3dgs.ply> <out_run_dir>

Our exports live in the training frame (utils/camera_utils.transform_poses_pca, fit on the training
cameras: rotate, recentre, scale into [-1,1]^3), rotated upright by utils/ply_export.UPRIGHT. The
original 3DGS keeps COLMAP's frame. The map is x -> s * U (R x + t): positions, orientations and
view-dependent colour are rotated by U R (utils/ply_export.rotate_vertices), positions are scaled
and shifted, and log-scales get log(s). <out_run_dir> gets the run's cameras.json and cfg_args and
the splat as <name>.ply, with <name> taken from output/<name>/<run>, which orbit_vkgs.py expects.
"""
import json
import os
import shutil
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from utils.ply_export import UPRIGHT, _names, read_vertices, rotate_vertices, write_vertices  # noqa: E402


def pca_transform(c2w):
    """Same maths as utils.camera_utils.transform_poses_pca (that module needs the GPU stack)."""
    t = c2w[:, :3, 3]
    t_mean = t.mean(0)
    eigval, eigvec = np.linalg.eig((t - t_mean).T @ (t - t_mean))
    rot = eigvec[:, np.argsort(eigval)[::-1]].T
    if np.linalg.det(rot) < 0:
        rot = np.diag([1, 1, -1]) @ rot
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rot, rot @ -t_mean
    posed = T @ c2w
    if posed.mean(0)[2, 1] < 0:
        T = np.diag([1, -1, -1, 1]) @ T
        posed = T @ c2w
    return T, 1.0 / np.max(np.abs(posed[:, :3, 3]))


def main(run_dir, ply_in, out_dir):
    cams = sorted(json.load(open(os.path.join(run_dir, "cameras.json"))), key=lambda c: c["img_name"])
    c2w = np.tile(np.eye(4), (len(cams), 1, 1))
    for i, c in enumerate(cams):
        c2w[i, :3, :3], c2w[i, :3, 3] = np.array(c["rotation"]), np.array(c["position"])
    train = np.array([i % 8 != 0 for i in range(len(cams))])  # e2e holdout: every 8th by name
    T, s = pca_transform(c2w[train])
    R, t = T[:3, :3].real, T[:3, 3].real

    arr = np.array(read_vertices(ply_in))  # writable copy
    rotate_vertices(arr, UPRIGHT @ R)
    shift = s * (UPRIGHT @ t)
    for i, c in enumerate("xyz"):
        arr[c] = s * arr[c].astype(np.float64) + shift[i]
    for n in _names(arr, "scale_"):
        arr[n] = arr[n] + np.log(s)

    os.makedirs(out_dir, exist_ok=True)
    for f in ("cameras.json", "cfg_args"):
        shutil.copy(os.path.join(run_dir, f), out_dir)
    name = os.path.basename(os.path.dirname(os.path.abspath(out_dir)))
    out = os.path.join(out_dir, f"{name}.ply")
    write_vertices(arr, out)
    print(f"{len(arr):,} Gaussians, scale {s:.4g} -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
