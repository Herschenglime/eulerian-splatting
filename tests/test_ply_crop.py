import os
import subprocess
import sys

import numpy as np

from tests.conftest import REPO_ROOT
from tests.gaussian_fixtures import activated_vertices
from utils.ply_crop import crop_vertices
from utils.ply_export import convert_vertices, read_vertices, write_vertices


def _gaussians_at(xyz):
    n = len(xyz)
    return activated_vertices(np.asarray(xyz, dtype=float), rgb=np.full((n, 3), 0.5),
                              opacity=np.full(n, 0.5), scale=np.full((n, 3), 0.01))


POINTS = [[0, 0, 0], [1.5, 0, 0], [0, 1.9, 0], [3, 0, 0], [1e6, 0, 0], [-2, 0, 0]]


def test_crop_keeps_points_within_radius():
    out = crop_vertices(_gaussians_at(POINTS), max_radius=2.0)
    np.testing.assert_allclose(out["x"], [0, 1.5, 0, -2])


def test_export_crops_through_the_same_module():
    out = convert_vertices(_gaussians_at(POINTS), max_radius=2.0)
    assert len(out) == 4


def test_cli(tmp_path):
    src, dst = tmp_path / "in.ply", tmp_path / "out.ply"
    write_vertices(_gaussians_at(POINTS), str(src))
    res = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "crop_ply.py"), str(src), str(dst),
                          "--max_radius", "2"], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "4 of 6" in res.stdout and len(read_vertices(str(dst))) == 4
