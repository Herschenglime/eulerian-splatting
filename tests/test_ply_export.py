import os
import subprocess
import sys

import numpy as np
import pytest

from tests.gaussian_fixtures import STANDARD_NAMES, activated_vertices, make_test_shapes
from tests.conftest import REPO_ROOT
from utils.ply_export import UPRIGHT, convert_vertices, read_vertices, rotate_vertices, write_vertices
from utils.sh_utils import eval_sh


def _random_gaussians(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return activated_vertices(
        xyz=rng.normal(size=(n, 3)),
        rgb=rng.random((n, 3)),
        opacity=rng.random(n) * 0.98 + 0.01,
        scale=np.exp(rng.uniform(-8, -1, size=(n, 3))),
        rot=rng.normal(size=(n, 4)),
        f_rest=rng.normal(size=(n, 3, 15)) * 0.1,
    )


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def test_round_trip_matches_viewer_activations():
    arr = _random_gaussians()
    out = convert_vertices(arr, upright=False)
    # A standard viewer applies sigmoid to opacity and exp to scale.
    np.testing.assert_allclose(_sigmoid(out["opacity"].astype(np.float64)), arr["opacity"], rtol=1e-5)
    for i in range(3):
        np.testing.assert_allclose(np.exp(out[f"scale_{i}"].astype(np.float64)), arr[f"scale_{i}"], rtol=1e-5)
    # Everything else passes through untouched.
    for name in STANDARD_NAMES:
        if name != "opacity" and not name.startswith("scale_"):
            np.testing.assert_array_equal(out[name], arr[name])


def test_output_layout_is_reference_3dgs():
    out = convert_vertices(_random_gaussians())
    assert list(out.dtype.names) == STANDARD_NAMES
    assert all(out.dtype[n] == np.float32 for n in out.dtype.names)


def test_rejects_already_converted_file():
    out = convert_vertices(_random_gaussians())
    with pytest.raises(ValueError, match="already in standard"):
        convert_vertices(out)
    convert_vertices(out, force=True)  # escape hatch still works


def test_extreme_opacities_stay_finite():
    arr = _random_gaussians(n=4)
    arr["opacity"] = [0.0, 1.0, 1e-9, 0.5]
    out = convert_vertices(arr)
    assert np.all(np.isfinite(out["opacity"]))


def test_min_opacity_and_max_points():
    arr = _random_gaussians(n=500)
    out = convert_vertices(arr, min_opacity=0.5)
    assert len(out) == int(np.sum(arr["opacity"] >= 0.5))
    assert _sigmoid(out["opacity"].astype(np.float64)).min() >= 0.5 - 1e-5

    out = convert_vertices(arr, max_points=50)
    assert len(out) == 50
    kept = np.sort(_sigmoid(out["opacity"].astype(np.float64)))
    np.testing.assert_allclose(kept, np.sort(arr["opacity"])[-50:], rtol=1e-5)
    # original order preserved: x values appear in the same relative order as in the input
    pos = [int(np.nonzero(arr["x"] == x)[0][0]) for x in out["x"]]
    assert pos == sorted(pos)


def test_max_radius_drops_far_gaussians():
    arr = _random_gaussians(n=6)
    arr["x"], arr["y"], arr["z"] = [0, 1.5, 0, 3, 1e6, -2], [0, 0, 1.9, 0, 0, 0], [0, 0, 0, 0, 0, 0]
    out = convert_vertices(arr, max_radius=2.0)
    np.testing.assert_allclose(out["x"], [0, 1.5, 0, -2])


def _quat_to_matrix(q):
    w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], 1)


def _sh(arr, c):
    """[N, 16] SH coefficients of color channel c (f_dc + 15 f_rest)."""
    return np.stack([arr[f"f_dc_{c}"]] + [arr[f"f_rest_{c * 15 + k}"] for k in range(15)], 1).astype(np.float64)


def test_upright_maps_scene_up_to_plus_y():
    # The training frame's up is -z; viewers expect +y.
    np.testing.assert_allclose(UPRIGHT @ [0, 0, -1], [0, 1, 0])
    assert np.isclose(np.linalg.det(UPRIGHT), 1)
    arr = _random_gaussians(n=1)
    arr["x"], arr["y"], arr["z"] = 0.5, 0.25, -2.0
    out = convert_vertices(arr)
    np.testing.assert_allclose([out["x"][0], out["y"][0], out["z"][0]], [0.5, 2.0, 0.25], rtol=1e-6)


@pytest.mark.parametrize("rot", [UPRIGHT, _quat_to_matrix(np.array([[0.3, -0.5, 0.7, 0.2]]))[0]],
                         ids=["upright", "arbitrary"])
def test_rotation_rotates_covariance_and_view_dependent_color(rot):
    arr = convert_vertices(_random_gaussians(n=50), upright=False)
    out = rotate_vertices(arr.copy(), rot)
    # Covariance: R S S^T R^T before, R_new S S^T R_new^T after, and the latter must equal rot @ before @ rot^T.
    s = np.exp(np.stack([arr[f"scale_{i}"] for i in range(3)], 1).astype(np.float64))
    def cov(a):
        r = _quat_to_matrix(np.stack([a[f"rot_{i}"] for i in range(4)], 1).astype(np.float64))
        return r @ (s[:, :, None] ** 2 * np.eye(3)) @ r.transpose(0, 2, 1)
    np.testing.assert_allclose(cov(out), rot @ cov(arr) @ rot.T, atol=1e-5)
    # Color: the rotated Gaussian seen along rot @ d looks like the original seen along d.
    dirs = np.random.default_rng(1).normal(size=(50, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    for c in range(3):
        before = eval_sh(3, _sh(arr, c)[:, None, :], dirs)
        after = eval_sh(3, _sh(out, c)[:, None, :], dirs @ rot.T)
        np.testing.assert_allclose(after, before, atol=1e-4)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_truncation_keeps_lowest_bands_per_channel(degree):
    arr = _random_gaussians()
    out = convert_vertices(arr, sh_degree=degree, upright=False)
    keep = (degree + 1) ** 2 - 1
    assert len([n for n in out.dtype.names if n.startswith("f_rest_")]) == 3 * keep
    for c in range(3):
        for k in range(keep):
            np.testing.assert_array_equal(out[f"f_rest_{c * keep + k}"], arr[f"f_rest_{c * 15 + k}"])


def test_cli_end_to_end_and_writes_visual_check_file(tmp_path):
    os.makedirs(os.path.join(REPO_ROOT, "tests", "out"), exist_ok=True)
    src = tmp_path / "shapes_activated.ply"
    dst = os.path.join(REPO_ROOT, "tests", "out", "test_shapes.ply")
    write_vertices(make_test_shapes(), str(src))

    result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "export_ply.py"), str(src), dst],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = read_vertices(dst)
    assert len(out) == len(make_test_shapes())
    np.testing.assert_allclose(np.exp(out["scale_0"].astype(np.float64)), 0.006, rtol=1e-5)

    # Running it again on the converted file must refuse.
    again = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "scripts", "export_ply.py"), dst,
                            str(tmp_path / "twice.ply")], capture_output=True, text=True)
    assert again.returncode != 0 and "already in standard" in again.stderr
