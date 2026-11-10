import os
import subprocess
import sys

import numpy as np
import pytest

from tests.gaussian_fixtures import STANDARD_NAMES, activated_vertices, make_test_shapes
from tests.conftest import REPO_ROOT
from utils.ply_export import convert_vertices, read_vertices, write_vertices


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
    out = convert_vertices(arr)
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


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_sh_truncation_keeps_lowest_bands_per_channel(degree):
    arr = _random_gaussians()
    out = convert_vertices(arr, sh_degree=degree)
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
