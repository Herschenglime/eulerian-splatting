"""Run the real COLMAP loader on a tiny synthetic scene (CPU only).

The GPU-only modules imported transitively by scene/ are stubbed, since the
loader itself never calls into them.
"""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))

from scene.colmap_loader import rotmat2qvec  # noqa: E402
from scene.dataset_readers import readColmapSceneInfo  # noqa: E402
from utils.subset_utils import select_subset  # noqa: E402

N_IMAGES = 40
W, H = 32, 24


def _look_at_w2c(cam_pos):
    """World-to-camera rotation/translation, COLMAP convention (x right, y down, z forward)."""
    forward = -cam_pos / np.linalg.norm(cam_pos)
    right = np.cross(forward, [0, 0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    R_c2w = np.stack([right, down, forward], axis=1)
    R_w2c = R_c2w.T
    return R_w2c, -R_w2c @ cam_pos


@pytest.fixture
def colmap_scene(tmp_path):
    """40 cameras on a circle around the origin, PINHOLE text model, no point_cloud/ dir."""
    os.makedirs(tmp_path / "sparse" / "0")
    os.makedirs(tmp_path / "images")
    with open(tmp_path / "sparse" / "0" / "cameras.txt", "w") as f:
        f.write(f"1 PINHOLE {W} {H} 30 30 {W / 2} {H / 2}\n")
    with open(tmp_path / "sparse" / "0" / "images.txt", "w") as f:
        for i in range(N_IMAGES):
            a = 2 * np.pi * i / N_IMAGES
            R, t = _look_at_w2c(np.array([3 * np.cos(a), 3 * np.sin(a), 0.5]))
            q = rotmat2qvec(R)
            name = f"frame_{i:03d}.png"
            f.write(f"{i + 1} {' '.join(map(str, q))} {' '.join(map(str, t))} 1 {name}\n\n")
            Image.fromarray(np.full((H, W, 3), i * 5, dtype=np.uint8)).save(tmp_path / "images" / name)
    return tmp_path


def _read(path, subset):
    return readColmapSceneInfo(str(path), "images", eval=True, init_type="random", train_subset=subset)


def test_select_subset_is_even_and_deterministic():
    items = list(range(35))
    assert select_subset(items, 5) == [0, 8, 17, 26, 34]
    assert select_subset(items, 5) == select_subset(items, 5)
    assert select_subset(items, -1) == items
    assert select_subset(items, 100) == items
    assert len(select_subset(items, 34)) == 34


def test_missing_point_cloud_dir_no_longer_crashes(colmap_scene):
    assert not (colmap_scene / "point_cloud").exists()
    _read(colmap_scene, -1)
    assert (colmap_scene / "point_cloud").is_dir()


@pytest.mark.parametrize("n", [30, 10, 3])
def test_subset_keeps_test_set_and_frame(colmap_scene, n):
    full = _read(colmap_scene, -1)
    sub = _read(colmap_scene, n)

    assert len(full.train_cameras) == 35 and len(full.test_cameras) == 5  # llffhold=8
    assert len(sub.train_cameras) == n
    assert [c.image_name for c in sub.test_cameras] == [c.image_name for c in full.test_cameras]
    # subset views come from the training set, never the test set, in name order
    names = [c.image_name for c in sub.train_cameras]
    assert set(names) <= {c.image_name for c in full.train_cameras}
    assert names == sorted(names)
    # identical scene normalization, so renders from different ladder runs line up
    np.testing.assert_array_equal(sub.nerf_normalization["transform"], full.nerf_normalization["transform"])
    assert sub.nerf_normalization["scale"] == full.nerf_normalization["scale"]


def test_scene_writes_train_subset_list(colmap_scene, tmp_path, monkeypatch):
    import scene as scene_mod

    # Camera objects move tensors to CUDA; replace that step with a passthrough.
    monkeypatch.setattr(scene_mod, "cameraList_from_camInfos", lambda infos, *a, **k: list(infos))
    model_path = tmp_path / "run"
    os.makedirs(model_path)
    args = SimpleNamespace(model_path=str(model_path), source_path=str(colmap_scene), images="images",
                           eval=True, init_type="random", white_background=False, train_subset=10)
    s = scene_mod.Scene(args, gaussians=None)

    assert len(s.getTrainCameras()) == 10
    listed = (model_path / "train_subset.txt").read_text().split()
    assert listed == [c.image_name for c in s.getTrainCameras()]
