"""Snapshot logic and CLI wiring on CPU, with fake Gaussians and a fake renderer."""

import os
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

# Stub GPU-only modules imported by train.py / gaussian_renderer / models.
sys.modules.setdefault("tinycudann", types.ModuleType("tinycudann"))
for name in ["slang_gaussian_rasterization", "slang_gaussian_rasterization.api",
             "slang_gaussian_rasterization.api.inria_3dgs", "slang_gaussian_rasterization.internal",
             "slang_gaussian_rasterization.internal.alphablend_tiled_slang"]:
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["slang_gaussian_rasterization.api.inria_3dgs"].common_properties_from_inria_GaussianModel = None
sys.modules["slang_gaussian_rasterization.api.inria_3dgs"].common_properties_from_inria_Camera = None
sys.modules["slang_gaussian_rasterization.internal.alphablend_tiled_slang"].render_alpha_blend_tiles_slang_raw = None

from scripts.make_timelapse import build  # noqa: E402
from utils.ply_export import read_vertices  # noqa: E402
from utils.snapshot_utils import snapshot_cameras, take_snapshot  # noqa: E402

N = 1000
SH = 16


class FakeSampledGaussians:
    """Mimics GaussianModel during the probabilistic phase (refinement=False)."""

    refinement = False
    render_extras = {"method": "reinforce"}

    def __init__(self):
        self.xyz = torch.randn(N, 3)
        self.xyz_norm = torch.rand(N, 3)
        self.queried = None

    @property
    def get_xyz(self):
        return self.xyz

    def query_hashgrid_attributes(self, xyz, xyz_norm):
        self.queried = len(xyz)
        m = len(xyz)
        return (torch.randn(m, SH, 3), torch.rand(m, 1), torch.rand(m, 3) * 0.01 + 1e-4,
                torch.nn.functional.normalize(torch.randn(m, 4), dim=-1))


class FakeRefinedGaussians:
    """Mimics GaussianModel after convert2refinement (explicit parameters)."""

    refinement = True
    render_extras = {"method": "default"}

    def __init__(self):
        self._xyz = torch.randn(N, 3)
        self._features_dc = torch.randn(N, 1, 3)
        self._features_rest = torch.randn(N, SH - 1, 3)
        self._opacity = torch.randn(N, 1)
        self._scaling = torch.randn(N, 3)
        self._rotation = torch.randn(N, 4)
        self.opacity_activation = torch.sigmoid
        self.scaling_activation = lambda s, x: torch.nn.functional.softplus(s)
        self.rotation_activation = lambda r: torch.nn.functional.normalize(r, dim=-1)

    @property
    def get_xyz(self):
        return self._xyz


class RecordingRender:
    def __init__(self):
        self.calls = []

    def __call__(self, cam, gaussians, pipe, bg, extras=None, override_scaling=None, override_opacity=None):
        self.calls.append(dict(extras=extras, scaling=override_scaling, opacity=override_opacity, bg=bg))
        return {"render": torch.rand(3, 12, 16)}


@pytest.mark.parametrize("model_cls", [FakeSampledGaussians, FakeRefinedGaussians])
def test_take_snapshot_outputs(tmp_path, model_cls):
    g = model_cls()
    render = RecordingRender()
    cams = [(0, "camA"), (3, "camB")]
    written = take_snapshot(1234, g, cams, render, pipe=None, out_dir=str(tmp_path), max_points=200)

    names = sorted(os.path.basename(p) for p in written)
    assert names == ["iter_001234_gaussians.ply",
                     "iter_001234_view0_centers.png", "iter_001234_view0_render.png",
                     "iter_001234_view3_centers.png", "iter_001234_view3_render.png"]

    # normal render has no overrides; centers render overrides cover all N Gaussians
    plain, dots = render.calls[0], render.calls[1]
    assert plain["scaling"] is None and plain["opacity"] is None
    assert dots["scaling"].shape == (N, 3) and dots["opacity"].shape == (N, 1)
    assert torch.all(dots["opacity"] == 1)
    assert all(c["extras"] is g.render_extras for c in render.calls)
    assert torch.all(plain["bg"] == 0)

    ply = read_vertices(str(tmp_path / "iter_001234_gaussians.ply"))
    assert len(ply) == 200
    assert not any(n.startswith("f_rest_") for n in ply.dtype.names)  # SH degree 0 to stay small
    assert np.all(np.isfinite(ply["opacity"])) and np.all(np.isfinite(ply["scale_0"]))
    if model_cls is FakeSampledGaussians:
        assert g.queried == 200  # only the subset hits the hash grids, not all N


def test_snapshot_cameras_fall_back_to_train_and_skip_bad_indices():
    scene = SimpleNamespace(getTestCameras=lambda: [], getTrainCameras=lambda: ["t0", "t1"])
    assert snapshot_cameras(scene, [0, 1, 5, -1]) == [(0, "t0"), (1, "t1")]
    scene = SimpleNamespace(getTestCameras=lambda: ["v0"], getTrainCameras=lambda: ["t0"])
    assert snapshot_cameras(scene, [0]) == [(0, "v0")]


def test_make_timelapse(tmp_path):
    snap = tmp_path / "snapshots"
    os.makedirs(snap)
    iters = [1, 300, 600]
    for it in iters:
        for kind in ("centers", "render"):
            Image.new("RGB", (16, 12), (it % 255, 0, 0)).save(snap / f"iter_{it:06d}_view0_{kind}.png")
    written = build(str(tmp_path))
    assert sorted(os.path.basename(p) for p in written) == ["contact_view0.png", "timelapse_view0.gif"]

    gif = Image.open(snap / "timelapse_view0.gif")
    assert gif.n_frames == len(iters)
    assert gif.size == (32, 12 + 22)
    assert Image.open(snap / "contact_view0.png").size == (32, (12 + 22) * len(iters))


def test_make_timelapse_without_snapshots(tmp_path):
    with pytest.raises(FileNotFoundError):
        build(str(tmp_path))


def test_cli_flags_reach_training(monkeypatch):
    import run

    captured = {}
    monkeypatch.setattr(run, "training", lambda *a, **k: captured.update(args=a, kwargs=k) or "model")
    run.train_model(cli_args=["--config=truck", "--source_path=/data/truck", "--train_subset=30",
                              "--snapshot_iterations", "1", "300", "600", "--snapshot_views", "0", "2"])

    snap = captured["kwargs"]["snapshot_args"]
    assert snap == {"iterations": [1, 300, 600], "views": [0, 2], "max_points": 500_000, "center_scale": 0.002}
    model_params = captured["args"][0]
    assert model_params.train_subset == 30
    assert captured["args"][2].iterations == 60_000  # scene YAML still applied

    captured.clear()
    run.train_model(cli_args=["--config=truck", "--source_path=/data/truck"])
    assert captured["kwargs"]["snapshot_args"]["iterations"] == []  # off by default
    assert captured["args"][0].train_subset == -1
