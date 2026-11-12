"""scripts/e2e.py orchestration on CPU, with COLMAP, training, timelapse and export stubbed out."""

import os
import sys
import types
from argparse import Namespace

import pytest
import torch
import yaml

from scripts import e2e


class FakeRun(types.ModuleType):
    """Stands in for run.py: resolves iteration counts like make_args and records train_model."""

    def __init__(self):
        super().__init__("run")
        self.calls = []

    def make_args(self, cli_args=None, **overrides):
        args = Namespace(iterations=60_000, refinement_iterations=5000)
        for k, v in overrides.items():
            setattr(args, k, v)
        cli = list(cli_args or [])
        for key in ("iterations", "refinement_iterations"):
            if f"--{key}" in cli:
                setattr(args, key, int(cli[cli.index(f"--{key}") + 1]))
        return args

    def train_model(self, cli_args=None, **kwargs):
        self.calls.append({"cli_args": cli_args, **kwargs})
        return kwargs["model_path"]


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A temp repo root with one photo capture and a config; returns a namespace of handles."""
    monkeypatch.setattr(e2e, "REPO", str(tmp_path))
    capture = tmp_path / "photos"
    capture.mkdir()
    config = tmp_path / "cap.yaml"
    config.write_text(yaml.safe_dump({"capture": str(capture), "name": "cap", "train": {"num_samples": 2_000_000}}))

    fake_run = FakeRun()
    monkeypatch.setitem(sys.modules, "run", fake_run)
    colmap_calls, timelapse_calls, export_calls = [], [], []
    import scripts
    for name, calls in (("run_colmap", colmap_calls), ("make_timelapse", timelapse_calls)):
        stub = types.SimpleNamespace(main=lambda argv, calls=calls: calls.append(argv))
        monkeypatch.setitem(sys.modules, f"scripts.{name}", stub)
        monkeypatch.setattr(scripts, name, stub, raising=False)  # `from scripts import x` reads this first

    def fake_convert(src, dst, **kwargs):
        export_calls.append((src, dst, kwargs))
        open(dst, "wb").close()
        return 10, 10

    import utils.ply_export
    monkeypatch.setattr(utils.ply_export, "convert_file", fake_convert)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a: (8, 6))
    return types.SimpleNamespace(root=tmp_path, config=str(config), capture=str(capture), run=fake_run,
                                 colmap=colmap_calls, timelapse=timelapse_calls, export=export_calls)


def test_snapshot_schedule():
    sched = e2e.snapshot_schedule(10, 60_000, 5000)
    assert sched[0] == 1 and 60_000 in sched and sched[-1] == 65_000
    assert sched == sorted(set(sched)) and len(sched) == 10
    assert e2e.snapshot_schedule(10, 600, 0)[-1] == 600


def test_precedence_preset_yaml_cli():
    cfg = {**e2e.DEFAULTS, "preset": "small", "train": {"num_samples": 2_000_000}, "holdout": True}
    overrides, last = e2e.train_overrides(cfg, [], FakeRun(), "8.6")
    assert overrides["num_samples"] == 2_000_000            # YAML beats preset
    assert overrides["images"] == "images_4"                # preset beats code defaults
    assert overrides["eval"] is True and overrides["torch_cuda_arch_list"] == "8.6"
    assert overrides["save_iterations"] == [60_000, 65_000] and last == 65_000
    assert overrides["checkpoint_iterations"][-1] < 65_000 and overrides["keep_last_checkpoint"]

    overrides, last = e2e.train_overrides(cfg, ["--num_samples", "3000000", "--iterations", "600"], FakeRun(), "8.6")
    assert "num_samples" not in overrides                   # given after `--`: left to the CLI
    assert last == 5600 and overrides["snapshot_iterations"][-1] == 5600


def test_new_run_end_to_end(env):
    model_path = e2e.main([env.config, "--", "--iterations", "600"])
    assert env.colmap == [[env.capture, os.path.join(env.root, "data", "cap")]]
    call = env.run.calls[0]
    assert call["cli_args"] == ["--iterations", "600"] and "iterations" not in call
    assert call["source_path"] == os.path.join(env.root, "data", "cap")
    assert call["model_path"] == model_path and model_path.startswith(os.path.join(env.root, "output", "cap"))
    assert call["eval"] is False and call["dry_run"] and call["disable_visualize"]
    assert env.timelapse == [[model_path]]
    src, dst, kwargs = env.export[0]
    assert src == os.path.join(model_path, "point_cloud", "iteration_5600.ply")
    assert dst == os.path.join(model_path, "cap.ply") and kwargs == {}
    saved = yaml.safe_load(open(os.path.join(model_path, "e2e_config.yaml")))
    assert saved["passthrough"] == ["--iterations", "600"] and saved["config"]["name"] == "cap"


def test_colmap_reused_unless_redo(env):
    os.makedirs(os.path.join(env.root, "data", "cap", "sparse", "0"))
    e2e.main([env.config])
    assert env.colmap == []
    e2e.main([env.config, "--redo_colmap", "--holdout"])
    assert len(env.colmap) == 1 and env.run.calls[-1]["eval"] is True


def test_resume_uses_saved_config_and_latest_checkpoint(env):
    model_path = e2e.main([env.config, "--preset", "full", "--", "--iterations", "600"])
    os.makedirs(os.path.join(model_path, "checkpoints"))
    for it in (200, 1000, 400):
        open(os.path.join(model_path, "checkpoints", f"checkpoint_{it}.pth"), "wb").close()
    assert e2e.main(["--resume", model_path]) == model_path
    first, resumed = env.run.calls
    assert resumed["load_checkpoint"] == os.path.join(model_path, "checkpoints", "checkpoint_1000.pth")
    assert resumed["cli_args"] == first["cli_args"] and resumed["images"] == "images_2"
    assert {k: v for k, v in resumed.items() if k != "load_checkpoint"} == first
    assert len(env.colmap) == 1                             # resume never re-runs COLMAP


def test_colmap_argv_and_config_errors(tmp_path):
    cfg = {**e2e.DEFAULTS, "capture": "/c", "colmap": {"fps": 3, "skip_input": True, "downsample": [2, 4],
                                                       "matcher": None}}
    assert e2e.colmap_argv(cfg, "/d") == ["/c", "/d", "--fps", "3", "--skip_input", "--downsample", "2", "4"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("capture: x\nname: y\niterations: 5\n")
    with pytest.raises(SystemExit, match="unknown keys"):
        e2e.load_config(str(bad))
