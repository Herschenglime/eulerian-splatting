"""One command from a capture to a viewable splat:
COLMAP -> training (with progress snapshots) -> timelapse -> upright, standard-format PLY.

Usage:
    python scripts/e2e.py configs/e2e/<name>.yaml [--preset small|full] [--holdout] [--redo_colmap]
                          [-- <extra train.py flags>]
    python scripts/e2e.py --resume output/<name>/<run>        # continue from the run's last checkpoint

Settings, lowest to highest priority: DEFAULTS below, the preset, the YAML file, the flags above,
then anything after `--` (passed to train.py as is). The resolved settings are saved as
<run>/e2e_config.yaml, which --resume reads back. See docs/e2e-pipeline.md.

Outputs:
    data/<name>/                    COLMAP dataset (reused on later runs unless --redo_colmap)
    output/<name>/<timestamp>/      the training run, with:
        <name>.ply                  final splat: standard 3DGS format, upright, nothing removed
        snapshots/                  progress renders, timelapse_view*.gif, contact_view*.png
        point_cloud/, checkpoints/  raw training outputs (the latest checkpoint only)
"""

import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # needed to fit 8 GB cards

import argparse  # noqa: E402
import copy  # noqa: E402
import glob  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DEFAULTS = {
    "capture": None,          # folder of photos, or a video file
    "name": None,             # dataset -> data/<name>, runs -> output/<name>/<timestamp>
    "preset": "small",
    "holdout": False,         # True: every 8th image becomes a test view (compared, never trained on)
    "colmap": {},             # run_colmap.py options, e.g. {fps: 2, max_size: 3200}
    "train": {},              # any train.py option, e.g. {iterations: 30000, train_subset: 30}
    "snapshots": {"count": 10, "views": [0], "max_points": 2_000_000},
    "checkpoint_every": 5000,
    "export": {},             # utils/ply_export.convert_file options, e.g. {max_radius: 1000}
}

PRESETS = {
    # Verified on an 8 GB RTX 3060 Ti (peak 7.0 GB including the desktop).
    "small": {"attrib_config": os.path.join(REPO, "configs", "attribute_grids_small.yaml"),
              "images": "images_4", "resolution": 1,
              "num_samples": 1_000_000, "num_samples_min": 300_000},
    # The paper's hash grids and sample count (code defaults); for HiPerGator-class GPUs.
    "full": {"images": "images_2", "resolution": 1},
}


def load_config(path):
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    unknown = set(user) - set(DEFAULTS)
    if unknown:
        sys.exit(f"{path}: unknown keys {sorted(unknown)} (known: {sorted(DEFAULTS)})")
    cfg = copy.deepcopy(DEFAULTS)
    for key, value in user.items():
        cfg[key] = {**cfg[key], **(value or {})} if isinstance(cfg[key], dict) else value
    return cfg


def explicit_keys(cli_args):
    """Option names given in a train.py argument list (same rule as run.make_args)."""
    return {a.lstrip("-").split("=")[0].replace("-", "_") for a in cli_args if a.startswith("-")}


def snapshot_schedule(count, iterations, refinement_iterations):
    """Log-spaced from 1 to the end of the probabilistic phase, plus the end of refinement."""
    sched = {int(round(i)) for i in np.geomspace(1, iterations, max(count - 1, 2))} | {iterations}
    if refinement_iterations > 0:
        sched.add(iterations + refinement_iterations)
    return sorted(sched)


def colmap_argv(cfg, data_dir):
    argv = [cfg["capture"], data_dir]
    for key, value in cfg["colmap"].items():
        if value is True:
            argv.append(f"--{key}")
        elif isinstance(value, (list, tuple)):
            argv += [f"--{key}", *map(str, value)]
        elif value not in (False, None):
            argv += [f"--{key}", str(value)]
    return argv


def latest_checkpoint(model_path):
    paths = glob.glob(os.path.join(model_path, "checkpoints", "checkpoint_*.pth"))
    if not paths:
        sys.exit(f"No checkpoints in {model_path}/checkpoints to resume from")
    return max(paths, key=lambda p: int(re.search(r"checkpoint_(\d+)\.pth$", p).group(1)))


def train_overrides(cfg, passthrough, run, gpu_arch):
    """kwargs for run.train_model. Keys given after `--` are left out, so the CLI wins."""
    overrides = {**PRESETS[cfg["preset"]], **cfg["train"]}
    explicit = explicit_keys(passthrough)
    args = run.make_args(cli_args=passthrough, config="capture",
                         **{k: v for k, v in overrides.items() if k not in explicit})
    iters, refine = args.iterations, args.refinement_iterations
    last = iters + refine
    snaps = cfg["snapshots"]
    derived = {
        "eval": bool(cfg["holdout"]),
        "torch_cuda_arch_list": gpu_arch,
        "save_iterations": sorted({iters, last}),
        "test_iterations": [iters],
        "snapshot_iterations": snapshot_schedule(snaps["count"], iters, refine),
        "snapshot_views": snaps["views"],
        "snapshot_max_points": snaps["max_points"],
        "checkpoint_iterations": list(range(cfg["checkpoint_every"], last, cfg["checkpoint_every"]))
                                 if cfg["checkpoint_every"] else [],
        "keep_last_checkpoint": True,
    }
    merged = {**derived, **overrides}
    return {k: v for k, v in merged.items() if k not in explicit}, last


def step(title):
    print(f"\n==== {title}", flush=True)
    return time.perf_counter()


def done(t0):
    print(f"==== done in {time.perf_counter() - t0:.0f}s", flush=True)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    ours, passthrough = (argv[:argv.index("--")], argv[argv.index("--") + 1:]) if "--" in argv else (argv, [])
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", nargs="?", help="YAML config (see configs/e2e/example.yaml)")
    parser.add_argument("--preset", choices=sorted(PRESETS))
    parser.add_argument("--holdout", action="store_true", help="hold out every 8th image as a test view")
    parser.add_argument("--redo_colmap", action="store_true", help="re-run COLMAP even if data/<name> exists")
    parser.add_argument("--resume", metavar="RUN_DIR", help="continue a run from its latest checkpoint")
    args = parser.parse_args(ours)

    if args.resume:
        model_path = os.path.abspath(args.resume)
        with open(os.path.join(model_path, "e2e_config.yaml")) as f:
            saved = yaml.safe_load(f)
        cfg, passthrough = saved["config"], saved["passthrough"]
    else:
        if not args.config:
            parser.error("give a config file, or --resume RUN_DIR")
        cfg = load_config(args.config)
        if args.preset:
            cfg["preset"] = args.preset
        if args.holdout:
            cfg["holdout"] = True
        if not cfg["capture"] or not cfg["name"]:
            sys.exit(f"{args.config}: 'capture' and 'name' are required")
        if not os.path.exists(cfg["capture"]):
            sys.exit(f"capture not found: {cfg['capture']}")
        cfg["capture"] = os.path.abspath(cfg["capture"])
        model_path = os.path.join(REPO, "output", cfg["name"], time.strftime("%Y%m%d-%H%M%S"))
    data_dir = os.path.join(REPO, "data", cfg["name"])
    started = time.perf_counter()

    if not args.resume and (args.redo_colmap or not os.path.isdir(os.path.join(data_dir, "sparse", "0"))):
        t0 = step(f"COLMAP: {cfg['capture']} -> {data_dir}")
        from scripts import run_colmap
        run_colmap.main(colmap_argv(cfg, data_dir))
        done(t0)
    else:
        print(f"\n==== COLMAP: reusing {data_dir} (--redo_colmap to re-run)")

    t0 = step(f"training -> {model_path}")
    import torch
    import run
    major, minor = torch.cuda.get_device_capability()
    overrides, last_iter = train_overrides(cfg, passthrough, run, f"{major}.{minor}")
    os.makedirs(model_path, exist_ok=True)
    if not args.resume:
        with open(os.path.join(model_path, "e2e_config.yaml"), "w") as f:
            yaml.safe_dump({"config": cfg, "passthrough": passthrough, "train_overrides": overrides}, f,
                           sort_keys=False)
    if args.resume:
        overrides["load_checkpoint"] = latest_checkpoint(model_path)
        print(f"resuming from {overrides['load_checkpoint']}")
    run.train_model(cli_args=passthrough, config="capture", source_path=data_dir, model_path=model_path,
                    dry_run=True, disable_visualize=True, **overrides)
    done(t0)

    t0 = step("timelapse")
    from scripts import make_timelapse
    make_timelapse.main([model_path])
    done(t0)

    t0 = step("export")
    from utils.ply_export import convert_file
    final_ply = os.path.join(model_path, f"{cfg['name']}.ply")
    n_in, n_out = convert_file(os.path.join(model_path, "point_cloud", f"iteration_{last_iter}.ply"), final_ply,
                               **cfg["export"])
    print(f"{final_ply}: {n_out:,} of {n_in:,} Gaussians, {os.path.getsize(final_ply) / 1e6:.0f} MB")
    done(t0)

    print(f"\nFinished in {(time.perf_counter() - started) / 60:.1f} min")
    print(f"  splat:     {final_ply}")
    print(f"  timelapse: {os.path.join(model_path, 'snapshots')}/timelapse_view*.gif, contact_view*.png")
    print(f"  COLMAP:    {os.path.join(data_dir, 'colmap_summary.json')}")
    return model_path


if __name__ == "__main__":
    main()
