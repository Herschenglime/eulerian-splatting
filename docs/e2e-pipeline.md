# One command from a capture to a viewable splat (`scripts/e2e.py`)

```bash
uv run python scripts/e2e.py configs/e2e/nemesys.yaml
```

It runs four steps:
1. **COLMAP** (`scripts/run_colmap.py`): photos or a video become a dataset in `data/<name>/`.
2. **Training** (`train.py`): with progress snapshots and periodic checkpoints.
3. **Timelapse** (`scripts/make_timelapse.py`): a GIF and a contact sheet of the snapshots.
4. **Export** (`utils/ply_export.py`): `output/<name>/<run>/<name>.ply`, in the standard 3DGS format and rotated upright, with nothing removed.

Rendering held-out views and computing metrics aren't part of it. `render.py` and `metrics.py` work on an e2e run folder afterwards (`--config=capture --source_path=data/<name> --model_path=<run>`).

## Config

A YAML file per capture. Only `capture` and `name` are required. `configs/e2e/example.yaml` lists every key with its default and a comment, and `configs/e2e/nemesys.yaml` is a real one.

| Key | Default | Meaning |
|---|---|---|
| `capture` | (required) | folder of photos (JPEG, PNG, HEIC), or a video file |
| `name` | (required) | dataset in `data/<name>`, runs in `output/<name>/<timestamp>` |
| `preset` | `small` | GPU sizing (below) |
| `holdout` | `false` | hold out every 8th image as a test view (below) |
| `colmap` | `{}` | `run_colmap.py` options, e.g. `{fps: 2, max_size: 3200}` |
| `train` | `{}` | any `train.py` option, e.g. `{iterations: 30000, train_subset: 30}` |
| `snapshots` | `{count: 10, views: [0], max_points: 2000000}` | progress snapshots |
| `checkpoint_every` | `5000` | checkpoint interval, for `--resume`; `0` turns them off |
| `export` | `{}` | `convert_file` options: `max_radius`, `min_opacity`, `max_points`, `sh_degree`, `upright` |

**Settings precedence**, lowest to highest:
1. the code defaults;
2. `configs/scenes/capture.yaml`, the shared scene config for our captures (60K + 5K iterations);
3. the preset;
4. the YAML's `train:`;
5. the flags `--preset`, `--holdout` and `--redo_colmap`;
6. anything after `--`, which goes to `train.py` as is.

The resolved settings are saved as `<run>/e2e_config.yaml`.

```bash
uv run python scripts/e2e.py configs/e2e/nemesys.yaml --holdout -- --train_subset 30
```

## Presets

| Preset | For | Settings |
|---|---|---|
| `small` | 8–12 GB GPUs | small hash grids (`configs/attribute_grids_small.yaml`), `images_4`, 1M samples (minimum 300K) |
| `full` | HiPerGator (B200, RTX PRO 6000) | the paper's hash grids, `images_2`, 15M samples |

`small` is the setting verified on the lab RTX 3060 Ti, where it peaked at 7.0 GB including the desktop. `full` hasn't been run through this tool yet.

## Held-out views (`holdout`)

**Off by default: training uses every registered image.**

With `holdout: true`, every 8th image (sorted by filename) is kept out of training. Training then renders those views at each evaluation and saves them next to the real photos: `test_visualization_*.png`, with the real photo, the render and the depth.

**This never changes training.** The optimiser only sees training views, the scene's coordinate frame is computed from training cameras only, and the evaluation runs without gradients and feeds nothing back. There's no early stopping and no checkpoint selection. Holding out only costs training data, in exchange for honest comparisons on views no run trained on. Use it for experiments that compare runs, e.g. the same capture trained on all, ~30 and ~10 views (`-- --train_subset 30`).

COLMAP always uses every image, so held-out photos still help compute the camera poses.

## Snapshots and disk use

Snapshots are taken at `count` iterations, log-spaced from 1 to the end of the first ("probabilistic") phase so the early condensing gets more frames, plus the end of refinement. Each one saves:
- a render and a "centers" image (every Gaussian drawn as a dot) from the first test view, or the first training view without holdout. These use all Gaussians.
- a PLY of up to `max_points` Gaussians, a random subset, with base colour only (spherical-harmonics degree 0, 68 bytes per Gaussian). 2M Gaussians come to 136 MB, so ~1.4 GB per run.

A subset never looks like the finished splat, because each Gaussian's size and opacity assume its neighbours exist. It shows where the Gaussians are.

Other outputs:
- **Checkpoints:** only the latest is kept. Measured on the `small` preset: 1.8 GB in the first phase and 676 MB in refinement (1M Gaussians). With the `full` preset's hash grids, a first-phase checkpoint should be ~12 GB: ~986M parameters × 12 bytes (each parameter plus Adam's two optimiser states).
- **Final PLY:** 248 bytes per Gaussian, all colour terms kept (248 MB for 1M).

## Resume

```bash
uv run python scripts/e2e.py --resume output/<name>/<run>
```

It continues from the run's latest checkpoint, with the settings in its `e2e_config.yaml`, then rebuilds the timelapse and the export. Checkpoints from either training phase work. Three bugs had to be fixed for this:
- `run.get_combined_args` swallowed a `load_checkpoint` keyword, so resuming from Python silently started over.
- Refinement-phase checkpoints crashed on restore.
- First-phase checkpoints restored `xyz` and `xyz_norm` swapped.

The random-number state isn't saved, so a resumed run isn't bit-identical to an uninterrupted one. In testing, a run resumed from iteration 525 of 700 finished at training-view PSNR 16.51, against 16.65 uninterrupted.

## COLMAP reuse and capture checks

- **Reuse:** if `data/<name>/sparse/0` exists, COLMAP is skipped (`--redo_colmap` forces it). The time saved is small. The point is that repeated runs of a capture share identical camera poses.
- **Capture checks:** `run_colmap.py` writes them into `data/<name>/colmap_summary.json` and prints them as warnings. Nothing is removed.
  - **`capture_settings` (photos):** each EXIF exposure, ISO, white-balance, aperture, focal-length and lens field, with its distinct values. It warns when one varies. The NemeSys photos switched between 1/60 s at ISO 200 (41 photos) and 1/40 s at ISO 160 (23 photos), a sign exposure wasn't locked. Phone videos carry no per-frame exposure data, so this isn't available for them.
  - **`blurry_images`:** sharpness (variance of the Laplacian) below half the capture's median. On NemeSys, the flagged photos include IMG_6232, which is visibly motion-blurred and one of the 4 that didn't register.
  - **`static_frames` (video):** kept frames whose mean absolute difference from the previous kept frame is below 2 (on a 0–255 scale, 64 px wide). That catches moments where the camera stopped. For calibration, raw frames 1/60 s apart in the NemeSys video differ by a median of 4.3 and as little as 0.4. The threshold hasn't been tested on footage where the camera truly stops.

## Cropping

The export keeps everything by default. To drop the far background tail:
- run `scripts/crop_ply.py in.ply out.ply --max_radius R` on any PLY, or
- set `export: {max_radius: R}`.

`utils/ply_crop.py` explains how to choose R. In short, 1000 removes only the points millions of units out that stop SuperSplat from zooming in.

## Future work

- **A GUI for reviewing and selecting images and video frames** before or after COLMAP. It would build on `blurry_images`, `static_frames` and per-image registration. As far as we know, COLMAP's own GUI inspects reconstructions (registered images, matches) but doesn't help select frames; that hasn't been checked.
- **Fly-through rendering.**
- **Choosing a desktop viewer for large splats** (e.g. SIBR, the original 3DGS viewer, which expects a different folder layout).
- **A SLURM wrapper around `e2e.py`** for HiPerGator.
