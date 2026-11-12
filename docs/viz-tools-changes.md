# Changes: environment update and visualization tools

This covers the first set of changes to our fork, committed as `aaface5` ("init claude run of utility creation"). It doesn't cover later commits, such as the COLMAP dependency added in `41e49cc`. The macOS support described in section 1 was later dropped, and the CUDA submodules are now built by `uv sync`: see `docs/colmap-setup.md`.

## Why

We have about a week to produce visual results for the final project:
- a fly-through of a reconstructed scene,
- a timelapse of training,
- a comparison of how quality drops as the number of camera views drops.

The upstream code could do none of these out of the box:
- Its saved Gaussian files don't open correctly in standard viewers.
- It had no clean way to train on fewer views.
- It had no way to capture training progress.

Its pinned versions (Python 3.9, PyTorch 2.4.1 with CUDA 12.1) also don't support the Blackwell GPUs on HiPerGator.

## 1. Environment: a uv project for Blackwell GPUs

**Files:** `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore`

- **What changed:**
  - The project is now set up with uv on Python 3.10, with `torch==2.7.1` and `torchvision==0.22.1`.
  - On Linux these come from PyTorch's CUDA 12.8 index. On macOS they resolve to CPU builds, which are only used to run the tests.
  - The rest of the README's dependencies are listed without the old version pins.
- **Why:** Blackwell GPUs (B200 = compute capability 10.0, RTX PRO 6000 = 12.0) need CUDA 12.8 and PyTorch 2.7 or newer. HiPerGator's A100s have been retired, and its current GPUs are L4, RTX PRO 6000 and B200.
- **The CUDA submodules stay out of the lockfile.** `tiny-cuda-nn` and `slang-gaussian-rasterization` are compiled against whichever torch is installed, so they're built separately after `uv sync`:
  ```bash
  uv sync
  uv pip install --no-build-isolation ./submodules/slang-gaussian-rasterization
  TCNN_HALF_PRECISION=0 uv pip install --no-build-isolation ./submodules/tiny-cuda-nn/bindings/torch
  ```
- **`slangtorch` is restricted to non-macOS platforms.** It has no macOS wheels, and the rasterizer only runs on a GPU anyway.
- **`.gitignore` fixes:**
  - The existing `*.lock` rule would have hidden `uv.lock` from git, so we added an exception for it.
  - `.venv/` and `tests/out/` are now ignored.

## 2. Exporting PLYs that viewers can open

**Files:** `utils/ply_export.py`, `scripts/export_ply.py`

- **The problem:** `GaussianModel.save_ply` writes *activated* values: opacity in [0, 1] and scales in world units. Standard 3DGS viewers like SuperSplat expect opacity as a logit and scales as logs, and apply sigmoid and exp themselves. Opened directly, our scenes show up as huge semi-transparent blobs. Positions, colors (spherical harmonics) and rotations already use the standard layout.
- **What the converter does:**
  - Opacity becomes `log(o / (1 - o))`, clipped so the result stays finite.
  - Each scale becomes `log(s)`.
  - Every other field is copied unchanged.
- **Safety check:** it refuses to convert a file that already looks converted (opacity outside [0, 1] or non-positive scales), unless you pass `--force`.
- **Options to shrink files for browser viewers,** which struggle above about 1 GB:
  - `--min_opacity` drops nearly invisible Gaussians.
  - `--max_points N` keeps only the N most opaque.
  - `--sh_degree 0` drops view-dependent color, making the file about 3.5× smaller.

## 3. `--train_subset N` for the views comparison

**Files:** `utils/subset_utils.py`, `arguments/__init__.py`, `scene/dataset_readers.py`, `scene/__init__.py`

- **The problem:** the existing `--load_ncameras` option keeps N+1 views instead of N, and it also shrinks the test set. Runs with different view counts would then be scored on different test images.
- **What `--train_subset N` does:**
  - It keeps N training views evenly spaced by filename. For video frames, that means evenly spaced along the camera path.
  - It's applied *after* the train/test split and *after* the scene's coordinate normalization is computed, so every run in the comparison shares the same test set and the same coordinate frame.
  - The kept image names are written to `<run>/train_subset.txt`.
- **A loader bugfix in the same files:** with the default random initialization, the loader writes a placeholder point cloud into `<dataset>/point_cloud/` but never creates that folder. It crashed on fresh mip-NeRF 360 and Tanks & Temples downloads, and now it creates the folder.

## 4. Training snapshots and timelapse

**Files:** `utils/snapshot_utils.py`, `train.py`, `run.py`, `scripts/make_timelapse.py`

- **Why:** the paper's central idea is that Gaussians start spread uniformly and probability "flows" toward surfaces during training (its Figure 1). A timelapse shows this directly. Saving full PLYs early in training would cost about 3–4 GB each, so snapshots save small images instead.
- **New flags:** `--snapshot_iterations` (off by default), `--snapshot_views`, `--snapshot_max_points` and `--snapshot_center_scale`.
- **What each snapshot writes,** into `<run>/snapshots/`, for each chosen view:
  - a normal render;
  - a "centers" render, where every Gaussian is drawn as a small opaque dot to show where samples are. It uses the renderer's existing scaling and opacity overrides.
  - a PLY of a random subset of the current Gaussians, in standard format with spherical harmonics reduced to degree 0.
- **Timing and safety:** snapshots run after the optimizer step and work in both the sampling and refinement phases. A failed snapshot logs a warning instead of stopping a multi-hour run.
- **`scripts/make_timelapse.py <run>`** turns the snapshots into an animated GIF (centers next to render, labeled with the iteration number) and a contact sheet with one row per iteration.

## 5. Tests and the GPU smoke test

**Files:** `tests/`, `scripts/gpu_smoke_test.sh`

- **CPU tests** (`uv run pytest`, 22 tests) stub out the GPU-only modules. They check:
  - **Export:** a viewer's sigmoid/exp recovers the original values, the field layout matches standard 3DGS, the double-convert check and the size options work, and the command-line tool works end to end.
  - **View subset:** the real COLMAP loader runs on a synthetic 40-camera scene. The test set and coordinate frame are unchanged, exactly N views are kept, the missing `point_cloud/` folder no longer crashes, and `Scene` writes `train_subset.txt`.
  - **Snapshots:** both training phases are tested with stand-in Gaussians and a stand-in renderer. They check the expected file names, that the overrides cover every Gaussian, that only the sampled subset queries the hash grids, the timelapse assembly, and that the new flags reach `training()`.
- **`tests/out/test_shapes.ply`** is written by the tests: RGB axes and a yellow cube in converted format. Open it in https://superspl.at/editor to check the export by eye. Sharp lines mean the conversion is right; fog or blobs mean it isn't.
- **`scripts/gpu_smoke_test.sh <truck> [arch]`** is for the first run on any GPU machine and takes about 15 minutes. It:
  - trains a tiny Tanks & Temples truck run with every new feature on;
  - renders before and after refinement;
  - scores, exports and builds a timelapse;
  - checks that every expected file exists.

  This is the only check of the GPU code paths.

## 6. Documentation

`CLAUDE.md` now records:
- the agreed project scope,
- the new environment,
- the test commands,
- how to use the new flags and scripts,
- the warning that saved PLYs need `export_ply.py` before viewing.

## Not yet verified

- None of this has run on a GPU yet; that's what `scripts/gpu_smoke_test.sh` is for.
- The SuperSplat visual check of `tests/out/test_shapes.ply` hasn't been done.
- The newer versions of `slangtorch` and `tiny-cuda-nn` haven't been built against torch 2.7.1 on a Blackwell card.
