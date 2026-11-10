# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Official code for *Eulerian Gaussian Splatting using Hashed Probability Pyramids* (Polansky et al., CVPR 2026, arXiv 2605.29136), cloned as our group's fork for the EEL5406 Computational Photography final project (UF, Fall 2026). `origin` is the group fork (Herschenglime/eulerian-splatting) and `upstream` is euleriansplatting/eulerian-splatting. Upstream has a single commit ("Initial code release").

Project goal: reproduce the paper's results, then run the method on our own captured scenes and find where it breaks. The instructor's feedback was to "push it to its boundaries", not just show it working. With about a week left, the goal is **visual results, not metric sweeps**: pick the experiments whose failures are easiest to see.

Agreed scope, in priority order:
1. Reproduce one benchmark scene (truck or bicycle) next to ground truth, plus a fly-through.
2. One well-captured "hero" scene of our own (tabletop with stuffed animals), with a fly-through.
3. A training timelapse showing samples condensing from a uniform cloud (`--snapshot_iterations`).
4. A views ladder on the hero capture (all → ~30 → ~10 views) using `--train_subset`, with poses from the full COLMAP run.
5. A mirror or glossy object (reflections reconstructed as a fake "room behind the glass").
- Stretch: a person holding still; the whole classroom.
- Cut: underwater, SLAM poses, low light, auto-exposure comparisons, checkerboard calibration. Poses come from COLMAP; phone captures use locked exposure/ISO/white balance/focus, one lens, stabilization off, shutter around 1/250 s or faster.

`docs/` records what we changed in the codebase and why (`docs/viz-tools-changes.md` covers the environment update and visualization tools; `docs/colmap-setup.md` covers COLMAP preprocessing, making `uv sync` build the whole environment, and a review of the HiPerGator work on `main`).

**Branches:** `main` holds a teammate's HiPerGator/B200 work (a notebook workflow, `progress_callback` in `train.py`, `utils/notebook_progress.py`), not merged into `colmap_setup` yet. Its environment is a venv on `/blue` that isn't recorded in git. `docs/colmap-setup.md` §5 has the review and the merge plan.

Course materials are outside this repo, in `/home/pgrau/Sync/School/EEL5406 Comp Phot/Final Project/`:
- `2605.29136v1(1).pdf`: the paper with its appendix. Read it with the PDF tooling.
- `Topic_ Final Projects.pdf`: the assignment post. `Initial Braindump(1).pdf`: the group's planning notes.

## Environment

- **Requires CUDA.** Device `"cuda"` is hard-coded throughout. The attribute grids use tiny-cuda-nn, the rasterizer is written in Slang, and sampling uses a CUDA kernel that is JIT-compiled at runtime. Training runs on Linux machines with NVIDIA GPUs: a lab RTX 3060 Ti (8 GB) for smoke tests, a personal RTX 4070 Super, and HiPerGator for full runs. **Linux x86_64 only**: the uv project doesn't resolve anywhere else. The paper's numbers come from an H200 (~3.5 h for 35K iterations, ~7.25 h for 65K).
- **Run everything through uv:** `uv run python train.py ...`. The plain `python` in the examples below assumes the project venv is active.
- **12 GB cards can't run the default config.** Estimated from the tcnn level sizes (not measured), the default attribute grids (`log2_hashmap_size: 23`) need ~14 GB once gradients and Adam state are counted. `configs/attribute_grids_small.yaml` (2^19 tables, ~1.1 GB) is what the smoke test uses; pass it with `--attrib_config` on small GPUs, along with lower `--num_samples`/`--num_samples_min` and `--resolution`.
- **Setup is `uv sync`, nothing else** (README "Installation" has the user-facing steps). The uv project (`pyproject.toml`, `uv.lock`) is the source of truth. It has Python 3.10 and `torch==2.7.1`/`torchvision==0.22.1` from the cu128 index, moved forward from the README's pins so it runs on Blackwell GPUs. It also includes both CUDA submodules as path dependencies, which `uv sync` builds and keeps. No `uv pip install` steps and no `--inexact`. Prerequisites:
  - **The submodules checked out:** `git submodule update --init --recursive`. `slang-gaussian-rasterization` points to the authors' fork (polanskers), because the custom gradient is implemented there.
  - **A CUDA 12.8 toolkit with `nvcc` on `PATH`,** both for `uv sync` and at runtime. tiny-cuda-nn is compiled during the sync, and the Slang rasterizer and the sampling kernel are JIT-compiled on first use. On HiPerGator, `module load cuda/12.8.1`. On a workstation, install NVIDIA's `cuda-toolkit-12-8` apt package (Ubuntu's own `nvidia-cuda-toolkit` is 12.0, too old) and add `export PATH=/usr/local/cuda-12.8/bin:$PATH` to the shell profile.
  - **Why not from pip:** pip has no CUDA 12 `nvcc` (the `nvidia-cuda-nvcc-cu12` wheel is only `ptxas`/`nvvm`). CUDA 13 has full nvcc wheels, but there's no GPU pycolmap for CUDA 13.
  - **tiny-cuda-nn build settings** live in `pyproject.toml` (`[tool.uv.extra-build-*]`): built against the runtime torch, `setuptools<81`, half precision off, and `TCNN_CUDA_ARCHITECTURES=86;89;100;120`. The fixed arch list matters because uv caches the built wheel and reuses it on every machine sharing `~/.cache/uv`. A new GPU type means adding its arch there. To force a rebuild after changing the submodule: `uv sync --reinstall-package tinycudann`.
  - The README's original Python 3.9 / PyTorch 2.4.1+cu121 setup is what the paper used. Blackwell (B200 = 10.0, RTX PRO 6000 = 12.0) needs CUDA 12.8 and PyTorch ≥ 2.7. HiPerGator's current GPUs are L4, RTX PRO 6000 and B200 (the A100s were retired).
- **8 GB cards** (the lab 3060 Ti) ran out of memory at 2M samples. Use `--num_samples=1000000 --num_samples_min=300000 --images=images_4 --resolution=1` with the small attribute grids, which peaked at 7.0 GB including the desktop.
- `--torch_cuda_arch_list` sets `TORCH_CUDA_ARCH_LIST` before the sampling kernel compiles. The default is `"8.0;8.9"`; use `"8.6"` for an RTX 30xx, `"8.9"` for an RTX 40xx, `"10.0"` for B200, `"12.0"` for RTX PRO 6000. If the compile fails, `sample_cuda` logs a warning and silently falls back to the much slower PyTorch `sample` path, so check the log.

## Commands

**Tests** run on the CPU (no GPU needed) with the GPU-only modules stubbed out:

```bash
uv run pytest                       # all tests
uv run pytest tests/test_snapshots.py -k timelapse   # one test
uv run pyflakes utils scripts tests
```

They cover our additions (PLY export, `--train_subset` through the real COLMAP loader, snapshot logic and CLI wiring), not the CUDA paths.

**On a new GPU machine, in this order:** `git submodule update --init --recursive`, make sure `nvcc --version` reports 12.8, `uv sync` (a few minutes the first time, for tiny-cuda-nn), then `scripts/gpu_smoke_test.sh /path/to/tandt/truck` before anything long. The smoke test is sized for a 12 GB card (small hash tables, half resolution, 2M samples, 700 iterations) and auto-detects the GPU's architecture. It trains a tiny truck run with every new feature switched on, then renders, scores, exports and builds a timelapse, and checks that all expected files exist. `tests/out/test_shapes.ply` (written by the tests) is a converted PLY of RGB axes and a yellow cube for checking the export by eye in https://superspl.at/editor.

The entry points are:

```bash
# full pipeline: train -> render -> metrics
python run.py --config=bicycle --source_path=/path/to/360_v2/bicycle --torch_cuda_arch_list="9.0" --dry_run

# or step by step
python train.py   --config=bicycle --source_path=... --torch_cuda_arch_list="9.0" --dry_run
python render.py  --config=bicycle --source_path=... --model_path=output/bicycle/<run> [--eval_iteration=30000] [--skip_train]
python metrics.py --model_paths output/bicycle/<run> [more runs...]

# resume
python train.py --config=bicycle --source_path=... --dry_run --load_checkpoint=<model_path>/checkpoints/checkpoint_N.pth

# our additions
python train.py ... --train_subset=30                             # views ladder
python train.py ... --snapshot_iterations 1 500 2000 5000 30000 [--snapshot_views 0 3]
python scripts/make_timelapse.py output/<scene>/<run>             # -> snapshots/timelapse_view*.gif, contact_view*.png
python scripts/export_ply.py <run>/point_cloud/iteration_N.ply out.ply [--min_opacity 0.005] [--sh_degree 0] [--max_radius 10]
```

- `--train_subset N` keeps N evenly spaced training views (by filename), applied *after* the test split and *after* the scene normalization, so every run of a views ladder has the same test set and coordinate frame. The kept names go to `<run>/train_subset.txt`. Use it instead of `--load_ncameras`.
- `--snapshot_iterations` (off by default) writes `<run>/snapshots/iter_N_view{k}_render.png`, `..._centers.png` (every Gaussian shrunk to an opaque dot, to show where samples are) and `iter_N_gaussians.ply` (a random subset, standard format, SH degree 0). Views index the test cameras. A failed snapshot logs a warning and never stops training.
- **Saved PLYs are not in the standard 3DGS format**: `save_ply` writes activated opacity (0–1) and world-unit scales, while viewers expect a logit and a log. Always run `scripts/export_ply.py` before opening one in SuperSplat or another viewer; otherwise the scene looks like huge semi-transparent blobs. Also pass `--max_radius` (around 10; the training cameras fit in [-1,1]^3). Background Gaussians sampled in the contracted outer shell un-contract to millions of units away; the NemeSys export's bounding box was ±8.4M. SuperSplat bases its zoom range on the whole bounding box (closest approach ≈ 1e-6 × bound radius / sin(42.5°)), so uncropped, the scene is a dot you can't zoom into. Exports are also rotated upright by default (`UPRIGHT` in `utils/ply_export.py`, −90° about x). The training frame puts the scene's up at −z (camera centers are PCA-aligned and flipped so the OpenCV camera y axis points +z). Standard 3DGS PLYs have up at −y, and SuperSplat rotates every PLY 180° about z on load to show it upright. The rotation covers positions, quaternions and SH coefficients. Pass `--no_upright` to keep the training frame. Snapshot PLYs get the same rotation.

- `--dry_run` turns wandb off. Without it, the run calls `wandb.login()`. Tensorboard logging is used if it is installed.
- `--test_interval=5000` evaluates more often. On headless machines, pass `--disable_visualize`; the test-grid PNG is still saved to the run folder.
- `--checkpoint_iterations` defaults to empty, so no `.pth` checkpoints are written unless you pass it.
- The same entry points can be called from Python with kwargs: `run.train_model(config=..., source_path=...)`, `render_model`, and `evaluate_model`. See `example.ipynb`.

## Config system (run.py `make_args` / `get_combined_args`)

- `--config <name>` is required. It is a **name**, not a path, and resolves to `configs/scenes/<name>.yaml`. Each new scene of our own needs a YAML in that folder.
- Precedence, lowest to highest: argparse defaults in `arguments/__init__.py`, then the scene YAML, then `configs/attribute_grids.yaml` (or the file given by `--attrib_config`), then flags explicitly typed on the CLI, then Python kwargs. Rendering and resuming also apply the run's saved `cfg_args`.
- Boolean params use `store_true`, and several default to True: `eval`, `normalize_cameras`, `contract_xyz`, `noise_samples`, `snap_to_center`, `random_background`. **You cannot turn these off from the CLI.** Set them to `false` in a YAML or pass a kwarg instead. The paper's Table 2 ablations map to these settings:
  - no sample rounding: `snap_to_center: false`
  - no defensive sampling: `noise_samples: false`
  - no opacity or scale regularization: `opacity_reg: 0` or `scale_reg: 0`
  - pathwise (autodiff) gradient: there is no switch. The estimator is chosen inside the rasterizer submodule (`render_method="reinforce"`).
- `num_samples_min` is required; training exits if it is -1. The provided YAMLs set it to 3DGS-MCMC's primitive count for each scene (the fairness lower bound in paper Sec. 5.1). Our scenes have no such count, so choose a value below `num_samples` (default 15M).
- `iterations` is the length of the probabilistic phase, and `refinement_iterations` (5000) runs after it. Outdoor mip-NeRF 360 scenes use 30K+5K; all other scenes use 60K+5K. `save_iterations` should include both the end of the probabilistic phase and the final iteration.
- Memory scales with `num_samples` (its buffers are preallocated) and with `max_render`, the per-frame cap of 7.5M that is hard-coded in `GaussianModel.__init__`. On a smaller GPU, lower `num_samples` and `num_samples_min`. Appendix Table 3 shows the quality/time trade-off.

## Scene data and our own captures

- `Scene` treats the source as a COLMAP scene if `<source_path>/sparse` exists, and as a Blender/NeRF-synthetic scene if `transforms_train.json` exists.
- For COLMAP scenes, the loader reads `sparse/0/{cameras,images,points3D}.bin` (or `.txt`) and the images in `<source_path>/images` (change the folder with `--images`). **Only PINHOLE and SIMPLE_PINHOLE cameras are accepted**, so images must be undistorted. See "COLMAP preprocessing" below.
- **Camera poses are still required.** What the method removes is the SfM *point cloud* initialization: `init_type` defaults to `"random"`. Setting `init_type=sfm` only sets `xyz`, which the first `sample()` overwrites. The COLMAP-initialized pyramid from paper App. A4.5 is not implemented.
- With `init_type=random`, the reader writes an unused PLY to `<source_path>/point_cloud/`, so the dataset folder must be writable. (Upstream crashed if that folder was missing; we now create it.)
- `eval=True` holds out every 8th image, sorted by filename, as the test set.
- `resolution` set to 1, 2, 4, or 8 downsamples the full-resolution `images/` in memory. `-1` caps the width at 1600, and any other value is used as a target width. All training images are preloaded to the GPU.
- For RGBA images, alpha is used as a mask, and masked-out ground-truth pixels are filled with the random training background.
- `--load_ncameras N` truncates the train and test lists separately and keeps N+1 of each (off by one). For few-view experiments use `--train_subset` instead.
- Normalization: training camera centers are PCA-aligned and scaled into [-1,1]^3 (`transform_poses_pca`, applied in `Camera.__init__`). The L∞ contraction then gives that cube 3/4 of the pyramid's 4096 bins per axis and squeezes everything outside it into the outer shell. A consequence of this design (not yet tested): in inside-out captures, such as a camera walking through a room, most of the content falls outside the camera hull and gets coarse resolution.

## COLMAP preprocessing (`scripts/run_colmap.py`)

```bash
uv run python scripts/run_colmap.py data/NemeSys data/nemesys                  # folder of photos (HEIC OK)
uv run python scripts/run_colmap.py "data/nemesys vid.mp4" data/nemesys_vid --fps 2   # video
```

Writes `<out>/images/` (undistorted) and `<out>/sparse/0/` (PINHOLE), plus `images_2/`/`images_4/`, `input/` (the prepared JPEGs), `distorted/` (database and raw model) and `colmap_summary.json` (registered count, unregistered names, reprojection error). Then add `configs/scenes/<name>.yaml` (see `nemesys.yaml`) and train with `--source_path=<out>`. Details and measured results are in `docs/colmap-setup.md`.

- COLMAP runs through **pycolmap** (`pycolmap-cuda12`, Linux only). pip installs Python bindings only, not a `colmap` command. Extraction and matching are forced onto the GPU with no CPU fallback, since HPC time is limited.
- HEIC is converted with `pillow-heif`, and EXIF rotation is applied. `--max_size` (default 3200) caps the long side, because iPhone 15 photos are 48 MP.
- Videos keep the sharpest frame (variance of the Laplacian) per 1/fps-second window.
- One shared OPENCV camera, whose intrinsics COLMAP estimates; no checkerboard calibration.
- Matching is exhaustive up to 300 images. Sequential matching has no loop closure without a vocab tree: it split the 96-frame NemeSys video into 4 models, while exhaustive matching registered all 96.
- If several disconnected models come out, the largest is kept with a warning. Look at `unregistered_images` in the summary; `--skip_input` re-runs SfM after you prune `input/` by hand.
- Unverified: whether pycolmap's GPU SIFT runs on Blackwell (B200, RTX PRO 6000). Check on the first HiPerGator run.

For the views ladder, run COLMAP once on **all** images of a capture and subsample afterwards with `--train_subset`, so pose quality doesn't change between runs.

## Architecture: paper to code

**One training iteration in the probabilistic phase (`train.py` `training`):**
1. `GaussianModel.sample` draws `num_samples` centers from the pyramid.
2. Each center is snapped to the center of its finest bin, and duplicates are removed. If fewer than `num_samples_min` unique centers remain, it resamples for up to 8 rounds.
3. `render()` frustum-culls the centers (10% margin, random cap of 7.5M), queries the attribute hash grids at the visible centers, and rasterizes with Slang.
4. The loss is L1 + D-SSIM plus the opacity, scale, and SH regularizers.
5. Backward, then an Adam step on the hash grids and the pyramid logits.

After `iterations`, `convert2refinement` turns one final sample into explicit per-Gaussian parameters. Positions are frozen through `position_lr=0`. Training then continues as plain 3DGS optimization, with no regularizers and no densification.

| Paper | Code |
|---|---|
| Hashed probability pyramid (Eq. 4–6, App. A1) | `models/sampling_modules/__init__.py` `HashedProbabilityPyramid`: 12 levels with `res=2` each (finest grid 4096³). Logits `_pxyz[l]` have shape `[ncomp,2,2,2]` and are softmax-normalized per 2×2×2 block. Levels 0–7 are dense (the "tiled" map is the identity there). Levels 8–11 are hashed into 2^21 blocks using the Instant-NGP primes. |
| Sampling (Eq. 7–8) | `sample_cuda` calls `csrc/sample_pyramid_cuda_kernel.cu`. `sample` is the pure-PyTorch reference version. |
| Contraction (Eq. 9) | `utils/coord_utils.py` `contract_cube`/`inv_contract_cube`, wrapped by `GaussianModel.contract`/`inv_contract` |
| Control-variate estimator (Eq. 13, 37, 42–43) | `get_visible_xyz_and_attributes` detaches the positions and appends `log_prob` as a 4th channel. With `render_method="reinforce"`, the rasterizer's backward sends `o_i·∂L/∂o_i` into that channel. **That backward is in the slang submodule, not in this repo.** |
| Loss (Eq. 15) | Inline in `train.py` |
| Defensive sampling (Sec. 5.1) | Noise branch of `GaussianModel.sample` (`prop_noise`, `add_noise_*`) |
| Sample rounding (Eq. 16) / duplicate removal | `snap_to_center`; `first_idx_and_num_dups_from_bins` |
| Attribute hash grids (Sec. 5.2, App. A3.1) | `models/hashgrid_modules/__init__.py`: three tcnn `NetworkWithInputEncoding`s (color, opacity, scale+rotation), configured by `configs/attribute_grids.yaml`. SH damping is in `sh_weights`. The contraction-Jacobian scale factor is `scaling_contract`. |
| Frustum culling, 7.5M cap (Sec. 5.5, App. A4.1) | `gaussians_in_frustum`, `max_render` |
| Refinement (App. A4.2) | `convert2refinement` |
| Scene normalization (Sec. 5.4) | `utils/camera_utils.py` `transform_poses_pca` |

## Where the code differs from the paper (checked by reading the code)

- Sec. 5.3 gives B = 2^18 with levels 0–6 dense. The code keeps levels 0–7 dense and gives levels 8–11 2^21 blocks each. The code's ~86M parameters match the count in App. A1.1, not Sec. 5.3.
- The code's `acontract=1.5` on a [-2,2] cube is equivalent to the paper's a = 3/4 on [-1,1].
- The defensive-sampling noise is `factor * torch.rand(...)`, which is uniform and non-negative, not the zero-mean Gaussian noise described in Sec. 5.1.
- `z_threshold` (near-plane cull, App. A4.3; the truck and train YAMLs set it to 0.05) is stored on `GaussianModel` but never passed to the renderer, so changing it has no effect.
- `center_xyz` (in `attribute_grids.yaml`) and `anisotropy_loss` are unused.
- `--load_ply` without `--refinement` crashes, because `Scene.load` calls `gaussians.sample()` without its required `iteration` argument.
- Resuming from a checkpoint saved during refinement fails. The model is rebuilt in probabilistic mode and `restore` expects the 5-tuple. Only checkpoint at iterations ≤ `iterations`.

## Outputs

Each run is written to `<save_path or repo>/output/<basename(source_path)>/<run_num>_<id>/` (`output/` is gitignored). It contains:
- `cfg_args`, `cameras.json`, `input.ply`
- `point_cloud/iteration_N.ply`: one frozen sample with its attributes
- `learned_grids/iteration_N.pt`: the pyramid and hash grids (probabilistic phase only)
- `checkpoints/` and the test-grid PNGs
- `renders/{train,test}/ours_N/{renders,gt}/`
- `results.json` and `per_view.json`

`render.py` always loads a PLY, so it renders a frozen sample. By default it uses the latest iteration, which is post-refinement. Pass `--eval_iteration=<iterations>` to get the pre-refinement column from paper Table 1. `metrics.py` scores every `ours_N` folder it finds. Test metrics printed during the probabilistic phase come from a single random sample.
