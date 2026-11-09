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

Course materials are outside this repo, in `/Users/paul/School/EEL5406 Comp Phot/Final Project/`:
- `2605.29136v1(1).pdf`: the paper with its appendix. Read it with the PDF tooling.
- `Topic_ Final Projects.pdf`: the assignment post. `Initial Braindump(1).pdf`: the group's planning notes.

## Environment

- **Requires CUDA.** Device `"cuda"` is hard-coded throughout. The attribute grids use tiny-cuda-nn, the rasterizer is written in Slang, and sampling uses a CUDA kernel that is JIT-compiled at runtime. Nothing here runs on this Mac, so edit the code here and train on a Linux machine with an NVIDIA GPU. The paper's numbers come from an H200 (~3.5 h for 35K iterations, ~7.25 h for 65K).
- **The submodules are not checked out in this clone.** `submodules/*` are empty, so run `git submodule update --init --recursive` before installing. `slang-gaussian-rasterization` points to the authors' fork (polanskers), not the upstream rasterizer, because the custom gradient is implemented there.
- **Our environment is a uv project** (`pyproject.toml`, `uv.lock`), moved forward from the README's pins so it runs on Blackwell GPUs: Python 3.10, `torch==2.7.1` / `torchvision==0.22.1` from the cu128 index on Linux (CPU wheels on macOS, for tests only). `uv sync` installs everything except the CUDA submodules, which are built against the installed torch:
  ```bash
  uv sync
  uv pip install --no-build-isolation ./submodules/slang-gaussian-rasterization
  TCNN_HALF_PRECISION=0 uv pip install --no-build-isolation ./submodules/tiny-cuda-nn/bindings/torch
  ```
  `slangtorch` has no macOS wheels, so it is only installed on Linux/Windows. The README's original Python 3.9 / PyTorch 2.4.1+cu121 setup is what the paper used; Blackwell (B200 = 10.0, RTX PRO 6000 = 12.0) needs CUDA 12.8 and PyTorch ≥ 2.7. HiPerGator's current GPUs are L4, RTX PRO 6000 and B200 (the A100s were retired).
- `--torch_cuda_arch_list` sets `TORCH_CUDA_ARCH_LIST` before the sampling kernel compiles. The default is `"8.0;8.9"`; use `"8.9"` for an RTX 40xx, `"10.0"` for B200, `"12.0"` for RTX PRO 6000. If the compile fails, `sample_cuda` logs a warning and silently falls back to the much slower PyTorch `sample` path, so check the log.

## Commands

**Tests** run on CPU (any machine, including the Mac) with the GPU-only modules stubbed out:

```bash
uv run pytest                       # all tests
uv run pytest tests/test_snapshots.py -k timelapse   # one test
uv run pyflakes utils scripts tests
```

They cover our additions (PLY export, `--train_subset` through the real COLMAP loader, snapshot logic and CLI wiring), not the CUDA paths. On a new GPU machine, run `scripts/gpu_smoke_test.sh /path/to/tandt/truck` first (~15 min). It trains a tiny truck run with every new feature switched on, then renders, scores, exports and builds a timelapse, and checks that all expected files exist. `tests/out/test_shapes.ply` (written by the tests) is a converted PLY of RGB axes and a yellow cube for checking the export by eye in https://superspl.at/editor.

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
python scripts/export_ply.py <run>/point_cloud/iteration_N.ply out.ply [--min_opacity 0.005] [--sh_degree 0]
```

- `--train_subset N` keeps N evenly spaced training views (by filename), applied *after* the test split and *after* the scene normalization, so every run of a views ladder has the same test set and coordinate frame. The kept names go to `<run>/train_subset.txt`. Use it instead of `--load_ncameras`.
- `--snapshot_iterations` (off by default) writes `<run>/snapshots/iter_N_view{k}_render.png`, `..._centers.png` (every Gaussian shrunk to an opaque dot, to show where samples are) and `iter_N_gaussians.ply` (a random subset, standard format, SH degree 0). Views index the test cameras. A failed snapshot logs a warning and never stops training.
- **Saved PLYs are not in the standard 3DGS format**: `save_ply` writes activated opacity (0–1) and world-unit scales, while viewers expect a logit and a log. Always run `scripts/export_ply.py` before opening one in SuperSplat or another viewer; otherwise the scene looks like huge semi-transparent blobs.

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
- For COLMAP scenes, the loader reads `sparse/0/{cameras,images,points3D}.bin` (or `.txt`) and the images in `<source_path>/images` (change the folder with `--images`). **Only PINHOLE and SIMPLE_PINHOLE cameras are accepted**, so images must be undistorted. The repo has no `convert.py`. Run COLMAP's feature_extractor, a matcher, mapper, and image_undistorter, then move the undistorter's `sparse/*.bin` into `sparse/0/` (the same steps as 3DGS's `convert.py`).
- **Camera poses are still required.** What the method removes is the SfM *point cloud* initialization: `init_type` defaults to `"random"`. Setting `init_type=sfm` only sets `xyz`, which the first `sample()` overwrites. The COLMAP-initialized pyramid from paper App. A4.5 is not implemented.
- With `init_type=random`, the reader writes an unused PLY to `<source_path>/point_cloud/`, so the dataset folder must be writable. (Upstream crashed if that folder was missing; we now create it.)
- `eval=True` holds out every 8th image, sorted by filename, as the test set.
- `resolution` set to 1, 2, 4, or 8 downsamples the full-resolution `images/` in memory. `-1` caps the width at 1600, and any other value is used as a target width. All training images are preloaded to the GPU.
- For RGBA images, alpha is used as a mask, and masked-out ground-truth pixels are filled with the random training background.
- `--load_ncameras N` truncates the train and test lists separately and keeps N+1 of each (off by one). For few-view experiments use `--train_subset` instead.
- Normalization: training camera centers are PCA-aligned and scaled into [-1,1]^3 (`transform_poses_pca`, applied in `Camera.__init__`). The L∞ contraction then gives that cube 3/4 of the pyramid's 4096 bins per axis and squeezes everything outside it into the outer shell. A consequence of this design (not yet tested): in inside-out captures, such as a camera walking through a room, most of the content falls outside the camera hull and gets coarse resolution.

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
