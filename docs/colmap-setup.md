# Changes: COLMAP preprocessing and a local GPU setup

This covers the `colmap_setup` branch after `3418e3c` (the handoff): the COLMAP pipeline, making a plain `uv sync` build the whole environment (CUDA submodules included), and a review of the HiPerGator work on `main`.

## Why

Training needs camera poses, and our own captures come from an iPhone as HEIC photos or HEVC video. We need one command that turns a capture into a dataset `Scene` can load: undistorted images plus a PINHOLE COLMAP model in `sparse/0/`. It should run anywhere the project venv runs, without a separate COLMAP install.

## 1. `scripts/run_colmap.py`

**Files:** `scripts/run_colmap.py`, `configs/scenes/nemesys.yaml`

```bash
uv run python scripts/run_colmap.py data/NemeSys data/nemesys                 # folder of photos
uv run python scripts/run_colmap.py "data/nemesys vid.mp4" data/nemesys_vid --fps 2   # video
```

- **Input preparation** (`<out>/input/`):
  - Photos (JPEG, PNG, TIFF, HEIC) are written as JPEGs. HEIC is decoded with `pillow-heif`, EXIF rotation is applied to the pixels, and the EXIF is kept, since COLMAP takes its focal-length prior from it.
  - Video: OpenCV reads every frame and keeps the sharpest one (variance of the Laplacian) in each 1/`fps`-second window, instead of every Nth frame. OpenCV applies the phone's rotation metadata, so portrait video comes out portrait.
  - `--max_size` (default 3200) caps the long side. An iPhone 15's photos are 48 MP (6048×8064), which makes COLMAP and the training loader slow and memory-hungry for no gain at our training resolutions. 3200 is also COLMAP's own SIFT limit.
  - `--skip_input` reuses `<out>/input/` as is, e.g. after deleting bad frames by hand.
- **SfM** (`<out>/distorted/`, via pycolmap): SIFT extraction and matching on the GPU (`Device.cuda`, so it fails loudly instead of falling back to the CPU), one shared OPENCV camera, then incremental mapping (`--mapper global` for GLOMAP). If COLMAP produces several disconnected models, the largest one is kept and a warning lists their sizes.
- **Matching:** exhaustive up to 300 images, sequential above that (`--matcher` overrides). Sequential only matches nearby frames and, without a vocab tree, has no loop closure, so one blurry stretch splits the model. On the 96-frame NemeSys video, sequential matching gave 4 disconnected models (59 frames in the largest); exhaustive registered all 96 in about 90 s.
- **Undistortion** (`<out>/images/`, `<out>/sparse/0/`): pycolmap's undistorter writes the model to `sparse/` directly, and the script moves it into `sparse/0/`, where our loader reads. COLMAP 4 also writes `frames.bin` and `rigs.bin`; our loader ignores them, and it parses the new `images.bin` correctly (checked).
- **`images_2/`, `images_4/`** (`--downsample`, default 2 and 4): downsampled copies, as in the benchmark datasets. Train on them with `--images images_N --resolution 1` instead of decoding full-size images and resizing in memory. The HiPerGator notebook does the same to stay under the SLURM job's RAM limit.
- **`<out>/colmap_summary.json`:** registered/total images, the names of unregistered images, number of points, mean reprojection error, track length, both cameras, and frame sharpness for videos.
- **`configs/scenes/nemesys.yaml`** is the scene config for the NemeSys photos. `num_samples_min: 1M` is a guess, since our scenes have no 3DGS-MCMC count to copy.

## 2. `uv sync` builds the whole environment

**Files:** `pyproject.toml`, `uv.lock`

- **The problem:**
  - The two CUDA submodules were installed by hand with `uv pip install --no-build-isolation`, outside the lockfile.
  - A plain `uv sync` makes the venv match the lock exactly, so every sync **uninstalled** them. After that, the rasterizer failed with `No module named 'slang_gaussian_rasterization'`.
  - The HiPerGator environment lives in a venv on `/blue` that git knows nothing about.
- **What changed:** `pyproject.toml` is now the single description of the environment.
  - `slang-gaussian-rasterization` and `tinycudann` are path dependencies on the submodules. Both are in `uv.lock`, so `uv sync` builds and keeps them. The rasterizer is pure Python; its Slang kernels compile at runtime.
  - **tiny-cuda-nn build settings:**
    - It builds against the same torch it runs with (`extra-build-dependencies`, `match-runtime`).
    - `setuptools<81` in its build environment, because its `setup.py` imports `pkg_resources`, which setuptools 81 removed. Upstream's README pins `setuptools==73.0.1`.
    - Half precision is off (`TCNN_HALF_PRECISION=0`, as before).
    - Static `dependency-metadata`, so resolving the lock doesn't run its `setup.py`.
  - **`TCNN_CUDA_ARCHITECTURES=86;89;100;120`** builds one module per GPU family: RTX 30xx, RTX 40xx/L4, B200 and RTX PRO 6000. uv caches the built wheel and doesn't key the cache on the GPU. Without a fixed list, a wheel built on one HiPerGator node type would be reused on the others, which share `~/.cache/uv`. A new GPU type means adding its arch. After changing the submodule, run `uv sync --reinstall-package tinycudann`.
  - **Linux x86_64 only** (`[tool.uv] environments`). The macOS markers and CPU-wheel fallbacks are gone; nothing runs on the Mac.
- **The one thing uv can't provide: a CUDA 12.8 toolkit with `nvcc` on `PATH`.** It's needed during `uv sync` and at runtime. Use `module load cuda/12.8.1` on HiPerGator, or NVIDIA's `cuda-toolkit-12-8` apt package plus `export PATH=/usr/local/cuda-12.8/bin:$PATH`. Ubuntu's own `nvidia-cuda-toolkit` is 12.0, too old. Why it can't come from pip:
  - The `nvidia-cuda-nvcc-cu12` wheel, in every 12.x version, only ships `ptxas` and `nvvm`, for JAX and Triton. It has no `nvcc` driver, `cudafe++` or `fatbinary`.
  - CUDA 13's `nvidia-cuda-nvcc` wheels do have the full compiler, and torch's cu130 builds pull them in through NVIDIA's `cuda-toolkit` metapackage. But `pycolmap-cuda12` requires `cuda-toolkit<13`, and there's no GPU pycolmap for CUDA 13. Moving to CUDA 13 would also mean torch 2.7 → 2.14, a CUDA 13 build of tiny-cuda-nn and the Slang rasterizer that nobody has tried, and driver ≥ 580 everywhere.
- **Dropped:** `321e1e1` first worked around the missing `nvcc` with `scripts/setup_cuda_home.sh` and `utils/cuda_env.py`. They downloaded NVIDIA's redist archives into `.venv/cuda` and pointed `CUDA_HOME` at them. Both are removed now that the toolkit comes from the system.

## 3. Other environment fixes

**Files:** `pyproject.toml`, `train.py`, `.gitignore`

- **New dependencies:** `pillow-heif` (HEIC), `opencv-python-headless` (video frames and sharpness), and `ninja` (torch's `cpp_extension` builds with it).
- **`--dry_run` crashed with current wandb** (`ValueError: invalid literal for int()` in `prepare_output_and_logger`). Upstream pins `wandb==0.17.5`. In newer versions, `mode="disabled"` returns a no-op run whose `offline` is False, so the code treated it as an online run and parsed a run number from its random name. It now checks `wandb.run.disabled` too.
- **`data/` is gitignored.** Captures and processed datasets stay out of git.

## 4. Verified on the lab RTX 3060 Ti

This card has **8 GB**, not the 12 GB the smoke-test settings assume; about 1.2 GB of it goes to the desktop. Commands as run:

```bash
git submodule update --init --recursive
nvcc --version    # must say 12.8
uv sync
```

The final setup was checked from an empty `.venv` with a plain `uv sync` and `nvcc` 12.8 on `PATH` (no `CUDA_HOME`). It took 1.5 min, including tiny-cuda-nn for all four archs. A second `uv sync` left both submodules installed, and the tests and a short training run passed afterwards. This machine has no system toolkit yet, so for the check `PATH` pointed at a standalone copy of NVIDIA's 12.8.1 compiler, cudart and CCCL, with the math-library headers from the matching wheels.

| Step | Result |
|---|---|
| pycolmap GPU | `has_cuda` True; SIFT extraction runs on the GPU ("Creating SIFT GPU feature extractor") |
| COLMAP, NemeSys photos (64 HEIC, 48 MP) | 60/64 registered, 20K points, 1.28 px mean reprojection error, 53 s (plus ~1 min of HEIC conversion). Focal length 2458 px at 2377×3173, which matches the 26 mm-equivalent lens. Unregistered: IMG_6226, 6227, 6231, 6232. |
| COLMAP, NemeSys video (48 s, 1080p60 HEVC, `--fps 2`) | 96 frames kept. Sequential: 59/96 in 4 pieces. Exhaustive: 96/96, 14.7K points, 1.13 px, 2 min. |
| Builds | tiny-cuda-nn ~1.5 min for 4 archs (24 cores); Slang kernels 68 s on first import; sampling kernel 22 s, no fallback to the PyTorch path. All are cached afterwards. |
| CPU tests | 22 passed |
| Training on NemeSys photos | 600 + 100 iterations, about 2.5 min at ~4.7 it/s, peak 7.0 GB GPU memory in use (including the desktop). Test PSNR 15.3 at iteration 700. The test grid shows the ROV, table and tank from the correct held-out viewpoints, so the poses are right; the timelapse shows samples condensing. |

The training settings that fit in 8 GB (`--num_samples=2000000` ran out of memory inside tiny-cuda-nn):

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python train.py \
  --config=nemesys --source_path=data/nemesys --torch_cuda_arch_list=8.6 --dry_run --disable_visualize \
  --attrib_config=configs/attribute_grids_small.yaml --images=images_4 --resolution=1 \
  --iterations=600 --refinement_iterations=100 --num_samples=1000000 --num_samples_min=300000 \
  --save_iterations 600 700 --test_iterations 600 --snapshot_iterations 1 300 600 --snapshot_views 0
```

**Viewing the result in SuperSplat** needs `scripts/export_ply.py --max_radius 10` (new). About 1% of this run's Gaussians are background samples from the contracted outer shell. Un-contracted, they land up to 1.4e7 units away (bounding box ±8.4M), while the ROV sits within radius ~2. SuperSplat sets its zoom range from the whole bounding box: `minZoom: 1e-6` in `src/scene-config.ts`, scaled by `sceneRadius = bound.halfExtents.length()` in `src/camera.ts`. Its closest approach was therefore ~21 units, and the view opened ~2e7 units away. With `--max_radius 10`, 895K of the 1M Gaussians are kept; with 3, 702K.

The README's Installation section now describes this setup (`uv sync`, plus NVIDIA's apt repository for `cuda-toolkit-12-8`) instead of upstream's conda instructions.

This machine is for checking that captures and the pipeline work, not for results: a full 65K-iteration schedule at these reduced settings would take about 4 h and still not be paper quality.

## 5. The HiPerGator work on `main`

`origin/main` has 5 commits (Nov 8) on top of `01e528a`, the same base as this branch. **None of them changes `pyproject.toml`, `uv.lock` or any other dependency file.** The B200 environment lives outside git, at `/blue/egn5216/Photo_Gauss_Test/envs/eulerian` (the notebook registers it as a Jupyter kernel with `module`-style `CUDA_HOME=/apps/compilers/cuda/12.8.1`). Its package versions aren't recorded anywhere we can see.

What changed:
- **`example.ipynb`** became a HiPerGator notebook. It has a hard-coded `/blue/...` repo path, uses the GPU's exact arch, and trains bicycle on `images_4` with `resolution=1` to stay under the SLURM RAM limit. `QUICK_TEST` gives a 1000 + 300-iteration schedule with 5M samples.
- **`train.py` / `run.py`:** an optional `progress_callback` with stage events and per-stage CUDA timings, and `training_report` now returns its metrics. With no callback, behaviour is unchanged.
- **`utils/notebook_progress.py`:** a live Jupyter dashboard (ETA, per-stage timings, GPU memory, host RAM against the SLURM cgroup limit, preview images). It needs `IPython` and optionally `psutil`, neither of which is in our `pyproject.toml`.
- **`run.py`:** the default `--torch_cuda_arch_list` gains `10.0`.
- **`tqdm` → `tqdm.auto`** in 5 files, for notebook progress bars.
- **Committed data that shouldn't be in git:**
  - `data/bicycle/sparse/0/*.bin` (images.bin is 50 MB), `poses_bounds.npy` and a random-init PLY. This is a partial add of the dataset; the images were meant to go through Git LFS, which was never set up.
  - `data/wget-log`, a leftover from a failed `wget` of a claude.ai login URL. No secrets, but it's junk.
  - The PLY is written by the loader when training starts, so a run did load the full scene on HiPerGator.
  - Nothing from that run is in git (`output/` is ignored), so there are no B200 results to look at from here.
  - Bicycle is the public mip-NeRF 360 scene, so redoing the run needs only the public download (`360_v2`), not his copy.

**Should this branch merge it now? Yes, the code, soon.** The trial merge (`git merge-tree`) conflicts in only two places: `training()` and `train_model()` each gained a keyword argument on both sides (`snapshot_args` here, `progress_callback` there). Keeping both is a two-line fix, and the training-loop changes auto-merge. Both branches keep editing the training loop, so the conflicts only get worse with time. When merging:
- `git rm --cached -r data/bicycle data/wget-log` (and preferably on `main` too), since `data/` is now ignored.
- Add `ipython`, `ipykernel` and `psutil` to `pyproject.toml`, so the dashboard and the Jupyter kernel run from the uv environment instead of the separate `/blue` env.
- Re-run `uv run pytest` and a short training run, since snapshots and the progress callback both hook the end of the iteration.

**Ask the teammate for a `pip freeze` of the HiPerGator env**, especially the torch, slangtorch, wandb and setuptools versions and how tiny-cuda-nn was built. "The deps are sorted" currently means that one env on `/blue`. Our `pyproject.toml` has never been tested on a B200, and the `--dry_run` crash above shows that unpinned wandb breaks the upstream code. That env probably still has the README's wandb 0.17.5.

## Not yet verified

- pycolmap's GPU SIFT on Blackwell (B200, RTX PRO 6000).
- `uv sync` on HiPerGator with `module load cuda/12.8.1`, in particular tiny-cuda-nn's sm_100/sm_120 modules on real Blackwell hardware.
- `--mapper global`, and the sequential matcher on a capture longer than 300 frames.
- `scripts/gpu_smoke_test.sh`, which needs Tanks & Temples truck (not on this machine) and is sized for 12 GB. On an 8 GB card it would need the `--num_samples` settings above.
