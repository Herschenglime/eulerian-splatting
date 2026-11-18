# COLMAP on your own Linux desktop, then train on HiPerGator

COLMAP's slowest stage (incremental mapping) runs on the CPU. Our HiPerGator QOS gives a job 1–2 cores, so a 12-core desktop is several times faster, and it costs the lab nothing. Feature extraction and matching run on the desktop's NVIDIA GPU. Training then runs on HiPerGator, where `scripts/e2e.py` reuses the uploaded dataset and skips COLMAP (it checks for `data/<name>/sparse/0`).

The dataset name must match the `name:` in the HiPerGator config (`configs/e2e/<name>.yaml`), since e2e looks for `data/<name>/`. The examples below use the office capture (`name: office`).

## 1. Code

In your existing clone: `git pull`. A fresh clone works too; the submodules aren't needed for COLMAP.

Put the capture in `data/` (gitignored), e.g. `data/office_vid.mp4`, or a folder of photos.

## 2. Run COLMAP

`scripts/run_colmap.py` only needs pycolmap, OpenCV, Pillow, pillow-heif and numpy. A throwaway environment with just those avoids building tiny-cuda-nn (which needs `nvcc` 12.8):

```bash
uv run --no-project --python 3.10 \
  --with pycolmap-cuda12==4.2.1 --with opencv-python-headless --with pillow --with pillow-heif --with "numpy<2.3" \
  python scripts/run_colmap.py data/office_vid.mp4 data/office --fps 2 --matcher exhaustive
```

If the project's full environment already works on that machine (`uv sync` done), `uv run python scripts/run_colmap.py ...` with the same arguments is equivalent.

- `--fps 2` keeps the sharpest frame in each half-second window. For photos, drop `--fps`.
- **`--matcher exhaustive`:** without it, the script switches to sequential matching above 300 frames (a video longer than 150 s at 2 fps). Sequential matching has no loop closure and split the 96-frame NemeSys video into 4 disconnected models. Exhaustive matching costs N²/2 image pairs (300 frames ≈ 45K pairs; the 170-photo Testudo capture did 14K pairs in 4.4 min on an L4), which a desktop GPU can afford.
- Optional, video length in seconds: `uv run --no-project --with opencv-python-headless python -c "import cv2; c = cv2.VideoCapture('data/office_vid.mp4'); print(c.get(7) / c.get(5))"`
- The script forces GPU SIFT and fails loudly instead of falling back to the CPU.
- After removing bad frames from `data/office/input/` by hand, re-run with `--skip_input` to redo only the SfM.

### NixOS

uv's managed Python and the pip wheels (pycolmap, OpenCV) are generic Linux binaries:
- They need `nix-ld` (`programs.nix-ld.enable = true;`). If a wheel reports a missing library (e.g. `libstdc++.so.6`, `libz.so.1`), add its package (`stdenv.cc.cc.lib`, `zlib`, ...) to `programs.nix-ld.libraries`.
- The NVIDIA driver's `libcuda.so` lives in `/run/opengl-driver/lib`. If feature extraction reports no CUDA device or a CUDA library error: `export LD_LIBRARY_PATH=/run/opengl-driver/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}`.

## 3. Check the result before uploading

`data/office/colmap_summary.json` (the warnings are also printed at the end of the run):

| Field | Look for |
|---|---|
| `num_registered` / `num_input_images` | close to all. Unregistered frames are listed in `unregistered_images`. |
| `mean_reprojection_error_px` | our captures: 1.13–1.40 px. Well above ~2 px suggests a bad lens model or bad matches. |
| a "several models" warning | the capture split into disconnected pieces; only the largest is kept. Common in rooms with blank walls. |
| `blurry_images`, `static_frames`, `capture_settings` | blur, stops in camera motion, unlocked exposure (photos only; videos carry no per-frame EXIF) |

## 4. Upload to HiPerGator

Training only reads `images*/`, `sparse/` and the summary, so skip `input/` (the extracted frames) and `distorted/` (the database):

```bash
rsync -av --exclude input --exclude distorted data/office/ <gatorlink>@hpg.rc.ufl.edu:/blue/mdjahiduislam/<gatorlink>/eulerian-splatting/data/office/
```

The capture itself (`capture:` in the config) must also exist on HiPerGator, because `e2e.py` checks the path even when it reuses COLMAP.

## 5. Train on HiPerGator

```bash
sbatch --cpus-per-task=1 --mem=19gb slurm/e2e_b200.sbatch configs/e2e/office.yaml
slurm/status.sh
```

Measured on Testudo (170 photos, full preset): ~3.3 it/s on a B200 (65K iterations ≈ 5.5 h), 0.9 CPU cores, ~10 GB host RAM while training plus ~4 GB while saving the 15M-Gaussian PLY.
