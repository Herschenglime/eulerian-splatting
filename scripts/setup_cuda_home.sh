#!/usr/bin/env bash
# Put a CUDA 12.8 toolkit (nvcc + headers) inside the project venv, for machines with no
# system CUDA toolkit and no sudo. tiny-cuda-nn, the Slang rasterizer and the sampling kernel
# all compile CUDA code against torch, so they need an nvcc that matches torch's CUDA (12.8).
#
# Usage: scripts/setup_cuda_home.sh [target_dir]      (default: .venv/cuda)
#
# The nvidia-cuda-nvcc-cu12 pip wheel only ships ptxas/nvvm (for JAX/Triton), not the nvcc
# driver, so this downloads NVIDIA's per-component redist archives (~80 MB) for the compiler,
# cudart and CCCL. Library headers (cuBLAS, cuSPARSE, ...) are symlinked from the nvidia-*
# wheels torch already installed, so their versions match torch exactly.
#
# utils/cuda_env.py points CUDA_HOME at .venv/cuda when CUDA_HOME isn't already set, so
# `uv run python train.py ...` picks it up. On HiPerGator, `module load cuda/12.8.1` instead.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${1:-$REPO/.venv/cuda}"
CUDA_VERSION=12.8.1
REDIST=https://developer.download.nvidia.com/compute/cuda/redist
COMPONENTS=(cuda_nvcc cuda_cudart cuda_cccl cuda_nvtx cuda_profiler_api)

SITE="$(cd "$REPO" && uv run --no-sync python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')/nvidia"
[ -d "$SITE" ] || { echo "No nvidia wheels in the venv ($SITE); run 'uv sync' first" >&2; exit 1; }

if [ -x "$TARGET/bin/nvcc" ]; then
    echo "Already set up: $("$TARGET/bin/nvcc" --version | tail -1)"
    exit 0
fi

mkdir -p "$TARGET"
manifest="$(curl -sfL "$REDIST/redistrib_$CUDA_VERSION.json")"
for comp in "${COMPONENTS[@]}"; do
    path="$(python3 -c 'import json,sys; print(json.loads(sys.stdin.read())[sys.argv[1]]["linux-x86_64"]["relative_path"])' "$comp" <<<"$manifest")"
    echo "Downloading $path"
    curl -sfL "$REDIST/$path" | tar -xJ -C "$TARGET" --strip-components=1
done

# Headers and shared libraries of the math libraries, from torch's own wheels.
for pkg in cublas cusparse cusolver curand cufft cuda_nvrtc nvjitlink; do
    [ -d "$SITE/$pkg" ] || continue
    for f in "$SITE/$pkg"/include/*; do
        [ -e "$f" ] && [ "$(basename "$f")" != __init__.py ] && ln -sfn "$f" "$TARGET/include/$(basename "$f")"
    done
    for f in "$SITE/$pkg"/lib/*.so.*; do
        [ -e "$f" ] || continue
        ln -sfn "$f" "$TARGET/lib/$(basename "$f")"
        ln -sfn "$f" "$TARGET/lib/$(basename "$f" | sed 's/\.so\..*/.so/')"   # unversioned name for -l<lib>
    done
done
# torch's cpp_extension links against $CUDA_HOME/lib64.
[ -e "$TARGET/lib64" ] || ln -s lib "$TARGET/lib64"

"$TARGET/bin/nvcc" --version | tail -1
echo "CUDA_HOME=$TARGET"
