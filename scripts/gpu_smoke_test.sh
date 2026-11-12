#!/usr/bin/env bash
# First-run check on a GPU machine (~15 min). Exercises everything the CPU tests can't:
# the CUDA submodules, the sampling kernel JIT, the snapshot overrides through the Slang
# rasterizer (probabilistic and refinement phases), render/metrics, and the PLY export.
#
# Usage:  scripts/gpu_smoke_test.sh /path/to/tandt/truck [cuda_arch]
#   cuda_arch defaults to the first GPU's compute capability (8.6 = RTX 30xx, 8.9 = RTX 40xx,
#   10.0 = B200, 12.0 = RTX PRO 6000). Sized to fit a 12 GB card; peak memory is an estimate.
set -euo pipefail

SCENE=${1:?usage: $0 /path/to/tandt/truck [cuda_arch]}
cd "$(dirname "$0")/.."

ARCH=${2:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')}
MODEL="output/smoke_test_$(date +%Y%m%d_%H%M%S)"
PY="uv run python"

echo "== environment (arch $ARCH)"
$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "torch cannot see a CUDA GPU"
print("torch", torch.__version__, "cuda", torch.version.cuda, "|", torch.cuda.get_device_name(0),
      "| capability", torch.cuda.get_device_capability(0),
      "| %.1f GB" % (torch.cuda.get_device_properties(0).total_memory / 1e9))
import tinycudann, slang_gaussian_rasterization  # noqa: F401  (built from submodules)
print("submodules import OK")
EOF

echo "== train (600 sampling + 100 refinement iterations, 30 views, snapshots in both phases)"
# Sized for a 12 GB card (RTX 3060): small hash tables, half-resolution images, 2M samples.
$PY train.py --config=truck --source_path="$SCENE" --model_path="$MODEL" \
    --torch_cuda_arch_list="$ARCH" --dry_run --disable_visualize \
    --attrib_config=configs/attribute_grids_small.yaml --resolution=2 \
    --iterations=600 --refinement_iterations=100 \
    --num_samples=2000000 --num_samples_min=500000 \
    --save_iterations 600 700 --test_iterations 600 \
    --train_subset=30 \
    --snapshot_iterations 1 300 600 650 --snapshot_views 0 1

echo "== render (post- and pre-refinement)"
$PY render.py --config=truck --source_path="$SCENE" --model_path="$MODEL" --skip_train
$PY render.py --config=truck --source_path="$SCENE" --model_path="$MODEL" --skip_train --eval_iteration=600

echo "== metrics, export, timelapse"
$PY metrics.py --model_paths "$MODEL"
$PY scripts/export_ply.py "$MODEL/point_cloud/iteration_700.ply" "$MODEL/export_700.ply" --min_opacity 0.005
$PY scripts/make_timelapse.py "$MODEL"

echo "== checking outputs"
fail=0
check() { if [ -e "$MODEL/$1" ]; then echo "  ok   $1"; else echo "  MISSING $1"; fail=1; fi; }
check cfg_args
check train_subset.txt
check point_cloud/iteration_600.ply
check point_cloud/iteration_700.ply
check learned_grids/iteration_600.pt
check renders/test/ours_700/renders
check renders/test/ours_600/renders
check results.json
check export_700.ply
for it in 000001 000300 000600 000650; do
  check "snapshots/iter_${it}_view0_render.png"
  check "snapshots/iter_${it}_view0_centers.png"
  check "snapshots/iter_${it}_gaussians.ply"
done
check snapshots/timelapse_view0.gif
check snapshots/contact_view0.png
[ "$(wc -l < "$MODEL/train_subset.txt")" -eq 30 ] && echo "  ok   train_subset.txt lists 30 views" \
  || { echo "  WRONG train_subset.txt does not list 30 views"; fail=1; }

echo
if [ "$fail" -eq 0 ]; then
  echo "SMOKE TEST PASSED -> $MODEL"
  echo "Look at: $MODEL/snapshots/contact_view0.png (dots should condense toward the truck),"
  echo "         $MODEL/results.json, and drag $MODEL/export_700.ply into https://superspl.at/editor"
else
  echo "SMOKE TEST FAILED -> $MODEL (also grep the log above for 'Snapshot at iteration')"
  exit 1
fi
