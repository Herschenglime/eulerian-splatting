#!/bin/bash
# One e2e.py run (training -> timelapse -> export) on a B200. COLMAP must already be done
# (slurm/colmap.sbatch), so the GPU never sits idle on CPU-only work.
# Usage: sbatch slurm/e2e_b200.sbatch configs/e2e/<name>.yaml [e2e.py flags] [-- train.py flags]
#    or: sbatch slurm/e2e_b200.sbatch --resume output/<name>/<run>
# Watch it with: slurm/status.sh   (iteration, it/s, ETA, peak RAM; GPU memory in logs/*_gpu.csv)
# Sized for the lab QOS (mdjahiduislam: 16 CPU / 125 GB shared by the whole lab).
#SBATCH --job-name=eus-e2e-b200
#SBATCH --account=mdjahiduislam
#SBATCH --qos=mdjahiduislam
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/mdjahiduislam/pgrau/eulerian-splatting/logs/%x_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail

module purge
module load conda cuda/12.8.1
unset UV_PYTHON          # the conda module points uv at Python 3.14; the project needs 3.10
export UV_NO_SYNC=1      # never rebuild the environment inside a GPU job
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /blue/mdjahiduislam/pgrau/eulerian-splatting
stage() { echo; echo "#### stage: $* ($(date '+%H:%M:%S'))"; }

stage "start: job $SLURM_JOB_ID on $(hostname): e2e.py $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
# GPU memory and utilisation every 60 s into a side file.
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader -l 60 \
  > "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" &
GPUMON=$!
trap 'kill $GPUMON 2>/dev/null || true' EXIT

uv run python scripts/e2e.py "$@"

stage "done"
echo "peak GPU memory: $(cut -d, -f2 "logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv" | sort -n | tail -1)"
