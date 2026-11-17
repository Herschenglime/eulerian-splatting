#!/bin/bash
# COLMAP for one capture on an L4 (GPU SIFT; the mapper is CPU-only).
# Usage: sbatch slurm/colmap.sbatch <capture_dir_or_video> <out_dir> [run_colmap.py options]
#   e.g. sbatch slurm/colmap.sbatch data/raw/testudo data/testudo
# Sized for the lab QOS (mdjahiduislam: 16 CPU / 125 GB shared by the whole lab).
#SBATCH --job-name=eus-colmap
#SBATCH --account=mdjahiduislam
#SBATCH --qos=mdjahiduislam
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --time=00:45:00
#SBATCH --output=/blue/mdjahiduislam/pgrau/eulerian-splatting/logs/%x_%j.log
set -euo pipefail

module purge
module load conda cuda/12.8.1
unset UV_PYTHON          # the conda module points uv at Python 3.14; the project needs 3.10
export UV_NO_SYNC=1      # never rebuild the environment inside a GPU job

cd /blue/mdjahiduislam/pgrau/eulerian-splatting
echo "job $SLURM_JOB_ID on $(hostname), $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

uv run python scripts/run_colmap.py "$@"
echo "done $(date)"
