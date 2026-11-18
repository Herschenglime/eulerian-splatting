#!/bin/bash
# Build the results zip (scripts/pull_results.py --local) for a run trained by slurm/e2e_b200.sbatch,
# only if that training job succeeded. Queue it right after the training job:
#   sbatch --dependency=afterok:<train_jobid> --kill-on-invalid-dep=yes slurm/zip_run.sbatch <train_jobid>
# afterok: starts only if training exited 0. --kill-on-invalid-dep: SLURM cancels this job if training
# fails, instead of leaving it pending forever. The run folder is read from the training job's log.
# Output: results/<name>_<run>.zip (download it, then delete it: it is a full copy on the lab's quota).
#
# Burst QOS: CPU-only, borrows idle cores, so it doesn't use the lab's 16 CPU / 125 GB investment.
# Zipping streams files (stored, not compressed), so it needs little memory.
#SBATCH --job-name=eus-zip
#SBATCH --account=mdjahiduislam
#SBATCH --qos=mdjahiduislam-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=00:45:00
#SBATCH --output=/blue/mdjahiduislam/pgrau/eulerian-splatting/logs/%x_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail

TRAIN=${1:?usage: sbatch slurm/zip_run.sbatch <train_jobid>}
cd /blue/mdjahiduislam/pgrau/eulerian-splatting

LOG=$(ls logs/*_"${TRAIN}".log 2>/dev/null | head -1 || true)
[ -n "$LOG" ] || { echo "ERROR: no log for job $TRAIN in logs/"; exit 1; }
RUN=$(grep -aoE 'output/[^/[:space:]]+/[0-9]{8}-[0-9]{6}' "$LOG" | head -1 || true)
[ -n "$RUN" ] || { echo "ERROR: no run folder (output/<name>/<run>) in $LOG"; exit 1; }
NAME=$(cut -d/ -f2 <<< "$RUN")

# Belt and braces on top of afterok: the job finished every stage and wrote the final splat.
grep -aq '#### stage: done' "$LOG" || { echo "ERROR: $LOG has no '#### stage: done'"; exit 1; }
[ -f "$RUN/$NAME.ply" ] || { echo "ERROR: $RUN/$NAME.ply missing"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname): zipping $RUN (training job $TRAIN), $(date)"
python3 -u scripts/pull_results.py --local . "$RUN"
echo "done $(date)"
