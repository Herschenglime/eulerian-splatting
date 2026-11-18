#!/bin/bash
# Upload the results zips (results/*.zip) to a cloud remote with rclone, e.g. Dropbox or OneDrive.
# Set up the remote once first (rclone config, see below), then:
#   sbatch --mail-user=<you>@ufl.edu slurm/rclone_upload.sbatch dropbox:EEL5406-splats
#   # after a later zip job:  --dependency=afterok:<zip_jobid> --kill-on-invalid-dep=yes
# rclone copy skips files already uploaded (same size and hash), so re-running uploads only new zips.
# Then share:  module load rclone/1.65.2 && rclone link dropbox:EEL5406-splats
#
# Remote setup (the login needs a browser, which HiPerGator lacks):
#   on your laptop:  rclone authorize "dropbox"     (prints a token)
#   on HiPerGator:   module load rclone/1.65.2 && rclone config
#                    -> new remote, type dropbox, blank client id/secret, no advanced config,
#                       "Use web browser?" No, paste the token. Saved in ~/.config/rclone/rclone.conf.
#
# Burst QOS: CPU-only, borrows idle cores, so it doesn't use the lab's 16 CPU / 125 GB investment.
# Memory: rclone buffers one chunk per transfer (4 x 128 MB).
#SBATCH --job-name=eus-upload
#SBATCH --account=mdjahiduislam
#SBATCH --qos=mdjahiduislam-b
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=04:00:00
#SBATCH --output=/blue/mdjahiduislam/pgrau/eulerian-splatting/logs/%x_%j.log
#SBATCH --mail-type=END,FAIL
set -euo pipefail

DEST=${1:?usage: sbatch slurm/rclone_upload.sbatch <remote>:<folder>}
module load rclone/1.65.2
cd /blue/mdjahiduislam/pgrau/eulerian-splatting

echo "job $SLURM_JOB_ID on $(hostname): results/*.zip -> $DEST, $(date)"
ls -la results/*.zip
rclone copy results "$DEST" --include '*.zip' \
  --transfers 4 --dropbox-chunk-size 128M --onedrive-chunk-size 120M \
  --stats 1m --stats-one-line -v
rclone ls "$DEST"
echo "done $(date)"
