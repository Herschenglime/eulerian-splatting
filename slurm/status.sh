#!/usr/bin/env bash
# Progress of our SLURM jobs: queue state, time used/left, the current stage, and stage progress
# (COLMAP: images registered; training: iteration, it/s and tqdm's ETA). Also lab QOS usage.
#   slurm/status.sh            all of $USER's jobs, plus jobs finished in the last 24 h
#   slurm/status.sh <jobid>    one job
# Logs are read from logs/<jobname>_<jobid>.log (the --output pattern in our sbatch files).
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=logs

job_progress() {  # $1 = log file
  local log=$1
  [ -f "$log" ] || { echo "    (no log yet: $log)"; return; }
  # Stage markers printed by our sbatch files ("#### stage: ...") and e2e.py ("==== ...").
  local stage
  stage=$(grep -aE '^(#### |==== |== )' "$log" | grep -av 'done in' | tail -1)
  [ -n "$stage" ] && echo "    stage:    ${stage#\#\#\#\# }"
  # tqdm writes carriage returns; split them so the last update is the last line.
  local bar
  bar=$(tr '\r' '\n' < "$log" | grep -aE 'Training progress: +[0-9]+%' | tail -1)
  if [ -n "$bar" ]; then
    # "Training progress:  12%|#  | 7800/65000 [10:00<1:13:00, 13.0it/s, Loss=...]"
    echo "    training: $(echo "$bar" | sed -E 's/.*\| *([0-9]+\/[0-9]+) \[([^<]*)<([^,]*), *([^,\]]*).*/\1 iterations, elapsed \2, ETA \3, \4/')"
  fi
  local reg total
  reg=$(grep -aoE 'num_reg_frames=[0-9]+' "$log" | tail -1 | cut -d= -f2)
  total=$(grep -aoE '[0-9]+ photos in ' "$log" | tail -1 | cut -d' ' -f1)
  [ -n "$reg" ] && echo "    COLMAP:   ${reg}/${total:-?} images registered"
  grep -aE 'Undistorting image \[' "$log" | tail -1 | sed -E 's/.*(\[[0-9]+\/[0-9]+\]).*/    COLMAP:   undistorting \1/'
  local err
  err=$(grep -aiE 'Traceback|out of memory|oom-kill|CUDA error|Error:' "$log" | tail -2)
  [ -n "$err" ] && echo "    ERRORS:   $err"
}

echo "== running / pending ($(date '+%H:%M:%S'))"
fmt="%i|%j|%P|%T|%M|%L|%r|%b"
if [ $# -ge 1 ]; then jobs=$(squeue -h -j "$1" -o "$fmt" 2>/dev/null); else jobs=$(squeue -h -u "$USER" -o "$fmt"); fi
if [ -z "$jobs" ]; then echo "  none"; fi
while IFS='|' read -r id name part state used left reason gres; do
  [ -z "$id" ] && continue
  echo "  $id $name on $part ($gres): $state, used $used, left $left${reason:+ [$reason]}"
  [ "$state" = PENDING ] || job_progress "$LOGDIR/${name}_${id}.log"
done <<< "$jobs"

echo "== finished in the last 24 h"
sacct -n -X -u "$USER" -S now-24hours ${1:+-j $1} -o JobID%10,JobName%22,State%11,Elapsed,ExitCode 2>/dev/null \
  | grep -vE 'RUNNING|PENDING' | sed 's/^/  /'
echo "  peak host RAM (MaxRSS):"
sacct -n -u "$USER" -S now-24hours ${1:+-j $1} -o JobID%16,MaxRSS 2>/dev/null \
  | awk '$2 ~ /[0-9]K$/ {printf "    %s %.1f GB\n", $1, ($2+0)/1048576}'

echo "== lab QOS (mdjahiduislam: 16 CPU / 125 GB / 15 GPU)"
squeue -A mdjahiduislam -h -t R -o "%u %C %m" | awk '{c[$1]+=$2; m[$1]+=$3+0; tc+=$2; tm+=$3+0}
  END {for (u in c) printf "  %-16s %3d CPU %4d GB\n", u, c[u], m[u]; printf "  free:            %3d CPU %4d GB\n", 16-tc, 125-tm}'
