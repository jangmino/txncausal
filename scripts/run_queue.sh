#!/usr/bin/env bash
# Job-queue runner for mvcp-train.
#   one job per line:  <model> <seed> [extra mvcp-train args...]   ('#' comments, blank lines ok)
#   usage: scripts/run_queue.sh [-P N] [-o OUT] QUEUE_FILE
#     -P N    concurrent jobs (default 12, the setting used on one RTX 4090 for Katcher's)
#     -o OUT  results dir (default results/katchers)
# Finished jobs are skipped (mvcp-train --skip-existing), so rerunning the same command resumes.
# Launch fully detached so a dropped session cannot kill it:
#   setsid nohup scripts/run_queue.sh -P 12 queue.txt > queue.log 2>&1 < /dev/null &
set -uo pipefail
P=12
OUT=results/katchers
while getopts "P:o:" opt; do
  case $opt in
    P) P=$OPTARG ;;
    o) OUT=$OPTARG ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
QUEUE=${1:?usage: run_queue.sh [-P N] [-o OUT] QUEUE_FILE}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
export OUT TRAIN="$ROOT/.venv/bin/mvcp-train"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}
# deterministic cuBLAS (torch.use_deterministic_algorithms); the setting of every run of the paper
export CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}
mkdir -p "$OUT/logs"

run_one() {
  local model=$1 seed=$2
  shift 2
  local tag="${model}_seed${seed}"
  [ $# -gt 0 ] && tag="${tag}_$(echo "$*" | tr -s ' ' '_' | tr -cd 'A-Za-z0-9._=-')"
  local log="$OUT/logs/$tag.log" t0=$SECONDS
  if "$TRAIN" --model "$model" --seed "$seed" --out "$OUT" --skip-existing "$@" \
      > "$log" 2>&1; then
    if grep -q '^\[skip\]' "$log"; then
      echo "[skip] $tag"
    else
      echo "[done] $tag $((SECONDS - t0))s"
    fi
  else
    echo "[FAIL] $tag rc=$? log=$log"
    return 1
  fi
}
export -f run_one

echo "[queue] $QUEUE -> $OUT  P=$P  $(date +%Y-%m-%dT%H:%M:%S%z)"
t0=$SECONDS
# strip comments/trailing blanks (xargs -L continues lines that end in a blank)
sed -e 's/#.*//' -e 's/[[:space:]]*$//' -e '/^$/d' "$QUEUE" \
  | xargs -P "$P" -L 1 bash -c 'run_one "$@"' _
rc=$?
echo "[queue] finished rc=$rc in $((SECONDS - t0))s  $(date +%Y-%m-%dT%H:%M:%S%z)"
exit $rc
