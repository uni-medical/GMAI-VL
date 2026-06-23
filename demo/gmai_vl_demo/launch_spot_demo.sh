#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-10083}"
PARTITION="${PARTITION:-Medisco}"
CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
TIME_LIMIT="${TIME_LIMIT:-4:00:00}"
QUOTATYPE="${QUOTATYPE:-spot}"
SRUN_QUOTA_ARGS=()
if [[ -n "${QUOTATYPE:-}" ]]; then
  SRUN_QUOTA_ARGS+=(--quotatype="$QUOTATYPE")
fi
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

exec srun -p "$PARTITION" \
  "${SRUN_QUOTA_ARGS[@]}" \
  --job-name=gmai_vl_demo \
  --gres=gpu:1 \
  --time "$TIME_LIMIT" \
  --cpus-per-task="$CPUS_PER_TASK" \
  bash -lc "cd '$SCRIPT_DIR' && PORT='$PORT' bash '$SCRIPT_DIR/run_demo.sh'" \
  2>&1 | tee "$LOG_DIR/gmai_vl_demo_${PORT}.log"
