#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=/home/UniAD
STAGE1_CFG=${1:-./projects/configs/stage1_track_map/base_track_map.py}
STAGE2_CFG=${2:-./projects/configs/stage2_e2e/base_e2e.py}
GPUS=${3:-8}
MAX_ITERS=${4:-400}
WARMUP_ITERS=${5:-80}
OUT_DIR=${6:-/home/UniAD/work_dirs/perf_bench}

mkdir -p "$OUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
cd "$ROOT_DIR"

run_one() {
  local name=$1
  local cfg=$2
  local log="$OUT_DIR/${name}_perf_${TS}.log"
  echo "[Perf] Start ${name}: cfg=${cfg} gpus=${GPUS} max_iters=${MAX_ITERS} warmup=${WARMUP_ITERS}"
  PERF_MAX_ITERS=${MAX_ITERS} \
    bash /home/UniAD/tools/uniad_dist_train.sh "$cfg" "$GPUS" \
    --cfg-options log_config.interval=20 \
    > "$log" 2>&1
  echo "[Perf] Finished ${name}, log: $log"
  /home/UniAD/tools/parse_compat_perf.py "$log" "$WARMUP_ITERS" | tee "$OUT_DIR/${name}_summary_${TS}.txt"
}

run_one stage1 "$STAGE1_CFG"
run_one stage2 "$STAGE2_CFG"

echo "[Perf] Done. Summaries:"
ls -1 "$OUT_DIR"/*_summary_${TS}.txt
