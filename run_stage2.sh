#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(pwd)"
CFG="${1:-./projects/configs/stage2_e2e/base_e2e.py}"
GPUS="${2:-8}"
shift $(( $# > 0 ? 1 : 0 ))
shift $(( $# > 0 ? 1 : 0 ))

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT_DIR}/logs/stage2_train_${TIMESTAMP}"
LOG_FILE="${LOG_DIR}/stage2_train_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"


echo "Stage2 config: ${CFG}" | tee "${LOG_FILE}"
echo "GPUs: ${GPUS}" | tee -a "${LOG_FILE}"
echo "Timestamp: ${TIMESTAMP}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"

export MUSA_EXECUTION_TIMEOUT=320000000
export MASTER_PORT="${MASTER_PORT:-28617}"
#export MUSA_LAUNCH_BLOCKING=1

#bash "${ROOT_DIR}/tools/uniad_dist_train.sh" "${CFG}" "${GPUS}" "$@" 2>&1 | tee -a "${LOG_FILE}"

export ENABLE_OPT_DEFORM_CONV=1 # 使能内部bev_kernels中优化过的kernel；
export ENABLE_CHANNEL_LAST=1

bash "${ROOT_DIR}/tools/uniad_dist_train.sh" "${CFG}" "${GPUS}" "$@"
