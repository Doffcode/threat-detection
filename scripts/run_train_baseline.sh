#!/usr/bin/env bash
# ==============================================================================
# scripts/run_train_baseline.sh
# Production launch script for Stage 1 Baseline Weapon Detector (YOLO11m @ 960px).
#
# Key Features:
# - No pre-created run directory (prevents Ultralytics from appending "-2").
# - Runs detached via setsid nohup (survives harness / session signals).
# - Fresh run name: baseline_m960_v1.
# - Preserves all legacy / previous run directories.
# - Checkpoints every epoch (save_period: 1) and early stopping (patience: 5).
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

PYTHON_BIN="${WORKSPACE_ROOT}/work/.venv/bin/python"
CONFIG_FILE="${WORKSPACE_ROOT}/configs/detection/yolo11m_960_baseline.yaml"
RUN_NAME="baseline_m960_v1"
RUN_DIR="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}"
LOG_FILE="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}.log"

echo "======================================================================"
echo "LAUNCHING STAGE 1 BASELINE TRAINING (15 EPOCHS)"
echo "Config        : ${CONFIG_FILE}"
echo "Run Name      : ${RUN_NAME}"
echo "Log File      : ${LOG_FILE}"
echo "======================================================================"

export PYTHONPATH="${WORKSPACE_ROOT}"

# Launch detached process using setsid
setsid "${PYTHON_BIN}" src/detection/train.py \
    --config "${CONFIG_FILE}" \
    --name "${RUN_NAME}" \
    "$@" \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[SUCCESS] Detached training process launched with PID: ${PID}"
echo "[MONITOR] Stream logs with: tail -f ${LOG_FILE}"

# Link log into the run folder once Ultralytics creates it
(
    for i in {1..60}; do
        if [[ -d "${RUN_DIR}" ]]; then
            ln -sf "${LOG_FILE}" "${RUN_DIR}/train.log"
            break
        fi
        sleep 1
    done
) >/dev/null 2>&1 &

echo "======================================================================"
