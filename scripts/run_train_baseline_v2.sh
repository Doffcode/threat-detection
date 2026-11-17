#!/usr/bin/env bash
# ==============================================================================
# scripts/run_train_baseline_v2.sh
# Production launch script for Stage 1 Baseline Weapon Detector v2 (YOLO11m @ 960px, 40 Epochs).
#
# Key Features:
# - No pre-created run directory (prevents Ultralytics "-2" naming collision).
# - Runs detached via setsid nohup (survives harness / session termination).
# - Fresh run name: baseline_m960_v2 (no resume from v1).
# - Config: configs/detection/yolo11m_960_v2.yaml (40 epochs, close_mosaic=10, patience=10).
# - Prints PID, log path, and exact resume command.
# - DO NOT EXECUTE UNTIL EXPLICITLY APPROVED BY USER.
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

PYTHON_BIN="${WORKSPACE_ROOT}/work/.venv/bin/python"
CONFIG_FILE="${WORKSPACE_ROOT}/configs/detection/yolo11m_960_v2.yaml"
RUN_NAME="baseline_m960_v2"
RUN_DIR="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}"
LOG_FILE="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}.log"

mkdir -p "${WORKSPACE_ROOT}/outputs/runs"

echo "======================================================================"
echo "STAGE 1 EXTENDED BASELINE TRAINING (v2 — 40 EPOCHS @ 960px)"
echo "Config          : ${CONFIG_FILE}"
echo "Run Name        : ${RUN_NAME}"
echo "Log File        : ${LOG_FILE}"
echo "Target Run Dir  : ${RUN_DIR}"
echo "======================================================================"

export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"

# Launch detached process using setsid nohup
setsid nohup env PYTHONPATH="${PYTHONPATH}" "${PYTHON_BIN}" src/detection/train.py \
    --config "${CONFIG_FILE}" \
    --name "${RUN_NAME}" \
    "$@" \
    < /dev/null > "${LOG_FILE}" 2>&1 &

PID=$!
echo "[SUCCESS] Detached training process launched with PID: ${PID}"
echo "[LOG PATH] Output log: ${LOG_FILE}"
echo "[MONITOR] Stream logs with:"
echo "          tail -f ${LOG_FILE}"
echo "[RESUME] Exact resume command if interrupted:"
echo "          ${PYTHON_BIN} src/detection/train.py --config ${CONFIG_FILE} --name ${RUN_NAME} --resume"

# Symlink log into run directory once created by Ultralytics
(
    for i in {1..120}; do
        if [[ -d "${RUN_DIR}" ]]; then
            ln -sf "${LOG_FILE}" "${RUN_DIR}/train.log"
            break
        fi
        sleep 1
    done
) >/dev/null 2>&1 &

echo "======================================================================"
