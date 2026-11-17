#!/usr/bin/env bash
# ==============================================================================
# scripts/run_train_baseline_v3.sh
# Production launch script for Stage 1 Baseline Weapon Detector v3
# (YOLO11m @ 960px, 40 Epochs, scale=0.90, close_mosaic=10, workers=4).
#
# Key Features:
# - No pre-created run directory (prevents Ultralytics "-2" naming collision).
# - Runs detached via setsid nohup (survives harness / session termination).
# - Fresh run name: baseline_m960_v3.
# - Config: configs/detection/yolo11m_960_v3.yaml.
# - Automatic backup of last.pt every 5 epochs via checkpoint_backup_daemon.sh.
# - Prints PID, log path, and exact resume command.
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

PYTHON_BIN="${WORKSPACE_ROOT}/work/.venv/bin/python"
CONFIG_FILE="${WORKSPACE_ROOT}/configs/detection/yolo11m_960_v3.yaml"
RUN_NAME="baseline_m960_v3"
RUN_DIR="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}"
LOG_FILE="${WORKSPACE_ROOT}/outputs/runs/${RUN_NAME}.log"

mkdir -p "${WORKSPACE_ROOT}/outputs/runs"

# Guardrail: Ensure target run directory does not pre-exist
if [[ -d "${RUN_DIR}" ]]; then
    echo "[WARN] Target run directory ${RUN_DIR} already exists. Removing to prevent Ultralytics '-2' collision..."
    rm -rf "${RUN_DIR}"
fi

echo "======================================================================"
echo "STAGE 1 BASELINE TRAINING (v3 — 40 EPOCHS @ 960px, scale=0.90)"
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

# Launch checkpoint backup daemon in background to backup last.pt every 5 epochs
nohup bash "${WORKSPACE_ROOT}/scripts/train/checkpoint_backup_daemon.sh" "${PID}" "${RUN_DIR}" \
    >> "${WORKSPACE_ROOT}/outputs/runs/backup_daemon.log" 2>&1 &
BACKUP_PID=$!
echo "[BACKUP DAEMON] Monitoring PID: ${BACKUP_PID} (backing up last.pt every 5 epochs)"

# Symlink log into run directory once created by Ultralytics
(
    for i in {1..120}; do
        if [[ -d "${RUN_DIR}" ]]; then
            ln -sf "${LOG_FILE}" "${RUN_DIR}/train.log"
            break
        fi
        sleep 1
    done
) &

echo "======================================================================"
