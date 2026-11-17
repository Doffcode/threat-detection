#!/usr/bin/env bash
# ==============================================================================
# scripts/queue_runner.sh
# Detached Queue Supervisor Launcher
# Runs baseline_m960_v3 and baseline_m960_v4 sequentially with full fault-tolerance.
# Survives terminal detachment, chat/harness signals.
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

mkdir -p "${WORKSPACE_ROOT}/outputs/queue"
QUEUE_LOG="${WORKSPACE_ROOT}/outputs/queue/queue.log"
QUEUE_PID_FILE="${WORKSPACE_ROOT}/outputs/queue/queue.pid"

# Check if queue runner is already running
if [[ -f "${QUEUE_PID_FILE}" ]]; then
    EXISTING_PID=$(cat "${QUEUE_PID_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "[WARNING] Queue runner is ALREADY RUNNING with PID ${EXISTING_PID}."
        echo "Log: ${QUEUE_LOG}"
        exit 0
    fi
fi

echo "======================================================================"
echo "LAUNCHING DETACHED QUEUE RUNNER"
echo "======================================================================"

# Ensure memory logger is running
if ! pgrep -f "scripts/mem_logger.sh" > /dev/null; then
    echo "[INFO] Starting memory telemetry logger..."
    nohup bash "${WORKSPACE_ROOT}/scripts/mem_logger.sh" > /dev/null 2>&1 &
fi

# Launch queue runner detached under setsid nohup
setsid nohup "${WORKSPACE_ROOT}/work/.venv/bin/python" "${WORKSPACE_ROOT}/src/queue/queue_runner.py" >> "${QUEUE_LOG}" 2>&1 &
RUNNER_PID=$!
echo "${RUNNER_PID}" > "${QUEUE_PID_FILE}"

sleep 2

if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    echo "[SUCCESS] Queue runner successfully started detached!"
    echo "  PID         : ${RUNNER_PID}"
    echo "  Log Path    : ${QUEUE_LOG}"
    echo "  Status File : ${WORKSPACE_ROOT}/outputs/queue/status.json"
    echo "  Stop File   : ${WORKSPACE_ROOT}/outputs/queue/STOP"
else
    echo "[ERROR] Queue runner failed to start! Last log lines:"
    tail -n 25 "${QUEUE_LOG}"
    exit 1
fi
