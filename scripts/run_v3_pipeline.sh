#!/usr/bin/env bash
# ==============================================================================
# scripts/run_v3_pipeline.sh
# End-to-end pipeline:
# 1. Launches v3 training detached (setsid nohup) with backup daemon.
# 2. Prints PID, log path, and resume command.
# 3. Monitors training without running any competing jobs.
# 4. If training fails/dies (OOM): reports evidence and stops (no restart).
# 5. If training finishes cleanly (40 epochs): runs post-training evaluation suite.
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

echo "======================================================================"
echo "LAUNCHING STAGE 1 BASELINE V3 DETACHED PIPELINE"
echo "======================================================================"

# Verify memory logger is active
if ! pgrep -f "scripts/mem_logger.sh" > /dev/null; then
    echo "[INFO] Starting memory telemetry logger..."
    nohup bash "${WORKSPACE_ROOT}/scripts/mem_logger.sh" > /dev/null 2>&1 &
fi

# Step 1: Launch training via run_train_baseline_v3.sh
bash "${WORKSPACE_ROOT}/scripts/run_train_baseline_v3.sh"

RUN_DIR="${WORKSPACE_ROOT}/outputs/runs/baseline_m960_v3"
LOG_FILE="${WORKSPACE_ROOT}/outputs/runs/baseline_m960_v3.log"

# Locate the training PID from ps
sleep 3
TRAIN_PID=$(pgrep -f "src/detection/train.py.*baseline_m960_v3" | head -n 1 || true)
if [[ -z "${TRAIN_PID}" ]]; then
    echo "[ERROR] Failed to find running training PID for baseline_m960_v3!"
    exit 1
fi

echo "[PIPELINE] Confirmed active training PID: ${TRAIN_PID}"
echo "[PIPELINE] Training is running detached under setsid nohup."
echo "[PIPELINE] Monitoring progress until completion (no competing CPU/GPU jobs allowed)..."

# Monitor loop
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    sleep 30
done

echo "[PIPELINE] Training process ${TRAIN_PID} has terminated at $(date -Iseconds)."
sleep 5

# Check if run succeeded (40 epochs in results.csv or best.pt present)
BEST_PT="${RUN_DIR}/weights/best.pt"
CSV_FILE="${RUN_DIR}/results.csv"

NUM_EPOCHS=0
if [[ -f "${CSV_FILE}" ]]; then
    NUM_ROWS=$(wc -l < "${CSV_FILE}")
    NUM_EPOCHS=$(( NUM_ROWS - 1 ))
fi

echo "[PIPELINE] Completed epochs recorded: ${NUM_EPOCHS} / 40"

if [[ -f "${BEST_PT}" && ${NUM_EPOCHS} -ge 40 ]]; then
    echo "[SUCCESS] Training completed all 40 epochs cleanly!"
    echo "[PIPELINE] Launching post-training evaluation suite (identical v1/v2 protocol)..."
    export PYTHONPATH="${WORKSPACE_ROOT}:${PYTHONPATH:-}"
    "${WORKSPACE_ROOT}/work/.venv/bin/python" "${WORKSPACE_ROOT}/scripts/eval/run_full_v3_evaluation.py"
    echo "[PIPELINE] Post-training evaluation completed successfully."
else
    echo "[ERROR] Training process terminated prematurely or failed before completing 40 epochs!"
    echo "--- LAST 50 LINES OF LOG ---"
    tail -n 50 "${LOG_FILE}" || true
    echo "--- MEMORY STATE ---"
    free -h || true
    echo "--- DMESG OOM EVIDENCE ---"
    dmesg -T | grep -i -E "oom|killed process" | tail -n 10 || true
    exit 137
fi
