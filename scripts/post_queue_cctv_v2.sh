#!/usr/bin/env bash
# ==============================================================================
# scripts/post_queue_cctv_v2.sh
# Detached post-queue evaluator for test_cctv_v2 benchmark.
# Triggers on: summary.md, failure_report.md, or status COMPLETED/FAILED/STOPPED.
# Evaluates whichever models (v1-v4) have best.pt:
# - Full-image vs SAHI (640 slices, 20% overlap)
# - Block-bootstrap CIs (contiguous 8-10 frame blocks)
# - Recall by box-size bucket
# - FP per image with and without ignore-frames
# - 92-frame visual contact sheet (GT green, predictions red)
# - Missed weapons manifest
# - Appends plain-numbers comparative table to docs/BASELINE_ERROR_ANALYSIS.md
# ==============================================================================

set -euo pipefail

WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
cd "${WORKSPACE_ROOT}"

POST_LOG="${WORKSPACE_ROOT}/outputs/queue/post_queue_cctv_v2.log"
POST_PID_FILE="${WORKSPACE_ROOT}/outputs/queue/post_queue_cctv_v2.pid"

mkdir -p "${WORKSPACE_ROOT}/outputs/queue"

# Check if already running
if [[ -f "${POST_PID_FILE}" ]]; then
    EXISTING_PID=$(cat "${POST_PID_FILE}")
    if kill -0 "${EXISTING_PID}" 2>/dev/null; then
        echo "[INFO] post_queue_cctv_v2 is already running with PID ${EXISTING_PID}."
        exit 0
    fi
fi

echo "======================================================================"
echo "LAUNCHING DETACHED POST-QUEUE CCTV V2 EVALUATOR"
echo "======================================================================"

setsid nohup "${WORKSPACE_ROOT}/work/.venv/bin/python" -u "${WORKSPACE_ROOT}/scripts/eval/eval_post_queue_cctv_v2.py" >> "${POST_LOG}" 2>&1 &
RUNNER_PID=$!
echo "${RUNNER_PID}" > "${POST_PID_FILE}"

sleep 1

if kill -0 "${RUNNER_PID}" 2>/dev/null; then
    echo "[SUCCESS] post_queue_cctv_v2 started detached!"
    echo "  PID      : ${RUNNER_PID}"
    echo "  Log Path : ${POST_LOG}"
else
    echo "[ERROR] post_queue_cctv_v2 failed to start!"
    exit 1
fi
