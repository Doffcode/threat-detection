#!/usr/bin/env bash
# ==============================================================================
# scripts/mem_logger.sh
# Lightweight background memory telemetry logger.
# Appends free -m and top 5 processes by RSS to outputs/runs/mem_log.txt every 60s.
# ==============================================================================

set -euo pipefail
WORKSPACE_ROOT="/home/shiva/Desktop/projects/akriti/weapon-detection"
LOG_FILE="${WORKSPACE_ROOT}/outputs/runs/mem_log.txt"

mkdir -p "${WORKSPACE_ROOT}/outputs/runs"

while true; do
    {
        echo "======================================================================"
        echo "TIMESTAMP: $(date -Iseconds)"
        echo "--- FREE MEMORY (MB) ---"
        free -m
        echo "--- TOP 5 PROCESSES BY RSS ---"
        ps -eo pid,user,%mem,rss,comm --sort=-rss | head -n 6
    } >> "${LOG_FILE}"
    sleep 60
done
