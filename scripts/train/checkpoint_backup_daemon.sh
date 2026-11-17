#!/usr/bin/env bash
# ==============================================================================
# scripts/train/checkpoint_backup_daemon.sh
# Monitors training and backs up last.pt every 5 epochs (5, 10, 15, 20, 25, 30, 35, 40).
# Exits when the target training PID terminates.
# ==============================================================================

set -euo pipefail

TRAIN_PID="${1:?Usage: $0 <TRAIN_PID> <RUN_DIR>}"
RUN_DIR="${2:?Usage: $0 <TRAIN_PID> <RUN_DIR>}"

LOG_PREFIX="[BACKUP_DAEMON]"
echo "${LOG_PREFIX} Started monitoring PID ${TRAIN_PID} in ${RUN_DIR}"

CSV_FILE="${RUN_DIR}/results.csv"
WEIGHTS_DIR="${RUN_DIR}/weights"
LAST_PT="${WEIGHTS_DIR}/last.pt"

while kill -0 "${TRAIN_PID}" 2>/dev/null; do
    if [[ -f "${CSV_FILE}" && -f "${LAST_PT}" ]]; then
        # Count non-header rows in results.csv to determine completed epochs
        NUM_ROWS=$(wc -l < "${CSV_FILE}")
        if [[ ${NUM_ROWS} -ge 2 ]]; then
            COMPLETED_EPOCHS=$(( NUM_ROWS - 1 ))
            # Check multiples of 5 up to completed epochs
            for (( ep=5; ep<=COMPLETED_EPOCHS; ep+=5 )); do
                BAK_FILE="${WEIGHTS_DIR}/last.pt.bak_epoch_${ep}"
                if [[ ! -f "${BAK_FILE}" ]]; then
                    cp "${LAST_PT}" "${BAK_FILE}"
                    echo "${LOG_PREFIX} $(date -Iseconds) Backed up last.pt to ${BAK_FILE} (Epoch ${ep} completed)"
                fi
            done
        fi
    fi
    sleep 30
done

# Final check upon process completion
if [[ -f "${CSV_FILE}" && -f "${LAST_PT}" ]]; then
    NUM_ROWS=$(wc -l < "${CSV_FILE}")
    if [[ ${NUM_ROWS} -ge 2 ]]; then
        COMPLETED_EPOCHS=$(( NUM_ROWS - 1 ))
        for (( ep=5; ep<=COMPLETED_EPOCHS; ep+=5 )); do
            BAK_FILE="${WEIGHTS_DIR}/last.pt.bak_epoch_${ep}"
            if [[ ! -f "${BAK_FILE}" ]]; then
                cp "${LAST_PT}" "${BAK_FILE}"
                echo "${LOG_PREFIX} $(date -Iseconds) Final backup of last.pt to ${BAK_FILE} (Epoch ${ep})"
            fi
        done
    fi
fi

echo "${LOG_PREFIX} Training process ${TRAIN_PID} has finished. Backup daemon exiting."
