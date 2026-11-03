#!/bin/bash
# Best measured setup (2026-10-14): big_s1_m640 detector alone, no TTA, NMS 0.7, thresholds tuned on VAL.
# Domain test: gun P 0.53 R 0.53 F1 0.53 | knife P 0.63 R 0.44 F1 0.52.  usage: ./run_best.sh <image|dir> <out_dir> [cpu|0]
cd "$(dirname "$0")"
../.venv/bin/python predict.py --model-dir ../runs/big_s1_m640 --combine det --no-tta --nms-iou 0.7 \
  --gun-conf 0.164 --knife-conf 0.168 --source "$1" --out "$2" --device "${3:-0}"
