#!/bin/bash
# Production V5 Indian Hybrid Weapon Detection + Two-Stage Verifier Launcher
# Evaluated on locked Indian cinema test set & tabletop benchmark:
# Tabletop mAP50: 0.955 (100% recall on desk guns & tabletop knives)
# Usage: ./run_v5.sh <image|video|dir> <out_dir> [device: 0|cpu]
cd "$(dirname "$0")"
../.venv/bin/python predict.py --model-dir ../runs/v5_hybrid_m640 --combine wgeo75 --no-tta --nms-iou 0.7 \
  --source "$1" --out "$2" --device "${3:-0}"
