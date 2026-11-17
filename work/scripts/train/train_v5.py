#!/usr/bin/env python
"""Train YOLO11m on v5_indian_hybrid dataset (40 epochs @ 640px) and evaluate on all benchmarks.

Benchmarks evaluated post-training:
  1. valid_domain (work/data/v5_indian_hybrid/valid_domain.txt) -> chooses optimal thresholds.json
  2. test_domain (work/data/v5_indian_hybrid/test_domain.txt) -> locked 144-image Indian cinema benchmark
  3. val_tabletop (work/data/v5_indian_hybrid/val_tabletop.txt) -> 35-image tabletop weapon benchmark
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
sys.path.insert(0, str(HERE))
from eval import run_eval  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default="v5_hybrid_m640")
    a = ap.parse_args()

    data_hybrid = WORK / "data" / "v5_indian_hybrid" / "data_hybrid.yaml"
    data_tabletop = WORK / "data" / "v5_indian_hybrid" / "data_tabletop.yaml"
    base_model = WORK / "raw" / "IntProj_Weapon_Dataset_annot" / "yolo11m.pt"
    project_dir = WORK / "runs"

    if not data_hybrid.exists():
        raise SystemExit(f"Data config not found: {data_hybrid}")
    if not base_model.exists():
        raise SystemExit(f"Base model not found: {base_model}")

    train_cfg = dict(
        data=str(data_hybrid.resolve()),
        epochs=a.epochs,
        batch=a.batch,
        imgsz=a.imgsz,
        optimizer="AdamW",
        lr0=0.005,
        lrf=0.01,
        cos_lr=True,
        close_mosaic=5,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        scale=0.5,
        translate=0.1,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        patience=20,
        device=a.device,
        workers=a.workers,
        project=str(project_dir.resolve()),
        name=a.name,
        exist_ok=True,
        plots=True,
        seed=0,
        deterministic=False,
    )

    print(f"[v5 train] Initializing YOLO11m from {base_model.resolve()}...")
    print(f"[v5 train] Config:\n" + json.dumps(train_cfg, indent=2, default=str))

    model = YOLO(str(base_model.resolve()))
    model.train(**train_cfg)

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    if not best_weights.exists():
        best_weights = save_dir / "weights" / "last.pt"
    print(f"\n[v5 train] Training complete! Best weights: {best_weights.resolve()}\n")

    # 1. Evaluate on valid_domain to tune thresholds
    print("[v5 eval] 1/3: Evaluating valid_domain and saving thresholds.json...")
    run_eval(best_weights, data_hybrid, "val", imgsz=a.imgsz, device=a.device,
             run_dir=save_dir, write_thresholds=True, workers=a.workers)

    # 2. Evaluate on test_domain (144 Indian cinema test images)
    print("\n[v5 eval] 2/3: Evaluating test_domain (Indian cinema benchmark)...")
    run_eval(best_weights, data_hybrid, "test", imgsz=a.imgsz, device=a.device,
             run_dir=save_dir, write_thresholds=False, workers=a.workers, tag="domain")

    # 3. Evaluate on val_tabletop (35 tabletop weapon images)
    print("\n[v5 eval] 3/3: Evaluating val_tabletop (tabletop weapon benchmark)...")
    run_eval(best_weights, data_tabletop, "val", imgsz=a.imgsz, device=a.device,
             run_dir=save_dir, write_thresholds=False, workers=a.workers, tag="tabletop")

    print("\n=======================================================")
    print(f"[v5 train & eval] All runs completed successfully!")
    print(f"Results directory: {save_dir}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
