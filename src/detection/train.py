"""
src/detection/train.py
======================
Config-driven, reproducible training script for Stage 1 Baseline Weapon Detector.
Supports:
- Ultralytics Albumentations injection with CCTV-realistic pipeline:
  (MotionBlur, GaussNoise, RandomBrightnessContrast, ImageCompression, ColorJitter)
- 1-epoch dry-run on 100% of rebuilt data with augmentations ON at 960px, batch 8, FP16.
- Full 15-epoch training with deterministic seeding (seed=2026).
- VRAM profiling and performance metrics logging.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
if str(_WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE_ROOT))

import yaml
import torch
import numpy as np

# 1. Inject Custom CCTV Albumentations into Ultralytics native pipeline
import albumentations as A
import ultralytics.data.augment as augment
from src.detection.albumentations_cctv import get_cctv_train_transform

cctv_transforms = get_cctv_train_transform().transforms
_orig_alb_init = augment.Albumentations.__init__

def patched_alb_init(self, p: float = 1.0, transforms: list | None = None, flip_idx: list[int] | None = None) -> None:
    if transforms is None:
        transforms = cctv_transforms
    _orig_alb_init(self, p=p, transforms=transforms, flip_idx=flip_idx)

augment.Albumentations.__init__ = patched_alb_init

from ultralytics import YOLO


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Train Baseline Weapon Detector")
    parser.add_argument("--config", type=str, default="configs/detection/yolo11m_960_baseline.yaml", help="Path to config yaml")
    parser.add_argument("--dry-run", action="store_true", help="Run 1-epoch dry run on 100% of rebuilt data")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    parser.add_argument("--batch", type=int, default=None, help="Override batch size from config")
    parser.add_argument("--imgsz", type=int, default=None, help="Override image size from config")
    parser.add_argument("--name", type=str, default=None, help="Override run name")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Command line overrides
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch is not None:
        cfg["batch"] = args.batch
    if args.imgsz is not None:
        cfg["imgsz"] = args.imgsz
    if args.name is not None:
        cfg["name"] = args.name
    if args.resume:
        cfg["resume"] = True

    if args.dry_run:
        cfg["epochs"] = 1
        cfg["name"] = "dry_run_rebuilt_m960"
        # 100% of training data used for full 1-epoch dry run

    print("=" * 65)
    print("STAGE 1 BASELINE DETECTOR: TRAINING INITIALIZATION")
    print("=" * 65)
    print(f"Git Commit        : {get_git_commit()}")
    print(f"PyTorch Version   : {torch.__version__}")
    print(f"CUDA Available    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name       : {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM (GB)   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}")
    print(f"Config File       : {config_path}")
    print(f"Model Backbone    : {cfg.get('model')}")
    print(f"Input Resolution  : {cfg.get('imgsz')}x{cfg.get('imgsz')} px")
    print(f"Batch Size        : {cfg.get('batch')}")
    print(f"Target Epochs     : {cfg.get('epochs')}")
    print(f"Dry Run Mode      : {args.dry_run}")
    print("\n[ACTIVE AUGMENTATIONS PROOF]")
    for t in cctv_transforms:
        print(f"  - {t.__class__.__name__} (p={t.p})")
    print("=" * 65)

    # Seed
    torch.manual_seed(cfg.get("seed", 2026))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.get("seed", 2026))
        torch.cuda.reset_peak_memory_stats()

    model_path = cfg.pop("model")
    start_time = time.perf_counter()
    if args.resume:
        last_ckpt = Path(cfg.get("project", "outputs/runs")) / cfg.get("name", "baseline_m960") / "weights" / "last.pt"
        if last_ckpt.exists():
            print(f"[INFO] Resuming training from checkpoint: {last_ckpt}")
            model = YOLO(str(last_ckpt))
            results = model.train(resume=True)
        else:
            print(f"[ERROR] Resume requested but checkpoint not found at {last_ckpt}")
            sys.exit(1)
    else:
        model = YOLO(model_path)
        results = model.train(**cfg)
    total_time = time.perf_counter() - start_time

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    print("\n" + "=" * 65)
    print("TRAINING SESSION COMPLETED")
    print("=" * 65)
    print(f"Total Wall Time   : {total_time:.2f} seconds ({total_time / 60:.2f} minutes)")
    print(f"Peak GPU VRAM     : {peak_vram_gb:.2f} GB / 12.47 GB")
    
    # Extract validation metrics
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print("\n=== Validation Results (mAP @ 0.5) ===")
        print(f"mAP50 (all)       : {rd.get('metrics/mAP50(B)', 0.0):.4f}")
        print(f"mAP50-95 (all)    : {rd.get('metrics/mAP50-95(B)', 0.0):.4f}")
        print(f"Box Loss          : {rd.get('train/box_loss', 0.0):.4f}")
        print(f"Cls Loss          : {rd.get('train/cls_loss', 0.0):.4f}")
        print(f"DFL Loss          : {rd.get('train/dfl_loss', 0.0):.4f}")
        
    save_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path("outputs/runs/dry_run_rebuilt_m960")
    print(f"\nRun Output Directory: {save_dir}")
    print("Augmented Batch Proof Files:")
    for b in ["train_batch0.jpg", "train_batch1.jpg", "train_batch2.jpg"]:
        bp = save_dir / b
        if bp.exists():
            print(f"  - {bp} (exists: {bp.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"  - {bp} (not found)")
    print("=" * 65)


if __name__ == "__main__":
    main()
