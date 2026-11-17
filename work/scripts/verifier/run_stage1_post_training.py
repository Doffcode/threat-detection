#!/usr/bin/env python
"""Stage 1 Post-Training Pipeline: Crop Generation, Verifier Training, and Benchmark Evaluation.

Steps:
  1. Verify v5_hybrid_m640 weights.
  2. Generate stage-2 verifier crops (make_crops.py) with 20% pad and 224px letterbox.
  3. Train YOLO11s-cls verifier (train_verifier.py) for 25 epochs.
  4. Run eval_pipeline.py to calibrate pipeline_thresholds.json for >= 90% precision.
  5. Run dual benchmarks:
     - 144-image Indian cinema test (test_domain.txt)
     - 35-image Tabletop weapon test (val_tabletop.txt)
  6. Generate comprehensive Stage 1 completion report.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = HERE.parent.parent
PYTHON = WORK / ".venv" / "bin" / "python"


def run_cmd(cmd: list[str], desc: str):
    print(f"\n=======================================================")
    print(f"[post-train] {desc}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"=======================================================\n", flush=True)
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[error] {desc} failed with exit code {res.returncode}", file=sys.stderr)
        sys.exit(res.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector-weights",
                    default=str(WORK / "runs" / "v5_hybrid_m640" / "weights" / "best.pt"))
    ap.add_argument("--data",
                    default=str(WORK / "data" / "v5_indian_hybrid" / "data_hybrid.yaml"))
    ap.add_argument("--device", default="0")
    a = ap.parse_args()

    detector_weights = Path(a.detector_weights)
    if not detector_weights.exists():
        detector_weights = detector_weights.parent / "last.pt"
    if not detector_weights.exists():
        raise SystemExit(f"Detector weights not found: {a.detector_weights}")

    crops_dir = WORK / "data" / "verifier_v5_crops"
    verifier_run_name = "verifier_v5"
    verifier_weights = WORK / "runs" / verifier_run_name / "weights" / "best.pt"

    # Step 1: Make crops
    run_cmd([
        str(PYTHON), str(WORK / "scripts" / "verifier" / "make_crops.py"),
        "--data", a.data,
        "--weights", str(detector_weights),
        "--out", str(crops_dir),
        "--overwrite",
    ], "Generating 224px stage-2 crops from hybrid validation set")

    # Step 2: Train verifier (YOLO11s-cls)
    run_cmd([
        str(PYTHON), str(WORK / "scripts" / "verifier" / "train_verifier.py"),
        "--data", str(crops_dir),
        "--name", verifier_run_name,
        "--epochs", "25",
        "--batch", "64",
        "--device", a.device,
    ], "Training YOLO11s-cls Stage-2 crop verifier (25 epochs)")

    # Step 3: Calibrate pipeline thresholds for high precision
    run_cmd([
        str(PYTHON), str(WORK / "scripts" / "verifier" / "eval_pipeline.py"),
        "--weights", str(detector_weights),
        "--verifier", str(verifier_weights),
        "--data", a.data,
        "--split", "val",
        "--write-thresholds",
    ], "Calibrating pipeline_thresholds.json on validation split")

    # Step 4: Evaluate pipeline on locked Indian cinema test set
    run_cmd([
        str(PYTHON), str(WORK / "scripts" / "verifier" / "eval_pipeline.py"),
        "--weights", str(detector_weights),
        "--verifier", str(verifier_weights),
        "--data", a.data,
        "--split", "test",
        "--tag", "domain_test",
    ], "Evaluating two-stage pipeline on locked 144-image Indian cinema test set")

    # Step 5: Evaluate detector on Tabletop benchmark
    tabletop_yaml = WORK / "data" / "v5_indian_hybrid" / "data_tabletop.yaml"
    run_cmd([
        str(PYTHON), str(WORK / "scripts" / "train" / "eval.py"),
        "--weights", str(detector_weights),
        "--data", str(tabletop_yaml),
        "--split", "val",
        "--tag", "tabletop",
    ], "Evaluating detector on 35-image dedicated tabletop benchmark")

    print("\n=======================================================")
    print("[post-train] Stage 1 post-training pipeline completed successfully!")
    print(f"Detector: {detector_weights}")
    print(f"Verifier: {verifier_weights}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
