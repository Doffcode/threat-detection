"""
src/detection/dry_run_gate_v4.py
================================
Profiles peak VRAM and step duration for v4 at imgsz 1280, batch 4 over 100 steps.
If peak VRAM > 11.2 GB, recommends batch 2.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

from ultralytics import YOLO


def run_dry_run_gate(batch_size: int = 4, n_steps: int = 100):
    print("=" * 70)
    print(f"RUNNING V4 DRY-RUN GATE: imgsz=1280, batch={batch_size}, steps={n_steps}")
    print("=" * 70)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available for dry run gate profiling")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    config_path = WORKSPACE_ROOT / "configs/detection/yolo11m_1280_v4.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Use small subset or 1 epoch with limited max steps via custom callback or dry run
    # Ultralytics train doesn't have max_steps directly, but we can measure during training or run a torch forward/backward loop on synthetic/dataloader batch
    from ultralytics.data import build_yolo_dataset, build_dataloader
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.loss import v8DetectionLoss

    data_cfg = check_det_dataset(cfg["data"])
    train_path = data_cfg["train"]

    model_weights = cfg["model"]
    model = YOLO(model_weights)
    nn_model = model.model.to("cuda:0")
    nn_model.train()

    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=0.001)
    loss_fn = v8DetectionLoss(nn_model)

    dataset = build_yolo_dataset(
        cfg=cfg,
        img_path=train_path,
        batch=batch_size,
        data=data_cfg,
        mode="train",
        rect=False,
        stride=32
    )

    loader = build_dataloader(dataset, batch=batch_size, workers=2, shuffle=True)

    print(f"[INFO] Dataloader created. Total batches in full epoch: {len(loader)}")
    
    start_time = time.perf_counter()
    step_times = []

    for step, batch in enumerate(loader):
        if step >= n_steps:
            break
        t0 = time.perf_counter()
        optimizer.zero_grad()
        
        imgs = batch["img"].to("cuda:0", non_blocking=True).float() / 255.0
        # resize/check shape
        preds = nn_model(imgs)
        loss, _ = loss_fn(preds, batch)
        loss.backward()
        optimizer.step()
        
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        step_times.append(dt)
        if (step + 1) % 25 == 0 or step == 0:
            cur_vram = torch.cuda.max_memory_allocated() / (1024**3)
            print(f"  Step {step + 1}/{n_steps}: step_time={dt:.3f}s | Peak VRAM={cur_vram:.2f} GB")

    total_time = time.perf_counter() - start_time
    peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    avg_step_time = sum(step_times) / len(step_times) if step_times else 0.0
    est_epoch_seconds = avg_step_time * len(loader)

    print("-" * 70)
    print(f"RESULTS FOR BATCH={batch_size}:")
    print(f"  Peak VRAM            : {peak_vram_gb:.2f} GB (Limit: 11.20 GB)")
    print(f"  Avg Step Time        : {avg_step_time:.3f} s")
    print(f"  Est. Seconds/Epoch   : {est_epoch_seconds:.1f} s (~{est_epoch_seconds/60:.1f} min)")
    
    use_batch = batch_size
    if peak_vram_gb > 11.2:
        print(f"[WARNING] Peak VRAM ({peak_vram_gb:.2f} GB) exceeds 11.2 GB limit! Recommending batch=2.")
        use_batch = 2
    else:
        print(f"[PASS] Peak VRAM ({peak_vram_gb:.2f} GB) is within 11.2 GB limit. Recommending batch={batch_size}.")

    del nn_model, optimizer, loss_fn, loader, dataset
    torch.cuda.empty_cache()

    result = {
        "tested_batch": batch_size,
        "n_steps": n_steps,
        "peak_vram_gb": round(peak_vram_gb, 3),
        "avg_step_time_s": round(avg_step_time, 4),
        "total_batches_per_epoch": len(loader) if 'loader' in locals() else 0,
        "est_epoch_seconds": round(est_epoch_seconds, 1),
        "recommended_batch": use_batch,
        "gate_status": "PASS" if peak_vram_gb <= 11.2 else "FALLBACK_BATCH_2",
    }
    return result


if __name__ == "__main__":
    res = run_dry_run_gate(batch_size=4, n_steps=50)
    out_path = WORKSPACE_ROOT / "outputs/queue/v4_dry_run_gate.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[SAVED] Results saved to {out_path}")
