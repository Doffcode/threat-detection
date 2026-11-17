"""
src/queue/queue_runner.py
=========================
Autonomous, fault-tolerant queue supervisor for Stage 1 Baseline training experiments.

Features:
- Sequential execution: baseline_m960_v3 -> baseline_m960_v4
- Status telemetry: outputs/queue/status.json and outputs/queue/queue.log
- Stop condition: checks outputs/queue/STOP between runs and every 10 minutes
- Crash/OOM recovery: max 2 retries per run; lowers workers by 1 (min 2) in checkpoint
- Diagnostic report: outputs/queue/failure_report.md on unrecoverable failures
- Thermal watchdog: logs nvidia-smi every 5m; pauses (SIGSTOP) if GPU >90C for 10m until <80C
- Free disk check: sets save_period=5 if free disk < 30 GB
- Detached execution resilience
"""

from __future__ import annotations

import collections
import datetime
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

WORKSPACE_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
os.chdir(WORKSPACE_ROOT)
sys.path.insert(0, str(WORKSPACE_ROOT))

PYTHON_BIN = WORKSPACE_ROOT / "work/.venv/bin/python"
QUEUE_DIR = WORKSPACE_ROOT / "outputs/queue"
STATUS_FILE = QUEUE_DIR / "status.json"
STOP_FILE = QUEUE_DIR / "STOP"
GPU_LOG_FILE = QUEUE_DIR / "gpu_log.txt"
FAILURE_REPORT_FILE = QUEUE_DIR / "failure_report.md"

QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamp() -> str:
    return datetime.datetime.now().astimezone().isoformat()


def write_status(state: Dict[str, Any]):
    state["updated_at"] = get_timestamp()
    tmp_file = STATUS_FILE.with_suffix(".tmp")
    with open(tmp_file, "w") as f:
        json.dump(state, f, indent=2)
    tmp_file.replace(STATUS_FILE)


def check_free_disk_gb() -> float:
    usage = shutil.disk_usage(WORKSPACE_ROOT)
    return usage.free / (1024 ** 3)


def ensure_mem_logger():
    try:
        res = subprocess.run(["pgrep", "-f", "scripts/mem_logger.sh"], capture_output=True, text=True)
        if not res.stdout.strip():
            print("[SUPERVISOR] Starting scripts/mem_logger.sh...")
            subprocess.Popen(
                ["bash", str(WORKSPACE_ROOT / "scripts/mem_logger.sh")],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        else:
            print(f"[SUPERVISOR] Confirmed active mem_logger PID(s): {res.stdout.strip()}")
    except Exception as e:
        print(f"[WARNING] Could not check/start mem_logger: {e}")


def get_gpu_telemetry() -> Tuple[float, float]:
    """Returns (temperature_c, memory_used_mb)."""
    try:
        cmd = ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used", "--format=csv,noheader,nounits"]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = res.stdout.strip().split(",")
        temp_c = float(parts[0].strip())
        mem_mb = float(parts[1].strip())
        return temp_c, mem_mb
    except Exception:
        return 0.0, 0.0


def get_host_memory_available_mb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def get_run_progress(run_dir: Path, target_epochs: int) -> Tuple[int, float, str]:
    """Reads results.csv, returns (completed_epochs, eta_seconds, eta_formatted)."""
    csv_file = run_dir / "results.csv"
    if not csv_file.exists():
        return 0, 0.0, "calculating"
    try:
        with open(csv_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        if len(lines) <= 1:
            return 0, 0.0, "calculating"
        
        epochs_done = len(lines) - 1
        if epochs_done >= target_epochs:
            return target_epochs, 0.0, "0m (finishing)"
            
        # Parse last few line times
        times = []
        for line in lines[-min(5, epochs_done):]:
            parts = line.split(",")
            if len(parts) > 1:
                try:
                    times.append(float(parts[1]))
                except ValueError:
                    pass
        
        avg_epoch_time = 680.0  # default ~11.3 min
        if len(times) >= 2:
            deltas = [times[i] - times[i-1] for i in range(1, len(times)) if times[i] > times[i-1]]
            if deltas:
                avg_epoch_time = sum(deltas) / len(deltas)
        elif len(times) == 1 and epochs_done > 0:
            avg_epoch_time = times[0] / epochs_done

        remaining_epochs = max(0, target_epochs - epochs_done)
        eta_seconds = remaining_epochs * avg_epoch_time
        hrs = int(eta_seconds // 3600)
        mins = int((eta_seconds % 3600) // 60)
        eta_str = f"{hrs}h {mins:02d}m" if hrs > 0 else f"{mins}m"
        return epochs_done, eta_seconds, eta_str
    except Exception:
        return 0, 0.0, "calculating"


def check_and_lower_workers(ckpt_path: Path) -> int:
    """Lowers workers by 1 (minimum 2) inside checkpoint train_args. Returns new worker count."""
    import torch
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ckpt.get("train_args", {})
    cur_workers = args.get("workers", 4)
    new_workers = max(2, cur_workers - 1)
    args["workers"] = new_workers
    ckpt["train_args"] = args
    torch.save(ckpt, ckpt_path)
    print(f"[SUPERVISOR] Checkpoint {ckpt_path.name}: workers lowered from {cur_workers} to {new_workers}.")
    return new_workers


def write_failure_report(run_name: str, reason: str):
    print(f"[SUPERVISOR] Writing failure report for {run_name} to {FAILURE_REPORT_FILE}...")
    oom_lines = "No journalctl OOM lines found."
    try:
        res = subprocess.run(["journalctl", "-k", "--no-pager", "-n", "40"], capture_output=True, text=True)
        filtered = [l for l in res.stdout.splitlines() if any(w in l.lower() for w in ["oom", "killed process", "out of memory"])]
        if filtered:
            oom_lines = "\n".join(filtered)
    except Exception as e:
        oom_lines = f"Error querying journalctl: {e}"

    free_m = "Unavailable"
    try:
        free_m = subprocess.check_output(["free", "-m"], text=True)
    except Exception:
        pass

    smi_out = "Unavailable"
    try:
        smi_out = subprocess.check_output(["nvidia-smi"], text=True)
    except Exception:
        pass

    content = f"""# Queue Failure Report: {run_name}
**Timestamp:** {get_timestamp()}
**Failure Reason:** {reason}

## 1. Kernel OOM Diagnostic Lines
```text
{oom_lines}
```

## 2. Host Memory State (`free -m`)
```text
{free_m}
```

## 3. GPU State (`nvidia-smi`)
```text
{smi_out}
```
"""
    with open(FAILURE_REPORT_FILE, "w") as f:
        f.write(content)


def run_experiment(run_name: str, config_file: Path, target_epochs: int, is_resume: bool = False):
    """Supervises training of one experiment with crash recovery, retry limits, and thermal monitoring."""
    run_dir = WORKSPACE_ROOT / "outputs/runs" / run_name
    weights_dir = run_dir / "weights"
    last_pt = weights_dir / "last.pt"
    
    retries = 0
    max_retries = 2
    thermal_overheat_count = 0  # number of consecutive 5-min checks >90C

    # Set up disk check
    free_disk = check_free_disk_gb()
    if free_disk < 30.0:
        print(f"[SUPERVISOR] Low disk space ({free_disk:.1f} GB < 30 GB). Setting save_period=5.")
        # inject save_period=5 into config
        import yaml
        with open(config_file) as f:
            cfg = yaml.safe_load(f)
        cfg["save_period"] = 5
        with open(config_file, "w") as f:
            yaml.safe_dump(cfg, f)

    while True:
        # Check STOP file before launching/resuming
        if STOP_FILE.exists():
            print(f"[SUPERVISOR] STOP file detected at {STOP_FILE}. Finishing checkpoint and exiting cleanly.")
            write_status({
                "status": "STOPPED_BY_USER",
                "current_run": run_name,
                "stage": "stopped",
                "retries": retries,
                "last_error": "STOP file encountered",
            })
            sys.exit(0)

        # Determine resume flag
        resume_flag = is_resume or (last_pt.exists() and run_dir.exists())

        # If resuming after crash or OOM, backup last.pt and adjust workers if needed
        if resume_flag and last_pt.exists():
            bak_path = weights_dir / f"last.pt.bak_supervisor_r{retries}_{int(time.time())}"
            try:
                shutil.copy2(last_pt, bak_path)
                print(f"[SUPERVISOR] Backed up last.pt to {bak_path.name}")
            except Exception as e:
                print(f"[WARNING] Backup of last.pt failed: {e}")

            # If retries > 0 or initial v3 resume after OOM, lower workers by 1
            if retries > 0 or (run_name == "baseline_m960_v3" and not Path("/tmp/v3_workers_adjusted").exists()):
                new_w = check_and_lower_workers(last_pt)
                Path("/tmp/v3_workers_adjusted").touch()

        # Build command
        cmd = [
            str(PYTHON_BIN),
            "src/detection/train.py",
            "--config", str(config_file),
            "--name", run_name,
        ]
        if resume_flag:
            cmd.append("--resume")

        print(f"\n[SUPERVISOR] Launching {run_name} (Attempt {retries + 1}/{max_retries + 1})...")
        print(f"Command: {' '.join(cmd)}")

        run_env = os.environ.copy()
        run_env["PYTHONPATH"] = f"{WORKSPACE_ROOT}:{run_env.get('PYTHONPATH', '')}"

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=run_env,
            preexec_fn=os.setsid,  # session leader
        )

        last_10m_check = time.time()
        last_5m_gpu_check = time.time()

        # Loop to monitor process output and watchdogs
        while proc.poll() is None:
            # Non-blocking read of process output
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                sys.stdout.flush()

            now = time.time()

            # 5-minute GPU check & thermal watchdog
            if now - last_5m_gpu_check >= 300:
                last_5m_gpu_check = now
                temp_c, mem_mb = get_gpu_telemetry()
                host_avail = get_host_memory_available_mb()
                epochs_done, eta_s, eta_str = get_run_progress(run_dir, target_epochs)

                # Log to gpu_log.txt
                with open(GPU_LOG_FILE, "a") as f:
                    f.write(f"{get_timestamp()} | GPU Temp: {temp_c} C | GPU VRAM: {mem_mb} MB | Host Avail: {host_avail:.1f} MB | Run: {run_name} | Ep: {epochs_done}/{target_epochs}\n")

                # Thermal check (>90C)
                if temp_c > 90.0:
                    thermal_overheat_count += 1
                    print(f"[THERMAL WARNING] GPU temp is {temp_c} C (>90 C)! Overheat count: {thermal_overheat_count}/2")
                    if thermal_overheat_count >= 2:
                        print(f"[THERMAL PAUSE] GPU temp remained >90 C for 10 minutes. Pausing training PID {proc.pid}...")
                        os.killpg(os.getpgid(proc.pid), signal.SIGSTOP)
                        write_status({
                            "status": "PAUSED_THERMAL",
                            "current_run": run_name,
                            "stage": "paused_thermal",
                            "gpu_temp_c": temp_c,
                            "epoch": epochs_done,
                            "total_epochs": target_epochs,
                            "eta_formatted": eta_str,
                            "retries": retries,
                        })
                        while temp_c >= 80.0:
                            time.sleep(60)
                            temp_c, _ = get_gpu_telemetry()
                            print(f"[THERMAL MONITOR] Cooling down... Current temp: {temp_c} C (target < 80 C)")
                        print(f"[THERMAL RESUME] GPU temp cooled to {temp_c} C. Resuming training PID {proc.pid}...")
                        os.killpg(os.getpgid(proc.pid), signal.SIGCONT)
                        thermal_overheat_count = 0
                else:
                    thermal_overheat_count = 0

                # Update status.json
                write_status({
                    "status": "RUNNING",
                    "current_run": run_name,
                    "stage": "training",
                    "epoch": epochs_done,
                    "total_epochs": target_epochs,
                    "progress_pct": round((epochs_done / target_epochs) * 100, 1),
                    "eta_seconds": round(eta_s, 1),
                    "eta_formatted": eta_str,
                    "gpu_temp_c": temp_c,
                    "gpu_vram_used_mb": mem_mb,
                    "host_mem_avail_mb": round(host_avail, 1),
                    "retries": retries,
                    "last_error": None,
                })

            # 10-minute STOP file check
            if now - last_10m_check >= 600:
                last_10m_check = now
                if STOP_FILE.exists():
                    print(f"[SUPERVISOR] STOP file detected during training! Allowing current epoch to finish...")
                    # Wait for next checkpoint save or terminate gracefully
                    time.sleep(30)
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=60)
                    write_status({
                        "status": "STOPPED_BY_USER",
                        "current_run": run_name,
                        "stage": "stopped",
                        "retries": retries,
                        "last_error": "STOP file encountered",
                    })
                    sys.exit(0)

        # Process terminated
        # Flush remaining stdout
        for line in proc.stdout:
            print(line, end="")
            sys.stdout.flush()

        ret_code = proc.returncode
        epochs_done, _, _ = get_run_progress(run_dir, target_epochs)
        print(f"\n[SUPERVISOR] Process {proc.pid} exited with code {ret_code}. Epochs completed: {epochs_done}/{target_epochs}")

        if ret_code == 0 and epochs_done >= target_epochs:
            print(f"[SUPERVISOR] {run_name} completed successfully!")
            return True

        # Failure / Crash handling
        err_msg = f"Process exited prematurely with exit code {ret_code} at epoch {epochs_done}/{target_epochs}"
        print(f"[SUPERVISOR ERROR] {err_msg}")

        # Check if crash was OOM
        is_oom = (ret_code == 137 or ret_code == -9)
        try:
            res = subprocess.run(["journalctl", "-k", "--no-pager", "--since", "-2min"], capture_output=True, text=True)
            if "oom-kill" in res.stdout.lower() or "killed process" in res.stdout.lower():
                is_oom = True
        except Exception:
            pass

        if is_oom:
            print("[SUPERVISOR] Failure diagnosed as LINUX KERNEL OUT-OF-MEMORY (OOM) KILL.")

        if retries < max_retries:
            retries += 1
            print(f"[SUPERVISOR] Retrying {run_name} (Retry {retries}/{max_retries})...")
            write_status({
                "status": "RETRYING",
                "current_run": run_name,
                "stage": "retrying_after_crash",
                "epoch": epochs_done,
                "total_epochs": target_epochs,
                "retries": retries,
                "last_error": err_msg,
            })
            time.sleep(10)
            is_resume = True
            continue
        else:
            print(f"[SUPERVISOR FATAL] Exceeded maximum retries ({max_retries}) for {run_name}. Halting queue.")
            write_failure_report(run_name, err_msg)
            write_status({
                "status": "FAILED",
                "current_run": run_name,
                "stage": "failed",
                "epoch": epochs_done,
                "total_epochs": target_epochs,
                "retries": retries,
                "last_error": err_msg,
            })
            sys.exit(1)


def main():
    print("=" * 80)
    print("AUTONOMOUS QUEUE SUPERVISOR INITIALIZATION")
    print(f"Timestamp: {get_timestamp()}")
    print(f"Workspace: {WORKSPACE_ROOT}")
    print("=" * 80)

    ensure_mem_logger()
    free_disk = check_free_disk_gb()
    print(f"[SUPERVISOR] Available Disk Space: {free_disk:.2f} GB")

    write_status({
        "status": "INITIALIZING",
        "current_run": None,
        "stage": "init",
        "retries": 0,
        "last_error": None,
    })

    # --------------------------------------------------------------------------
    # STEP 2: RUN V3 (baseline_m960_v3)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("STEP 2: RUN V3 (baseline_m960_v3) - 40 EPOCHS @ 960px (SCALE=0.90)")
    print("#" * 80)

    v3_cfg = WORKSPACE_ROOT / "configs/detection/yolo11m_960_v3.yaml"
    # Execute v3 training (will resume from epoch 12)
    run_experiment(
        run_name="baseline_m960_v3",
        config_file=v3_cfg,
        target_epochs=40,
        is_resume=True
    )

    # Post-training evaluation for v3
    print("\n[SUPERVISOR] Launching full evaluation suite for baseline_m960_v3...")
    write_status({
        "status": "RUNNING",
        "current_run": "baseline_m960_v3",
        "stage": "evaluation",
        "epoch": 40,
        "total_epochs": 40,
        "progress_pct": 100.0,
        "eta_formatted": "0m (evaluating)",
        "retries": 0,
        "last_error": None,
    })
    try:
        subprocess.run([str(PYTHON_BIN), "scripts/eval/run_full_v3_evaluation.py"], check=True)
        # Also run test_cctv_v2 evaluation
        subprocess.run([str(PYTHON_BIN), "scripts/eval/eval_cctv_test_v2.py"], check=True)
    except Exception as e:
        print(f"[WARNING] Post-v3 evaluation error: {e}")

    # Check STOP file before Step 3
    if STOP_FILE.exists():
        print(f"[SUPERVISOR] STOP file detected between runs. Exiting cleanly.")
        write_status({
            "status": "STOPPED_BY_USER",
            "current_run": "baseline_m960_v3",
            "stage": "stopped_between_runs",
            "retries": 0,
            "last_error": "STOP file encountered",
        })
        sys.exit(0)

    # --------------------------------------------------------------------------
    # STEP 3: RUN V4 (baseline_m960_v4)
    # --------------------------------------------------------------------------
    print("\n" + "#" * 80)
    print("STEP 3: RUN V4 (baseline_m960_v4) - 30 EPOCHS @ 1280px")
    print("#" * 80)

    v4_cfg = WORKSPACE_ROOT / "configs/detection/yolo11m_1280_v4.yaml"

    # 1. Dry-run gate for v4
    print("\n[SUPERVISOR] Executing V4 Dry-Run Gate (Profiling VRAM & Epoch Time)...")
    write_status({
        "status": "RUNNING",
        "current_run": "baseline_m960_v4",
        "stage": "dry_run_gate",
        "epoch": 0,
        "total_epochs": 30,
        "progress_pct": 0.0,
        "eta_formatted": "profiling",
        "retries": 0,
        "last_error": None,
    })

    try:
        from src.detection.dry_run_gate_v4 import run_dry_run_gate
        gate_res = run_dry_run_gate(batch_size=4, n_steps=60)
        gate_out = QUEUE_DIR / "v4_dry_run_gate.json"
        with open(gate_out, "w") as f:
            json.dump(gate_res, f, indent=2)
        print(f"[SUPERVISOR] Gate completed. Peak VRAM: {gate_res['peak_vram_gb']} GB. Recommended batch: {gate_res['recommended_batch']}")
        
        # Adjust batch in v4 config if recommended
        if gate_res["recommended_batch"] != 4:
            import yaml
            with open(v4_cfg) as f:
                c = yaml.safe_load(f)
            c["batch"] = gate_res["recommended_batch"]
            with open(v4_cfg, "w") as f:
                yaml.safe_dump(c, f)
            print(f"[SUPERVISOR] Updated {v4_cfg.name} with batch: {gate_res['recommended_batch']}")
    except Exception as e:
        print(f"[WARNING] V4 dry run gate profiling failed with error: {e}. Defaulting to batch=4.")

    # 2. Run v4 training
    run_experiment(
        run_name="baseline_m960_v4",
        config_file=v4_cfg,
        target_epochs=30,
        is_resume=False
    )

    # Post-training evaluation for v4
    print("\n[SUPERVISOR] Launching full evaluation suite for baseline_m960_v4...")
    write_status({
        "status": "RUNNING",
        "current_run": "baseline_m960_v4",
        "stage": "evaluation",
        "epoch": 30,
        "total_epochs": 30,
        "progress_pct": 100.0,
        "eta_formatted": "0m (evaluating)",
        "retries": 0,
        "last_error": None,
    })
    try:
        # Evaluate v4 using evaluate_detector
        from src.eval.evaluate_detector import run_full_evaluation
        v4_best = WORKSPACE_ROOT / "outputs/runs/baseline_m960_v4/weights/best.pt"
        data_root = WORKSPACE_ROOT / "data/processed/stage1_baseline"
        out_dir_v4 = WORKSPACE_ROOT / "outputs/eval/v4"
        out_dir_v4.mkdir(parents=True, exist_ok=True)
        splits = ["valid", "test_public", "test_domain", "test_cctv_v1", "test_hard_negatives"]
        run_full_evaluation(
            weights_path=v4_best,
            data_root=data_root,
            splits=splits,
            model_tag="v4",
            imgsz=1280,
            fixed_conf=0.25,
            iou_op=0.50,
            output_dir=out_dir_v4,
            device="cuda:0",
        )
        print("[SUPERVISOR] V4 evaluation completed successfully.")
    except Exception as e:
        print(f"[WARNING] Post-v4 evaluation error: {e}")

    # Complete
    print("\n" + "=" * 80)
    print("ALL QUEUED EXPERIMENTS COMPLETED CLEANLY!")
    print("=" * 80)
    write_status({
        "status": "COMPLETED",
        "current_run": "baseline_m960_v4",
        "stage": "done",
        "epoch": 30,
        "total_epochs": 30,
        "progress_pct": 100.0,
        "eta_seconds": 0.0,
        "eta_formatted": "0m (completed)",
        "retries": 0,
        "last_error": None,
    })


if __name__ == "__main__":
    main()
