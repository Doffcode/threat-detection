"""
scripts/freeze_test_splits.py
=============================
Freezes the exact membership of all test splits (test_domain, test_cctv,
test_hard_negatives, test_public) into configs/detection/reproducibility_manifest.json
with SHA256 hashes of the file listings and the master manifest.
"""

from pathlib import Path
import json
import hashlib
import pandas as pd

def freeze_splits():
    root = Path("data/processed/stage1_baseline")
    out_file = Path("configs/detection/reproducibility_manifest.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    
    manifest_p = root / "manifest.csv"
    with open(manifest_p, "rb") as f:
        manifest_sha = hashlib.sha256(f.read()).hexdigest()
        
    df = pd.read_csv(manifest_p)
    
    test_splits = ["test_domain", "test_cctv_v1", "test_hard_negatives", "test_public"]
    frozen_data = {
        "manifest_sha256": manifest_sha,
        "total_active_images": len(df),
        "classes": {0: "handgun", 1: "rifle", 2: "knife"},
        "test_splits": {}
    }
    
    for split_name in test_splits:
        split_dir = root / split_name / "images"
        filenames = sorted([p.name for p in split_dir.glob("*.*")])
        
        # Compute SHA256 of the concatenated sorted filenames
        list_str = "\n".join(filenames).encode("utf-8")
        split_hash = hashlib.sha256(list_str).hexdigest()
        
        frozen_data["test_splits"][split_name] = {
            "count": len(filenames),
            "sha256_filelist": split_hash,
            "filenames": filenames
        }
        print(f"Frozen {split_name:20s}: {len(filenames):4d} images | SHA256: {split_hash[:12]}...")

    with open(out_file, "w") as f:
        json.dump(frozen_data, f, indent=2)
        
    print(f"\n[SUCCESS] Reproducibility manifest saved to {out_file}")

if __name__ == "__main__":
    freeze_splits()
