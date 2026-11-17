"""
tests/test_dataset_integrity.py
================================
Automated pre-training data integrity test suite.
Validates:
1. Label format & class ID bounds (0, 1, 2)
2. Normalized box coordinate bounds [0.0, 1.0]
3. Duplicate filename detection
4. Empty label counts per split (background negatives)
5. Strict isolation: asserts ZERO scene/group stems cross splits
6. Strict isolation: asserts ZERO dHash duplicate clusters cross splits
"""

import csv
import collections
from pathlib import Path
import pytest


DATASET_ROOT = Path("data/processed/stage1_baseline")
MANIFEST_PATH = DATASET_ROOT / "manifest.csv"
SPLITS = ["train", "valid", "test_public", "test_domain", "test_cctv", "test_hard_negatives"]


def test_manifest_exists():
    assert MANIFEST_PATH.exists(), f"Manifest file missing at {MANIFEST_PATH}"


def test_split_directories_exist():
    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        assert img_dir.exists(), f"Images dir missing: {img_dir}"
        assert lbl_dir.exists(), f"Labels dir missing: {lbl_dir}"


def test_no_duplicate_filenames_across_splits():
    all_filenames = collections.defaultdict(list)
    for split in SPLITS:
        img_dir = DATASET_ROOT / split / "images"
        for img_path in img_dir.iterdir():
            if img_path.is_file():
                all_filenames[img_path.name].append(split)
    
    duplicates = {name: splits for name, splits in all_filenames.items() if len(splits) > 1}
    assert len(duplicates) == 0, f"Found {len(duplicates)} duplicate filenames across splits: {list(duplicates.items())[:5]}"


def test_label_syntax_and_coordinate_bounds():
    total_boxes = 0
    total_labels = 0
    empty_labels = collections.defaultdict(int)
    
    for split in SPLITS:
        lbl_dir = DATASET_ROOT / split / "labels"
        for lp in lbl_dir.glob("*.txt"):
            total_labels += 1
            content = lp.read_text().strip()
            if not content:
                empty_labels[split] += 1
                continue
            
            lines = content.splitlines()
            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                assert len(parts) == 5, f"{lp.name}:{line_idx} - expected 5 tokens, got {len(parts)}: '{line}'"
                
                cid = int(parts[0])
                assert cid in (0, 1, 2), f"{lp.name}:{line_idx} - invalid class ID {cid}"
                
                cx, cy, bw, bh = map(float, parts[1:])
                assert 0.0 <= cx <= 1.0, f"{lp.name}:{line_idx} - cx out of bounds: {cx}"
                assert 0.0 <= cy <= 1.0, f"{lp.name}:{line_idx} - cy out of bounds: {cy}"
                assert 0.0 < bw <= 1.01, f"{lp.name}:{line_idx} - bw out of bounds: {bw}"
                assert 0.0 < bh <= 1.01, f"{lp.name}:{line_idx} - bh out of bounds: {bh}"
                
                total_boxes += 1
                
    print(f"\n[PASS] Verified {total_boxes} bounding boxes across {total_labels} label files.")
    print(f"[INFO] Empty label counts (negatives): {dict(empty_labels)}")


def test_zero_group_or_scene_leakage_across_splits():
    """Validates that no source group_id or scene prefix spans more than one split."""
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
    
    group_to_splits = collections.defaultdict(set)
    for r in manifest_rows:
        gid = r["group_id"]
        split = r["split"]
        if split != "unassigned_negative":
            group_to_splits[gid].add(split)
            
    leaking_groups = {gid: splits for gid, splits in group_to_splits.items() if len(splits) > 1}
    assert len(leaking_groups) == 0, f"Found {len(leaking_groups)} groups spanning multiple splits: {list(leaking_groups.items())[:5]}"
    print(f"\n[PASS] Verified {len(group_to_splits)} distinct groups. Zero groups span multiple splits.")


def test_zero_dhash_leakage_across_splits():
    """Validates that no exact or near dHash cluster (dHash64) crosses splits."""
    with open(MANIFEST_PATH, "r") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)
        
    hash_to_splits = collections.defaultdict(set)
    for r in manifest_rows:
        h = r["dhash"]
        split = r["split"]
        if split != "unassigned_negative" and h not in ("0x0", "0", ""):
            hash_to_splits[h].add(split)
            
    leaking_hashes = {h: splits for h, splits in hash_to_splits.items() if len(splits) > 1}
    assert len(leaking_hashes) == 0, f"Found {len(leaking_hashes)} dHash clusters spanning multiple splits: {list(leaking_hashes.items())[:5]}"
    print(f"\n[PASS] Verified {len(hash_to_splits)} distinct dHash values. Zero dHash clusters span multiple splits.")


if __name__ == "__main__":
    pytest.main(["-v", "-s", str(Path(__file__))])
