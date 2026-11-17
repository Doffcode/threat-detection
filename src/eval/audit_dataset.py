"""
src/eval/audit_dataset.py
=========================
Step 1: Comprehensive Dataset Audit Script.
Analyzes annotation formats, class balance, bounding box dimensions,
small-object ratios (COCO small <32px, CCTV 20-60px), and label noise/conflicts.
"""

import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np

def audit_sohas(sohas_root: Path):
    """Audit the Sohas / OD-WeaponDetection dataset."""
    print("=" * 60)
    print("1. AUDITING OD-WEAPONDETECTION (SOHAS)")
    print(f"Path: {sohas_root}")
    print("=" * 60)
    
    xml_files = list(sohas_root.glob("**/*.xml"))
    print(f"Found {len(xml_files)} XML annotation files.")
    
    class_counts = Counter()
    box_sizes = []
    box_widths = []
    box_heights = []
    aspect_ratios = []
    image_sizes = Counter()
    degenerate_boxes = 0
    images_with_boxes = 0
    
    for xml_path in xml_files:
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size_elem = root.find("size")
            if size_elem is not None:
                w = int(size_elem.find("width").text or 0)
                h = int(size_elem.find("height").text or 0)
                image_sizes[(w, h)] += 1
            else:
                w, h = 640, 480  # fallback
                
            objs = root.findall("object")
            if objs:
                images_with_boxes += 1
                
            for obj in objs:
                name = obj.find("name").text.strip().lower()
                class_counts[name] += 1
                
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                bw = xmax - xmin
                bh = ymax - ymin
                
                if bw <= 0 or bh <= 0:
                    degenerate_boxes += 1
                    continue
                    
                area = bw * bh
                box_sizes.append(area)
                box_widths.append(bw)
                box_heights.append(bh)
                aspect_ratios.append(bw / bh if bh > 0 else 0)
        except Exception as e:
            print(f"Error parsing {xml_path}: {e}")
            
    # Compute size breakdowns
    side_lengths = [np.sqrt(a) for a in box_sizes]
    coco_small = sum(1 for s in side_lengths if s < 32)
    coco_medium = sum(1 for s in side_lengths if 32 <= s <= 96)
    coco_large = sum(1 for s in side_lengths if s > 96)
    cctv_small_20_60 = sum(1 for s in side_lengths if 20 <= s <= 60)
    
    total_boxes = len(box_sizes)
    print("\n[Sohas Class Balance]")
    for cname, count in class_counts.most_common():
        print(f"  - {cname:15s}: {count:5d} ({count/total_boxes*100:5.1f}%)")
        
    print("\n[Sohas Box Scale Breakdown]")
    print(f"  - Total valid boxes: {total_boxes}")
    print(f"  - Degenerate boxes (w<=0 or h<=0): {degenerate_boxes}")
    print(f"  - COCO Small (<32 px): {coco_small} ({coco_small/total_boxes*100:5.1f}%)")
    print(f"  - COCO Medium (32-96 px): {coco_medium} ({coco_medium/total_boxes*100:5.1f}%)")
    print(f"  - COCO Large (>96 px): {coco_large} ({coco_large/total_boxes*100:5.1f}%)")
    print(f"  - CCTV Critical Range (20-60 px): {cctv_small_20_60} ({cctv_small_20_60/total_boxes*100:5.1f}%)")
    print(f"  - Median Box Dimension: {np.median(box_widths):.1f} x {np.median(box_heights):.1f} px")
    print(f"  - Common Image Resolutions: {image_sizes.most_common(3)}")
    
    return {
        "dataset": "Sohas (OD-WeaponDetection)",
        "xml_files": len(xml_files),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "coco_small_pct": coco_small / total_boxes * 100,
        "coco_medium_pct": coco_medium / total_boxes * 100,
        "coco_large_pct": coco_large / total_boxes * 100,
        "cctv_small_pct": cctv_small_20_60 / total_boxes * 100,
        "degenerate_boxes": degenerate_boxes,
    }

def audit_indian_raw(raw_root: Path):
    """Audit the raw Indian cinema/Roboflow weapon dataset."""
    print("\n" + "=" * 60)
    print("2. AUDITING RAW INDIAN WEAPON DATASET")
    print(f"Path: {raw_root}")
    print("=" * 60)
    
    label_files = list(raw_root.glob("**/*.txt"))
    # filter out data.yaml, readme, etc.
    label_files = [f for f in label_files if "labels" in str(f) or "train" in str(f) or "valid" in str(f) or "test" in str(f)]
    label_files = [f for f in label_files if not f.name.endswith(".dataset.txt") and not f.name.endswith(".roboflow.txt")]
    
    print(f"Found {len(label_files)} YOLO annotation label files.")
    
    class_counts = Counter()
    box_sizes = []
    box_widths = []
    box_heights = []
    degenerate_boxes = 0
    duplicate_boxes = 0
    
    # Check for near duplicates or multi-copy leaks across splits
    stem_to_splits = defaultdict(set)
    
    for lf in label_files:
        split_name = "train" if "train" in str(lf) else ("valid" if "valid" in str(lf) else "test")
        stem = lf.stem.split(".rf.")[0].split("-Copy")[0]
        stem_to_splits[stem].add(split_name)
        
        try:
            lines = lf.read_text().strip().splitlines()
            seen_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                
                class_counts[cls_id] += 1
                
                if w <= 0 or h <= 0:
                    degenerate_boxes += 1
                    continue
                    
                # check duplicates
                for sc, sx, sy, sw, sh in seen_boxes:
                    if sc == cls_id and abs(cx - sx) < 0.01 and abs(cy - sy) < 0.01 and abs(w - sw) < 0.01 and abs(h - sh) < 0.01:
                        duplicate_boxes += 1
                seen_boxes.append((cls_id, cx, cy, w, h))
                
                # Assume 640x640 standard YOLO resolution
                pw, ph = w * 640, h * 640
                box_widths.append(pw)
                box_heights.append(ph)
                box_sizes.append(pw * ph)
        except Exception as e:
            print(f"Error parsing {lf}: {e}")
            
    total_boxes = len(box_sizes)
    side_lengths = [np.sqrt(a) for a in box_sizes]
    coco_small = sum(1 for s in side_lengths if s < 32)
    coco_medium = sum(1 for s in side_lengths if 32 <= s <= 96)
    coco_large = sum(1 for s in side_lengths if s > 96)
    cctv_small_20_60 = sum(1 for s in side_lengths if 20 <= s <= 60)
    
    leaking_stems = sum(1 for s, splits in stem_to_splits.items() if len(splits) > 1)
    
    print("\n[Indian Raw Class Balance (0=Armed Person, 1=Gun, 2=Knife)]")
    for cls_id, count in sorted(class_counts.items()):
        name = {0: "armed person", 1: "gun", 2: "knife"}.get(cls_id, f"class_{cls_id}")
        print(f"  - Class {cls_id} ({name:12s}): {count:5d} ({count/total_boxes*100:5.1f}%)")
        
    print("\n[Indian Raw Scale Breakdown (@ 640px norm)]")
    print(f"  - Total valid boxes: {total_boxes}")
    print(f"  - Degenerate boxes: {degenerate_boxes}")
    print(f"  - Exact duplicate boxes in same file: {duplicate_boxes}")
    print(f"  - COCO Small (<32 px): {coco_small} ({coco_small/total_boxes*100:5.1f}%)")
    print(f"  - COCO Medium (32-96 px): {coco_medium} ({coco_medium/total_boxes*100:5.1f}%)")
    print(f"  - COCO Large (>96 px): {coco_large} ({coco_large/total_boxes*100:5.1f}%)")
    print(f"  - CCTV Critical Range (20-60 px): {cctv_small_20_60} ({cctv_small_20_60/total_boxes*100:5.1f}%)")
    print(f"  - Median Box Dimension: {np.median(box_widths):.1f} x {np.median(box_heights):.1f} px")
    
    print("\n[Indian Raw Integrity & Data Leakage]")
    print(f"  - Unique base image stems: {len(stem_to_splits)}")
    print(f"  - Stems leaked across multiple splits (train/val/test): {leaking_stems} ({leaking_stems/len(stem_to_splits)*100:.1f}%)")
    
    return {
        "dataset": "Raw Indian Cinema & Roboflow (IntProj)",
        "label_files": len(label_files),
        "total_boxes": total_boxes,
        "class_counts": dict(class_counts),
        "coco_small_pct": coco_small / total_boxes * 100,
        "cctv_small_pct": cctv_small_20_60 / total_boxes * 100,
        "leaking_stems": leaking_stems,
        "duplicate_boxes": duplicate_boxes,
    }

if __name__ == "__main__":
    work_dir = Path("/home/shiva/Desktop/projects/akriti/weapon-detection/work")
    sohas_path = work_dir / "downloads" / "odw" / "extracted" / "OD-WeaponDetection-master"
    raw_indian_path = work_dir / "raw" / "IntProj_Weapon_Dataset_annot"
    
    if sohas_path.exists():
        audit_sohas(sohas_path)
    if raw_indian_path.exists():
        audit_indian_raw(raw_indian_path)
