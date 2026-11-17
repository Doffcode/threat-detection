"""
scripts/generate_occlusion_template.py
======================================
Generates a clean single-reviewer occlusion annotation CSV template for the
final held-out evaluation sets (test_domain and test_cctv).
"""

from pathlib import Path
import csv

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}

def generate_template():
    base_dir = Path("data/processed/stage1_baseline")
    output_path = Path("configs/detection/occlusion_annotation_template.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    
    for split_name in ["test_domain", "test_cctv"]:
        split_dir = base_dir / split_name
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        valid_exts = {".jpg", ".jpeg", ".png", ".webp"}
        img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in valid_exts])
        
        for img_p in img_files:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                continue
            
            lines = [l.strip() for l in lbl_p.read_text().strip().splitlines() if l.strip()]
            for box_idx, line in enumerate(lines):
                parts = line.split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    cname = CLASS_NAMES.get(cid, f"class_{cid}")
                    bbox_str = f"{cx:.6f},{cy:.6f},{w:.6f},{h:.6f}"
                    
                    rows.append({
                        "split": split_name,
                        "image_filename": img_p.name,
                        "box_index": box_idx,
                        "class_name": cname,
                        "bbox_yolo_cx_cy_w_h": bbox_str,
                        "occlusion_level": "",      # none / light_<25% / medium_25-50% / heavy_>50%
                        "occluder_type": "",        # hand / clothing / body / furniture / other
                        "visible_features": "",     # grip / barrel_blade / silhouette / other
                        "notes": ""
                    })

    fieldnames = [
        "split",
        "image_filename",
        "box_index",
        "class_name",
        "bbox_yolo_cx_cy_w_h",
        "occlusion_level",
        "occluder_type",
        "visible_features",
        "notes"
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated single-reviewer occlusion template at {output_path}")
    print(f"Total rows: {len(rows)}")
    
    # Class breakdown
    by_split_class = {}
    for r in rows:
        key = (r["split"], r["class_name"])
        by_split_class[key] = by_split_class.get(key, 0) + 1
    for k, v in sorted(by_split_class.items()):
        print(f"  - {k[0]:12s} | {k[1]:10s} : {v} boxes")

if __name__ == "__main__":
    generate_template()
