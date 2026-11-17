"""
scripts/import_cctv_test_v2_labels.py
====================================
Automated importer for CCTV Test v2 annotations.
Solves the manual renaming problem: matches exported YOLO annotations from CVAT,
Roboflow, Label Studio, or any tool back to the exact target filenames in
data/cctv_test_v2_candidates/labels/ and syncs with configs/detection/cctv_test_v2_occlusion_template.csv.

Usage:
    python scripts/import_cctv_test_v2_labels.py --input /path/to/exported_labels_or_zip
    python scripts/import_cctv_test_v2_labels.py --input exported.zip --dry-run
"""

import argparse
import os
import re
import shutil
import zipfile
from pathlib import Path
import pandas as pd
import yaml

TARGET_CLASSES = {
    0: "handgun",
    1: "rifle",
    2: "knife"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Import and map YOLO labels to CCTV test v2 candidates")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to exported zip file or directory containing label .txt files")
    parser.add_argument("--workspace", "-w", type=str,
                        default="/home/shiva/Desktop/projects/akriti/weapon-detection",
                        help="Weapon detection workspace directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Simulate the import and show matches without writing files")
    return parser.parse_args()

def get_class_mapper(source_names):
    """
    Maps source class names (e.g. ['blade', 'handgun', 'knife', 'scissor'])
    to target classes: 0: handgun, 1: rifle, 2: knife
    """
    if not source_names:
        # Default identity mapping if classes are already [handgun, rifle, knife]
        return lambda cid: (cid, TARGET_CLASSES.get(cid, f"class_{cid}"))
        
    mapping = {}
    for cid, name in enumerate(source_names):
        name_lower = str(name).lower().strip()
        if any(w in name_lower for w in ["handgun", "gun", "pistol", "revolver"]):
            target_id = 0
        elif any(w in name_lower for w in ["rifle", "shotgun", "long gun", "sniper"]):
            target_id = 1
        elif any(w in name_lower for w in ["knife", "blade", "scissor", "dagger", "sword", "weapon"]):
            target_id = 2
        else:
            target_id = 2  # default fallback
        mapping[cid] = (target_id, TARGET_CLASSES[target_id])
        
    print("[CLASS MAPPER] Detected source classes and mapped to target YOLO schema:")
    for cid, name in enumerate(source_names):
        tid, tname = mapping[cid]
        print(f"  Source {cid} ('{name}') -> Target {tid} ('{tname}')")
        
    return lambda cid: mapping.get(cid, (cid, TARGET_CLASSES.get(cid, f"class_{cid}")))

def extract_labels_and_yaml(input_path: Path):
    labels_dict = {}
    source_names = None
    
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(input_path, "r") as z:
            # Check for data.yaml
            for name in z.namelist():
                if Path(name).name.lower() in ["data.yaml", "dataset.yaml"]:
                    try:
                        cfg = yaml.safe_load(z.read(name).decode("utf-8", errors="replace"))
                        if isinstance(cfg, dict) and "names" in cfg:
                            raw_names = cfg["names"]
                            if isinstance(raw_names, dict):
                                source_names = [raw_names[k] for k in sorted(raw_names.keys())]
                            elif isinstance(raw_names, list):
                                source_names = raw_names
                    except Exception as e:
                        print(f"[WARNING] Could not parse yaml: {e}")
                        
            for zip_info in z.infolist():
                if zip_info.filename.endswith("/") or not zip_info.filename.lower().endswith(".txt"):
                    continue
                base = Path(zip_info.filename).name
                if base.lower() in {"classes.txt", "notes.txt", "readme.txt", "labels.txt", "readme.roboflow.txt", "readme.dataset.txt"}:
                    continue
                content = z.read(zip_info).decode("utf-8", errors="replace")
                labels_dict[zip_info.filename] = (base, content)
                
    elif input_path.is_dir():
        yaml_candidates = list(input_path.rglob("data.yaml")) + list(input_path.rglob("dataset.yaml"))
        if yaml_candidates:
            try:
                cfg = yaml.safe_load(yaml_candidates[0].read_text(encoding="utf-8", errors="replace"))
                if isinstance(cfg, dict) and "names" in cfg:
                    raw_names = cfg["names"]
                    if isinstance(raw_names, dict):
                        source_names = [raw_names[k] for k in sorted(raw_names.keys())]
                    elif isinstance(raw_names, list):
                        source_names = raw_names
            except Exception as e:
                print(f"[WARNING] Could not parse yaml: {e}")
                
        for txt_file in input_path.rglob("*.txt"):
            base = txt_file.name
            if base.lower() in {"classes.txt", "notes.txt", "readme.txt", "labels.txt", "readme.roboflow.txt", "readme.dataset.txt"}:
                continue
            content = txt_file.read_text(encoding="utf-8", errors="replace")
            rel = str(txt_file.relative_to(input_path))
            labels_dict[rel] = (base, content)
    else:
        raise FileNotFoundError(f"Input path does not exist or is not a zip/dir: {input_path}")
        
    return labels_dict, source_names

def parse_and_convert_yolo(content: str, filename: str, mapper):
    boxes = []
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    for line_idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            raw_cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
        except ValueError:
            continue
            
        target_cid, target_cname = mapper(raw_cid)
        
        # Clamp bounds
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        
        boxes.append((target_cid, target_cname, cx, cy, w, h))
    return boxes

def main():
    args = parse_args()
    workspace = Path(args.workspace)
    cctv_dir = workspace / "data/cctv_test_v2_candidates"
    images_dir = cctv_dir / "images"
    labels_dir = cctv_dir / "labels"
    template_path = workspace / "configs/detection/cctv_test_v2_occlusion_template.csv"
    
    if not template_path.exists():
        raise FileNotFoundError(f"Template CSV not found at: {template_path}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found at: {labels_dir}")
        
    df_template = pd.read_csv(template_path)
    print(f"Loaded template with {len(df_template)} candidate frames.")
    
    # Load exported labels & class definitions
    input_path = Path(args.input)
    raw_labels, source_names = extract_labels_and_yaml(input_path)
    mapper = get_class_mapper(source_names)
    print(f"Extracted {len(raw_labels)} label text files from {input_path.name}.")
    
    # Build candidate lookup
    candidate_rows = []
    for idx, row in df_template.iterrows():
        cand_img = row["image_filename"]
        cand_stem = Path(cand_img).stem
        orig_img = row["orig_filename"]
        orig_stem = Path(orig_img).stem
        candidate_rows.append({
            "idx": idx,
            "row": row,
            "cand_img": cand_img,
            "cand_stem": cand_stem,
            "cand_lbl": Path(cand_img).with_suffix(".txt").name,
            "orig_stem": orig_stem,
            "index_prefix": f"cctv_v2_{idx:03d}"
        })
        
    # Matching
    matched_labels = {}  # idx -> (content, source_name)
    unmatched_sources = []
    
    for src_rel, (base_name, content) in raw_labels.items():
        base_stem = Path(base_name).stem
        matched_idx = None
        
        # Strategy A: Candidate index prefix match (e.g. cctv_v2_002)
        m = re.search(r"cctv_v2_(\d{3})", base_stem)
        if m:
            idx_val = int(m.group(1))
            if 0 <= idx_val < len(candidate_rows):
                matched_idx = idx_val
                
        # Strategy B: Exact candidate stem match
        if matched_idx is None:
            for c in candidate_rows:
                if base_stem == c["cand_stem"]:
                    matched_idx = c["idx"]
                    break
                    
        # Strategy C: Original filename stem match
        if matched_idx is None:
            for c in candidate_rows:
                if base_stem == c["orig_stem"] or base_stem in c["cand_stem"] or c["orig_stem"] in base_stem:
                    matched_idx = c["idx"]
                    break
                    
        # Strategy D: Sequential frame match (frame_000000)
        if matched_idx is None:
            m_cvat = re.search(r"frame_0*(\d+)", base_stem)
            if m_cvat:
                frame_idx = int(m_cvat.group(1))
                if 0 <= frame_idx < len(candidate_rows):
                    matched_idx = frame_idx
                    
        if matched_idx is not None:
            matched_labels[matched_idx] = (content, base_name)
        else:
            unmatched_sources.append(src_rel)
            
    print(f"\nSuccessfully matched: {len(matched_labels)} / {len(candidate_rows)} candidate frames.")
    
    # Process all candidates
    stats = {
        "with_weapons": 0,
        "clean_negatives": 0,
        "classes": {"handgun": 0, "rifle": 0, "knife": 0}
    }
    
    updated_template_rows = []
    
    for c in candidate_rows:
        idx = c["idx"]
        dest_lbl_path = labels_dir / c["cand_lbl"]
        row_dict = dict(c["row"])
        
        if idx in matched_labels:
            raw_content, src_name = matched_labels[idx]
            boxes = parse_and_convert_yolo(raw_content, src_name, mapper)
            
            if len(boxes) > 0:
                stats["with_weapons"] += 1
                row_dict["weapon_present"] = True
                
                # Format converted YOLO lines for the label file
                yolo_lines = [f"{b[0]} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f} {b[5]:.6f}" for b in boxes]
                if not args.dry_run:
                    dest_lbl_path.write_text("\n".join(yolo_lines) + "\n")
                    
                # Format CSV metadata
                cls_names = [b[1] for b in boxes]
                row_dict["weapon_class"] = ";".join(cls_names)
                
                bbox_strs = [f"{b[2]:.6f},{b[3]:.6f},{b[4]:.6f},{b[5]:.6f}" for b in boxes]
                row_dict["bbox_normalized_xywh"] = ";".join(bbox_strs)
                
                for b in boxes:
                    cname = b[1]
                    stats["classes"][cname] = stats["classes"].get(cname, 0) + 1
            else:
                # File present in export but has no bounding box
                stats["clean_negatives"] += 1
                if not args.dry_run:
                    dest_lbl_path.write_text("")
                row_dict["weapon_present"] = False
                row_dict["weapon_class"] = ""
                row_dict["bbox_normalized_xywh"] = ""
                if not row_dict.get("occlusion_level") or pd.isna(row_dict.get("occlusion_level")):
                    row_dict["occlusion_level"] = "none"
        else:
            # Candidate not in export: human skipped as unclear or negative
            stats["clean_negatives"] += 1
            if not args.dry_run:
                dest_lbl_path.write_text("")
            row_dict["weapon_present"] = False
            row_dict["weapon_class"] = ""
            row_dict["bbox_normalized_xywh"] = ""
            if not row_dict.get("occlusion_level") or pd.isna(row_dict.get("occlusion_level")):
                row_dict["occlusion_level"] = "none"
            if not row_dict.get("reviewer_notes") or pd.isna(row_dict.get("reviewer_notes")):
                row_dict["reviewer_notes"] = "negative / skipped"
                
        updated_template_rows.append(row_dict)
        
    if not args.dry_run:
        df_new = pd.DataFrame(updated_template_rows)
        df_new.to_csv(template_path, index=False)
        print(f"\n[SUCCESS] Updated {template_path}")
        print(f"[SUCCESS] Wrote all 92 label files to {labels_dir} (preserved original names!)")
    else:
        print("\n[DRY-RUN] No files written.")
        
    print("\n================ CCTV TEST V2 SUMMARY ================")
    print(f"Total candidate frames:   {len(candidate_rows)}")
    print(f"Frames with weapons:      {stats['with_weapons']}")
    print(f"Clean negative frames:    {stats['clean_negatives']}")
    print("Weapon instance counts by target class:")
    for cname, count in stats["classes"].items():
        print(f"  - {cname:10s} (class {list(TARGET_CLASSES.keys())[list(TARGET_CLASSES.values()).index(cname)]}): {count}")
    print("======================================================")

if __name__ == "__main__":
    main()
