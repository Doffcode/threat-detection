"""
src/detection/audit_and_reconcile.py
====================================
Comprehensive pre-training data integrity, count reconciliation,
embedding-based leak detection, and negative split balancing.
"""

from __future__ import annotations

import collections
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
import torch
from torchvision import models, transforms
from PIL import Image

PROJECT_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "stage1_baseline"
DEBUG_DIR = PROJECT_ROOT / "outputs" / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


def reconcile_counts() -> dict:
    """1. Reconcile total candidates, dropped, and split counts."""
    manifest_path = DATA_DIR / "manifest.csv"
    records = []
    with open(manifest_path) as f:
        reader = csv.DictReader(f)
        records = list(reader)
        
    counts_by_split = collections.Counter(r["split"] for r in records)
    hard_negs_by_split = collections.Counter(r["split"] for r in records if r["is_hard_neg"] == "1")
    positives_by_split = collections.Counter(r["split"] for r in records if r["is_hard_neg"] == "0")
    
    # Candidate numbers from build script:
    total_pos_candidates = 17215
    total_neg_candidates = 3409
    total_candidates = total_pos_candidates + total_neg_candidates  # 20624
    
    dropped_leak_guard = 354
    # Positive records retained in clean pool:
    # 17215 - 354 = 16861
    # Train pos: 13383, Valid pos: 1674, Test_pub pos: 1668, Test_domain pos: 144, Test_cctv pos: 200
    # Sum of pos in splits: 13383 + 1674 + 1668 + 144 + 200 = 17069? Let's check exact counts.
    
    table = {
        "total_positive_candidates": total_pos_candidates,
        "total_negative_candidates": total_neg_candidates,
        "total_raw_candidates": total_candidates,
        "dropped_by_leak_guard": dropped_leak_guard,
        "clean_pool_retained": total_candidates - dropped_leak_guard,
        "splits_breakdown": {}
    }
    
    for s in ["train", "valid", "test_public", "test_domain", "test_cctv", "test_hard_negatives"]:
        table["splits_breakdown"][s] = {
            "total_images": counts_by_split[s],
            "positives": positives_by_split[s],
            "hard_negatives": hard_negs_by_split[s],
        }
        
    total_in_splits = sum(counts_by_split.values())
    table["total_images_in_splits"] = total_in_splits
    table["unassigned_hard_negatives"] = total_candidates - dropped_leak_guard - total_in_splits
    
    return table


def add_negatives_to_valid():
    """4. Add weapon-free negatives to `valid` (10% of valid pos, ~167 images)."""
    # Check current negatives in valid
    valid_lbls = list((DATA_DIR / "valid" / "labels").glob("*.txt"))
    empty_in_val = sum(1 for lp in valid_lbls if len(lp.read_text().strip()) == 0)
    print(f"Current empty labels in valid: {empty_in_val}")
    
    if empty_in_val >= 150:
        print("Valid already has sufficient weapon-free negatives.")
        return
        
    # We draw from unused Sohas distractors or Indian street negatives
    manifest_path = DATA_DIR / "manifest.csv"
    with open(manifest_path) as f:
        reader = list(csv.DictReader(f))
        existing_filenames = {row["filename"] for row in reader}
        existing_paths = {str(Path(row["orig_path"]).resolve()) for row in reader}
        
    sohas_base = Path("/home/shiva/Desktop/projects/akriti/weapon-detection/work/downloads/odw/extracted/OD-WeaponDetection-master/Weapons and similar handled objects/Sohas_weapon-Detection")
    candidate_negs = []
    for x in sohas_base.glob("**/*.xml"):
        try:
            tree = ET.parse(x)
            root = tree.getroot()
            names = [obj.find("name").text.strip().lower() for obj in root.findall("object")]
            if names and all(n in ("smartphone", "monedero", "billete", "tarjeta") for n in names):
                fname = root.find("filename").text if root.find("filename") is not None else x.stem + ".jpg"
                for folder in ["images", "images_test"]:
                    cand = sohas_base / folder / fname
                    if cand.is_file() and str(cand.resolve()) not in existing_paths:
                        candidate_negs.append(cand)
                        break
        except Exception:
            continue
                
    print(f"Found {len(candidate_negs)} unassigned clean distractor negatives.")
    n_to_add = 167
    to_add = candidate_negs[:n_to_add]
    
    val_img_dir = DATA_DIR / "valid" / "images"
    val_lbl_dir = DATA_DIR / "valid" / "labels"
    
    new_rows = []
    for idx, p in enumerate(to_add):
        safe_name = f"val_neg_{idx:04d}_{p.name}"
        tgt_img = val_img_dir / safe_name
        tgt_lbl = val_lbl_dir / f"{tgt_img.stem}.txt"
        
        if tgt_img.exists():
            tgt_img.unlink()
        os.symlink(p.resolve(), tgt_img)
        tgt_lbl.write_text("")  # empty label file
        
        new_rows.append({
            "split": "valid",
            "filename": safe_name,
            "source": "sohas_distractor_val",
            "orig_path": str(p),
            "n_boxes": 0,
            "is_hard_neg": 1,
            "dhash": "0x0",
        })
        
    # Append to manifest
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(new_rows[0].keys()))
        writer.writerows(new_rows)
        
    print(f"Successfully added {len(new_rows)} weapon-free negatives to valid/!")


def check_deep_embedding_leakage() -> Tuple[int, List[dict]]:
    """2c. Embedding cosine similarity check using pretrained ResNet50 between train and eval sets."""
    print("\nRunning ResNet50 deep-feature embedding leak detection...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load feature extractor
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Collect eval images (test_domain, test_cctv, valid sample)
    eval_splits = ["test_domain", "test_cctv", "valid"]
    eval_images = []
    for s in eval_splits:
        for p in (DATA_DIR / s / "images").glob("*.*"):
            eval_images.append((s, p))
            
    print(f"Extracting embeddings for {len(eval_images)} evaluation images...")
    eval_feats = []
    valid_eval_records = []
    with torch.no_grad():
        for s, p in eval_images:
            try:
                img = Image.open(p).convert("RGB")
                tensor = preprocess(img).unsqueeze(0).to(device)
                feat = model(tensor).cpu().numpy().flatten()
                feat = feat / np.linalg.norm(feat)
                eval_feats.append(feat)
                valid_eval_records.append((s, p))
            except Exception:
                continue
                
    eval_feats = np.array(eval_feats)  # (N_eval, 2048)
    
    # Extract train embeddings in batches
    train_images = list((DATA_DIR / "train" / "images").glob("*.*"))
    print(f"Extracting embeddings for {len(train_images)} training images...")
    
    train_feats = []
    valid_train_paths = []
    batch_tensors = []
    batch_paths = []
    
    with torch.no_grad():
        for idx, p in enumerate(train_images):
            try:
                img = Image.open(p).convert("RGB")
                batch_tensors.append(preprocess(img))
                batch_paths.append(p)
                
                if len(batch_tensors) >= 64 or idx == len(train_images) - 1:
                    batch = torch.stack(batch_tensors).to(device)
                    feats = model(batch).cpu().numpy()
                    norms = np.linalg.norm(feats, axis=1, keepdims=True)
                    feats = feats / np.maximum(norms, 1e-12)
                    train_feats.extend(feats)
                    valid_train_paths.extend(batch_paths)
                    batch_tensors = []
                    batch_paths = []
            except Exception:
                continue
                
    train_feats = np.array(train_feats)  # (N_train, 2048)
    print(f"Computed embeddings: Train={train_feats.shape}, Eval={eval_feats.shape}")
    
    # Compute cosine similarity matrix (eval x train) in chunks
    print("Computing cosine similarity matrix...")
    chunk_size = 500
    high_sim_pairs = []
    
    for i in range(0, len(eval_feats), chunk_size):
        eval_chunk = eval_feats[i:i + chunk_size]
        sim_matrix = np.dot(eval_chunk, train_feats.T)  # (chunk, N_train)
        
        # Find pairs with sim >= 0.92
        matches = np.argwhere(sim_matrix >= 0.92)
        for e_idx_local, t_idx in matches:
            e_idx = i + e_idx_local
            sim = float(sim_matrix[e_idx_local, t_idx])
            high_sim_pairs.append({
                "eval_split": valid_eval_records[e_idx][0],
                "eval_path": str(valid_eval_records[e_idx][1]),
                "train_path": str(valid_train_paths[t_idx]),
                "similarity": sim,
            })
            
    print(f"Total pairs with deep feature similarity >= 0.92: {len(high_sim_pairs)}")
    
    # Save 20 borderline pairs as side-by-side images
    borderline_dir = DEBUG_DIR / "borderline_pairs"
    borderline_dir.mkdir(parents=True, exist_ok=True)
    
    high_sim_pairs.sort(key=lambda x: x["similarity"], reverse=True)
    top_20 = high_sim_pairs[:20]
    
    for idx, pair in enumerate(top_20):
        try:
            im1 = cv2.imread(pair["eval_path"])
            im2 = cv2.imread(pair["train_path"])
            if im1 is None or im2 is None:
                continue
            h = 400
            w1 = int(im1.shape[1] * (h / im1.shape[0]))
            w2 = int(im2.shape[1] * (h / im2.shape[0]))
            r1 = cv2.resize(im1, (w1, h))
            r2 = cv2.resize(im2, (w2, h))
            
            # Put text
            cv2.putText(r1, f"{pair['eval_split']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(r2, f"train (sim={pair['similarity']:.3f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            combined = np.hstack([r1, r2])
            out_img_path = borderline_dir / f"pair_{idx:02d}_sim_{pair['similarity']:.3f}.jpg"
            cv2.imwrite(str(out_img_path), combined)
        except Exception:
            continue
            
    return len(high_sim_pairs), top_20


if __name__ == "__main__":
    import xml.etree.ElementTree as ET
    reconciliation = reconcile_counts()
    print("\n" + "=" * 60)
    print("COUNT RECONCILIATION SUMMARY")
    print("=" * 60)
    print(json.dumps(reconciliation, indent=2))
    
    add_negatives_to_valid()
    n_pairs, top_pairs = check_deep_embedding_leakage()
    print(f"\nSaved borderline pairs to: {DEBUG_DIR / 'borderline_pairs'}")
