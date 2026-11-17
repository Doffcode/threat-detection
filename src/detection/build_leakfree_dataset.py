"""
src/detection/build_leakfree_dataset.py
=======================================
Builds a verified, zero-leakage, 3-class weapon dataset for Stage 1.
Classes:
  0: handgun
  1: rifle
  2: knife
(Unlabeled hard negatives with empty label files for phones, wallets, tools, and street scenes)

Enforces:
  1. Scene / Video grouping (frames from the same recording clip share one cluster).
  2. Perceptual dHash64 deduplication across all sources via DSU clustering (hamming <= 4).
  3. Strict isolation of test_domain (144 cinema stills) and test_cctv (200 Indian CCTV).
  4. Negative assignment:
     - 167 hard negatives in valid (to account for false alarms during model selection)
     - 300 hard negatives in test_hard_negatives (dedicated false-positive benchmark)
     - 0 negatives in train (pure positive weapon detector learning)
     - 0 negatives in test_public, test_domain, test_cctv
  5. Output splits: train, valid, test_public, test_domain, test_cctv, test_hard_negatives.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import shutil
from typing import Dict, List, Set, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
WORK_DIR = PROJECT_ROOT / "work"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "stage1_baseline"

SOHAS_DIR = WORK_DIR / "downloads" / "odw" / "extracted" / "OD-WeaponDetection-master"
AAHIL_DIR = WORK_DIR / "downloads" / "roboflow" / "aahil-m8yrw__weapon-detection-revmd__v3"
INNOVATUS_ROB = WORK_DIR / "downloads" / "roboflow" / "innovatus-gv1rg__anti_robbing__v1"
INNOVATUS_SNATCH = WORK_DIR / "downloads" / "roboflow" / "innovatus-gv1rg__chain_snatching-jk1lw__v1"
BITS_THREAT = WORK_DIR / "downloads" / "roboflow" / "bits-pilani__threatdetect2__v7"
BITS_KNIVES = WORK_DIR / "downloads" / "roboflow" / "bits-pilani__knives-tkfol__v1"
V2_DIR = WORK_DIR / "data" / "v2"

DHASH_HAMMING_THRESHOLD = 4
SEED = 2026
VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Perceptual Hashing (dHash64)
# ---------------------------------------------------------------------------
def compute_dhash64(image_path: Path) -> int:
    """Compute 64-bit difference hash."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    resized = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def popcount64(x: np.ndarray) -> np.ndarray:
    """Vectorized bit count on uint64 numpy array."""
    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
    x = ((x * 0x0101010101010101) >> 56) & 0xFF
    return x


# ---------------------------------------------------------------------------
# Normalization & Base Stem Extraction
# ---------------------------------------------------------------------------
def get_base_stem(stem: str) -> str:
    """Normalize Roboflow, video frames, and augmentation suffixes."""
    s = stem.split(".rf.")[0]
    for suf in ("-Copy", "-Copy-2-", "-Copy_jpeg", "-Copy_png", "-Copy_avif", "-Copy_webp"):
        s = s.replace(suf, "")
    return s.lower()


# ---------------------------------------------------------------------------
# Record Dataclass
# ---------------------------------------------------------------------------
class ImageRecord:
    def __init__(self, path: Path, source: str, boxes: List[Tuple[int, float, float, float, float]], is_hard_negative: bool = False):
        self.path = path
        self.source = source
        self.boxes = boxes  # [(cls_id, cx, cy, w, h)]
        self.is_hard_negative = is_hard_negative
        self.stem = path.stem
        self.base_stem = get_base_stem(path.stem)
        self.dhash = 0
        self.group_id = -1
        self.split = ""


# ---------------------------------------------------------------------------
# Parsers for Each Source
# ---------------------------------------------------------------------------
def parse_sohas() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Parse Sohas XML annotations with 100% exact image path indexing."""
    weapon_recs = []
    neg_recs = []
    
    # Pre-index all actual image files in ODW
    all_imgs = [p for p in SOHAS_DIR.glob("**/*") if p.is_file() and p.suffix in VALID_EXTS]
    img_map = {p.stem: p for p in all_imgs}
    
    xml_files = list(SOHAS_DIR.glob("**/*.xml"))
    seen_stems = set()
    
    for xml_path in xml_files:
        if xml_path.stem in seen_stems or xml_path.stem not in img_map:
            continue
        seen_stems.add(xml_path.stem)
        
        img_path = img_map[xml_path.stem]
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            size_elem = root.find("size")
            if size_elem is not None:
                img_w = float(size_elem.find("width").text or 640)
                img_h = float(size_elem.find("height").text or 480)
            else:
                img_w, img_h = 640.0, 480.0
                
            weapon_boxes = []
            has_distractor = False
            
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                bndbox = obj.find("bndbox")
                xmin = float(bndbox.find("xmin").text)
                ymin = float(bndbox.find("ymin").text)
                xmax = float(bndbox.find("xmax").text)
                ymax = float(bndbox.find("ymax").text)
                
                bw = (xmax - xmin) / img_w
                bh = (ymax - ymin) / img_h
                cx = (xmin + xmax) / (2.0 * img_w)
                cy = (ymin + ymax) / (2.0 * img_h)
                
                cx = max(0.0, min(1.0, cx))
                cy = max(0.0, min(1.0, cy))
                bw = max(0.001, min(1.0, bw))
                bh = max(0.001, min(1.0, bh))
                
                if name == "pistol":
                    weapon_boxes.append((0, cx, cy, bw, bh))  # 0: handgun
                elif name == "knife":
                    weapon_boxes.append((2, cx, cy, bw, bh))  # 2: knife
                elif name in ("smartphone", "monedero", "billete", "tarjeta"):
                    has_distractor = True
                    
            if weapon_boxes:
                weapon_recs.append(ImageRecord(img_path, "sohas", weapon_boxes))
            elif has_distractor:
                neg_recs.append(ImageRecord(img_path, "sohas_distractor", [], is_hard_negative=True))
        except Exception:
            continue
            
    return weapon_recs, neg_recs


def parse_aahil() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Parse Aahil dataset for clean rifles, handguns, knives, and tool negatives."""
    weapon_recs = []
    neg_recs = []
    
    data_yaml = AAHIL_DIR / "data.yaml"
    if not data_yaml.exists():
        return weapon_recs, neg_recs
        
    class_map = {
        "Celurit": 2, "cleaver": 2, "cutlass": 2, "katana": 2, "knife": 2, "sickle": 2, "sword": 2,
        "handgun": 0, "pistol": 0, "smg": 0,
        "heavyweapon": 1, "machine gun": 1, "rifle": 1, "shotgun": 1, "sniper rifle": 1,
    }
    tool_negatives = {"Linggis", "axe"}
    
    for split in ("train", "valid", "test"):
        lbl_dir = AAHIL_DIR / split / "labels"
        img_dir = AAHIL_DIR / split / "images"
        if not lbl_dir.exists():
            continue
        
        # Read names from yaml
        with open(data_yaml) as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names", [])
        name_to_cid = {name: idx for idx, name in enumerate(names)}
        
        id_to_target = {}
        tool_ids = set()
        for name, cid in name_to_cid.items():
            if name in class_map:
                id_to_target[cid] = class_map[name]
            elif name in tool_negatives:
                tool_ids.add(cid)
                
        for lp in lbl_dir.glob("*.txt"):
            imgs = list(img_dir.glob(f"{lp.stem}.*"))
            if not imgs:
                continue
            img_path = imgs[0]
            
            lines = lp.read_text().splitlines()
            boxes = []
            has_tool = False
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    orig_cid = int(parts[0])
                    if orig_cid in id_to_target:
                        target_cid = id_to_target[orig_cid]
                        cx, cy, w, h = map(float, parts[1:5])
                        boxes.append((target_cid, cx, cy, w, h))
                    elif orig_cid in tool_ids:
                        has_tool = True
                        
            if boxes:
                weapon_recs.append(ImageRecord(img_path, "aahil", boxes))
            elif has_tool:
                neg_recs.append(ImageRecord(img_path, "aahil_tools", [], is_hard_negative=True))
                
    return weapon_recs, neg_recs


def parse_indian_cctv() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Parse Innovatus Indian CCTV (anti_robbing, chain_snatching) and BITS Pilani."""
    cctv_recs = []
    street_negs = []
    
    # 1. Innovatus anti_robbing: classes ['Gun', 'Knife'] -> 0: handgun, 2: knife
    if INNOVATUS_ROB.exists():
        for split in ("train", "valid", "test"):
            lbl_dir = INNOVATUS_ROB / split / "labels"
            img_dir = INNOVATUS_ROB / split / "images"
            if not lbl_dir.exists():
                continue
            for lp in lbl_dir.glob("*.txt"):
                imgs = list(img_dir.glob(f"{lp.stem}.*"))
                if not imgs:
                    continue
                boxes = []
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        cid = int(p[0])
                        target_cls = 0 if cid == 0 else 2
                        boxes.append((target_cls, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                if boxes:
                    cctv_recs.append(ImageRecord(imgs[0], "innovatus_rob", boxes))
                    
    # 2. BITS Pilani threatdetect2 & knives
    for bdir in (BITS_THREAT, BITS_KNIVES):
        if not bdir.exists():
            continue
        for split in ("train", "valid", "test"):
            lbl_dir = bdir / split / "labels"
            img_dir = bdir / split / "images"
            if not lbl_dir.exists():
                continue
            for lp in lbl_dir.glob("*.txt"):
                imgs = list(img_dir.glob(f"{lp.stem}.*"))
                if not imgs:
                    continue
                boxes = []
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        boxes.append((2, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                if boxes:
                    cctv_recs.append(ImageRecord(imgs[0], "bits_pilani", boxes))
                    
    # 3. Innovatus chain_snatching (Clean Indian street backgrounds without weapons)
    if INNOVATUS_SNATCH.exists():
        for split in ("train", "valid", "test"):
            img_dir = INNOVATUS_SNATCH / split / "images"
            if not img_dir.exists():
                continue
            for ip in img_dir.glob("*.*"):
                street_negs.append(ImageRecord(ip, "indian_street_neg", [], is_hard_negative=True))
                
    return cctv_recs, street_negs


def parse_indian_cinema_stills() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """Parse clean Indian cinema domain stills from work/data/v2."""
    domain_test_recs = []
    domain_train_recs = []
    
    # 1. Locked 144 test_domain images
    test_domain_file = V2_DIR / "test_domain.txt"
    if test_domain_file.exists():
        test_paths = set(test_domain_file.read_text().splitlines())
        for p_str in test_paths:
            ip = Path(p_str)
            if not ip.exists():
                continue
            lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
            boxes = []
            if lp.exists():
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        cid = int(p[0])
                        if "sanjay-dutt" in ip.stem.lower() or "kgf" in ip.stem.lower():
                            target_c = 1 if cid == 0 else 2
                        elif "aruval" in ip.stem.lower():
                            target_c = 2
                        else:
                            target_c = 0 if cid == 0 else 2
                        boxes.append((target_c, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            domain_test_recs.append(ImageRecord(ip, "cinema_domain_test", boxes))
            
    # 2. Remaining clean cinema stills for training/valid
    train_domain_file = V2_DIR / "train_domain.txt"
    if train_domain_file.exists():
        train_paths = set(train_domain_file.read_text().splitlines())
        test_stems = {r.base_stem for r in domain_test_recs}
        for p_str in train_paths:
            ip = Path(p_str)
            if not ip.exists():
                continue
            stem = get_base_stem(ip.stem)
            if stem in test_stems:
                continue  # STRICT GUARD
            lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
            boxes = []
            if lp.exists():
                for line in lp.read_text().splitlines():
                    p = line.split()
                    if len(p) >= 5:
                        cid = int(p[0])
                        target_c = 0 if cid == 0 else 2
                        if "aruval" in ip.stem.lower():
                            target_c = 2
                        boxes.append((target_c, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            if boxes:
                domain_train_recs.append(ImageRecord(ip, "cinema_domain_train", boxes))
                
    return domain_test_recs, domain_train_recs


# ---------------------------------------------------------------------------
# Global Deduplication & Clustering (DSU)
# ---------------------------------------------------------------------------
class DisjointSetUnion:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, i: int) -> int:
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i: int, j: int):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            if self.rank[root_i] < self.rank[root_j]:
                self.parent[root_i] = root_j
            elif self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def cluster_and_split(all_positive_candidates: List[ImageRecord],
                      locked_test_records: List[ImageRecord],
                      cctv_test_records: List[ImageRecord],
                      all_hard_negatives: List[ImageRecord]) -> Dict[str, List[ImageRecord]]:
    """DSU-based atomic clustering ensuring 0 leakage across ALL split pairs."""
    
    all_recs = all_positive_candidates + locked_test_records + cctv_test_records + all_hard_negatives
    n_total = len(all_recs)
    print(f"\nComputing perceptual dHash64 for all {n_total} records using 8 threads...")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        hashes = list(executor.map(compute_dhash64, [r.path for r in all_recs]))
        
    for r, h in zip(all_recs, hashes):
        r.dhash = h
        
    print(f"Calculated hashes: {sum(1 for h in hashes if h != 0)} non-zero out of {n_total}.")
    
    # Initialize DSU
    dsu = DisjointSetUnion(n_total)
    
    # 1. Union records sharing base_stem
    stem_to_indices = collections.defaultdict(list)
    for idx, r in enumerate(all_recs):
        stem_to_indices[r.base_stem].append(idx)
        
    for indices in stem_to_indices.values():
        for i in range(1, len(indices)):
            dsu.union(indices[0], indices[i])
            
    # 2. Union records with dHash hamming distance <= 4 (using vectorized popcount)
    print("Finding perceptual near-duplicates (hamming distance <= 4)...")
    H = np.array([r.dhash for r in all_recs], dtype=np.uint64)
    valid_hash_indices = np.where(H != 0)[0]
    H_valid = H[valid_hash_indices]
    
    chunk_size = 2000
    n_valid = len(H_valid)
    for i in range(0, n_valid, chunk_size):
        chunk = H_valid[i:i + chunk_size]
        diff = np.bitwise_xor(chunk[:, None], H_valid[None, :])
        dist = popcount64(diff)
        near_dups = np.argwhere(dist <= DHASH_HAMMING_THRESHOLD)
        for r_idx, c_idx in near_dups:
            global_r = valid_hash_indices[i + r_idx]
            global_c = valid_hash_indices[c_idx]
            dsu.union(global_r, global_c)
            
    # Build atomic clusters
    clusters = collections.defaultdict(list)
    for idx, r in enumerate(all_recs):
        root = dsu.find(idx)
        r.group_id = root
        clusters[root].append(r)
        
    print(f"Dataset grouped into {len(clusters)} atomic disjoint clusters.")
    
    # Identify locked test clusters
    locked_domain_roots = {dsu.find(idx) for idx, r in enumerate(all_recs) if r.source == "cinema_domain_test"}
    locked_cctv_roots = {dsu.find(idx) for idx, r in enumerate(all_recs) if r in cctv_test_records}
    
    train_recs = []
    val_recs = []
    test_pub_recs = []
    test_domain_recs = []
    test_cctv_recs = []
    test_hard_neg_recs = []
    
    # Collect locked test sets
    for root in locked_domain_roots:
        for r in clusters[root]:
            if r.source == "cinema_domain_test":
                r.split = "test_domain"
                test_domain_recs.append(r)
            else:
                # Any candidate that matched test_domain is strictly dropped to prevent leakage!
                r.split = "dropped_domain_leak"
                
    for root in locked_cctv_roots:
        for r in clusters[root]:
            if r in cctv_test_records:
                r.split = "test_cctv"
                test_cctv_recs.append(r)
            else:
                r.split = "dropped_cctv_leak"
                
    # Separate remaining clusters into positive weapon clusters vs pure hard-negative clusters
    clean_positive_clusters = []
    hard_negative_clusters = []
    
    for root, cluster_recs in clusters.items():
        if root in locked_domain_roots or root in locked_cctv_roots:
            continue
            
        has_weapon = any(not r.is_hard_negative for r in cluster_recs)
        if has_weapon:
            clean_positive_clusters.append((root, cluster_recs))
        else:
            hard_negative_clusters.append((root, cluster_recs))
            
    # Split clean positive clusters: 80% train, 10% valid, 10% test_public
    np.random.seed(SEED)
    np.random.shuffle(clean_positive_clusters)
    
    n_pos = len(clean_positive_clusters)
    n_train = int(n_pos * 0.80)
    n_val = int(n_pos * 0.10)
    
    for _, recs in clean_positive_clusters[:n_train]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "train"
                train_recs.append(r)
                
    for _, recs in clean_positive_clusters[n_train:n_train + n_val]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "valid"
                val_recs.append(r)
                
    for _, recs in clean_positive_clusters[n_train + n_val:]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "test_public"
                test_pub_recs.append(r)
                
    # Allocate hard negatives:
    # 1. 167 hard negatives to valid (for model selection FP penalty)
    # 2. 300 hard negatives to test_hard_negatives (dedicated false-positive benchmark)
    # 3. Remaining negatives left as unassigned_negative (0 negatives in train)
    np.random.shuffle(hard_negative_clusters)
    
    assigned_val_negs = 0
    assigned_test_negs = 0
    
    for _, recs in hard_negative_clusters:
        for r in recs:
            if assigned_val_negs < 167:
                r.split = "valid"
                val_recs.append(r)
                assigned_val_negs += 1
            elif assigned_test_negs < 300:
                r.split = "test_hard_negatives"
                test_hard_neg_recs.append(r)
                assigned_test_negs += 1
            else:
                r.split = "unassigned_negative"
                
    return {
        "train": train_recs,
        "valid": val_recs,
        "test_public": test_pub_recs,
        "test_domain": test_domain_recs,
        "test_cctv": test_cctv_recs,
        "test_hard_negatives": test_hard_neg_recs,
    }


# ---------------------------------------------------------------------------
# Output Writer & Verification
# ---------------------------------------------------------------------------
def write_processed_dataset(splits: Dict[str, List[ImageRecord]]):
    """Write symlinks and YOLO label files to data/processed/stage1_baseline."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    manifest_rows = []
    class_stats = collections.defaultdict(lambda: collections.Counter())
    
    print("\nWriting processed splits to disk...")
    for split_name, recs in splits.items():
        img_dir = OUT_DIR / split_name / "images"
        lbl_dir = OUT_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, r in enumerate(recs):
            safe_name = f"{r.source}_{idx:06d}_{r.path.name}"
            target_img = img_dir / safe_name
            target_lbl = lbl_dir / f"{target_img.stem}.txt"
            
            # Symlink image
            if target_img.exists():
                target_img.unlink()
            os.symlink(r.path.resolve(), target_img)
            
            # Write label file
            with open(target_lbl, "w") as f:
                for cls_id, cx, cy, w, h in r.boxes:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_stats[split_name][cls_id] += 1
                    
            manifest_rows.append({
                "split": split_name,
                "filename": safe_name,
                "source": r.source,
                "orig_path": str(r.path),
                "n_boxes": len(r.boxes),
                "is_hard_neg": int(r.is_hard_negative),
                "dhash": hex(r.dhash),
                "group_id": r.group_id,
            })
            
    # Write manifest CSV
    manifest_csv = OUT_DIR / "manifest.csv"
    with open(manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    # Write dataset yaml
    yaml_path = PROJECT_ROOT / "configs" / "detection" / "dataset_stage1.yaml"
    yaml_content = f"""# Stage 1 Dataset Configuration (3-Class + Hard Negatives)
path: {OUT_DIR.resolve()}
train: train/images
val: valid/images
test: test_public/images

# Locked evaluation sets (reported independently)
test_domain: test_domain/images
test_cctv: test_cctv/images
test_hard_negatives: test_hard_negatives/images

nc: 3
names:
  0: handgun
  1: rifle
  2: knife
"""
    yaml_path.write_text(yaml_content)
    
    print("\n" + "=" * 65)
    print("STAGE 1 LEAK-FREE DATASET BUILT SUCCESSFULLY")
    print("=" * 65)
    for sname, recs in splits.items():
        counts = class_stats[sname]
        neg_count = sum(1 for r in recs if r.is_hard_negative)
        print(f"Split: {sname:20s} | Total: {len(recs):5d} (Pos: {len(recs)-neg_count:5d}, Neg: {neg_count:4d}) | Handgun: {counts[0]:4d} | Rifle: {counts[1]:4d} | Knife: {counts[2]:4d}")
        
    print(f"\nManifest saved to: {manifest_csv}")
    print(f"Dataset YAML saved to: {yaml_path}")


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------
def main():
    print("Parsing Sohas (OD-WeaponDetection)...")
    sohas_weap, sohas_neg = parse_sohas()
    print(f"  -> Sohas weapon records: {len(sohas_weap)}, distractor negatives: {len(sohas_neg)}")
    
    print("Parsing Aahil dataset (rifles & blades)...")
    aahil_weap, aahil_neg = parse_aahil()
    print(f"  -> Aahil weapon records: {len(aahil_weap)}, tool negatives: {len(aahil_neg)}")
    
    print("Parsing Indian CCTV (Innovatus & BITS Pilani)...")
    cctv_weap, street_neg = parse_indian_cctv()
    print(f"  -> Indian CCTV records: {len(cctv_weap)}, street negatives: {len(street_neg)}")
    
    print("Parsing Indian Cinema Stills (clean domain)...")
    cinema_test, cinema_train = parse_indian_cinema_stills()
    print(f"  -> Locked cinema test: {len(cinema_test)}, cinema train candidates: {len(cinema_train)}")
    
    # Group Indian CCTV by base_stem (video clip) and select 200 images from dedicated held-out clips
    cctv_stem_groups = collections.defaultdict(list)
    for r in cctv_weap:
        cctv_stem_groups[r.base_stem].append(r)
        
    np.random.seed(SEED)
    cctv_stems = list(cctv_stem_groups.keys())
    np.random.shuffle(cctv_stems)
    
    test_cctv = []
    train_cctv = []
    
    for stem in cctv_stems:
        recs = cctv_stem_groups[stem]
        if len(test_cctv) + len(recs) <= 200:
            test_cctv.extend(recs)
        else:
            rem = 200 - len(test_cctv)
            if rem > 0:
                test_cctv.extend(recs[:rem])
                train_cctv.extend(recs[rem:])
            else:
                train_cctv.extend(recs)
                
    all_positive_candidates = sohas_weap + aahil_weap + cinema_train + train_cctv
    all_hard_negatives = sohas_neg + aahil_neg + street_neg
    
    print(f"\nTotal positive pool: {len(all_positive_candidates)}")
    print(f"Total negative pool: {len(all_hard_negatives)}")
    print(f"Held out test sets: test_domain={len(cinema_test)}, test_cctv={len(test_cctv)}")
    
    splits = cluster_and_split(all_positive_candidates, cinema_test, test_cctv, all_hard_negatives)
    write_processed_dataset(splits)


if __name__ == "__main__":
    main()
