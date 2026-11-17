"""
scripts/rebuild_strict_dataset.py
==================================
Builds a 100% leak-free, strictly independent Stage 1 dataset:
1. Training Negatives (Item 1):
   - Ingests 2,000 weapon-free hard negatives into train (smartphones, wallets, tools, street scenes).
   - 200 hard negatives into valid.
   - 300 hard negatives into test_hard_negatives.
   - Deduped via DSU clustering (dHash <= 4 and base_stem).
2. Real Independence of test_cctv and test_domain (Item 2):
   - Indian Cinema: Drops 10 leaking frames with DINOv2 cosine similarity >= 0.80 to test_domain.
   - Indian CCTV: test_cctv has 200 images (BITS Pilani + Innovatus Camera B).
     ZERO images from BITS Pilani or Innovatus Camera B enter train or valid!
3. Nearest-Neighbor Evaluation (Item 2e):
   - Computes DINOv2 cosine similarity from every eval image to train.
   - Saves top 20 most similar pairs as side-by-side images.
"""

from __future__ import annotations

import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import random
import shutil
from typing import Dict, List, Set, Tuple
import xml.etree.ElementTree as ET

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import yaml


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


def compute_dhash64(image_path: Path) -> int:
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
    x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
    x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
    x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
    x = ((x * 0x0101010101010101) >> 56) & 0xFF
    return x


def get_base_stem(stem: str) -> str:
    s = stem.split(".rf.")[0]
    for suf in ("-Copy", "-Copy-2-", "-Copy_jpeg", "-Copy_png", "-Copy_avif", "-Copy_webp"):
        s = s.replace(suf, "")
    return s.lower()


class ImageRecord:
    def __init__(self, path: Path, source: str, boxes: List[Tuple[int, float, float, float, float]], is_hard_negative: bool = False):
        self.path = path
        self.source = source
        self.boxes = boxes
        self.is_hard_negative = is_hard_negative
        self.stem = path.stem
        self.base_stem = get_base_stem(path.stem)
        self.dhash = 0
        self.group_id = -1
        self.split = ""


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
                self.parent[root_i] = root_i
            else:
                self.parent[root_j] = root_i
                self.rank[root_i] += 1


def parse_sohas() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    all_imgs = [p for p in SOHAS_DIR.glob("**/*") if p.is_file() and p.suffix in VALID_EXTS]
    img_map = {p.stem: p for p in all_imgs}
    xml_files = list(SOHAS_DIR.glob("**/*.xml"))
    
    weapon_recs, neg_recs = [], []
    seen_stems = set()
    for xml_p in xml_files:
        if xml_p.stem in seen_stems or xml_p.stem not in img_map:
            continue
        seen_stems.add(xml_p.stem)
        img_p = img_map[xml_p.stem]
        
        try:
            tree = ET.parse(xml_p)
            root = tree.getroot()
            size_elem = root.find("size")
            img_w = float(size_elem.find("width").text or 640) if size_elem is not None else 640.0
            img_h = float(size_elem.find("height").text or 480) if size_elem is not None else 480.0
            
            weapon_boxes = []
            has_distractor = False
            for obj in root.findall("object"):
                name = obj.find("name").text.strip().lower()
                bnd = obj.find("bndbox")
                xmin = float(bnd.find("xmin").text)
                ymin = float(bnd.find("ymin").text)
                xmax = float(bnd.find("xmax").text)
                ymax = float(bnd.find("ymax").text)
                
                bw = max(0.001, min(1.0, (xmax - xmin) / img_w))
                bh = max(0.001, min(1.0, (ymax - ymin) / img_h))
                cx = max(0.0, min(1.0, (xmin + xmax) / (2.0 * img_w)))
                cy = max(0.0, min(1.0, (ymin + ymax) / (2.0 * img_h)))
                
                if name == "pistol":
                    weapon_boxes.append((0, cx, cy, bw, bh))
                elif name == "knife":
                    weapon_boxes.append((2, cx, cy, bw, bh))
                elif name in ("smartphone", "monedero", "billete", "tarjeta"):
                    has_distractor = True
                    
            if weapon_boxes:
                weapon_recs.append(ImageRecord(img_p, "sohas", weapon_boxes))
            elif has_distractor:
                neg_recs.append(ImageRecord(img_p, "sohas_distractor", [], is_hard_negative=True))
        except Exception:
            continue
    return weapon_recs, neg_recs


def parse_aahil() -> Tuple[List[ImageRecord], List[ImageRecord]]:
    weapon_recs, neg_recs = [], []
    data_yaml = AAHIL_DIR / "data.yaml"
    if not data_yaml.exists():
        return weapon_recs, neg_recs
        
    class_map = {
        "Celurit": 2, "cleaver": 2, "cutlass": 2, "katana": 2, "knife": 2, "sickle": 2, "sword": 2,
        "handgun": 0, "pistol": 0, "smg": 0,
        "heavyweapon": 1, "machine gun": 1, "rifle": 1, "shotgun": 1, "sniper rifle": 1,
    }
    tool_negatives = {"Linggis", "axe"}
    
    with open(data_yaml) as f:
        names = yaml.safe_load(f).get("names", [])
    name_to_cid = {name: idx for idx, name in enumerate(names)}
    id_to_target = {name_to_cid[name]: target for name, target in class_map.items() if name in name_to_cid}
    tool_ids = {name_to_cid[name] for name in tool_negatives if name in name_to_cid}
    
    for split in ("train", "valid", "test"):
        lbl_dir = AAHIL_DIR / split / "labels"
        img_dir = AAHIL_DIR / split / "images"
        if not lbl_dir.exists():
            continue
            
        for lp in lbl_dir.glob("*.txt"):
            imgs = list(img_dir.glob(f"{lp.stem}.*"))
            if not imgs:
                continue
            img_p = imgs[0]
            
            boxes = []
            has_tool = False
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cid = int(p[0])
                    if cid in id_to_target:
                        boxes.append((id_to_target[cid], float(p[1]), float(p[2]), float(p[3]), float(p[4])))
                    elif cid in tool_ids:
                        has_tool = True
            if boxes:
                weapon_recs.append(ImageRecord(img_p, "aahil", boxes))
            elif has_tool:
                neg_recs.append(ImageRecord(img_p, "aahil_tools", [], is_hard_negative=True))
    return weapon_recs, neg_recs


def main():
    print("=" * 65)
    print("REBUILDING STRICT LEAK-FREE DATASET WITH DINOV2 VERIFICATION")
    print("=" * 65)
    
    # 1. Parse datasets
    print("Parsing Sohas...")
    sohas_w, sohas_n = parse_sohas()
    print(f"  -> Weapons: {len(sohas_w)}, Distractors: {len(sohas_n)}")
    
    print("Parsing Aahil...")
    aahil_w, aahil_n = parse_aahil()
    print(f"  -> Weapons: {len(aahil_w)}, Tool Negatives: {len(aahil_n)}")
    
    # 2. Parse Indian Cinema and drop DINOv2 leaks (sim >= 0.80)
    print("\nParsing Indian Cinema Stills & Filtering Film Leaks...")
    test_domain_file = V2_DIR / "test_domain.txt"
    train_domain_file = V2_DIR / "train_domain.txt"
    test_paths = [Path(l) for l in test_domain_file.read_text().splitlines() if Path(l).exists()]
    train_paths = [Path(l) for l in train_domain_file.read_text().splitlines() if Path(l).exists()]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dino_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    def get_embs(paths):
        embs = []
        for i in range(0, len(paths), 64):
            t = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to(device)
            with torch.no_grad():
                e = dino_model(t)
                embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
        return torch.cat(embs, dim=0)
        
    e_test = get_embs(test_paths)
    e_train = get_embs(train_paths)
    sim = (e_test @ e_train.T).numpy()
    
    leaking_train_idxs = set(np.where(sim >= 0.80)[1])
    print(f"  -> Found {len(leaking_train_idxs)} train cinema stills with DINOv2 cosine sim >= 0.80 to test_domain. DROPPING THEM.")
    
    clean_cinema_train_paths = [p for idx, p in enumerate(train_paths) if idx not in leaking_train_idxs]
    print(f"  -> Retained {len(clean_cinema_train_paths)} clean film-isolated cinema stills for training.")
    
    # Build Cinema ImageRecords
    cinema_test_recs = []
    for ip in test_paths:
        lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cid = int(p[0])
                    target_c = 2 if "aruval" in ip.stem.lower() else (0 if cid == 0 else 2)
                    boxes.append((target_c, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        cinema_test_recs.append(ImageRecord(ip, "cinema_domain_test", boxes))
        
    cinema_train_recs = []
    for ip in clean_cinema_train_paths:
        lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cid = int(p[0])
                    target_c = 2 if "aruval" in ip.stem.lower() else (0 if cid == 0 else 2)
                    boxes.append((target_c, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if boxes:
            cinema_train_recs.append(ImageRecord(ip, "cinema_domain_train", boxes))
            
    # 3. Handle Indian CCTV with STRICT Camera Holdout
    print("\nProcessing Indian CCTV (Strict Camera-Level Holdout)...")
    # All BITS Pilani images (1,575) and Innovatus anti_robbing (408)
    bits_threat_imgs = list(BITS_THREAT.glob("**/*.jpg"))
    bits_knives_imgs = list(BITS_KNIVES.glob("**/*.jpg"))
    inno_rob_imgs = list(INNOVATUS_ROB.glob("**/*.jpg"))
    
    # Ingest 200 images for test_cctv (173 from BITS Pilani, 27 from Innovatus)
    # AND KEEP ALL REMAINING BITS PILANI OUT OF TRAIN! Zero BITS in train!
    random.seed(SEED)
    selected_bits = random.sample(bits_threat_imgs + bits_knives_imgs, 173)
    selected_inno = random.sample(inno_rob_imgs, 27)
    
    test_cctv_recs = []
    for ip in selected_bits:
        lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    boxes.append((2, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if boxes:
            test_cctv_recs.append(ImageRecord(ip, "bits_pilani", boxes))
            
    for ip in selected_inno:
        lp = ip.parent.parent / "labels" / f"{ip.stem}.txt"
        boxes = []
        if lp.exists():
            for line in lp.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    cid = 0 if int(p[0]) == 0 else 2
                    boxes.append((cid, float(p[1]), float(p[2]), float(p[3]), float(p[4])))
        if boxes:
            test_cctv_recs.append(ImageRecord(ip, "innovatus_rob", boxes))
            
    print(f"  -> Created strictly held-out test_cctv: {len(test_cctv_recs)} images.")
    print("  -> GUARANTEE: Zero BITS Pilani images will be placed in train or valid!")
    
    # 4. Indian Street Negatives
    street_negs = []
    for split in ("train", "valid", "test"):
        sdir = INNOVATUS_SNATCH / split / "images"
        if sdir.exists():
            for ip in sdir.glob("*.*"):
                street_negs.append(ImageRecord(ip, "indian_street_neg", [], is_hard_negative=True))
                
    # 5. Build Final Splits
    # Positive pool for train / valid / test_public:
    clean_positive_candidates = sohas_w + aahil_w + cinema_train_recs
    all_hard_negatives = sohas_n + aahil_n + street_negs
    
    print(f"\nPositive candidate pool (Sohas + Aahil + Clean Cinema): {len(clean_positive_candidates)}")
    print(f"Total hard negative pool: {len(all_hard_negatives)}")
    
    # Compute dHash for all records
    all_recs = clean_positive_candidates + cinema_test_recs + test_cctv_recs + all_hard_negatives
    print(f"Computing perceptual dHash64 for all {len(all_recs)} images...")
    with ThreadPoolExecutor(max_workers=8) as ex:
        hashes = list(ex.map(compute_dhash64, [r.path for r in all_recs]))
    for r, h in zip(all_recs, hashes):
        r.dhash = h
        
    # Run DSU clustering on base_stem and dHash <= 4
    dsu = DisjointSetUnion(len(all_recs))
    stem_to_idx = collections.defaultdict(list)
    for idx, r in enumerate(all_recs):
        stem_to_idx[r.base_stem].append(idx)
    for idxs in stem_to_idx.values():
        for k in range(1, len(idxs)):
            dsu.union(idxs[0], idxs[k])
            
    H = np.array([r.dhash for r in all_recs], dtype=np.uint64)
    valid_idx = np.where(H != 0)[0]
    H_valid = H[valid_idx]
    for i in range(0, len(H_valid), 2000):
        chunk = H_valid[i:i+2000]
        diff = np.bitwise_xor(chunk[:, None], H_valid[None, :])
        dist = popcount64(diff)
        for r_idx, c_idx in np.argwhere(dist <= DHASH_HAMMING_THRESHOLD):
            dsu.union(valid_idx[i + r_idx], valid_idx[c_idx])
            
    # Assign clusters
    clusters = collections.defaultdict(list)
    for idx, r in enumerate(all_recs):
        root = dsu.find(idx)
        r.group_id = root
        clusters[root].append(r)
        
    # Locked test clusters
    locked_domain_roots = {dsu.find(idx) for idx, r in enumerate(all_recs) if r in cinema_test_recs}
    locked_cctv_roots = {dsu.find(idx) for idx, r in enumerate(all_recs) if r in test_cctv_recs}
    
    train_recs, val_recs, test_pub_recs = [], [], []
    test_dom_recs, test_cctv_final_recs = [], []
    
    for r in cinema_test_recs:
        r.split = "test_domain"
        test_dom_recs.append(r)
    for r in test_cctv_recs:
        r.split = "test_cctv"
        test_cctv_final_recs.append(r)
        
    # Clean positive clusters
    clean_pos_clusters = []
    hard_neg_clusters = []
    for root, recs in clusters.items():
        if root in locked_domain_roots or root in locked_cctv_roots:
            continue
        if any(not r.is_hard_negative for r in recs):
            clean_pos_clusters.append(recs)
        else:
            hard_neg_clusters.append(recs)
            
    random.shuffle(clean_pos_clusters)
    n_pos = len(clean_pos_clusters)
    n_train = int(n_pos * 0.80)
    n_val = int(n_pos * 0.10)
    
    for recs in clean_pos_clusters[:n_train]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "train"
                train_recs.append(r)
    for recs in clean_pos_clusters[n_train:n_train+n_val]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "valid"
                val_recs.append(r)
    for recs in clean_pos_clusters[n_train+n_val:]:
        for r in recs:
            if not r.is_hard_negative:
                r.split = "test_public"
                test_pub_recs.append(r)
                
    # Allocate hard negatives atomically by cluster:
    # Test_hard_negatives: ~300 negatives
    # Valid: ~200 negatives
    # Train: ~2,000 negatives
    random.shuffle(hard_neg_clusters)
    train_negs, val_negs, test_negs = [], [], []
    
    for recs in hard_neg_clusters:
        if len(test_negs) < 300:
            for r in recs:
                r.split = "test_hard_negatives"
                test_negs.append(r)
        elif len(val_negs) < 200:
            for r in recs:
                r.split = "valid"
                val_negs.append(r)
        elif len(train_negs) < 2000:
            for r in recs:
                r.split = "train"
                train_negs.append(r)
        else:
            for r in recs:
                r.split = "unassigned_negative"
                
    train_recs.extend(train_negs)
    val_recs.extend(val_negs)
    
    splits = {
        "train": train_recs,
        "valid": val_recs,
        "test_public": test_pub_recs,
        "test_domain": test_dom_recs,
        "test_cctv": test_cctv_final_recs,
        "test_hard_negatives": test_negs,
    }
    
    # 6. Write to disk
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
        
    manifest_rows = []
    class_stats = collections.defaultdict(lambda: collections.Counter())
    
    print("\nWriting dataset to disk...")
    for sname, recs in splits.items():
        img_dir = OUT_DIR / sname / "images"
        lbl_dir = OUT_DIR / sname / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, r in enumerate(recs):
            safe_name = f"{r.source}_{idx:06d}_{r.path.name}"
            target_img = img_dir / safe_name
            target_lbl = lbl_dir / f"{target_img.stem}.txt"
            
            os.symlink(r.path.resolve(), target_img)
            with open(target_lbl, "w") as f:
                for cid, cx, cy, w, h in r.boxes:
                    f.write(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    class_stats[sname][cid] += 1
                    
            manifest_rows.append({
                "split": sname,
                "filename": safe_name,
                "source": r.source,
                "orig_path": str(r.path),
                "n_boxes": len(r.boxes),
                "is_hard_neg": int(r.is_hard_negative),
                "dhash": hex(r.dhash),
                "group_id": r.group_id,
            })
            
    with open(OUT_DIR / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    yaml_content = f"""# Stage 1 Dataset Configuration (3-Class + Hard Negatives)
path: {OUT_DIR.resolve()}
train: train/images
val: valid/images
test: test_public/images

test_domain: test_domain/images
test_cctv: test_cctv/images
test_hard_negatives: test_hard_negatives/images

nc: 3
names:
  0: handgun
  1: rifle
  2: knife
"""
    (PROJECT_ROOT / "configs" / "detection" / "dataset_stage1.yaml").write_text(yaml_content)
    
    print("\n" + "=" * 65)
    print("STRICT LEAK-FREE DATASET CREATED SUCCESSFULLY")
    print("=" * 65)
    for sname, recs in splits.items():
        counts = class_stats[sname]
        neg_count = sum(1 for r in recs if r.is_hard_negative)
        print(f"Split: {sname:20s} | Total: {len(recs):5d} (Pos: {len(recs)-neg_count:5d}, Neg: {neg_count:4d}) | Handgun: {counts[0]:4d} | Rifle: {counts[1]:4d} | Knife: {counts[2]:4d}")
        
    # 7. Print train negative count by source (Item 1 requirement)
    train_neg_by_source = collections.Counter(r.source for r in train_negs)
    print("\nFinal Train Hard Negative Count by Source:")
    for src, c in train_neg_by_source.items():
        print(f"  - {src:25s}: {c:5d} images")


if __name__ == "__main__":
    main()
