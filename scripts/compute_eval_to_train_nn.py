"""
scripts/compute_eval_to_train_nn.py
====================================
Independent verification of eval-to-train independence using DINOv2 (ViT-S/14).
Computes:
1. Cosine similarity distribution from each eval image (test_domain, test_cctv) to the train set.
2. Identifies any near-duplicate pairs (threshold >= 0.80).
3. Saves the top 20 most similar pairs as side-by-side images with annotations.
"""

from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

root = Path("data/processed/stage1_baseline")
test_dom_imgs = sorted(list((root / "test_domain" / "images").glob("*.*")))
test_cctv_imgs = sorted(list((root / "test_cctv" / "images").glob("*.*")))
train_imgs = sorted(list((root / "train" / "images").glob("*.*")))

print(f"Eval sets: test_domain={len(test_dom_imgs)}, test_cctv={len(test_cctv_imgs)}")
print(f"Train pool: {len(train_imgs)} images")

def get_embs(paths):
    embs = []
    for i in range(0, len(paths), 64):
        t = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths[i:i+64]]).to(device)
        with torch.no_grad():
            e = model(t)
            embs.append((e / e.norm(dim=-1, keepdim=True)).cpu())
    return torch.cat(embs, dim=0)

print("\nExtracting DINOv2 embeddings...")
e_dom = get_embs(test_dom_imgs)
e_cctv = get_embs(test_cctv_imgs)
e_train = get_embs(train_imgs)

# 1. test_domain to train
sim_dom = (e_dom @ e_train.T).numpy()
nn_dom = sim_dom.max(axis=1)

print("\n" + "=" * 60)
print("INDEPENDENT CHECK: test_domain -> train Nearest Neighbors")
print("=" * 60)
print(f"Min similarity:    {nn_dom.min():.4f}")
print(f"Median similarity: {np.median(nn_dom):.4f}")
print(f"95th percentile:   {np.percentile(nn_dom, 95):.4f}")
print(f"Max similarity:    {nn_dom.max():.4f}")
print(f"Pairs with sim >= 0.85: {(sim_dom >= 0.85).sum()}")
print(f"Pairs with sim >= 0.80: {(sim_dom >= 0.80).sum()}")

# 2. test_cctv to train
sim_cctv = (e_cctv @ e_train.T).numpy()
nn_cctv = sim_cctv.max(axis=1)

print("\n" + "=" * 60)
print("INDEPENDENT CHECK: test_cctv -> train Nearest Neighbors")
print("=" * 60)
print(f"Min similarity:    {nn_cctv.min():.4f}")
print(f"Median similarity: {np.median(nn_cctv):.4f}")
print(f"95th percentile:   {np.percentile(nn_cctv, 95):.4f}")
print(f"Max similarity:    {nn_cctv.max():.4f}")
print(f"Pairs with sim >= 0.85: {(sim_cctv >= 0.85).sum()}")
print(f"Pairs with sim >= 0.80: {(sim_cctv >= 0.80).sum()}")

# 3. Save Top 20 Most Similar (Borderline) Pairs
out_dir = Path("outputs/debug/nearest_neighbors")
out_dir.mkdir(parents=True, exist_ok=True)

# Combine candidate pairs
all_pairs = []
for i, dom_p in enumerate(test_dom_imgs):
    j = np.argmax(sim_dom[i])
    all_pairs.append((sim_dom[i, j], "test_domain", dom_p, train_imgs[j]))
    
for i, cctv_p in enumerate(test_cctv_imgs):
    j = np.argmax(sim_cctv[i])
    all_pairs.append((sim_cctv[i, j], "test_cctv", cctv_p, train_imgs[j]))
    
all_pairs.sort(key=lambda x: x[0], reverse=True)

print(f"\nSaving top 20 most similar pairs to {out_dir}/...")
for rank, (score, eval_set, ep, tp) in enumerate(all_pairs[:20], 1):
    img_eval = cv2.imread(str(ep))
    img_train = cv2.imread(str(tp))
    
    # Resize to common height
    h = 360
    w_eval = int(img_eval.shape[1] * (h / img_eval.shape[0]))
    w_train = int(img_train.shape[1] * (h / img_train.shape[0]))
    
    r_eval = cv2.resize(img_eval, (w_eval, h))
    r_train = cv2.resize(img_train, (w_train, h))
    
    combined = np.hstack([r_eval, r_train])
    
    # Overlay banner
    banner = np.zeros((40, combined.shape[1], 3), dtype=np.uint8)
    text = f"Rank {rank:02d} | Sim: {score:.4f} | LEFT ({eval_set}): {ep.name[:30]}... | RIGHT (train): {tp.name[:30]}..."
    cv2.putText(banner, text, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    final_img = np.vstack([banner, combined])
    
    out_p = out_dir / f"pair_{rank:02d}_{score:.3f}_{eval_set}.jpg"
    cv2.imwrite(str(out_p), final_img)
    
print(f"[PASS] Successfully saved 20 pairs to {out_dir}/")
