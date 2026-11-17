"""
scripts/review_cctv_test_v2.py
==============================
Streamlit visual inspector and occlusion annotation helper for CCTV Test v2.
Allows rapid review of all 92 frames, visual verification of bounding boxes,
and 1-click tagging of occlusion levels and occluder types into:
configs/detection/cctv_test_v2_occlusion_template.csv

Run with:
    work/.venv/bin/streamlit run scripts/review_cctv_test_v2.py
"""

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

WORKSPACE = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
IMAGES_DIR = WORKSPACE / "data/cctv_test_v2_candidates/images"
LABELS_DIR = WORKSPACE / "data/cctv_test_v2_candidates/labels"
TEMPLATE_CSV = WORKSPACE / "configs/detection/cctv_test_v2_occlusion_template.csv"

CLASS_NAMES = {0: "handgun", 1: "rifle", 2: "knife"}
CLASS_COLORS = {0: (0, 128, 255), 1: (0, 200, 0), 2: (255, 50, 50)}  # RGB

st.set_page_config(page_title="CCTV Test v2 Reviewer", layout="wide")

@st.cache_data
def load_template():
    return pd.read_csv(TEMPLATE_CSV)

def save_template(df):
    df.to_csv(TEMPLATE_CSV, index=False)
    st.cache_data.clear()

def parse_yolo_file(lbl_path):
    boxes = []
    if not lbl_path.exists():
        return boxes
    lines = [l.strip() for l in lbl_path.read_text().strip().splitlines() if l.strip()]
    for line in lines:
        parts = line.split()
        if len(parts) >= 5:
            cid = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((cid, cx, cy, w, h))
    return boxes

def draw_boxes_on_image(img_pil, boxes):
    w_img, h_img = img_pil.size
    img_cv = np.array(img_pil)
    
    for cid, cx, cy, bw, bh in boxes:
        cname = CLASS_NAMES.get(cid, f"cls_{cid}")
        color = CLASS_COLORS.get(cid, (255, 255, 0))
        
        x1 = int((cx - bw / 2) * w_img)
        y1 = int((cy - bh / 2) * h_img)
        x2 = int((cx + bw / 2) * w_img)
        y2 = int((cy + bh / 2) * h_img)
        
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_img - 1, x2), min(h_img - 1, y2)
        
        cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)
        label_txt = f"{cname} ({bw*bh*100:.2f}%)"
        cv2.putText(img_cv, label_txt, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
    return Image.fromarray(img_cv)

# App UI
st.title("🎯 CCTV Test v2 — Rapid Annotation & Occlusion Review")

df = load_template()
total_frames = len(df)

if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])

with col_nav1:
    if st.button("⬅️ Previous") and st.session_state.frame_idx > 0:
        st.session_state.frame_idx -= 1
        st.rerun()

with col_nav3:
    if st.button("Next ➡️") and st.session_state.frame_idx < total_frames - 1:
        st.session_state.frame_idx += 1
        st.rerun()

with col_nav2:
    st.session_state.frame_idx = st.slider(
        f"Frame {st.session_state.frame_idx + 1} of {total_frames}",
        0, total_frames - 1, st.session_state.frame_idx
    )

idx = st.session_state.frame_idx
row = df.iloc[idx]
cand_img_name = row["image_filename"]
img_path = IMAGES_DIR / cand_img_name
lbl_path = LABELS_DIR / Path(cand_img_name).with_suffix(".txt").name

boxes = parse_yolo_file(lbl_path)

col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(f"Frame #{idx:03d} — {cand_img_name}")
    st.caption(f"Source: **{row['source']}** | Sequence: **{row['sequence_id']}** | Frame Num: **{row.get('frame_number', 'N/A')}**")
    
    if img_path.exists():
        raw_img = Image.open(img_path)
        annotated_img = draw_boxes_on_image(raw_img, boxes)
        st.image(annotated_img, use_container_width=True)
    else:
        st.error(f"Image not found at: {img_path}")
        
    st.info(f"**Bounding Boxes in label file:** {len(boxes)}")
    for b_idx, (cid, cx, cy, bw, bh) in enumerate(boxes):
        st.code(f"Box {b_idx}: Class {cid} ({CLASS_NAMES.get(cid)}) | cx={cx:.4f}, cy={cy:.4f}, w={bw:.4f}, h={bh:.4f} (Area={bw*bh*100:.2f}%)")

with col_right:
    st.subheader("📋 Occlusion Metadata")
    
    curr_occ = str(row["occlusion_level"]) if pd.notna(row["occlusion_level"]) else "none"
    curr_notes = str(row["reviewer_notes"]) if pd.notna(row["reviewer_notes"]) else ""
    
    # Auto-detect default if empty
    occ_options = ["none", "under 50%", "over 50%"]
    default_occ_idx = occ_options.index(curr_occ) if curr_occ in occ_options else 0
    
    selected_occ = st.radio(
        "Occlusion Level:",
        occ_options,
        index=default_occ_idx,
        help="How much of the weapon is hidden? (none, under 50%, over 50%)"
    )
    
    common_occluders = ["attacker's hand", "victim's body", "desk", "clothing", "clean/unoccluded", "other"]
    selected_occluder = st.selectbox(
        "Occluder Quick Pick:",
        ["(Select or type below)"] + common_occluders
    )
    
    notes_val = curr_notes
    if selected_occluder != "(Select or type below)":
        notes_val = selected_occluder
        
    notes_input = st.text_input("Reviewer Notes (What hides it / details):", value=notes_val)
    
    st.divider()
    
    if st.button("💾 Save Occlusion & Advance ➡️", type="primary", use_container_width=True):
        # Update row in dataframe
        df.at[idx, "weapon_present"] = len(boxes) > 0
        if len(boxes) > 0:
            df.at[idx, "weapon_class"] = ";".join([CLASS_NAMES.get(b[0], str(b[0])) for b in boxes])
            df.at[idx, "bbox_normalized_xywh"] = ";".join([f"{b[1]:.6f},{b[2]:.6f},{b[3]:.6f},{b[4]:.6f}" for b in boxes])
        else:
            df.at[idx, "weapon_class"] = ""
            df.at[idx, "bbox_normalized_xywh"] = ""
            
        df.at[idx, "occlusion_level"] = selected_occ
        df.at[idx, "reviewer_notes"] = notes_input
        
        save_template(df)
        st.success(f"Saved Frame #{idx:03d}!")
        if st.session_state.frame_idx < total_frames - 1:
            st.session_state.frame_idx += 1
            st.rerun()

st.divider()
st.subheader("📊 Dataset Progress Overview")
st.dataframe(
    df[["image_filename", "source", "weapon_present", "weapon_class", "occlusion_level", "reviewer_notes"]],
    use_container_width=True,
    height=250
)
