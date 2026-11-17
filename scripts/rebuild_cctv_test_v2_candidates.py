"""
scripts/rebuild_cctv_test_v2_candidates.py
=========================================
Rebuilds the CCTV test v2 candidate pool strictly following user guidelines:
1. ~60-80 Innovatus frames by temporal stride, max ~25-30 per sequence.
   - Selected: Seq 02 (20 frames, stride 2), Seq 07 (26 frames, stride 3), Seq 08 (26 frames, stride 4).
   - Total Innovatus = 72 frames.
   - Camera: All Innovatus frames originate from a single camera (Camera B - office robbery footage).
   - Holdout Guarantee: Sequences 02, 07, and 08 are permanently quarantined as test-only (ZERO overlap with future fine-tuning pool).
   - Overlap with test_cctv_v1: 0 frames.
2. ~20 BITS Pilani frames:
   - Selected: 10 from bits-pilani__knives-tkfol__v1, 10 from bits-pilani__threatdetect2__v7.
   - Overlap with test_cctv_v1: 0 frames.
3. Grand total: 92 frames.
4. Generates empty label files and occlusion annotation template.
"""

from pathlib import Path
import shutil
import re
import json
import pandas as pd

def main():
    workspace = Path("/home/shiva/Desktop/projects/akriti/weapon-detection")
    manifest_path = workspace / "data/processed/stage1_baseline/manifest.csv"
    df = pd.read_csv(manifest_path)

    # 1. Innovatus Frames from held_out_cctv (excluding test_cctv_v1)
    inno = df[(df["source"] == "innovatus_rob") & (df["disposition"] == "held_out_cctv")].copy()

    def parse_frame_num(fname):
        m = re.search(r"innovatus_rob_\d+_(\d+)_jpg", fname)
        if m:
            return int(m.group(1))
        m2 = re.search(r"^(\d+)_jpg", fname)
        if m2:
            return int(m2.group(1))
        return -1

    inno["frame_num"] = inno["filename"].apply(parse_frame_num)
    inno = inno.sort_values("frame_num")

    # Select sequences by temporal stride
    # Seq02: 213-279 (40 held-out frames) -> stride 2 -> 20 frames
    seq02 = inno[(inno["frame_num"] >= 213) & (inno["frame_num"] <= 279)].iloc[::2].copy()
    seq02["sequence_id"] = "innovatus_cam_b_seq02"
    seq02["camera_id"] = "innovatus_cam_b"

    # Seq07: 440-523 (77 held-out frames) -> stride 3 -> 26 frames
    seq07 = inno[(inno["frame_num"] >= 440) & (inno["frame_num"] <= 523)].iloc[::3].copy()
    seq07["sequence_id"] = "innovatus_cam_b_seq07"
    seq07["camera_id"] = "innovatus_cam_b"

    # Seq08: 542-666 (101 held-out frames) -> stride 4 -> 26 frames
    seq08 = inno[(inno["frame_num"] >= 542) & (inno["frame_num"] <= 666)].iloc[::4].copy()
    seq08["sequence_id"] = "innovatus_cam_b_seq08"
    seq08["camera_id"] = "innovatus_cam_b"

    inno_selected = pd.concat([seq02, seq07, seq08])

    # 2. BITS Pilani Frames from held_out_cctv
    bits_tkfol = df[(df["orig_path"].str.contains("knives-tkfol", na=False)) & (df["disposition"] == "held_out_cctv")].sort_values("filename")
    bits_threat = df[(df["orig_path"].str.contains("threatdetect2", na=False)) & (df["disposition"] == "held_out_cctv")].sort_values("filename")

    stride_tk = len(bits_tkfol) // 10
    stride_th = len(bits_threat) // 10
    sel_tkfol = bits_tkfol.iloc[::stride_tk].head(10).copy()
    sel_tkfol["sequence_id"] = "bits_knives_tkfol"
    sel_tkfol["camera_id"] = "bits_lab_cctv"

    sel_threat = bits_threat.iloc[::stride_th].head(10).copy()
    sel_threat["sequence_id"] = "bits_threatdetect2"
    sel_threat["camera_id"] = "bits_lab_cctv"

    bits_selected = pd.concat([sel_tkfol, sel_threat])

    total_selected = pd.concat([inno_selected, bits_selected]).reset_index(drop=True)

    # Verify zero overlap with test_cctv_v1
    v1_files = set(df[df["split"] == "test_cctv_v1"]["filename"])
    overlap_v1 = set(total_selected["filename"]) & v1_files
    assert len(overlap_v1) == 0, f"Found unexpected overlap with test_cctv_v1: {overlap_v1}"

    print(f"[VERIFIED] Total selected: {len(total_selected)} frames")
    print(f"  - Innovatus Seq02 (stride 2): {len(seq02)} frames (Frames: {seq02['frame_num'].min()}..{seq02['frame_num'].max()})")
    print(f"  - Innovatus Seq07 (stride 3): {len(seq07)} frames (Frames: {seq07['frame_num'].min()}..{seq07['frame_num'].max()})")
    print(f"  - Innovatus Seq08 (stride 4): {len(seq08)} frames (Frames: {seq08['frame_num'].min()}..{seq08['frame_num'].max()})")
    print(f"  - Total Innovatus: {len(inno_selected)} frames (all from Camera B)")
    print(f"  - BITS Pilani: {len(bits_selected)} frames (10 tkfol + 10 threatdetect2)")
    print(f"  - Overlap with test_cctv_v1: {len(overlap_v1)}")

    # Clean out and rebuild candidates dir
    out_dir = workspace / "data/cctv_test_v2_candidates"
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"

    # Remove existing files
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    template_rows = []
    manifest_rows = []

    for idx, row in total_selected.iterrows():
        src_path = Path(row["orig_path"])
        cand_name = f"cctv_v2_{idx:03d}_{row['filename']}"
        dest_img = out_img / cand_name
        dest_lbl = out_lbl / cand_name.replace(".jpg", ".txt").replace(".png", ".txt")

        shutil.copy2(src_path, dest_img)
        dest_lbl.write_text("")  # Empty label for human annotation

        template_rows.append({
            "split": "cctv_test_v2_candidate",
            "image_filename": cand_name,
            "orig_filename": row["filename"],
            "source": row["source"],
            "camera_id": row["camera_id"],
            "sequence_id": row["sequence_id"],
            "frame_number": row.get("frame_num", -1),
            "weapon_present": "",
            "weapon_class": "",
            "bbox_normalized_xywh": "",
            "occlusion_level": "",
            "reviewer_notes": ""
        })

    # Save occlusion template
    df_template = pd.DataFrame(template_rows)
    template_path = workspace / "configs/detection/cctv_test_v2_occlusion_template.csv"
    df_template.to_csv(template_path, index=False)
    print(f"[SUCCESS] Wrote {len(df_template)} candidates to {template_path}")

    # Write holdout guarantee json
    guarantee = {
        "benchmark": "test_cctv_v2_candidate_pool",
        "total_candidate_frames": len(total_selected),
        "innovatus_summary": {
            "total_frames": len(inno_selected),
            "camera": "Camera B (single camera, office robbery footage)",
            "held_out_sequences_for_testing": [
                {
                    "sequence_id": "innovatus_cam_b_seq02",
                    "frame_range": "213-279",
                    "total_sequence_frames": 43,
                    "sampled_for_v2": len(seq02),
                    "stride": 2
                },
                {
                    "sequence_id": "innovatus_cam_b_seq07",
                    "frame_range": "440-523",
                    "total_sequence_frames": 80,
                    "sampled_for_v2": len(seq07),
                    "stride": 3
                },
                {
                    "sequence_id": "innovatus_cam_b_seq08",
                    "frame_range": "542-666",
                    "total_sequence_frames": 107,
                    "sampled_for_v2": len(seq08),
                    "stride": 4
                }
            ],
            "fine_tuning_reserve_guarantee": (
                "Sequences seq02, seq07, and seq08 (230 total frames) are strictly held out for evaluation. "
                "They will NEVER appear in any future fine-tuning or training pool. "
                "Any future fine-tuning is restricted strictly to sequences seq01 (0-172), seq03 (290-308), "
                "seq04 (321-335), seq05 (360-366), and seq06 (384-403) — total 178 frames outside test sets."
            )
        },
        "bits_pilani_summary": {
            "total_frames": len(bits_selected),
            "sources": [
                {"source": "bits-pilani__knives-tkfol__v1", "sampled_frames": 10},
                {"source": "bits-pilani__threatdetect2__v7", "sampled_frames": 10}
            ],
            "overlap_with_test_cctv_v1": 0
        }
    }
    guarantee_path = workspace / "configs/detection/cctv_v2_holdout_guarantee.json"
    guarantee_path.write_text(json.dumps(guarantee, indent=2))
    print(f"[SUCCESS] Wrote holdout guarantee to {guarantee_path}")

if __name__ == "__main__":
    main()
