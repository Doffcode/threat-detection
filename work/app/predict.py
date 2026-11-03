"""Run the engine on an image, a directory, a .txt list of image paths, or a video.

    python predict.py --model-dir ../runs/indian_strict_m960 --source img_dir/ --out out/
    python predict.py --model-dir ../runs/indian_strict_m960 --verifier ../runs/verifier_quick/weights/best.pt \
        --source ../data/v2/test_domain.txt --limit 20 --out out/ --device cpu
    python predict.py --config app_config.json --source clip.mp4 --out out/

--model-dir picks up weights/best.pt, thresholds (pipeline[_tta]/pipeline_thresholds.json or thresholds.json),
imgsz and the verifier named in pipeline_thresholds.json. Images run in image mode (no temporal smoothing,
TTA on) and videos in video mode (TTA off), unless --mode / --tta / --no-tta say otherwise. Every image is a
separate predict call (never a batched list). Writes annotated images / an annotated .mp4 plus a
detections.csv (one row per detection: det_conf, verifier_p and the final combined score) to --out.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_intent_engine import (  # noqa: E402
    ARMED,
    COMBINE_CHOICES,
    DEFAULT_PERSON_CONF,
    DEFAULT_PERSON_WEIGHTS,
    SuspiciousBehaviorEngine,
    WeightsError,
)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
CSV_HEADER = ["source", "frame", "class", "det_conf", "verifier_p", "score",
              "x1", "y1", "x2", "y2", "threat_score", "flags"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=None,
                   help="training run dir: weights/best.pt + thresholds + imgsz (+ verifier from pipeline_thresholds.json)")
    p.add_argument("--config", default=None, help="JSON of engine arguments (see README); CLI flags override it")
    p.add_argument("--weights", default=None, help="weapon detector .pt (default: --model-dir, $WEAPON_WEIGHTS or newest work/runs/*/weights/best.pt)")
    p.add_argument("--source", required=True, help="image file, directory, .txt list of image paths, or video file")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--limit", type=int, default=0, help="only the first N sources (0 = all)")
    p.add_argument("--verifier", default=None,
                   help="YOLO classification .pt (gun / knife / not_weapon); 'none' disables; "
                        "default with --model-dir: the one in pipeline_thresholds.json")
    p.add_argument("--thresholds", default=None, help="use exactly this thresholds json (either format)")
    p.add_argument("--combine", default=None, choices=COMBINE_CHOICES,
                   help="detector x verifier score mode (default: pipeline_thresholds.json selected_mode, else prod)")
    p.add_argument("--nms-iou", type=float, default=None, help="detector NMS IoU (default 0.5)")
    p.add_argument("--person-weights", default=None,
                   help="COCO person detector for the derived 'armed person' ('none' to disable). "
                        f"Default {DEFAULT_PERSON_WEIGHTS}. Unused with old 3-class models.")
    p.add_argument("--person-det-conf", type=float, default=None, help=f"person detector confidence ({DEFAULT_PERSON_CONF})")
    p.add_argument("--mode", default=None, choices=("image", "video"),
                   help="force a mode for every source (default: image for images, video for videos)")
    p.add_argument("--tta", default=None, action=argparse.BooleanOptionalAction,
                   help="force TTA on (--tta) or off (--no-tta) for every source. Default: on for images, off for video")
    p.add_argument("--pose-weights", default=None)
    p.add_argument("--conf", type=float, default=None, help="fallback threshold for classes without a specific one")
    p.add_argument("--knife-conf", type=float, default=None)
    p.add_argument("--gun-conf", type=float, default=None)
    p.add_argument("--person-conf", type=float, default=None, help="'armed person' threshold (old 3-class models)")
    p.add_argument("--imgsz", type=int, default=None, help="detector size (default: thresholds file / run args.yaml / checkpoint)")
    p.add_argument("--tile", action="store_true", default=None)
    p.add_argument("--contour-edge", action="store_true", default=None, help="only if the model was trained on contour-edge images")
    p.add_argument("--camouflage", action="store_true", default=None, help="only if the model was trained on camouflage-enhanced images")
    p.add_argument("--device", default=None, help="'cpu', 0, ...")
    p.add_argument("--max-frames", type=int, default=0, help="stop videos after N frames (0 = all)")
    p.add_argument("--allow-any-model", action="store_true", default=None, help=argparse.SUPPRESS)  # testing only
    return p.parse_args(argv)


def engine_kwargs(a) -> dict:
    """CLI -> engine kwargs; None = not given (engine / config default)."""
    class_conf = {k: v for k, v in (("knife", a.knife_conf), ("gun", a.gun_conf), (ARMED, a.person_conf))
                  if v is not None}
    person = a.person_weights
    if person is not None and str(person).lower() in ("", "none"):
        person = ""  # '' disables (None would mean "default")
    return dict(model_dir=a.model_dir, weights=a.weights, verifier_weights=a.verifier, thresholds_file=a.thresholds,
                combine=a.combine, nms_iou=a.nms_iou, person_weights=person, person_conf=a.person_det_conf,
                augment=a.tta, pose_weights=a.pose_weights, conf_thresh=a.conf, class_conf=class_conf or None,
                enable_contour_edge=a.contour_edge, enable_camouflage=a.camouflage, imgsz=a.imgsz, tile=a.tile,
                device=a.device, allow_any_model=a.allow_any_model)


def build_engine(a) -> SuspiciousBehaviorEngine:
    kw = engine_kwargs(a)
    if a.config:
        return SuspiciousBehaviorEngine.from_config(a.config, **kw)
    return SuspiciousBehaviorEngine(**{k: v for k, v in kw.items() if v is not None})


def read_list(txt: Path):
    """Image paths from a .txt list (one per line; relative paths are relative to the list's directory)."""
    for line in txt.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        p = Path(line)
        yield p if p.is_absolute() else (txt.parent / p)


def iter_sources(src: Path):
    if src.suffix.lower() == ".txt":
        yield from read_list(src)
    elif src.is_dir():
        for f in sorted(src.iterdir()):
            if f.suffix.lower() in IMG_EXT | VID_EXT:
                yield f
    else:
        yield src


def _fmt(v, nd=4):
    return "" if v is None else f"{v:.{nd}f}"


def write_rows(writer, name, idx, meta):
    common = [meta["threat_score"], ";".join(meta["flags"])]
    for d in meta["weapons"]:
        writer.writerow([name, idx, d["name"], _fmt(d["det_conf"]), _fmt(d["verifier_p"]), _fmt(d["conf"]),
                         *[f"{v:.1f}" for v in d["box"]], *common])
    for b in meta["armed_persons"]:  # derived boxes carry no score of their own
        writer.writerow([name, idx, ARMED, "", "", "", *[f"{v:.1f}" for v in b], *common])


def run_image(engine, path, out_dir, writer, mode="image"):
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"[skip] cannot read {path}")
        return
    engine.reset()
    annotated, meta = engine.process_frame(frame, mode=mode)
    cv2.imwrite(str(out_dir / path.name), annotated)
    write_rows(writer, path.name, 0, meta)
    print(f"{path.name}: score={meta['threat_score']:.0f} flags={meta['flags']} dets={meta['raw_counts']}")


def run_video(engine, path, out_dir, writer, max_frames=0, mode="video"):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"[skip] cannot open {path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out_dir / (path.stem + "_annotated.mp4")
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    engine.reset()
    idx, peak = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        annotated, meta = engine.process_frame(frame, mode=mode)
        vw.write(annotated)
        peak = max(peak, meta["threat_score"])
        write_rows(writer, path.name, idx, meta)
        idx += 1
    cap.release()
    vw.release()
    print(f"{path.name}: {idx} frames, peak score={peak:.0f} -> {out_path}")


def main(argv=None):
    a = parse_args(argv)
    try:
        engine = build_engine(a)
    except (WeightsError, FileNotFoundError, ValueError) as e:
        sys.exit(f"ERROR: {e}")
    print(engine.describe())

    src, out_dir = Path(a.source), Path(a.out)
    if not src.exists():
        sys.exit(f"ERROR: source not found: {src}")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_dir / "detections.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for path in iter_sources(src):  # one source (and one predict call per image) at a time
            if a.limit and n >= a.limit:
                break
            if path.suffix.lower() in VID_EXT:
                run_video(engine, path, out_dir, writer, a.max_frames, mode=a.mode or "video")
            elif path.suffix.lower() in IMG_EXT:
                run_image(engine, path, out_dir, writer, mode=a.mode or "image")
            else:
                continue
            n += 1
    print(f"{n} source(s); outputs in {out_dir}")


if __name__ == "__main__":
    main()
