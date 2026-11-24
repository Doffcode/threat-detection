"""Score a fixed labeled clip manifest using pinned upstream model source."""

import argparse
import csv
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch

MODELS = {
    "x3d": ("x3d_m", 16, 0.4),
    "r2plus1d": ("r2plus1d_18", 32, 0.5),
}


def load_behavior_model(name, source, checkpoint, device):
    sys.path.insert(0, str(source.resolve()))
    from sv_model import create_model

    backbone, _, _ = MODELS[name]
    model = create_model(backbone=backbone, num_classes=2, pretrained=False)
    scalar = getattr(np, "_core", np.core).multiarray.scalar
    metadata_types = [scalar, np.dtype, *(type(np.dtype(kind)) for kind in ("float32", "float64", "int64"))]
    allow_metadata = getattr(torch.serialization, "safe_globals", lambda _: nullcontext())
    with allow_metadata(metadata_types):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=MODELS)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = load_behavior_model(args.model, args.source, args.checkpoint, args.device)
    from video_utils import preprocess_video

    _, frames, threshold = MODELS[args.model]
    with args.manifest.open(newline="") as file:
        clips = list(csv.DictReader(file))

    @torch.inference_mode()
    def predict(path):
        start = time.perf_counter()
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps <= 0 or frame_count <= 0:
                raise ValueError("Missing clip duration")
        finally:
            capture.release()
        tensor = preprocess_video(path, num_frames=frames).to(args.device)
        probability = model(tensor).softmax(dim=1)[0, 1].item()
        return probability, time.perf_counter() - start, frame_count / fps

    for label in ("0", "1"):
        clip = next(row for row in clips if row["label"] == label)
        predict(args.manifest.parent / clip["path"])

    with args.output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("clip", "label", "model", "score", "prediction", "seconds", "duration", "error"))
        for clip in clips:
            prefix = (clip["clip"], clip["label"], args.model)
            try:
                score, seconds, duration = predict(args.manifest.parent / clip["path"])
                writer.writerow((*prefix, score, int(score >= threshold), seconds, duration, ""))
            except (RuntimeError, ValueError, OSError) as error:
                writer.writerow((*prefix, "", "", "", "", str(error)))


if __name__ == "__main__":
    main()
