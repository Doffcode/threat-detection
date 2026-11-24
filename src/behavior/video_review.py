"""Clip-level fight scores and independent approach-and-retreat evidence."""

import subprocess
import tempfile
from collections import deque
from pathlib import Path
from itertools import permutations

import cv2
import numpy as np
import torch


class ApproachRetreat:
    def __init__(self, fps):
        self.history = deque(maxlen=max(2, round(0.5 * fps) + 1))
        self.active = set()

    def update(self, tracks):
        boxes = {track.track_id: np.asarray(track.box) for track in tracks if track.track_id >= 0}
        self.history.append(boxes)
        if len(self.history) < self.history.maxlen:
            return []
        visible = set.intersection(*(set(frame) for frame in self.history))
        first = self.history[0]
        active = set()
        events = []
        for a, b in permutations(sorted(visible), 2):
            start_a, start_b, end_a, end_b = (first[a], first[b], boxes[a], boxes[b])
            def foot(box):
                return np.array([(box[0] + box[2]) / 2, box[3]])
            start = foot(start_b) - foot(start_a)
            distance = np.linalg.norm(start)
            height = np.mean([box[3] - box[1] for box in (start_a, start_b, end_a, end_b)])
            if distance <= 0 or height <= 0:
                continue
            direction = start / distance
            approach = np.dot(foot(end_a) - foot(start_a), direction) / height
            retreat = np.dot(foot(end_b) - foot(start_b), direction) / height
            before = distance / height
            after = np.linalg.norm(foot(end_b) - foot(end_a)) / height
            if approach >= 0.5 and retreat >= 0.15 and before - after >= 0.3 and after <= 2:
                pair = tuple(sorted((a, b)))
                if pair not in active and pair not in self.active:
                    events.append(dict(approaching=a, retreating=b, approach=approach,
                                       retreat=retreat, separation_start=before, separation_end=after))
                active.add(pair)
        self.active = active
        return events


@torch.inference_mode()
def review_video(path, model, device, tracker=None):
    capture = cv2.VideoCapture(str(path))
    windows, events, frames = [], [], []
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no readable frame rate.")
        window_size = max(1, round(5 * fps))
        movement = ApproachRetreat(fps)
        radius = max(1, round(fps))
        recent = deque(maxlen=radius + 1)
        pending = deque()
        if tracker is not None:
            tracker.reset()

        def score(end_frame):
            if model is None:
                frames.clear()
                return
            sample = np.linspace(0, len(frames) - 1, 16).astype(int)
            array = np.stack([frames[i] for i in sample])
            tensor = torch.from_numpy(array).permute(3, 0, 1, 2).contiguous().float()
            tensor = ((tensor / 255 - 0.45) / 0.225).unsqueeze(0).to(device)
            probability = model(tensor).softmax(dim=1)[0, 1].item()
            windows.append(dict(start_s=(end_frame - len(frames)) / fps, end_s=end_frame / fps,
                                score=probability, label="Fight detected" if probability >= 0.4 else "No fight detected"))
            frames.clear()

        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.resize(frame, (224, 224))[:, :, ::-1].copy())
            if tracker is not None:
                tracks = tracker.track(frame)
                detected = movement.update(tracks)
                boxes = movement.history[-1]
                recent.append((index, boxes))
                while pending and index > pending[0][0]:
                    pending.popleft()
                for _, replay, ids in pending:
                    replay[index] = {key: boxes[key] for key in ids if key in boxes}
                for event in detected:
                    still = frame.copy()
                    for role, color in (("approaching", (255, 230, 40)), ("retreating", (0, 165, 255))):
                        track_id = event[role]
                        points = np.asarray([
                            ((box[0] + box[2]) / 2, box[3])
                            for frame_boxes in movement.history
                            for box in (frame_boxes[track_id],)
                        ], dtype=np.int32)
                        cv2.polylines(still, [points], False, color, 2)
                        x1, y1, x2, y2 = map(int, movement.history[-1][track_id])
                        cv2.rectangle(still, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(still, f"Track {track_id}", (x1, max(y1, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    encoded, data = cv2.imencode(".jpg", still)
                    ids = (event["approaching"], event["retreating"])
                    replay = {
                        frame_index: {key: frame_boxes[key] for key in ids if key in frame_boxes}
                        for frame_index, frame_boxes in recent
                    }
                    pending.append((index + radius, replay, ids))
                    events.append((dict(timestamp_s=index / fps, **event),
                                   data.tobytes() if encoded else None, replay))
            index += 1
            if len(frames) == window_size:
                score(index)
        if frames:
            score(index)
        if index == 0:
            raise ValueError("Video contains no readable frames.")
        return windows, events
    finally:
        capture.release()


def encode_cue_replay(video, event, replay):
    """Render the recorded evidence boxes without running the tracker again."""
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / "source.mp4"
        raw = Path(temp) / "frames.bgr"
        output = Path(temp) / "replay.mp4"
        source.write_bytes(video)
        capture = cv2.VideoCapture(str(source))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            if not np.isfinite(fps) or fps <= 0 or not replay:
                raise ValueError("Replay has no readable frames.")
            start = min(replay)
            if start and not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise ValueError("Could not seek to the cue.")
            with raw.open("wb") as file:
                for index in range(start, max(replay) + 1):
                    ok, frame = capture.read()
                    if not ok:
                        raise ValueError("Could not decode a cue frame.")
                    height, width = frame.shape[:2]
                    for role, color in (("approaching", (255, 230, 40)), ("retreating", (0, 165, 255))):
                        track_id = event[role]
                        box = replay[index].get(track_id)
                        if box is None:
                            continue
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"Track {track_id}", (x1, max(y1, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    file.write(frame.tobytes())
        finally:
            capture.release()
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", str(raw),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ], check=True, capture_output=True, timeout=30)
        return output.read_bytes()
