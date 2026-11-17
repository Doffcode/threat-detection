"""
action_intent_engine.py
=======================

Two-stage inference engine for the weapon-detection project.

Stage 1, weapon detector (YOLO detect). Classes: ``gun``, ``knife``,
``handheld_object``. ``handheld_object`` is never reported. It exists only so
the detector has somewhere to put phones, wallets, tools, ... instead of
calling them knives or guns. Old 3-class models (``armed person``, ``gun``,
``knife``) are still accepted.

Stage 2, verifier (optional, YOLO classify). Classes: ``gun``, ``knife``,
``not_weapon``. It runs on a square crop of every gun / knife candidate
(padded 20% per side, letterboxed to 224 px; ``square_crop``, identical to
scripts/verifier/common.py), and the detector and verifier scores are combined
exactly like scripts/verifier/eval_pipeline.py::

    det    : det_conf                       cls    : p(detected class)
    prod   : det_conf * p                   geo    : sqrt(det_conf * p)
    wgeo25 : det_conf**0.75 * p**0.25       wgeo75 : det_conf**0.25 * p**0.75
    (aliases: product = prod, verifier = cls, geomean = geo)

Per-class thresholds are applied to the combined score (``score >= thr``).

Armed person (derived). A COCO person detector (default
``work/raw/IntProj_Weapon_Dataset_annot/yolo11m.pt``, class 0, conf 0.35) runs
only when at least one confirmed weapon box exists. A person box becomes
'armed person' if some confirmed weapon box has
``intersection / weapon_area >= 0.3`` with it. With an old 3-class model the
detector's own 'armed person' boxes are used and the COCO model is not loaded.

Thresholds, in priority order: ``class_conf`` argument > ``thresholds_file`` >
(with a verifier) ``<run>/pipeline_tta/pipeline_thresholds.json`` for TTA calls or
``<run>/pipeline/pipeline_thresholds.json`` otherwise (eval_pipeline.py output;
its ``selected_mode`` + ``per_mode`` thresholds, ``det_conf_floor``, ``pad``,
``cls_imgsz``, ``det_imgsz``) > ``thresholds.json`` > built-in defaults. Without a
verifier ``thresholds.json`` (detector scores) wins, then the pipeline file's
detector-only thresholds.

Inference recipe: NMS IoU 0.5, TTA (ultralytics augment=True + verifier h-flip)
on for still images and off for video, detector imgsz from the run, one image
per predict call. ``model_dir=`` points the engine at a run directory.

Modes: ``mode='image'`` (default) means each call is independent and has no
temporal smoothing. ``mode='video'`` uses k-of-n smoothing, so a class must
appear in ``smooth_min_hits`` of the last ``smooth_window`` frames.

Public API (what dashboard.py / predict.py use)::

    engine = SuspiciousBehaviorEngine(model_dir="work/runs/<run>", verifier_weights=None, mode="image", ...)
    annotated_bgr, meta = engine.process_frame(frame_bgr)          # or (frame, mode="video")
    meta = {
        "threat_score": float (0-100),
        "threat_level": str,
        "weapons": [{"name", "conf", "box": [x1, y1, x2, y2], "cls",
                     "det_conf", "verifier_p", "confirmed"}, ...],   # conf = combined score
        "armed_persons": [[x1, y1, x2, y2], ...],
        "flags": [str, ...],          # 'KNIFE_DETECTED', 'GUN_DETECTED', 'ARMED_PERSON', ...
        "raw_counts": {name: n},      # per-frame detections (gun / knife / armed person), before smoothing
        "confirmed": [name, ...],     # classes that passed temporal smoothing (all present ones in image mode)
        "mode": "image" | "video", "tta": bool, "combine": str,
    }
"""

from __future__ import annotations

import glob
import json
import math
import os
import warnings
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR.parent

ARMED = "armed person"
IGNORED_CLASSES = ("handheld_object",)
REQUIRED_CLASSES = ("knife", "gun")
WEAPON_CLASSES = ("gun", "knife")          # the only detector classes ever reported as weapons
DEFAULT_THRESHOLDS: Dict[str, float] = {ARMED: 0.30, "gun": 0.25, "knife": 0.20}

# Score modes, named exactly as in scripts/verifier/eval_pipeline.py (MODES there):
#   det     det_conf                         cls     p_class
#   prod    det_conf * p_class               geo     sqrt(det_conf * p_class)
#   wgeo25  det_conf**0.75 * p_class**0.25   wgeo75  det_conf**0.25 * p_class**0.75
COMBINE_MODES = ("det", "cls", "prod", "geo", "wgeo25", "wgeo75")
COMBINE_ALIASES = {"product": "prod", "verifier": "cls", "geomean": "geo", "detector": "det"}
COMBINE_CHOICES = COMBINE_MODES + tuple(COMBINE_ALIASES)   # everything accepted on the CLI
DEFAULT_COMBINE = "prod"
MODES = ("image", "video")

DEFAULT_PERSON_WEIGHTS = str(WORK_DIR / "raw" / "IntProj_Weapon_Dataset_annot" / "yolo11m.pt")
DEFAULT_PERSON_CONF = 0.35
DEFAULT_ARMED_IOA = 0.30
DEFAULT_VERIFIER_PAD = 0.20      # eval_pipeline.py --pad (must match make_crops.py)
DEFAULT_VERIFIER_IMGSZ = 224     # eval_pipeline.py --cls-imgsz
DEFAULT_NMS_IOU = 0.5            # measured: removes duplicate boxes, gun P 0.54 -> 0.67 at the same R
DEFAULT_IMGSZ = 960              # only if neither the thresholds file, args.yaml nor the checkpoint say
VERIFIER_BATCH = 32
NONE_VALUES = ("", "none", "off", "false", "0")

PIPELINE_THRESHOLDS_FILE = "pipeline_thresholds.json"
THRESHOLDS_FILE = "thresholds.json"
PIPELINE_DIRS = {False: "pipeline", True: "pipeline_tta"}   # eval_pipeline.py default output dirs

# Threat scoring (based on the notebook: knife +45, gun +50)
SCORE_KNIFE = 45.0
SCORE_GUN = 50.0
SCORE_ARMED_PERSON = 20.0
SCORE_RAISED_ARM = 10.0          # raised arm alone
SCORE_RAISED_ARM_WITH_WEAPON = 15.0  # extra when a weapon is also confirmed
LEVEL_CRITICAL = 75.0
LEVEL_SUSPICIOUS = 45.0

CLASS_COLORS = {  # BGR
    "knife": (0, 0, 255),
    "gun": (0, 140, 255),
    ARMED: (255, 0, 200),
}

FLAG_FOR_CLASS = {
    "knife": "KNIFE_DETECTED",
    "gun": "GUN_DETECTED",
    ARMED: "ARMED_PERSON",
}


# ----------------------------------------------------------------------------
# Preprocessors
# NOTE: enable these ONLY if the weights were trained on images produced by
# the same preprocessor (e.g. dataset_camouflage_enhanced / dataset_contour_enhanced).
# ----------------------------------------------------------------------------
class FrequencyAwareCamouflageEnhancer:
    """Copied from Phase1_Knife_Accuracy_Colab.ipynb (Step 3)."""

    def __init__(self, high_pass_weight=0.4, contrast_clip=3.0):
        self.high_pass_weight = high_pass_weight
        self.clahe = cv2.createCLAHE(clipLimit=contrast_clip, tileGridSize=(8, 8))

    def extract_frequency_camouflage_map(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        equalized = self.clahe.apply(gray)
        blurred = cv2.GaussianBlur(equalized, (21, 21), 0)
        high_pass = cv2.subtract(equalized, blurred)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        gradient = cv2.morphologyEx(high_pass, cv2.MORPH_GRADIENT, kernel)
        freq_map = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
        return freq_map

    def enhance_camouflaged_frame(self, img_bgr):
        freq_map = self.extract_frequency_camouflage_map(img_bgr)
        freq_3ch = cv2.cvtColor(freq_map, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(img_bgr, 1.0 - self.high_pass_weight, freq_3ch, self.high_pass_weight, 0)

    __call__ = enhance_camouflaged_frame


class ContourEdgeKnifePreprocessor:
    """CLAHE on luminance + Canny/contour edge overlay to emphasise thin,
    high-contrast blade outlines. Same spirit as the camouflage enhancer.

    Original implementation was not shipped with the project, so this is a
    reconstruction. If a model was trained on ``dataset_contour_enhanced``
    made with a different recipe, results will not match exactly.
    """

    def __init__(self, clip_limit=2.5, tile=(8, 8), canny_lo=50, canny_hi=150,
                 edge_weight=0.35, min_contour_len=25):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
        self.canny_lo = canny_lo
        self.canny_hi = canny_hi
        self.edge_weight = edge_weight
        self.min_contour_len = min_contour_len

    def edge_map(self, img_bgr):
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_eq = self.clahe.apply(lab[:, :, 0])
        l_blur = cv2.GaussianBlur(l_eq, (3, 3), 0)
        edges = cv2.Canny(l_blur, self.canny_lo, self.canny_hi)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(edges)
        long_contours = [c for c in contours if cv2.arcLength(c, False) >= self.min_contour_len]
        cv2.drawContours(mask, long_contours, -1, 255, 1)
        return l_eq, lab, mask

    def enhance(self, img_bgr):
        l_eq, lab, mask = self.edge_map(img_bgr)
        lab = lab.copy()
        lab[:, :, 0] = l_eq
        contrast = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        edges_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        return cv2.addWeighted(contrast, 1.0, edges_3ch, self.edge_weight, 0)

    __call__ = enhance


# ----------------------------------------------------------------------------
# Weights / thresholds helpers
# ----------------------------------------------------------------------------
class WeightsError(RuntimeError):
    pass


def find_default_weights() -> str:
    """WEAPON_WEIGHTS env var, else newest work/runs/**/weights/best.pt.
    Never falls back to a COCO checkpoint."""
    env = os.environ.get("WEAPON_WEIGHTS")
    if env:
        if not os.path.isfile(env):
            raise WeightsError(f"WEAPON_WEIGHTS={env!r} does not exist.")
        return env
    patterns = [
        str(WORK_DIR / "runs" / "*" / "weights" / "best.pt"),
        str(WORK_DIR / "runs" / "*" / "*" / "weights" / "best.pt"),  # runs/detect/<name>/...
    ]
    # Run dirs starting with '_' (e.g. runs/_smoke_train_tmp) are scratch runs and ignored.
    cands = sorted(
        {p for pat in patterns for p in glob.glob(pat)
         if not any(part.startswith("_") for part in Path(p).relative_to(WORK_DIR / "runs").parts)},
        key=os.path.getmtime,
    )
    if not cands:
        raise WeightsError(
            "No trained weapon weights found. Set WEAPON_WEIGHTS=/path/to/best.pt or pass "
            f"weights=..., or train a model so that {WORK_DIR}/runs/<name>/weights/best.pt exists. "
            "(Refusing to fall back to stock COCO yolo11n.pt: it has no 'gun' class and its "
            "'knife' is kitchen cutlery.)"
        )
    return cands[-1]


def resolve_model_dir(model_dir) -> dict:
    """A training run dir (or a weights file inside one) -> {"weights", "run_dir", "imgsz"}.

    weights = <dir>/weights/best.pt (else <dir>/best.pt); imgsz from <dir>/args.yaml (None if absent).
    Thresholds / verifier are looked up later relative to the weights (see threshold_candidates)."""
    p = Path(model_dir).expanduser()
    if p.is_file():
        weights = p
    else:
        weights = next((c for c in (p / "weights" / "best.pt", p / "best.pt") if c.is_file()), None)
        if weights is None:
            raise WeightsError(f"--model-dir {model_dir}: no weights/best.pt (or best.pt) inside.")
    run = run_dir_of(weights)
    return {"weights": str(weights), "run_dir": str(run), "imgsz": read_run_imgsz(run)}


def run_dir_of(weights) -> Path:
    """runs/<name>/weights/best.pt -> runs/<name>; a .pt elsewhere -> its directory."""
    wdir = Path(weights).resolve().parent
    return wdir.parent if wdir.name == "weights" else wdir


def read_run_imgsz(run_dir) -> Optional[int]:
    """imgsz from the run's args.yaml (training size), None if unavailable."""
    path = Path(run_dir) / "args.yaml"
    if not path.is_file():
        return None
    try:
        import yaml
        v = (yaml.safe_load(path.read_text()) or {}).get("imgsz")
        return int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None
    except Exception:  # noqa: BLE001  (unreadable args.yaml is not fatal)
        return None


MIN_THRESHOLD = 0.05  # floor for detector-score files: best-F1 search can return 0.0 (accept everything)
_THR_KEYS = ("thr", "score_thr", "threshold", "conf", "det_conf_thr")
_CANDIDATE_KEYS = ("candidate_conf", "det_conf", "det_min")


def _norm_name(name) -> str:
    return str(name).strip().lower()


def parse_thresholds(data: dict) -> Dict[str, float]:
    """Lenient per-class threshold parser. Accepts:

    * flat: ``{"armed person": x, "gun": y, "knife": z}`` (non-numeric values such as
      ``"combine"`` are ignored)
    * eval.py: ``{"source_split": "val", "classes": {"knife": {"class_id": 2, "conf": z}, ...}}``
    * pipeline: ``{"combine": "product", "classes": {"gun": {"thr": y}, "knife": {"thr": z}}}``
      (inner key ``thr`` / ``threshold`` / ``conf``, or a bare number)
    * ``{"thresholds": {...}}`` with any of the above inside.
    """
    if not isinstance(data, dict):
        return {}
    inner = data.get("classes", data.get("thresholds"))
    if isinstance(inner, dict):
        data = inner
    out = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = next((v[key] for key in _THR_KEYS if isinstance(v.get(key), (int, float))), None)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[_norm_name(k)] = max(MIN_THRESHOLD, float(v))
    return out


def _read_json(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError("top level is not an object")
        return data
    except (OSError, ValueError, TypeError) as e:
        warnings.warn(f"Ignoring unreadable {path}: {e}")
        return None


def threshold_candidates(weights: Optional[str], filename: str, tta: bool = False) -> List[Path]:
    """Where `filename` is looked for, in order. For pipeline_thresholds.json:
    <run>/pipeline_tta/ (when tta) or <run>/pipeline/ first, then the other one, then the weights dir and the
    run dir. For thresholds.json: the weights dir, then the run dir."""
    if not weights or str(weights).startswith("<"):
        return []
    wdir = Path(weights).resolve().parent
    run = run_dir_of(weights)
    cands = []
    if filename == PIPELINE_THRESHOLDS_FILE:
        cands += [run / PIPELINE_DIRS[bool(tta)] / filename, run / PIPELINE_DIRS[not tta] / filename]
    cands += [wdir / filename, wdir.parent / filename, run / filename]
    out = []
    for c in cands:
        if c not in out:
            out.append(c)
    return out


def _find_file(weights: Optional[str], filename: str, tta: bool = False) -> Optional[Path]:
    return next((c for c in threshold_candidates(weights, filename, tta) if c.is_file()), None)


def canonical_combine(mode) -> str:
    """'product' -> 'prod' etc. Raises ValueError for unknown modes."""
    m = str(mode).strip().lower()
    m = COMBINE_ALIASES.get(m, m)
    if m not in COMBINE_MODES:
        raise ValueError(f"unknown combine mode {mode!r}; expected one of {COMBINE_CHOICES}")
    return m


def is_pipeline_format(data: dict) -> bool:
    """True for the file written by scripts/verifier/eval_pipeline.py."""
    return isinstance(data, dict) and ("selected_mode" in data or "per_mode" in data)


def parse_pipeline_file(data: dict, has_verifier: bool, combine: Optional[str] = None,
                        path: str = "") -> dict:
    """eval_pipeline.py's pipeline_thresholds.json -> engine config. Mirrors its --thresholds loader:
    thr[mode][cls] = per_mode[mode][cls]["thr"] and mode = selected_mode (or the explicit `combine`).
    Without a verifier the detector-only thresholds ("det" mode) are used. Thresholds are NOT clamped
    (they are real scores from the VAL best-F1 search; clamping would break the match with the eval)."""
    per_mode = {}
    for m, cls in (data.get("per_mode") or {}).items():
        try:
            mm = canonical_combine(m)
        except ValueError:
            continue
        per_mode[mm] = {_norm_name(c): float(v["thr"]) for c, v in (cls or {}).items()
                        if isinstance(v, dict) and isinstance(v.get("thr"), (int, float))}
    selected = data.get("selected_mode", "det")
    try:
        selected = canonical_combine(selected)
    except ValueError:
        warnings.warn(f"{path}: unknown selected_mode {selected!r}; using 'det'.")
        selected = "det"
    if selected not in per_mode and isinstance(data.get("classes"), dict):
        per_mode[selected] = {_norm_name(c): float(v.get("score_thr", v.get("thr")))
                              for c, v in data["classes"].items()
                              if isinstance(v, dict) and isinstance(v.get("score_thr", v.get("thr")), (int, float))}
    if "det" not in per_mode and isinstance(data.get("detector_only"), dict):
        per_mode["det"] = {_norm_name(c): float(v["det_conf_thr"]) for c, v in data["detector_only"].items()
                           if isinstance(v, dict) and isinstance(v.get("det_conf_thr"), (int, float))}

    if not has_verifier:
        mode = "det"
    elif combine:
        mode = canonical_combine(combine)
    else:
        mode = selected
    if mode not in per_mode:
        fallback = selected if has_verifier and selected in per_mode else None
        warnings.warn(f"{path}: no thresholds for mode {mode!r} (has {sorted(per_mode)}); "
                      + (f"using the selected mode {fallback!r}." if fallback else "using built-in defaults."))
        mode = fallback
    thr = {c: max(1e-6, t) for c, t in per_mode.get(mode, {}).items()} if mode else {}

    def num(key, cast=float):
        v = data.get(key)
        return cast(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    return {"thresholds": thr, "combine": mode, "candidate_conf": num("det_conf_floor"),
            "imgsz": num("det_imgsz", int), "verifier_imgsz": num("cls_imgsz", int), "pad": num("pad"),
            "tuned_tta": data.get("tta") if isinstance(data.get("tta"), bool) else None,
            "tuned_nms_iou": num("nms_iou"), "verifier": data.get("verifier") if has_verifier else None,
            "pipeline_verifier": data.get("verifier")}


def _empty_cfg() -> dict:
    return {"thresholds": dict(DEFAULT_THRESHOLDS), "combine": None, "candidate_conf": None, "source": None,
            "imgsz": None, "verifier_imgsz": None, "pad": None, "tuned_tta": None, "tuned_nms_iou": None,
            "pipeline_verifier": None, "format": None}


def load_pipeline_config(weights: Optional[str], prefer_pipeline: bool = True, tta: bool = False,
                         combine: Optional[str] = None, thresholds_file: Optional[str] = None) -> dict:
    """Resolve thresholds and score settings for a detector checkpoint.

    Returns ``{"thresholds": {cls: thr}, "combine": str|None, "candidate_conf": float|None, "source": str|None,
    "imgsz", "verifier_imgsz", "pad", "tuned_tta", "tuned_nms_iou", "pipeline_verifier", "format"}``.
    ``thresholds`` always contains the built-in defaults, overridden by the first file found.

    ``prefer_pipeline`` (a verifier is loaded): pipeline_thresholds.json (``<run>/pipeline_tta/`` first when
    ``tta`` else ``<run>/pipeline/``; see threshold_candidates) before thresholds.json; otherwise
    thresholds.json first and a pipeline file's detector-only thresholds as the fallback.
    ``thresholds_file``: use exactly this file (either format).
    """
    cfg = _empty_cfg()
    if thresholds_file:
        if not Path(thresholds_file).is_file():
            raise FileNotFoundError(f"thresholds file not found: {thresholds_file}")
        paths = [Path(thresholds_file)]
    else:
        order = (PIPELINE_THRESHOLDS_FILE, THRESHOLDS_FILE) if prefer_pipeline else (THRESHOLDS_FILE, PIPELINE_THRESHOLDS_FILE)
        paths = [p for f in order for p in [_find_file(weights, f, tta)] if p is not None]
    for path in paths:
        data = _read_json(path)
        if data is None:
            continue
        cfg["source"] = str(path)
        if is_pipeline_format(data):
            pc = parse_pipeline_file(data, prefer_pipeline, combine, str(path))
            cfg.update({k: v for k, v in pc.items() if k not in ("thresholds", "verifier")})
            cfg["thresholds"].update(pc["thresholds"])
            cfg["format"] = "pipeline"
            if prefer_pipeline and cfg["tuned_tta"] is not None and cfg["tuned_tta"] != bool(tta):
                warnings.warn(f"{path} was tuned with tta={cfg['tuned_tta']} but this run uses tta={bool(tta)}; "
                              f"run eval_pipeline.py {'with' if tta else 'without'} --tta for exact thresholds.")
            break
        # legacy / simple formats
        cfg["format"] = "simple"
        cfg["thresholds"].update(parse_thresholds(data))
        comb = data.get("combine")
        if isinstance(comb, str):
            try:
                cfg["combine"] = canonical_combine(comb)
            except ValueError:
                warnings.warn(f"{path}: unknown combine {comb!r}; expected one of {COMBINE_CHOICES}.")
        for key in _CANDIDATE_KEYS:
            if isinstance(data.get(key), (int, float)):
                cfg["candidate_conf"] = max(0.001, float(data[key]))
                break
        if path.name == PIPELINE_THRESHOLDS_FILE and not prefer_pipeline:
            warnings.warn(f"Using {path} without a verifier: these thresholds were calibrated on "
                          "combined detector x verifier scores, so they are likely too strict/loose "
                          "for detector-only scores.")
        if path.name == THRESHOLDS_FILE and prefer_pipeline:
            warnings.warn(f"Verifier loaded but no {PIPELINE_THRESHOLDS_FILE} found; using detector-calibrated "
                          f"{path} on the combined score (stricter than tuned). Run "
                          "scripts/verifier/eval_pipeline.py to get pipeline thresholds.")
        break
    return cfg


def load_thresholds(weights: Optional[str], prefer_pipeline: bool = False) -> Dict[str, float]:
    """Built-in defaults overridden by thresholds.json / pipeline_thresholds.json
    next to the weights (or one directory up). See load_pipeline_config."""
    return load_pipeline_config(weights, prefer_pipeline=prefer_pipeline)["thresholds"]


def combine_scores(det_conf: float, p: Optional[float], mode: str = DEFAULT_COMBINE) -> float:
    """Combine detector confidence with the verifier's p(detected class), exactly like
    eval_pipeline.py's MODES. ``p=None`` (no verifier) returns det_conf unchanged."""
    m = canonical_combine(mode)
    if p is None:
        return float(det_conf)
    d, p = float(det_conf), float(p)
    if m == "det":
        return d
    if m == "cls":
        return p
    if m == "prod":
        return d * p
    if m == "geo":
        return math.sqrt(max(0.0, d * p))
    if m == "wgeo25":
        return d ** 0.75 * p ** 0.25
    return d ** 0.25 * p ** 0.75  # wgeo75


def min_det_conf_for(thr: float, mode: str, has_verifier: bool) -> float:
    """Lowest detector confidence that can still reach a combined score of `thr`
    (verifier p <= 1). Used as the detector's candidate threshold; boxes below it can never pass, so
    raising the candidate floor up to this value does not change the output."""
    m = canonical_combine(mode)
    if not has_verifier or m in ("det", "prod"):
        return thr
    if m == "geo":
        return thr * thr
    if m == "wgeo25":
        return thr ** (4.0 / 3.0)
    if m == "wgeo75":
        return thr ** 4
    return 0.0  # 'cls': det_conf does not enter the final score


def box_area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection_over_a(a: Sequence[float], b: Sequence[float]) -> float:
    """|a ∩ b| / |a| (0 if a has no area)."""
    area = box_area(a)
    if area <= 0:
        return 0.0
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    return (iw * ih) / area


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Standard IoU |a ∩ b| / |a ∪ b|."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def derive_armed_persons(person_boxes: Sequence[Sequence[float]], weapon_boxes: Sequence[Sequence[float]],
                         min_ioa: float = DEFAULT_ARMED_IOA) -> List[List[float]]:
    """Person boxes that contain at least `min_ioa` of some weapon box's area."""
    return [list(map(float, p)) for p in person_boxes
            if any(intersection_over_a(w, p) >= min_ioa for w in weapon_boxes)]


def pad_box(box: Sequence[float], pad: float, w: int, h: int) -> Tuple[int, int, int, int]:
    """Expand a box by `pad` x its width/height on every side, clip to the image,
    and make sure it is at least 1 px."""
    x1, y1, x2, y2 = box
    bw, bh = max(0.0, x2 - x1), max(0.0, y2 - y1)
    x1 = int(max(0, math.floor(x1 - pad * bw)))
    y1 = int(max(0, math.floor(y1 - pad * bh)))
    x2 = int(min(w, math.ceil(x2 + pad * bw)))
    y2 = int(min(h, math.ceil(y2 + pad * bh)))
    x1, y1 = min(x1, max(0, w - 1)), min(y1, max(0, h - 1))
    return x1, y1, max(x2, x1 + 1), max(y2, y1 + 1)


# Verifier crops: copied verbatim from scripts/verifier/common.py (crop_region / square_crop), which
# make_crops.py (training data) and eval_pipeline.py (thresholds) use. test_engine.py checks they stay identical.
def crop_region(box, W: int, H: int, pad: float = 0.2) -> Tuple[int, int, int, int]:
    """Pad the box by `pad` of its size on every side, grow the short side to the long side (square),
    shift it back inside the image where possible, clip. Returns int xyxy."""
    x0, y0, x1, y1 = map(float, box)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(w, h) * (1 + 2 * pad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    sw, sh = min(side, W), min(side, H)
    nx0 = min(max(cx - sw / 2, 0), W - sw)
    ny0 = min(max(cy - sh / 2, 0), H - sh)
    return int(round(nx0)), int(round(ny0)), int(round(nx0 + sw)), int(round(ny0 + sh))


def square_crop(img: np.ndarray, box, pad: float = 0.2, size: int = 224, fill: int = 114) -> np.ndarray:
    """BGR crop around box (see crop_region), letterboxed to size x size (only non-square near borders)."""
    H, W = img.shape[:2]
    x0, y0, x1, y1 = crop_region(box, W, H, pad)
    c = img[y0:max(y1, y0 + 1), x0:max(x1, x0 + 1)]
    s = size / max(c.shape[:2])
    nw, nh = max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))
    c = cv2.resize(c, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if nw == size and nh == size:
        return c
    out = np.full((size, size, 3), fill, np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = c
    return out


def _np(x):
    return x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)


def _names_dict(names) -> Dict[int, str]:
    return dict(names) if isinstance(names, dict) else dict(enumerate(names))


def _nms_numpy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> List[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thr]
    return keep


def nms_per_class(dets: List[dict], iou_thr: float = 0.5) -> List[dict]:
    out = []
    for name in {d["name"] for d in dets}:
        group = [d for d in dets if d["name"] == name]
        boxes = np.array([d["box"] for d in group], dtype=np.float32)
        scores = np.array([d["conf"] for d in group], dtype=np.float32)
        out.extend(group[i] for i in _nms_numpy(boxes, scores, iou_thr))
    return sorted(out, key=lambda d: -d["conf"])


def make_tiles(h: int, w: int, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """2x2 overlapping tiles as (x1, y1, x2, y2)."""
    tw = int(np.ceil(w * (0.5 + overlap / 2)))
    th = int(np.ceil(h * (0.5 + overlap / 2)))
    xs = [0, w - tw]
    ys = [0, h - th]
    return [(x, y, x + tw, y + th) for y in ys for x in xs]


# ----------------------------------------------------------------------------
# Engine
# ----------------------------------------------------------------------------
class SuspiciousBehaviorEngine:
    """
    Parameters
    ----------
    model_dir : training run dir (e.g. work/runs/indian_strict_m960). Picks up
        <dir>/weights/best.pt, thresholds (<dir>/pipeline[_tta]/pipeline_thresholds.json,
        <dir>/thresholds.json), imgsz (thresholds file, else <dir>/args.yaml) and, when
        ``verifier_weights`` is None, the verifier recorded in pipeline_thresholds.json.
    weights : weapon detector .pt (gun / knife / handheld_object, or the old
        armed person / gun / knife). Default: model_dir, $WEAPON_WEIGHTS, else newest
        work/runs/*/weights/best.pt. With an injected ``model`` it is only used to
        locate threshold files.
    verifier_weights : YOLO classification .pt (gun / knife / not_weapon). 'none' disables;
        'auto' (the default when model_dir is given) = the one in pipeline_thresholds.json, if any.
    combine : 'det' | 'cls' | 'prod' | 'geo' | 'wgeo25' | 'wgeo75' (eval_pipeline.py names; the old
        'product' / 'verifier' / 'geomean' are aliases). None = the file's selected_mode, else 'prod'.
        With a pipeline file the thresholds of that mode (per_mode) are used. Ignored without a verifier.
    thresholds_file : use exactly this thresholds file instead of the lookup.
    person_weights : COCO person detector for the derived 'armed person'
        (default yolo11m.pt in raw/). None / '' disables it. Not loaded for old
        3-class models, which have their own 'armed person' class.
    person_conf : person detector confidence (default 0.35).
    armed_ioa : min intersection / weapon-area for a weapon to arm a person (0.3).
    mode : 'image' (default; no temporal smoothing) or 'video' (k-of-n smoothing).
    tta_image / tta_video : test-time augmentation (ultralytics augment=True for the detector, plus
        h-flip averaging in the verifier, as eval_pipeline.py --tta) per mode. Default: on for images,
        off for video. ``augment`` (bool) forces both. Thresholds follow: pipeline_tta/ for TTA calls,
        pipeline/ otherwise. ``verifier_tta`` (bool) decouples the verifier flip from the detector TTA.
    nms_iou : detector NMS IoU (default 0.5).
    pose_weights : optional path to yolo11n-pose.pt; enables RAISED_ARM flag.
    conf_thresh : fallback threshold for any class without a specific threshold.
    class_conf : dict overriding per-class thresholds, e.g. {"knife": 0.15}.
        Thresholds apply to the combined score when a verifier is used.
    candidate_conf : detector confidence floor for candidates sent to the verifier.
        None = det_conf_floor from pipeline_thresholds.json; it is raised to the lowest detector score
        that could still pass (thr for prod, thr^2 for geo, ...), which never changes the output.
    enable_contour_edge / enable_camouflage : preprocessing before detection. Only
        if the weights were trained on images preprocessed the same way.
    imgsz : detector inference size. None = thresholds file det_imgsz, run args.yaml, checkpoint, 960.
    verifier_imgsz / verifier_pad : None = from pipeline_thresholds.json, else 224 / 0.2.
    tile : run full frame + 2x2 overlapping tiles (one predict call each) and merge with NMS.
    smooth_window / smooth_min_hits : k-of-n temporal smoothing (video mode).
    allow_any_model : skip the class-name validation (TESTING ONLY).
    device : passed to ultralytics (None = auto, 'cpu', 0, ...).
    model / verifier_model / person_model : injected models (tests).
    """

    def __init__(self, weights: Optional[str] = None, pose_weights: Optional[str] = None,
                 conf_thresh: float = 0.25, class_conf: Optional[Dict[str, float]] = None,
                 enable_contour_edge: bool = False, enable_camouflage: bool = False,
                 imgsz: Optional[int] = None, tile: bool = False, tile_overlap: float = 0.2,
                 nms_iou: float = DEFAULT_NMS_IOU, smooth_window: int = 5, smooth_min_hits: int = 2,
                 allow_any_model: bool = False, device=None, model=None,
                 verifier_weights: Optional[str] = None, verifier_model=None,
                 combine: Optional[str] = None, verifier_pad: Optional[float] = None,
                 verifier_imgsz: Optional[int] = None, candidate_conf: Optional[float] = None,
                 person_weights: Optional[str] = DEFAULT_PERSON_WEIGHTS, person_model=None,
                 person_conf: float = DEFAULT_PERSON_CONF, person_imgsz: int = 640,
                 armed_ioa: float = DEFAULT_ARMED_IOA, mode: str = "image", augment: Optional[bool] = None,
                 tta_image: bool = True, tta_video: bool = False, verifier_tta: Optional[bool] = None,
                 model_dir: Optional[str] = None, thresholds_file: Optional[str] = None,
                 max_det: int = 300):
        from ultralytics import YOLO  # imported lazily so helpers work without torch

        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.mode = mode
        if augment is not None:
            tta_image = tta_video = bool(augment)
        self.tta = {"image": bool(tta_image), "video": bool(tta_video)}
        self.verifier_tta = verifier_tta

        # ---------------- detector
        self.model_dir = None
        if model_dir:
            md = resolve_model_dir(model_dir)
            self.model_dir = md["run_dir"]
            weights = weights or md["weights"]
            if verifier_weights is None:
                verifier_weights = "auto"
        if model is not None:  # injected model (tests)
            self.weights = str(weights) if weights else (getattr(model, "ckpt_path", None) or "<injected>")
            self.model = model
            thr_anchor = str(weights) if weights else None
        else:
            self.weights = str(weights) if weights else find_default_weights()
            if not os.path.isfile(self.weights):
                raise WeightsError(f"Weights file not found: {self.weights}")
            self.model = YOLO(self.weights)
            thr_anchor = self.weights

        self.names: Dict[int, str] = _names_dict(self.model.names)
        name_set = {_norm_name(n) for n in self.names.values()}
        missing = [c for c in REQUIRED_CLASSES if c not in name_set]
        if missing and not allow_any_model:
            hint = ""
            if len(self.names) == 80:
                hint = (" This looks like a stock COCO checkpoint (e.g. yolo11n.pt); COCO has no "
                        "'gun' and its 'knife' (class 43) is kitchen cutlery.")
            raise WeightsError(
                f"Model {self.weights!r} is not a weapon model: classes {missing} missing from "
                f"model.names ({len(self.names)} classes).{hint} Use weights trained on "
                "gun / knife / handheld_object (or the old armed person / gun / knife)."
            )
        # Old 3-class model: its own 'armed person' boxes replace the COCO person model.
        self.legacy_armed = ARMED in name_set
        self.weapon_classes = set(WEAPON_CLASSES)
        self.detect_classes = self.weapon_classes | ({ARMED} if self.legacy_armed else set())
        # passed to ultralytics as classes=..., like eval_pipeline.py (handheld_object never reaches NMS output)
        self.detect_class_ids = sorted(i for i, n in self.names.items() if _norm_name(n) in self.detect_classes)

        # ---------------- verifier
        default_tta = self.tta[self.mode]
        if isinstance(verifier_weights, str) and verifier_weights.strip().lower() in NONE_VALUES:
            verifier_weights = None
        if isinstance(verifier_weights, str) and verifier_weights.strip().lower() == "auto":
            verifier_weights = None
            if verifier_model is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pc = load_pipeline_config(thr_anchor, True, default_tta, thresholds_file=thresholds_file) \
                        if (thr_anchor or thresholds_file) else _empty_cfg()
                cand = pc.get("pipeline_verifier")
                if cand and os.path.isfile(str(cand)):
                    verifier_weights = str(cand)
                elif cand:
                    warnings.warn(f"Verifier {cand} named in {pc['source']} not found; running without verifier.")
        self.verifier = None
        self.verifier_weights = None
        self.verifier_index: Dict[str, int] = {}
        if verifier_model is not None or verifier_weights:
            if verifier_model is not None:
                self.verifier = verifier_model
                self.verifier_weights = str(verifier_weights) if verifier_weights else "<injected>"
            else:
                if not os.path.isfile(str(verifier_weights)):
                    raise WeightsError(f"Verifier weights not found: {verifier_weights}")
                self.verifier = YOLO(str(verifier_weights))
                self.verifier_weights = str(verifier_weights)
            task = getattr(self.verifier, "task", "classify")
            if task != "classify":
                raise WeightsError(f"Verifier {self.verifier_weights!r} is a {task!r} model; "
                                   "expected a YOLO classification model (gun / knife / not_weapon).")
            vnames = _names_dict(self.verifier.names)
            self.verifier_index = {_norm_name(n): int(i) for i, n in vnames.items()}
            vmissing = [c for c in REQUIRED_CLASSES if c not in self.verifier_index]
            if vmissing:
                raise WeightsError(f"Verifier {self.verifier_weights!r} lacks classes {vmissing} "
                                   f"(has {sorted(self.verifier_index)}).")

        # ---------------- thresholds / combine: one profile per TTA state (pipeline_tta/ vs pipeline/)
        if combine is not None:
            combine = canonical_combine(combine)
        self.conf_thresh = float(conf_thresh)
        self.nms_iou = float(nms_iou)
        self._profiles: Dict[bool, dict] = {}
        for t in sorted({self.tta["image"], self.tta["video"]}, key=lambda t: t != default_tta):
            with warnings.catch_warnings():
                if t != default_tta:   # warn once, for the default profile only
                    warnings.simplefilter("ignore")
                cfg = load_pipeline_config(thr_anchor, prefer_pipeline=self.verifier is not None, tta=t,
                                           combine=combine, thresholds_file=thresholds_file)
            if class_conf:
                cfg["thresholds"].update({_norm_name(k): float(v) for k, v in class_conf.items()})
            cfg["combine"] = combine or cfg["combine"] or DEFAULT_COMBINE
            if candidate_conf is not None:
                cfg["candidate_conf"] = float(candidate_conf)
            cfg["pad"] = float(verifier_pad if verifier_pad is not None else cfg["pad"] or DEFAULT_VERIFIER_PAD)
            cfg["verifier_imgsz"] = int(verifier_imgsz or cfg["verifier_imgsz"] or DEFAULT_VERIFIER_IMGSZ)
            self._profiles[t] = cfg
        self._active = self._profiles[default_tta]
        if (self._active.get("tuned_nms_iou") is not None and self.verifier is not None
                and abs(self._active["tuned_nms_iou"] - self.nms_iou) > 1e-9):
            self.notes = [f"thresholds tuned with nms_iou={self._active['tuned_nms_iou']}, running {self.nms_iou}"]
        else:
            self.notes = []

        # ---------------- image size: explicit > thresholds file > run args.yaml > checkpoint > 960
        ckpt_imgsz = (getattr(self.model, "overrides", None) or {}).get("imgsz")
        run_imgsz = read_run_imgsz(run_dir_of(thr_anchor)) if thr_anchor and not str(thr_anchor).startswith("<") else None
        for cand, src in ((imgsz, "argument"), (self._active.get("imgsz"), "thresholds file"),
                          (run_imgsz, "args.yaml"), (ckpt_imgsz, "checkpoint"), (DEFAULT_IMGSZ, "default")):
            if isinstance(cand, (int, float)) and not isinstance(cand, bool) and cand > 0:
                self.imgsz, self.imgsz_source = int(cand), src
                break
        self.max_det = int(max_det)

        # ---------------- person detector (derived armed person)
        self.person_model = None
        self.person_class = 0
        self.person_conf = float(person_conf)
        self.person_imgsz = int(person_imgsz)
        self.armed_ioa = float(armed_ioa)
        if not self.legacy_armed:
            if person_model is not None:
                self.person_model = person_model
            elif person_weights:
                if os.path.isfile(str(person_weights)):
                    self.person_model = YOLO(str(person_weights))
                elif str(person_weights) == DEFAULT_PERSON_WEIGHTS:
                    warnings.warn(f"Default person model {person_weights} not found; "
                                  "'armed person' derivation disabled.")
                else:
                    raise WeightsError(f"Person weights not found: {person_weights}")
            if self.person_model is not None:
                pnames = {_norm_name(n): int(i) for i, n in _names_dict(self.person_model.names).items()}
                self.person_class = pnames.get("person", 0)
        if self.person_model is None:
            self.person_weights = None
        else:
            self.person_weights = "<injected>" if person_model is not None else str(person_weights)

        # ---------------- preprocessing (OFF by default; see module docstring)
        self.enable_contour_edge = bool(enable_contour_edge)
        self.enable_camouflage = bool(enable_camouflage)
        self.contour_pre = ContourEdgeKnifePreprocessor() if enable_contour_edge else None
        self.camo_pre = FrequencyAwareCamouflageEnhancer() if enable_camouflage else None
        if (enable_contour_edge or enable_camouflage) and model is None:
            warnings.warn("Preprocessing enabled: make sure these weights were trained on images "
                          "preprocessed the same way, otherwise accuracy will drop.")

        self.tile = bool(tile)
        self.tile_overlap = float(tile_overlap)
        self.device = device

        self.smooth_window = max(1, int(smooth_window))
        self.smooth_min_hits = max(1, int(smooth_min_hits))
        self.history: deque = deque(maxlen=self.smooth_window)
        self.pose_history: deque = deque(maxlen=self.smooth_window)
        self.unattended_tracks: List[dict] = []
        self.unattended_tier1_frames: int = 15
        self.unattended_tier2_frames: int = 45

        self.pose_model = None
        if pose_weights:
            try:
                pm = YOLO(str(pose_weights))
                if getattr(pm, "task", None) != "pose":
                    warnings.warn(f"{pose_weights} is not a pose model; pose disabled.")
                else:
                    self.pose_model = pm
            except Exception as e:  # pose is optional, never fatal
                warnings.warn(f"Could not load pose model {pose_weights}: {e}")

    @classmethod
    def from_config(cls, path, **overrides) -> "SuspiciousBehaviorEngine":
        """Engine from a JSON file of constructor kwargs, e.g.
        ``{"model_dir": "../runs/big_s2_m960", "verifier_weights": "../runs/verifier_quick/weights/best.pt",
        "nms_iou": 0.5, "tta_image": true, "device": "cpu"}``. Relative paths are relative to the file.
        ``overrides`` (non-None values) win over the file."""
        path = Path(path)
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object of engine arguments")
        for k in ("model_dir", "weights", "verifier_weights", "person_weights", "pose_weights", "thresholds_file"):
            v = data.get(k)
            if isinstance(v, str) and v and v.lower() not in NONE_VALUES + ("auto",) and not os.path.isabs(v):
                data[k] = str((path.parent / v).resolve())
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data)

    # ------------------------------------------------------------------ active profile (per TTA state)
    @property
    def class_conf(self) -> Dict[str, float]:
        return self._active["thresholds"]

    @property
    def combine(self) -> str:
        return self._active["combine"]

    @property
    def threshold_source(self) -> Optional[str]:
        return self._active["source"]

    @property
    def verifier_pad(self) -> float:
        return self._active["pad"]

    @property
    def verifier_imgsz(self) -> int:
        return self._active["verifier_imgsz"]

    @property
    def augment(self) -> bool:
        """TTA state of the default mode (kept for backwards compatibility)."""
        return self.tta[self.mode]

    def tta_for(self, mode: Optional[str] = None) -> bool:
        return self.tta[mode or self.mode]

    def use_profile(self, tta: bool) -> dict:
        """Select the thresholds / combine profile for a TTA state (process_frame does this per call)."""
        self._active = self._profiles.get(bool(tta), self._active)
        return self._active

    def describe(self) -> str:
        lines = [f"weights={self.weights} imgsz={self.imgsz} ({self.imgsz_source}) nms_iou={self.nms_iou} "
                 f"tta image={self.tta['image']} video={self.tta['video']} tile={self.tile} device={self.device}",
                 f"verifier={self.verifier_weights}"
                 + (f" pad={self.verifier_pad} imgsz={self.verifier_imgsz}" if self.verifier is not None else ""),
                 f"person={'legacy armed-person class' if self.legacy_armed else self.person_weights}"]
        for t, p in sorted(self._profiles.items()):
            thr = {k: round(v, 4) for k, v in p["thresholds"].items() if k in self.detect_classes}
            lines.append(f"[tta={t}] combine={p['combine'] if self.verifier is not None else 'det'} thresholds={thr} "
                         f"candidate_floor={p['candidate_conf']} from {p['source'] or 'defaults'}")
        return "\n".join(lines + self.notes)

    # ------------------------------------------------------------------
    def reset(self):
        self.history.clear()
        self.pose_history.clear()
        self.unattended_tracks.clear()

    def threshold_for(self, name: str) -> float:
        return self.class_conf.get(name, self.conf_thresh)

    def candidate_conf_for(self, name: str) -> float:
        """Detector confidence a box of class `name` needs to be considered at all."""
        if name == ARMED or self.verifier is None:
            return self.threshold_for(name)
        need = min_det_conf_for(self.threshold_for(name), self.combine, True)
        floor = self._active["candidate_conf"]
        if floor is None:
            return max(MIN_THRESHOLD, need)
        return max(float(floor), need)

    def preprocess(self, frame):
        out = frame
        if self.camo_pre is not None:
            out = self.camo_pre.enhance_camouflaged_frame(out)
        if self.contour_pre is not None:
            out = self.contour_pre.enhance(out)
        return out

    def _kw(self, **extra):
        kw = dict(verbose=False, **extra)
        if self.device is not None:
            kw["device"] = self.device
        return kw

    def _predict_kwargs(self, augment: Optional[bool] = None):
        # Run at the lowest per-class candidate threshold, then filter per class.
        augment = self.augment if augment is None else augment
        kw = self._kw(conf=min(self.candidate_conf_for(n) for n in self.detect_classes), imgsz=self.imgsz,
                      iou=self.nms_iou, max_det=self.max_det)
        if self.detect_class_ids:
            kw["classes"] = list(self.detect_class_ids)
        if augment:
            kw["augment"] = True
        return kw

    def _results_to_dets(self, res, dx=0, dy=0) -> List[dict]:
        dets = []
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return dets
        xyxy, confs, clss = _np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls)
        for b, c, k in zip(xyxy, confs, clss):
            name = _norm_name(self.names.get(int(k), int(k)))
            if name not in self.detect_classes:  # handheld_object and anything else: never reported
                continue
            if float(c) < self.candidate_conf_for(name):
                continue
            x1, y1, x2, y2 = (float(v) for v in b)
            dets.append({"name": name, "cls": int(k), "conf": float(c), "det_conf": float(c),
                         "verifier_p": None, "box": [x1 + dx, y1 + dy, x2 + dx, y2 + dy]})
        return dets

    def detect_candidates(self, frame, augment: Optional[bool] = None) -> List[dict]:
        """Stage 1: detector boxes (gun / knife / legacy armed person) after NMS (IoU ``nms_iou``),
        filtered at the candidate threshold. ``conf`` == ``det_conf`` here. One image per predict call
        (a list source is batched at once by ultralytics and can run out of RAM)."""
        inp = self.preprocess(frame)
        kw = self._predict_kwargs(augment)
        if not self.tile:
            # ultralytics already ran NMS at iou=nms_iou; a second (numpy) NMS pass is not exactly idempotent
            # with it and would drop boxes eval_pipeline.py keeps.
            res = self.model(inp, **kw)[0]
            return sorted(self._results_to_dets(res), key=lambda d: -d["conf"])
        h, w = inp.shape[:2]
        tiles = make_tiles(h, w, self.tile_overlap)
        crops = [inp] + [np.ascontiguousarray(inp[y1:y2, x1:x2]) for (x1, y1, x2, y2) in tiles]
        offsets = [(0, 0)] + [(x1, y1) for (x1, y1, _, _) in tiles]
        dets = []
        for crop, (dx, dy) in zip(crops, offsets):
            dets.extend(self._results_to_dets(self.model(crop, **kw)[0], dx, dy))
        return nms_per_class(dets, self.nms_iou)

    def _verifier_probs(self, crops: List[np.ndarray]) -> np.ndarray:
        out = []
        for i in range(0, len(crops), VERIFIER_BATCH):
            res = self.verifier(crops[i:i + VERIFIER_BATCH], **self._kw(imgsz=self.verifier_imgsz))
            out += [_np(r.probs.data).astype(np.float64).ravel() for r in res]
        return np.stack(out)

    def verify(self, frame, dets: List[dict], tta: Optional[bool] = None) -> None:
        """Stage 2: fill ``verifier_p`` for gun / knife dets (in place). Crops are taken from the original
        (not preprocessed) frame exactly like eval_pipeline.py: square_crop(pad, verifier_imgsz). With
        ``tta`` the probabilities are averaged with those of the h-flipped crops (eval_pipeline --tta)."""
        targets = [d for d in dets if d["name"] in self.weapon_classes]
        if self.verifier is None or not targets:
            return
        if tta is None:
            tta = self.verifier_tta if self.verifier_tta is not None else self.augment
        crops = [square_crop(frame, d["box"], self.verifier_pad, self.verifier_imgsz) for d in targets]
        probs = self._verifier_probs(crops)
        if tta:
            probs = (probs + self._verifier_probs([np.ascontiguousarray(c[:, ::-1]) for c in crops])) / 2
        for d, p in zip(targets, probs):
            d["verifier_p"] = float(p[self.verifier_index[d["name"]]])

    def detect(self, frame, mode: Optional[str] = None) -> List[dict]:
        """Both stages: confirmed-by-threshold gun / knife (and legacy armed person)
        dets with ``conf`` = combined score. Uses the TTA setting / thresholds profile of `mode`."""
        tta = self.tta_for(mode)
        self.use_profile(tta)
        dets = self.detect_candidates(frame, augment=tta)
        self.verify(frame, dets, tta=self.verifier_tta if self.verifier_tta is not None else tta)
        out = []
        for d in dets:
            d["conf"] = combine_scores(d["det_conf"], d["verifier_p"], self.combine)
            if d["conf"] >= self.threshold_for(d["name"]):
                out.append(d)
        return sorted(out, key=lambda d: -d["conf"])

    def detect_persons(self, frame) -> List[List[float]]:
        if self.person_model is None:
            return []
        res = self.person_model(frame, **self._kw(conf=self.person_conf, imgsz=self.person_imgsz,
                                                  classes=[self.person_class]))[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        return [[float(v) for v in b] for b, c, k in zip(_np(boxes.xyxy), _np(boxes.conf), _np(boxes.cls))
                if int(k) == self.person_class and float(c) >= self.person_conf]

    def _raised_arm(self, frame) -> bool:
        """True if any person has a wrist clearly above the same-side shoulder."""
        if self.pose_model is None:
            return False
        try:
            res = self.pose_model(frame, **self._kw(conf=0.3))[0]
            kp = getattr(res, "keypoints", None)
            if kp is None or kp.data is None or len(kp.data) == 0:
                return False
            data = kp.data.cpu().numpy()  # (N, 17, 3) x, y, conf
            for person in data:
                for sh, wr in ((5, 9), (6, 10)):  # (shoulder, wrist) left / right
                    if person[sh, 2] > 0.5 and person[wr, 2] > 0.5 and person[wr, 1] < person[sh, 1]:
                        return True
        except Exception as e:  # keep pose fully optional
            warnings.warn(f"Pose inference failed, disabling pose: {e}")
            self.pose_model = None
        return False

    # ------------------------------------------------------------------
    def process_frame(self, frame, mode: Optional[str] = None) -> Tuple[np.ndarray, dict]:
        mode = mode or self.mode
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        video = mode == "video"

        dets = self.detect(frame, mode=mode)
        weapons = [d for d in dets if d["name"] in self.weapon_classes]
        legacy_armed = [d for d in dets if d["name"] == ARMED]
        present = {d["name"] for d in dets}

        if video:
            self.history.append(present)
            confirmed = {n for n in self.detect_classes
                         if sum(n in s for s in self.history) >= self.smooth_min_hits}
        else:
            confirmed = set(present)
        for d in weapons:
            d["confirmed"] = d["name"] in confirmed

        # Armed person and person-association
        persons = []
        if self.legacy_armed:
            armed_persons = [list(d["box"]) for d in legacy_armed]
        else:
            confirmed_boxes = [d["box"] for d in weapons if d["confirmed"]]
            persons = self.detect_persons(frame) if confirmed_boxes else []
            armed_persons = derive_armed_persons(persons, confirmed_boxes, self.armed_ioa)
            if armed_persons:
                present.add(ARMED)
                confirmed.add(ARMED)  # derived from already-confirmed weapons: no second smoothing pass

        # Determine association (HELD vs UNATTENDED) for confirmed weapons
        for d in weapons:
            if not d["confirmed"]:
                d["association"] = "UNCONFIRMED"
                continue
            is_held = any(intersection_over_a(d["box"], p) >= self.armed_ioa for p in persons) if (not self.legacy_armed and persons) else bool(armed_persons)
            d["association"] = "HELD" if is_held else "UNATTENDED"

        # Two-tier unattended weapon tracking
        unattended_tier1 = False
        unattended_tier2 = False
        if video:
            unattended_boxes = [d["box"] for d in weapons if d["confirmed"] and d.get("association") == "UNATTENDED"]
            new_tracks = []
            matched_indices = set()
            for ub in unattended_boxes:
                best_iou, best_idx = 0.0, -1
                for idx, tr in enumerate(self.unattended_tracks):
                    if idx in matched_indices:
                        continue
                    iou = box_iou(ub, tr["box"])
                    if iou > best_iou:
                        best_iou, best_idx = iou, idx
                if best_iou >= 0.3 and best_idx >= 0:
                    matched_indices.add(best_idx)
                    tr = self.unattended_tracks[best_idx]
                    tr["frames"] += 1
                    tr["missing"] = 0
                    tr["box"] = ub
                    new_tracks.append(tr)
                else:
                    new_tracks.append({"box": ub, "frames": 1, "missing": 0})
            for idx, tr in enumerate(self.unattended_tracks):
                if idx not in matched_indices:
                    tr["missing"] += 1
                    if tr["missing"] <= 5:
                        new_tracks.append(tr)
            self.unattended_tracks = new_tracks
            for tr in self.unattended_tracks:
                if tr["frames"] >= self.unattended_tier2_frames:
                    unattended_tier2 = True
                elif tr["frames"] >= self.unattended_tier1_frames:
                    unattended_tier1 = True
        else:
            if any(d["confirmed"] and d.get("association") == "UNATTENDED" for d in weapons):
                unattended_tier1 = True

        raised = self._raised_arm(frame)
        if video:
            self.pose_history.append(raised)
            raised_confirmed = sum(self.pose_history) >= self.smooth_min_hits
        else:
            raised_confirmed = raised

        confirmed_list = sorted(confirmed)
        flags = [FLAG_FOR_CLASS[n] for n in ("knife", "gun", ARMED) if n in confirmed]
        score = 0.0
        if "knife" in confirmed:
            score += SCORE_KNIFE
        if "gun" in confirmed:
            score += SCORE_GUN
        if ARMED in confirmed:
            score += SCORE_ARMED_PERSON
        if unattended_tier2:
            flags.append("UNATTENDED_WEAPON_CONFIRMED")
            score += 25.0
        elif unattended_tier1 and video:
            flags.append("UNATTENDED_WEAPON_DETECTED")
            score += 15.0
        weapon_confirmed = bool(self.weapon_classes & confirmed)
        if raised_confirmed:
            flags.append("RAISED_ARM")
            score += SCORE_RAISED_ARM
            if weapon_confirmed:
                flags.append("RAISED_ARM_WITH_WEAPON")
                score += SCORE_RAISED_ARM_WITH_WEAPON
        # Unconfirmed single-frame hits are reported, but do not score.
        for n in sorted(present - confirmed):
            flags.append(f"{n.upper().replace(' ', '_')}_UNCONFIRMED")
        score = min(100.0, score)
        level = self.level_for(score)

        annotated = self.draw(frame, weapons, score, level, confirmed_list, armed_persons, flags)
        counts: Dict[str, int] = {}
        for d in weapons:
            counts[d["name"]] = counts.get(d["name"], 0) + 1
        if armed_persons:
            counts[ARMED] = len(armed_persons)
        meta = {"threat_score": score, "threat_level": level, "weapons": weapons,
                "armed_persons": armed_persons, "flags": flags, "raw_counts": counts,
                "confirmed": confirmed_list, "mode": mode,
                "tta": self.tta_for(mode), "combine": self.combine if self.verifier is not None else "det"}
        return annotated, meta

    @staticmethod
    def level_for(score: float) -> str:
        if score >= LEVEL_CRITICAL:
            return "CRITICAL THREAT / HARM INTENDED"
        if score >= LEVEL_SUSPICIOUS:
            return "SUSPICIOUS MOVEMENT"
        return "NORMAL"

    @staticmethod
    def draw(frame, dets: Sequence[dict], score: float, level: str,
             confirmed: Iterable[str] = (), armed_persons: Sequence[Sequence[float]] = (),
             flags: Iterable[str] = ()) -> np.ndarray:
        out = frame.copy()
        confirmed = set(confirmed)
        flag_set = set(flags)
        h, w = out.shape[:2]
        th = max(1, int(round(min(h, w) / 400)))
        fs = max(0.4, min(h, w) / 1000)

        banner = f"Threat {score:.0f}/100 | {level}"
        (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, th)

        def box_label(box, color, label, thick):
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            ty = max(tht + 4, y1)
            if ty - tht - 4 < bh + 12 and x1 < bw + 12:   # would hide under the banner: put it below
                ty = bh + 12 + tht + 4
            cv2.rectangle(out, (x1, ty - tht - 4), (x1 + tw + 4, ty), color, -1)
            cv2.putText(out, label, (x1 + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255, 255, 255), th, cv2.LINE_AA)

        for p in armed_persons:
            box_label(p, CLASS_COLORS[ARMED], ARMED, th + 1)
        for d in dets:
            label = f"{d['name']} {d['conf']:.2f}"
            if d.get("verifier_p") is not None:
                label += f" (d{d['det_conf']:.2f} v{d['verifier_p']:.2f})"
            color = CLASS_COLORS.get(d["name"], (0, 255, 0))
            if d.get("association") == "UNATTENDED":
                if "UNATTENDED_WEAPON_CONFIRMED" in flag_set:
                    color = (0, 0, 255)
                    label = f"[ALERT: UNATTENDED] {label}"
                else:
                    color = (0, 191, 255)
                    label = f"[UNATTENDED] {label}"
            box_label(d["box"], color, label,
                      th + (1 if d["name"] in confirmed else 0))
        bcolor = (0, 0, 255) if score >= LEVEL_CRITICAL else (0, 170, 255) if score >= LEVEL_SUSPICIOUS else (0, 160, 0)
        cv2.rectangle(out, (0, 0), (bw + 12, bh + 12), bcolor, -1)
        cv2.putText(out, banner, (6, bh + 6), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1,
                    (255, 255, 255), th, cv2.LINE_AA)
        return out


__all__ = [
    "SuspiciousBehaviorEngine", "ContourEdgeKnifePreprocessor", "FrequencyAwareCamouflageEnhancer",
    "WeightsError", "find_default_weights", "load_thresholds", "load_pipeline_config", "parse_thresholds",
    "parse_pipeline_file", "resolve_model_dir", "square_crop", "crop_region", "canonical_combine",
    "combine_scores", "derive_armed_persons", "intersection_over_a", "DEFAULT_THRESHOLDS",
    "COMBINE_MODES", "COMBINE_CHOICES", "DEFAULT_NMS_IOU", "DEFAULT_PERSON_WEIGHTS", "LEVEL_CRITICAL", "LEVEL_SUSPICIOUS",
]
