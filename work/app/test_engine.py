"""Tiny CPU tests for action_intent_engine. Run:  CUDA_VISIBLE_DEVICES= python test_engine.py  (or pytest)."""
import ast
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import action_intent_engine as aie  # noqa: E402
from action_intent_engine import SuspiciousBehaviorEngine, WeightsError  # noqa: E402

WORK = HERE.parent
RAW = WORK / "raw" / "IntProj_Weapon_Dataset_annot"
COCO = RAW / "yolo11n.pt"
POSE = RAW / "yolo11n-pose.pt"
TEST_IMAGES = sorted((RAW / "test" / "images").glob("*.jpg"))[:2]
DET_RUN = WORK / "runs" / "indian_strict_m960"             # read-only: real detector run
VERIFIER_PT = WORK / "runs" / "verifier_quick" / "weights" / "best.pt"
DOMAIN_LIST = WORK / "data" / "v2" / "test_domain.txt"
sys.path.insert(0, str(WORK / "scripts" / "verifier"))     # eval_pipeline.py / common.py (reference impl.)


# ---------------------------------------------------------------- fake models
class FakeBoxes:
    def __init__(self, dets):
        self.xyxy = np.array([d[2] for d in dets], dtype=np.float32).reshape(-1, 4)
        self.conf = np.array([d[1] for d in dets], dtype=np.float32)
        self.cls = np.array([d[0] for d in dets], dtype=np.float32)

    def __len__(self):
        return len(self.conf)


class FakeModel:
    """Mimics ultralytics YOLO (detect) enough for the engine. `script` is a list of
    per-call detections [(cls, conf, [x1,y1,x2,y2]), ...]. Old 3-class names."""
    names = {0: "armed person", 1: "gun", 2: "knife"}
    task = "detect"

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def __call__(self, inp, **kw):
        self.calls.append(kw)
        dets = self.script.pop(0) if self.script else []
        batch = inp if isinstance(inp, list) else [inp]
        # first image in a batch gets the scripted dets, tiles get nothing
        return [SimpleNamespace(boxes=FakeBoxes(dets if i == 0 else [])) for i in range(len(batch))]


class FakeDetector(FakeModel):
    """New-style weapon detector."""
    names = {0: "gun", 1: "knife", 2: "handheld_object"}


class FakePerson(FakeModel):
    """COCO-like person detector: same box script every call."""
    names = {0: "person", 1: "bicycle"}

    def __init__(self, persons):
        super().__init__([])
        self.persons = list(persons)

    def __call__(self, inp, **kw):
        self.calls.append(kw)
        return [SimpleNamespace(boxes=FakeBoxes(self.persons))]


class FakeVerifier:
    """YOLO classifier stand-in. `probs_for(crop_index, crop)` -> [p_gun, p_knife, p_not_weapon]."""
    names = {0: "gun", 1: "knife", 2: "not_weapon"}
    task = "classify"

    def __init__(self, probs):
        self.probs = list(probs)  # one vector per crop, in order of verify() calls
        self.crops = []
        self.calls = []

    def __call__(self, crops, **kw):
        self.calls.append(kw)
        self.batches = getattr(self, "batches", []) + [list(crops)]
        out = []
        for c in crops:
            self.crops.append(c.shape)
            p = self.probs.pop(0) if self.probs else [0.0, 0.0, 1.0]
            out.append(SimpleNamespace(probs=SimpleNamespace(data=np.array(p, np.float32))))
        return out


FRAME = np.zeros((480, 640, 3), np.uint8)
# legacy 3-class ids: 0 armed person, 1 gun, 2 knife
KNIFE = (2, 0.22, [10, 10, 50, 80])
GUN_LOW = (1, 0.22, [100, 100, 150, 150])
GUN = (1, 0.60, [100, 100, 150, 150])
ARMED_BOX = [80, 40, 300, 400]
LEGACY_ARMED = (0, 0.50, ARMED_BOX)
# new-style ids: 0 gun, 1 knife, 2 handheld_object
N_GUN = (0, 0.60, [100, 100, 150, 150])
N_KNIFE = (1, 0.40, [300, 200, 340, 280])
N_HANDHELD = (2, 0.90, [400, 50, 450, 100])
PERSON_AROUND_GUN = (0, 0.9, [80, 40, 300, 400])     # contains N_GUN fully
PERSON_ELSEWHERE = (0, 0.9, [500, 300, 630, 470])    # contains nothing


def new_engine(script, verifier=None, persons=None, **kw):
    """New-style engine with fakes; never loads the real COCO person model. TTA off unless asked for
    (the verifier h-flip pass would consume extra scripted probabilities)."""
    kw.setdefault("augment", False)
    det = FakeDetector(script)
    pm = FakePerson(persons) if persons is not None else None
    eng = SuspiciousBehaviorEngine(model=det, verifier_model=verifier, person_model=pm,
                                   person_weights=None, **kw)
    return eng, det, pm


# ---------------------------------------------------------------- original tests
def test_refuses_coco():
    try:
        SuspiciousBehaviorEngine(weights=str(COCO), device="cpu")
    except WeightsError as e:
        msg = str(e)
        assert "gun" in msg and "COCO" in msg, msg
        print("  refused COCO with:", msg[:160], "...")
    else:
        raise AssertionError("engine accepted stock COCO yolo11n.pt")


def test_no_default_weights_raises():
    old = os.environ.pop("WEAPON_WEIGHTS", None)
    try:
        aie.find_default_weights()
        print("  (skipped: real trained weights exist)")
        return
    except WeightsError:
        pass
    try:
        SuspiciousBehaviorEngine()
    except WeightsError as e:
        assert "yolo11n" in str(e)
    else:
        raise AssertionError("no error without weights")
    finally:
        if old is not None:
            os.environ["WEAPON_WEIGHTS"] = old


def test_per_class_threshold_and_smoothing():
    # Design change: smoothing is video-mode only now (image mode is the default).
    m = FakeModel([[KNIFE, GUN_LOW], [], [KNIFE], [KNIFE, GUN], [GUN], []])
    eng = SuspiciousBehaviorEngine(model=m, mode="video")
    _, meta = eng.process_frame(FRAME)
    assert m.calls[0]["conf"] == 0.20, m.calls[0]          # run at min threshold
    assert [d["name"] for d in meta["weapons"]] == ["knife"], meta  # gun 0.22 < 0.25 filtered
    assert "KNIFE_DETECTED" not in meta["flags"] and "KNIFE_UNCONFIRMED" in meta["flags"]
    assert meta["threat_score"] == 0 and meta["threat_level"] == "NORMAL"
    _, meta = eng.process_frame(FRAME)                     # nothing
    assert meta["threat_score"] == 0
    _, meta = eng.process_frame(FRAME)                     # 2nd knife in window -> confirmed
    assert "KNIFE_DETECTED" in meta["flags"] and meta["threat_score"] == 45
    assert meta["threat_level"] == "SUSPICIOUS MOVEMENT"
    _, meta = eng.process_frame(FRAME)                     # knife + gun (gun only 1 hit)
    assert meta["threat_score"] == 45 and "GUN_UNCONFIRMED" in meta["flags"]
    _, meta = eng.process_frame(FRAME)                     # gun 2nd hit
    assert set(meta["confirmed"]) == {"knife", "gun"} and meta["threat_score"] == 95
    assert meta["threat_level"].startswith("CRITICAL")
    for k in ("threat_score", "threat_level", "weapons", "flags", "raw_counts", "confirmed", "armed_persons"):
        assert k in meta
    for d in meta["weapons"]:
        assert {"name", "conf", "box", "det_conf", "verifier_p"} <= set(d)
        assert d["verifier_p"] is None and d["conf"] == d["det_conf"]


def test_class_conf_override_and_thresholds_json():
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "weights"
        w.mkdir()
        (w / "thresholds.json").write_text(json.dumps({"armed person": 0.4, "gun": 0.35, "knife": 0.1}))
        th = aie.load_thresholds(str(w / "best.pt"))
        assert th == {"armed person": 0.4, "gun": 0.35, "knife": 0.1}, th
        # nested format from scripts/train/eval.py, with the 0.0 floor applied
        (w.parent / "thresholds.json").write_text(json.dumps({"source_split": "val", "classes": {
            "knife": {"class_id": 2, "conf": 0.0, "f1": 0.2}, "gun": {"class_id": 1, "conf": 0.31}}}))
        (w / "thresholds.json").unlink()
        th = aie.load_thresholds(str(w / "best.pt"))
        assert th == {"armed person": 0.30, "gun": 0.31, "knife": aie.MIN_THRESHOLD}, th
    assert aie.load_thresholds(None) == aie.DEFAULT_THRESHOLDS
    m = FakeModel([[KNIFE]])
    eng = SuspiciousBehaviorEngine(model=m, class_conf={"knife": 0.3})
    _, meta = eng.process_frame(FRAME)
    assert meta["weapons"] == [], meta                     # 0.22 < 0.30


def test_tiling_merges_and_offsets():
    m = FakeModel([[KNIFE]])
    eng = SuspiciousBehaviorEngine(model=m, tile=True)
    _, meta = eng.process_frame(FRAME)
    assert len(meta["weapons"]) == 1
    tiles = aie.make_tiles(480, 640)
    assert len(tiles) == 4 and tiles[-1][2] == 640 and tiles[-1][3] == 480
    # NMS keeps the best of overlapping same-class boxes
    dets = [{"name": "knife", "conf": 0.5, "box": [0, 0, 10, 10]},
            {"name": "knife", "conf": 0.9, "box": [1, 1, 11, 11]},
            {"name": "gun", "conf": 0.4, "box": [0, 0, 10, 10]}]
    kept = aie.nms_per_class(dets, 0.5)
    assert {(d["name"], d["conf"]) for d in kept} == {("knife", 0.9), ("gun", 0.4)}


def test_preprocessors_shape():
    img = (np.random.default_rng(0).random((120, 160, 3)) * 255).astype(np.uint8)
    for pre in (aie.FrequencyAwareCamouflageEnhancer().enhance_camouflaged_frame,
                aie.ContourEdgeKnifePreprocessor().enhance):
        out = pre(img)
        assert out.shape == img.shape and out.dtype == np.uint8


# ---------------------------------------------------------------- score combination
def test_combine_scores_function():
    assert aie.combine_scores(0.6, None, "product") == 0.6     # no verifier: det_conf unchanged
    assert math.isclose(aie.combine_scores(0.6, 0.5, "product"), 0.30)
    assert aie.combine_scores(0.6, 0.5, "verifier") == 0.5
    assert math.isclose(aie.combine_scores(0.6, 0.5, "geomean"), math.sqrt(0.30))
    try:
        aie.combine_scores(0.6, 0.5, "max")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown combine accepted")
    assert aie.min_det_conf_for(0.3, "product", True) == 0.3
    assert math.isclose(aie.min_det_conf_for(0.3, "geomean", True), 0.09)
    assert aie.min_det_conf_for(0.3, "verifier", True) == 0.0
    assert aie.min_det_conf_for(0.3, "verifier", False) == 0.3


def test_verifier_combination_modes():
    # gun det 0.60, p_gun 0.50 | knife det 0.40, p_knife 0.90
    probs = [[0.50, 0.10, 0.40], [0.05, 0.90, 0.05]]
    expected = {"product": {"gun": 0.30, "knife": 0.36},
                "verifier": {"gun": 0.50, "knife": 0.90},
                "geomean": {"gun": math.sqrt(0.30), "knife": 0.60}}
    cand = {"product": 0.20, "geomean": 0.05, "verifier": 0.05}  # min over thr (0.20 knife, ...)
    for mode, exp in expected.items():
        v = FakeVerifier(probs)
        eng, det, _ = new_engine([[N_GUN, N_KNIFE]], verifier=v, combine=mode,
                                 class_conf={"gun": 0.32, "knife": 0.20})
        _, meta = eng.process_frame(FRAME)
        assert math.isclose(det.calls[0]["conf"], cand[mode]), (mode, det.calls[0])
        got = {d["name"]: d for d in meta["weapons"]}
        assert got["knife"]["det_conf"] == np.float32(0.40) and math.isclose(got["knife"]["verifier_p"], 0.90, rel_tol=1e-6)
        assert math.isclose(got["knife"]["conf"], exp["knife"], rel_tol=1e-5), (mode, got)
        if exp["gun"] >= 0.32:
            assert math.isclose(got["gun"]["conf"], exp["gun"], rel_tol=1e-5), (mode, got)
        else:
            assert "gun" not in got, (mode, got)          # product: 0.30 < 0.32 -> dropped
        assert eng.combine == aie.canonical_combine(mode)
        assert len(v.crops) == 2 and v.calls[0]["imgsz"] == 224


def test_verifier_rejects_and_crop_padding():
    v = FakeVerifier([[0.02, 0.03, 0.95]])                 # verifier says not_weapon
    eng, _, _ = new_engine([[N_GUN]], verifier=v)
    _, meta = eng.process_frame(FRAME)
    assert meta["weapons"] == [] and meta["threat_score"] == 0 and meta["flags"] == []
    assert v.crops == [(224, 224, 3)], v.crops             # square 224 crop, like eval_pipeline.py
    assert aie.crop_region([100, 100, 150, 150], 640, 480, 0.2) == (90, 90, 160, 160)  # 50x50 box + 20% each side
    assert aie.pad_box([100, 100, 150, 150], 0.2, 640, 480) == (90, 90, 160, 160)
    assert aie.pad_box([0, 0, 10, 10], 0.2, 640, 480) == (0, 0, 12, 12)          # clipped
    assert aie.pad_box([635, 475, 640, 480], 0.2, 640, 480) == (634, 474, 640, 480)
    x1, y1, x2, y2 = aie.pad_box([700, 500, 700, 500], 0.2, 640, 480)            # degenerate
    assert x2 > x1 and y2 > y1


def test_verifier_validation():
    class NoKnifeVerifier(FakeVerifier):
        names = {0: "gun", 1: "not_weapon"}

    class DetectVerifier(FakeVerifier):
        task = "detect"

    for bad in (NoKnifeVerifier([]), DetectVerifier([])):
        try:
            new_engine([], verifier=bad)
        except WeightsError:
            pass
        else:
            raise AssertionError(f"accepted bad verifier {type(bad).__name__}")
    try:
        SuspiciousBehaviorEngine(model=FakeDetector([]), person_weights=None, verifier_weights="/nope/cls.pt")
    except WeightsError as e:
        assert "Verifier" in str(e)
    else:
        raise AssertionError("missing verifier file accepted")


# ---------------------------------------------------------------- threshold files
def test_threshold_file_formats():
    with tempfile.TemporaryDirectory() as td:
        w = Path(td) / "run" / "weights"
        w.mkdir(parents=True)
        best = str(w / "best.pt")
        # pipeline format
        (w / "pipeline_thresholds.json").write_text(json.dumps(
            {"combine": "geomean", "candidate_conf": 0.07,
             "classes": {"gun": {"thr": 0.41, "f1": 0.8}, "knife": {"thr": 0.33}}}))
        cfg = aie.load_pipeline_config(best, prefer_pipeline=True)
        assert cfg["thresholds"] == {"armed person": 0.30, "gun": 0.41, "knife": 0.33}, cfg
        assert cfg["combine"] == "geo" and cfg["candidate_conf"] == 0.07
        assert cfg["source"].endswith("pipeline_thresholds.json")
        # flat lenient format, with a 'combine' string that must not break parsing
        (w / "pipeline_thresholds.json").write_text(json.dumps({"gun": 0.45, "knife": 0.01, "combine": "verifier"}))
        cfg = aie.load_pipeline_config(best, prefer_pipeline=True)
        assert cfg["thresholds"]["gun"] == 0.45 and cfg["thresholds"]["knife"] == aie.MIN_THRESHOLD
        assert cfg["combine"] == "cls" and cfg["candidate_conf"] is None
        # other lenient variants
        assert aie.parse_thresholds({"classes": {"gun": 0.5, "Knife": {"threshold": 0.2}}}) == {"gun": 0.5, "knife": 0.2}
        assert aie.parse_thresholds({"thresholds": {"gun": {"conf": 0.6}}}) == {"gun": 0.6}
        assert aie.parse_thresholds({"gun": True, "knife": "x"}) == {}
        # priority: with a verifier pipeline wins, without one thresholds.json wins (with a warning
        # when only the pipeline file exists)
        (w / "thresholds.json").write_text(json.dumps({"gun": 0.27, "knife": 0.18}))
        assert aie.load_pipeline_config(best, prefer_pipeline=True)["thresholds"]["gun"] == 0.45
        cfg = aie.load_pipeline_config(best, prefer_pipeline=False)
        assert cfg["thresholds"]["gun"] == 0.27 and cfg["combine"] is None, cfg
        (w / "thresholds.json").unlink()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = aie.load_pipeline_config(best, prefer_pipeline=False)
        assert cfg["thresholds"]["gun"] == 0.45 and any("without a verifier" in str(r.message) for r in rec)
        # pipeline file in the run dir (one level up) is found too
        (w / "pipeline_thresholds.json").rename(w.parent / "pipeline_thresholds.json")
        assert aie.load_pipeline_config(best, True)["thresholds"]["gun"] == 0.45
        # unreadable file -> warning, defaults
        (w.parent / "pipeline_thresholds.json").write_text("{not json")
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = aie.load_pipeline_config(best, True)
        assert cfg["thresholds"] == aie.DEFAULT_THRESHOLDS and cfg["source"] is None and rec
        # unknown combine -> warning, None
        (w.parent / "pipeline_thresholds.json").write_text(json.dumps({"combine": "max", "gun": 0.5}))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = aie.load_pipeline_config(best, True)
        assert cfg["combine"] is None and cfg["thresholds"]["gun"] == 0.5 and rec

        # engine: thresholds + combine + candidate_conf from the file next to the weights
        (w / "pipeline_thresholds.json").write_text(json.dumps(
            {"combine": "verifier", "candidate_conf": 0.12, "classes": {"gun": {"thr": 0.55}, "knife": {"thr": 0.5}}}))
        v = FakeVerifier([[0.60, 0.1, 0.3]])
        det = FakeDetector([[N_GUN]])
        eng = SuspiciousBehaviorEngine(model=det, weights=best, verifier_model=v, person_weights=None,
                                       augment=False)
        assert eng.combine == "cls" and eng.threshold_for("gun") == 0.55
        assert eng.threshold_source.endswith(str(Path("weights") / "pipeline_thresholds.json"))
        _, meta = eng.process_frame(FRAME)
        assert det.calls[0]["conf"] == 0.12
        assert len(meta["weapons"]) == 1 and math.isclose(meta["weapons"][0]["conf"], 0.60, rel_tol=1e-6)
        # explicit args beat the file
        eng = SuspiciousBehaviorEngine(model=FakeDetector([]), weights=best, verifier_model=FakeVerifier([]),
                                       person_weights=None, combine="product", class_conf={"gun": 0.2})
        assert eng.combine == "prod" and eng.threshold_for("gun") == 0.2
        # without a verifier the pipeline combine is irrelevant: score == det_conf
        eng = SuspiciousBehaviorEngine(model=FakeDetector([[N_GUN]]), weights=best, person_weights=None,
                                       class_conf={"gun": 0.5})
        _, meta = eng.process_frame(FRAME)
        assert meta["weapons"][0]["conf"] == meta["weapons"][0]["det_conf"]


# ---------------------------------------------------------------- handheld_object / validation
def test_handheld_object_suppressed():
    v = FakeVerifier([[0.1, 0.9, 0.0]])
    eng, _, pm = new_engine([[N_HANDHELD, N_KNIFE]], verifier=v, persons=[])
    _, meta = eng.process_frame(FRAME)
    assert [d["name"] for d in meta["weapons"]] == ["knife"], meta
    assert "handheld_object" not in meta["raw_counts"] and "handheld_object" not in meta["confirmed"]
    assert not any("HANDHELD" in f for f in meta["flags"]), meta["flags"]
    assert len(v.crops) == 1                               # verifier never sees handheld boxes
    # handheld alone -> nothing at all
    eng, _, pm = new_engine([[N_HANDHELD]], persons=[PERSON_AROUND_GUN])
    _, meta = eng.process_frame(FRAME)
    assert meta["weapons"] == [] and meta["flags"] == [] and meta["threat_score"] == 0
    assert meta["armed_persons"] == [] and pm.calls == []  # no weapon -> person model not even run


def test_detector_validation():
    eng, _, _ = new_engine([])
    assert not eng.legacy_armed and eng.weapon_classes == {"gun", "knife"}
    assert eng.detect_classes == {"gun", "knife"}

    class GunOnly(FakeModel):
        names = {0: "gun", 1: "handheld_object"}
    try:
        SuspiciousBehaviorEngine(model=GunOnly([]), person_weights=None)
    except WeightsError as e:
        assert "knife" in str(e)
    else:
        raise AssertionError("accepted model without knife")
    # explicitly given person weights that do not exist -> error
    try:
        SuspiciousBehaviorEngine(model=FakeDetector([]), person_weights="/nope/person.pt")
    except WeightsError:
        pass
    else:
        raise AssertionError("missing person weights accepted")


# ---------------------------------------------------------------- derived armed person
def test_ioa_helpers():
    person = [0, 0, 100, 100]
    assert aie.intersection_over_a([10, 10, 20, 20], person) == 1.0       # fully inside
    assert aie.intersection_over_a([90, 0, 130, 10], person) == 0.25      # 25% inside
    assert math.isclose(aie.intersection_over_a([70, 0, 170, 10], person), 0.30)
    assert aie.intersection_over_a([200, 200, 210, 210], person) == 0.0   # outside
    assert aie.intersection_over_a([5, 5, 5, 20], person) == 0.0          # zero-area weapon
    assert aie.derive_armed_persons([], [[10, 10, 20, 20]]) == []         # no person
    assert aie.derive_armed_persons([person], []) == []                   # no weapon
    assert aie.derive_armed_persons([person], [[90, 0, 130, 10]]) == []   # 0.25 < 0.3
    assert aie.derive_armed_persons([person], [[70, 0, 170, 10]]) == [person]  # 0.30 >= 0.3
    two = aie.derive_armed_persons([person, [300, 300, 400, 400]], [[10, 10, 20, 20], [50, 50, 60, 60]])
    assert two == [person]                                                # listed once


def test_armed_person_derived():
    # weapon inside a person -> armed person, +20
    eng, _, pm = new_engine([[N_GUN]], persons=[PERSON_AROUND_GUN, PERSON_ELSEWHERE])
    assert eng.person_class == 0
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [PERSON_AROUND_GUN[2]], meta
    assert "ARMED_PERSON" in meta["flags"] and "armed person" in meta["confirmed"]
    assert meta["threat_score"] == 50 + 20 and meta["raw_counts"]["armed person"] == 1
    assert pm.calls[0]["conf"] == aie.DEFAULT_PERSON_CONF and pm.calls[0]["classes"] == [0]
    # weapon outside every person box -> no armed person, weapon still reported
    eng, _, pm = new_engine([[N_KNIFE]], persons=[PERSON_ELSEWHERE, PERSON_AROUND_GUN])
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [] and "ARMED_PERSON" not in meta["flags"]
    assert "KNIFE_DETECTED" in meta["flags"] and meta["threat_score"] == 45 and len(pm.calls) == 1
    # no person at all
    eng, _, pm = new_engine([[N_GUN]], persons=[])
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [] and meta["threat_score"] == 50 and "GUN_DETECTED" in meta["flags"]
    # low-confidence person boxes are ignored
    eng, _, _ = new_engine([[N_GUN]], persons=[(0, 0.2, PERSON_AROUND_GUN[2])])
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == []
    # person model disabled -> still works, no armed person
    eng, _, _ = new_engine([[N_GUN]])
    assert eng.person_model is None
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [] and meta["threat_score"] == 50
    # weapon rejected by verifier does not arm anyone
    eng, _, pm = new_engine([[N_GUN]], verifier=FakeVerifier([[0.0, 0.0, 1.0]]), persons=[PERSON_AROUND_GUN])
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [] and pm.calls == []


def test_legacy_three_class_model():
    pm = FakePerson([PERSON_ELSEWHERE])
    m = FakeModel([[LEGACY_ARMED, GUN]])
    eng = SuspiciousBehaviorEngine(model=m, person_model=pm)  # default person_weights, but legacy -> unused
    assert eng.legacy_armed and eng.person_model is None and eng.person_weights is None
    assert eng.detect_classes == {"gun", "knife", "armed person"}
    _, meta = eng.process_frame(FRAME)
    assert [d["name"] for d in meta["weapons"]] == ["gun"]           # armed person not in weapons
    assert meta["armed_persons"] == [ARMED_BOX] and pm.calls == []
    assert set(meta["flags"]) >= {"GUN_DETECTED", "ARMED_PERSON"} and meta["threat_score"] == 70
    # legacy armed-person threshold (0.30) applies
    eng = SuspiciousBehaviorEngine(model=FakeModel([[(0, 0.25, ARMED_BOX)]]))
    _, meta = eng.process_frame(FRAME)
    assert meta["armed_persons"] == [] and meta["threat_score"] == 0
    # legacy armed person goes through temporal smoothing in video mode
    eng = SuspiciousBehaviorEngine(model=FakeModel([[LEGACY_ARMED], [LEGACY_ARMED]]), mode="video")
    _, meta = eng.process_frame(FRAME)
    assert "ARMED_PERSON_UNCONFIRMED" in meta["flags"] and meta["threat_score"] == 0
    _, meta = eng.process_frame(FRAME)
    assert "ARMED_PERSON" in meta["flags"] and meta["threat_score"] == 20
    # legacy model + verifier: verifier sees gun/knife crops only
    v = FakeVerifier([[0.8, 0.1, 0.1]])
    eng = SuspiciousBehaviorEngine(model=FakeModel([[LEGACY_ARMED, GUN]]), verifier_model=v, augment=False)
    _, meta = eng.process_frame(FRAME)
    assert len(v.crops) == 1 and math.isclose(meta["weapons"][0]["conf"], 0.48, rel_tol=1e-5)
    assert meta["armed_persons"] == [ARMED_BOX]


# ---------------------------------------------------------------- modes / TTA
def test_image_vs_video_mode():
    eng, _, _ = new_engine([[N_KNIFE], [N_KNIFE]])
    assert eng.mode == "image"
    _, meta = eng.process_frame(FRAME)                      # image: confirmed immediately
    assert meta["mode"] == "image" and "KNIFE_DETECTED" in meta["flags"] and meta["threat_score"] == 45
    assert meta["weapons"][0]["confirmed"] and len(eng.history) == 0
    _, meta = eng.process_frame(FRAME, mode="video")        # per-call override: 1 of 2 hits
    assert "KNIFE_UNCONFIRMED" in meta["flags"] and meta["threat_score"] == 0
    assert not meta["weapons"][0]["confirmed"]

    eng, _, pm = new_engine([[N_GUN], [N_GUN], []], persons=[PERSON_AROUND_GUN], mode="video")
    _, meta = eng.process_frame(FRAME)                      # unconfirmed weapon -> no armed person yet
    assert meta["threat_score"] == 0 and meta["armed_persons"] == [] and pm.calls == []
    _, meta = eng.process_frame(FRAME)                      # confirmed -> armed person
    assert meta["threat_score"] == 70 and meta["armed_persons"] == [PERSON_AROUND_GUN[2]]
    _, meta = eng.process_frame(FRAME)                      # no box this frame: gun still confirmed
    assert "GUN_DETECTED" in meta["flags"] and meta["armed_persons"] == []
    for bad in ("stream", "VIDEO"):
        try:
            eng.process_frame(FRAME, mode=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"mode {bad!r} accepted")
    try:
        new_engine([], mode="live")
    except ValueError:
        pass
    else:
        raise AssertionError("bad constructor mode accepted")


def test_tta_flag():
    eng, det, _ = new_engine([[N_GUN]], augment=True)
    eng.process_frame(FRAME)
    assert det.calls[0].get("augment") is True
    eng, det, _ = new_engine([[N_GUN]])                     # augment=False forces off
    eng.process_frame(FRAME)
    assert "augment" not in det.calls[0]
    # default recipe: TTA on for still images, off for video; per call, following the mode
    eng, det, _ = new_engine([[N_GUN], [N_GUN], [N_GUN]], augment=None)
    assert eng.tta == {"image": True, "video": False}
    _, meta = eng.process_frame(FRAME)
    assert det.calls[0].get("augment") is True and meta["tta"] is True
    _, meta = eng.process_frame(FRAME, mode="video")
    assert "augment" not in det.calls[1] and meta["tta"] is False
    eng, det, _ = new_engine([[N_GUN]], augment=None, mode="video")
    eng.process_frame(FRAME)
    assert "augment" not in det.calls[0]
    eng, det, _ = new_engine([[N_GUN]], augment=None, tta_image=False, tta_video=True, mode="video")
    eng.process_frame(FRAME)
    assert det.calls[0].get("augment") is True


def test_verifier_tta_flip_average():
    # TTA: verifier probs averaged with the h-flipped crops (eval_pipeline.py --tta / common.Verifier)
    frame = (np.random.default_rng(1).random((480, 640, 3)) * 255).astype(np.uint8)
    v = FakeVerifier([[0.8, 0.1, 0.1], [0.4, 0.3, 0.3]])     # call 1: original crop, call 2: flipped crop
    eng, det, _ = new_engine([[N_GUN]], verifier=v, augment=True, class_conf={"gun": 0.05})
    _, meta = eng.process_frame(frame)
    assert len(v.calls) == 2, v.calls
    a, b = v.batches[0][0], v.batches[1][0]
    assert np.array_equal(a[:, ::-1], b)                    # second pass = horizontal flip
    assert math.isclose(meta["weapons"][0]["verifier_p"], 0.6, rel_tol=1e-6)
    # no TTA: single pass
    v = FakeVerifier([[0.8, 0.1, 0.1]])
    eng, _, _ = new_engine([[N_GUN]], verifier=v, class_conf={"gun": 0.05})
    _, meta = eng.process_frame(frame)
    assert len(v.calls) == 1 and math.isclose(meta["weapons"][0]["verifier_p"], 0.8, rel_tol=1e-6)
    # verifier_tta=False decouples it from the detector TTA
    v = FakeVerifier([[0.8, 0.1, 0.1]])
    eng, det, _ = new_engine([[N_GUN]], verifier=v, augment=True, verifier_tta=False, class_conf={"gun": 0.05})
    eng.process_frame(frame)
    assert len(v.calls) == 1 and det.calls[0].get("augment") is True


# ---------------------------------------------------------------- final inference recipe
def _pipeline_json(selected, thr, det_thr, tta, verifier=None, floor=0.01, pad=0.2, cls_imgsz=224,
                   det_imgsz=960, nms_iou=0.7):
    """A pipeline_thresholds.json exactly as scripts/verifier/eval_pipeline.py writes it.
    thr: {mode: {"gun": t, "knife": t}} (must include `selected`); det_thr: detector-only thresholds."""
    per_mode = {m: {c: {"thr": t, "val_f1": 0.5, "val_ap50": 0.5} for c, t in v.items()} for m, v in thr.items()}
    per_mode["det"] = {c: {"thr": t, "val_f1": 0.4, "val_ap50": 0.4} for c, t in det_thr.items()}
    return {"selected_mode": selected, "formula": {"det": "det_conf", "prod": "det_conf*p_class"},
            "classes": {c: {"score_thr": t, "val_f1": 0.5} for c, t in thr[selected].items()},
            "detector_only": {c: {"det_conf_thr": t, "val_f1": 0.4} for c, t in det_thr.items()},
            "per_mode": per_mode, "det_weights": "/x/best.pt", "verifier": verifier, "val_data": "/x/data.yaml",
            "val_split": "val", "det_conf_floor": floor, "det_imgsz": det_imgsz, "cls_imgsz": cls_imgsz,
            "pad": pad, "tta": tta, "nms_iou": nms_iou, "ignored_classes": ["handheld_object"]}


def _make_run(td, args_imgsz=800):
    run = Path(td) / "myrun"
    (run / "weights").mkdir(parents=True)
    (run / "weights" / "best.pt").write_bytes(b"")          # never loaded: models are injected
    (run / "args.yaml").write_text(f"model: yolo11m.pt\nimgsz: {args_imgsz}\nepochs: 1\n")
    return run


def test_square_crop_matches_eval_pipeline():
    import common  # scripts/verifier/common.py: the crop used for verifier training and threshold tuning
    rng = np.random.default_rng(0)
    img = (rng.random((300, 500, 3)) * 255).astype(np.uint8)
    boxes = [[100, 100, 150, 150], [0, 0, 10, 40], [480, 250, 500, 300], [10, 10, 490, 290], [200, 5, 205, 295],
             [250.4, 120.7, 251.1, 121.2]]
    for _ in range(30):
        x = np.sort(rng.uniform(0, 500, 2))
        y = np.sort(rng.uniform(0, 300, 2))
        boxes.append([x[0], y[0], x[1], y[1]])
    for b in boxes:
        for pad, size in ((0.2, 224), (0.1, 160)):
            assert aie.crop_region(b, 500, 300, pad) == common.crop_region(b, 500, 300, pad), b
            assert np.array_equal(aie.square_crop(img, b, pad, size), common.square_crop(img, b, pad, size)), b


def test_combine_matches_eval_pipeline():
    from eval_pipeline import MODES as EVAL_MODES
    assert set(EVAL_MODES) == set(aie.COMBINE_MODES)
    rng = np.random.default_rng(0)
    d, p = rng.uniform(0.01, 1, 200), rng.uniform(0, 1, 200)
    for m, f in EVAL_MODES.items():
        ref = f(d, p)
        got = np.array([aie.combine_scores(a, b, m) for a, b in zip(d, p)])
        assert np.allclose(ref, got, rtol=0, atol=1e-15), m
        for thr in (0.05, 0.3, 0.7):   # candidate floor: the lowest det_conf that can still reach thr with p = 1
            dmin = aie.min_det_conf_for(thr, m, True)
            if m != "cls":
                assert math.isclose(aie.combine_scores(dmin, 1.0, m), thr, rel_tol=1e-9), (m, thr)
    for alias, canon in aie.COMBINE_ALIASES.items():
        assert aie.canonical_combine(alias) == canon and aie.canonical_combine(alias.upper()) == canon


def test_pipeline_thresholds_file_eval_format():
    with tempfile.TemporaryDirectory() as td:
        run = _make_run(td)
        best = str(run / "weights" / "best.pt")
        (run / "pipeline").mkdir()
        (run / "pipeline_tta").mkdir()
        (run / "pipeline" / "pipeline_thresholds.json").write_text(json.dumps(_pipeline_json(
            "wgeo25", {"wgeo25": {"gun": 0.031, "knife": 0.27}, "prod": {"gun": 0.2, "knife": 0.15}},
            {"gun": 0.16, "knife": 0.22}, tta=False, pad=0.25, cls_imgsz=192, det_imgsz=1024)))
        (run / "pipeline_tta" / "pipeline_thresholds.json").write_text(json.dumps(_pipeline_json(
            "prod", {"prod": {"gun": 0.45, "knife": 0.35}}, {"gun": 0.5, "knife": 0.4}, tta=True)))
        # helper level: no clamping of pipeline thresholds (0.031 < MIN_THRESHOLD stays 0.031)
        cfg = aie.load_pipeline_config(best, prefer_pipeline=True, tta=False)
        assert cfg["format"] == "pipeline" and cfg["combine"] == "wgeo25"
        assert cfg["thresholds"]["gun"] == 0.031 and cfg["thresholds"]["knife"] == 0.27
        assert cfg["candidate_conf"] == 0.01 and cfg["pad"] == 0.25 and cfg["verifier_imgsz"] == 192
        assert cfg["imgsz"] == 1024 and cfg["source"].endswith("pipeline/pipeline_thresholds.json")
        cfg = aie.load_pipeline_config(best, prefer_pipeline=True, tta=True)
        assert cfg["combine"] == "prod" and cfg["thresholds"]["gun"] == 0.45
        assert cfg["source"].endswith("pipeline_tta/pipeline_thresholds.json")
        # explicit combine -> that mode's per_mode thresholds
        cfg = aie.load_pipeline_config(best, prefer_pipeline=True, tta=False, combine="product")
        assert cfg["combine"] == "prod" and cfg["thresholds"] == {**aie.DEFAULT_THRESHOLDS, "gun": 0.2, "knife": 0.15}
        # mode not in the file -> warning, selected mode kept
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = aie.load_pipeline_config(best, prefer_pipeline=True, tta=False, combine="geo")
        assert cfg["combine"] == "wgeo25" and any("no thresholds for mode" in str(r.message) for r in rec)
        # no verifier -> the file's detector-only thresholds (no thresholds.json present)
        cfg = aie.load_pipeline_config(best, prefer_pipeline=False, tta=False)
        assert cfg["combine"] == "det" and cfg["thresholds"]["gun"] == 0.16 and cfg["thresholds"]["knife"] == 0.22
        # ... but thresholds.json wins without a verifier
        (run / "thresholds.json").write_text(json.dumps({"source_split": "val", "classes": {
            "gun": {"class_id": 0, "conf": 0.1602}, "knife": {"class_id": 1, "conf": 0.2232}}}))
        cfg = aie.load_pipeline_config(best, prefer_pipeline=False)
        assert cfg["thresholds"]["gun"] == 0.1602 and cfg["source"].endswith("myrun/thresholds.json")
        # only the other-TTA pipeline file exists -> used, with a warning
        (run / "pipeline_tta" / "pipeline_thresholds.json").unlink()
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            cfg = aie.load_pipeline_config(best, prefer_pipeline=True, tta=True)
        assert cfg["combine"] == "wgeo25" and any("tuned with tta=False" in str(r.message) for r in rec)

        # engine: image mode (TTA) vs video mode (no TTA) profiles, decisions exactly as eval_pipeline
        (run / "pipeline_tta" / "pipeline_thresholds.json").write_text(json.dumps(_pipeline_json(
            "prod", {"prod": {"gun": 0.45, "knife": 0.35}}, {"gun": 0.5, "knife": 0.4}, tta=True)))
        gun_hi = (0, 0.80, [100, 100, 150, 150])
        knife_lo = (1, 0.30, [300, 200, 340, 280])
        # image call (TTA): only the gun is a candidate (knife 0.30 < 0.35), verified twice (orig + flip);
        # video call (no TTA, floor 0.01): gun + knife, one pass
        v = FakeVerifier([[0.7, 0.2, 0.1], [0.5, 0.2, 0.3],
                          [0.6, 0.2, 0.2], [0.05, 0.9, 0.05]])
        det = FakeDetector([[gun_hi, knife_lo], [gun_hi, knife_lo]])
        eng = SuspiciousBehaviorEngine(model=det, weights=best, verifier_model=v, person_weights=None)
        # imgsz from the default (image, TTA) profile file: det_imgsz 960 there, 1024 in pipeline/
        assert eng.imgsz == 960 and eng.imgsz_source == "thresholds file" and eng.nms_iou == 0.5
        _, meta = eng.process_frame(FRAME)                   # TTA profile: prod, gun 0.45, knife 0.35
        assert meta["tta"] and meta["combine"] == "prod"
        assert det.calls[0]["augment"] is True and det.calls[0]["iou"] == 0.5
        assert math.isclose(det.calls[0]["conf"], 0.35)     # max(floor 0.01, min thr) for prod
        assert v.calls[0]["imgsz"] == 224 and v.crops[0] == (224, 224, 3)
        got = {d["name"]: d for d in meta["weapons"]}
        # gun: p = (0.7 + 0.5) / 2 = 0.6 -> 0.48 >= 0.45 kept; knife det 0.30 < 0.35 can never pass
        assert set(got) == {"gun"} and math.isclose(got["gun"]["conf"], 0.8 * 0.6, rel_tol=1e-6)
        _, meta = eng.process_frame(FRAME, mode="video")     # no TTA profile: wgeo25, gun 0.031, knife 0.27
        assert not meta["tta"] and meta["combine"] == "wgeo25" and "augment" not in det.calls[1]
        assert math.isclose(det.calls[1]["conf"], 0.01)     # wgeo25 floor (0.031^(4/3) < 0.01)
        assert v.calls[-1]["imgsz"] == 192 and v.crops[-1] == (192, 192, 3)  # cls_imgsz / pad from the file
        exp_gun = 0.8 ** 0.75 * 0.6 ** 0.25
        exp_knife = 0.3 ** 0.75 * 0.9 ** 0.25
        got = {d["name"]: d["conf"] for d in meta["weapons"]}
        assert math.isclose(got["gun"], exp_gun, rel_tol=1e-6) and math.isclose(got["knife"], exp_knife, rel_tol=1e-6)
        assert exp_knife >= 0.27
        assert "imgsz" in eng.describe() and "pipeline_tta" in eng.describe()
        # without verifier: thresholds.json (detector scores), score == det_conf
        eng = SuspiciousBehaviorEngine(model=FakeDetector([[gun_hi]]), weights=best, person_weights=None)
        assert eng.threshold_for("gun") == 0.1602 and eng.verifier is None
        # explicit thresholds file
        eng = SuspiciousBehaviorEngine(model=FakeDetector([]), weights=best, person_weights=None,
                                       verifier_model=FakeVerifier([]),
                                       thresholds_file=str(run / "pipeline" / "pipeline_thresholds.json"))
        assert eng.combine == "wgeo25" and eng.threshold_for("gun") == 0.031


def test_model_dir_and_config():
    with tempfile.TemporaryDirectory() as td:
        run = _make_run(td, args_imgsz=800)
        md = aie.resolve_model_dir(run)
        assert md["weights"] == str(run / "weights" / "best.pt") and md["imgsz"] == 800
        assert aie.resolve_model_dir(run / "weights" / "best.pt")["run_dir"] == str(run.resolve())
        try:
            aie.resolve_model_dir(Path(td) / "nope")
        except WeightsError:
            pass
        else:
            raise AssertionError("missing model dir accepted")
        # imgsz from args.yaml when no thresholds file says otherwise; thresholds.json picked up
        (run / "thresholds.json").write_text(json.dumps({"gun": 0.33, "knife": 0.22}))
        eng = SuspiciousBehaviorEngine(model_dir=str(run), model=FakeDetector([]), person_weights=None)
        assert eng.weights == str(run / "weights" / "best.pt") and eng.imgsz == 800
        assert eng.imgsz_source == "args.yaml" and eng.threshold_for("gun") == 0.33 and eng.verifier is None
        assert SuspiciousBehaviorEngine(model_dir=str(run), model=FakeDetector([]), person_weights=None,
                                        imgsz=640).imgsz == 640
        # verifier named in pipeline_thresholds.json is loaded automatically (real yolo11s-cls, CPU)
        (run / "pipeline").mkdir()
        vpath = str(VERIFIER_PT) if VERIFIER_PT.exists() else "/nope/cls.pt"
        (run / "pipeline" / "pipeline_thresholds.json").write_text(json.dumps(_pipeline_json(
            "prod", {"prod": {"gun": 0.3, "knife": 0.25}}, {"gun": 0.16, "knife": 0.22}, tta=False, verifier=vpath)))
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            eng = SuspiciousBehaviorEngine(model_dir=str(run), model=FakeDetector([]), person_weights=None,
                                           device="cpu")
        if VERIFIER_PT.exists():
            assert eng.verifier_weights == vpath and eng.verifier is not None and eng.combine == "prod"
            assert eng.threshold_for("gun") == 0.3 and eng.verifier_index["not_weapon"] >= 0
        else:
            assert eng.verifier is None and any("not found" in str(r.message) for r in rec)
        eng = SuspiciousBehaviorEngine(model_dir=str(run), model=FakeDetector([]), person_weights=None,
                                       verifier_weights="none")
        assert eng.verifier is None and eng.threshold_for("gun") == 0.33   # detector-only -> thresholds.json
        # JSON config: kwargs, relative paths resolved against the config file, overrides win
        cfgp = Path(td) / "app_config.json"
        cfgp.write_text(json.dumps({"model_dir": "myrun", "verifier_weights": "none", "nms_iou": 0.6,
                                    "tta_image": False, "person_weights": ""}))
        eng = SuspiciousBehaviorEngine.from_config(cfgp, model=FakeDetector([]), nms_iou=None)
        assert eng.weights == str((run / "weights" / "best.pt").resolve()) and eng.nms_iou == 0.6
        assert eng.tta == {"image": False, "video": False} and eng.person_model is None
        eng = SuspiciousBehaviorEngine.from_config(cfgp, model=FakeDetector([]), nms_iou=0.45)
        assert eng.nms_iou == 0.45
    # the real detector run resolves (read-only; nothing is loaded)
    if DET_RUN.exists():
        md = aie.resolve_model_dir(DET_RUN)
        assert md["weights"].endswith("indian_strict_m960/weights/best.pt") and md["imgsz"] == 960


def test_predict_kwargs_nms_classes_and_one_image_per_call():
    eng, det, _ = new_engine([[N_GUN]], tile=True)
    assert eng.nms_iou == aie.DEFAULT_NMS_IOU == 0.5
    eng.process_frame(FRAME)
    kw = det.calls[0]
    assert kw["iou"] == 0.5 and kw["classes"] == [0, 1] and kw["max_det"] == 300   # handheld_object (2) excluded
    assert len(det.calls) == 5                               # full frame + 4 tiles, one predict call each

    class RecordingDetector(FakeDetector):
        def __call__(self, inp, **kw):
            assert isinstance(inp, np.ndarray), "a list/batch was passed to predict"
            return super().__call__(inp, **kw)
    eng = SuspiciousBehaviorEngine(model=RecordingDetector([[N_GUN]]), person_weights=None, tile=True, nms_iou=0.6)
    eng.process_frame(FRAME)
    assert eng.model.calls[0]["iou"] == 0.6
    # verifier crops go in batches of at most VERIFIER_BATCH
    many = [(0, 0.9, [10 * i, 10, 10 * i + 8, 30]) for i in range(40)]
    v = FakeVerifier([[0.9, 0.05, 0.05]] * 40)
    eng, _, _ = new_engine([many], verifier=v, nms_iou=0.99)
    eng.process_frame(FRAME)
    assert [len(b) for b in v.batches] == [32, 8]


def test_real_pipeline_matches_eval_cpu():
    """Real detector + real verifier on CPU: the engine reproduces eval_pipeline.collect() scores exactly."""
    if not (DET_RUN / "weights" / "best.pt").exists() or not VERIFIER_PT.exists() or not DOMAIN_LIST.exists():
        print("  (skipped: real weights / test list missing)")
        return
    import cv2
    import common
    from ultralytics import YOLO
    img_path = Path(DOMAIN_LIST.read_text().split()[0])
    frame = cv2.imread(str(img_path))
    det = YOLO(str(DET_RUN / "weights" / "best.pt"))
    ver = common.Verifier(str(VERIFIER_PT), "cpu", 224, tta=False)
    did = common.name_ids(det.names)
    res = det.predict(str(img_path), imgsz=960, conf=0.01, iou=0.5, device="cpu", verbose=False, max_det=300,
                      augment=False, classes=[did["gun"], did["knife"]])[0]
    pb = res.boxes.xyxy.cpu().numpy()
    pc = res.boxes.cls.cpu().numpy().astype(int)
    ps = res.boxes.conf.cpu().numpy()
    probs = ver.probs([common.square_crop(res.orig_img, b, 0.2, 224) for b in pb])
    names = {did["gun"]: "gun", did["knife"]: "knife"}
    ref = sorted((names[c], round(float(s) * float(probs[i, ver.ids[names[c]]]), 6), float(s))
                 for i, (c, s) in enumerate(zip(pc, ps)))
    eng = SuspiciousBehaviorEngine(model_dir=str(DET_RUN), verifier_weights=str(VERIFIER_PT), device="cpu",
                                   person_weights=None, augment=False, combine="prod", candidate_conf=0.01,
                                   class_conf={"gun": 1e-6, "knife": 1e-6})   # keep everything above the floor
    assert eng.imgsz == 960
    _, meta = eng.process_frame(frame)
    got = sorted((d["name"], round(d["conf"], 6), d["det_conf"]) for d in meta["weapons"])
    print(f"  {img_path.name[:40]}: {len(ref)} eval candidates, {len(got)} engine dets")
    assert got == ref, (got[:5], ref[:5])


# ---------------------------------------------------------------- real ultralytics / CLI / dashboard
def test_real_yolo_full_path_cpu():
    """Full ultralytics path with COCO yolo11n via allow_any_model (testing only),
    including the real COCO person model for armed-person derivation."""
    import cv2
    assert TEST_IMAGES, "no test images"
    for tile in (False, True):
        eng = SuspiciousBehaviorEngine(weights=str(COCO), allow_any_model=True, device="cpu",
                                       imgsz=640, tile=tile, pose_weights=str(POSE) if POSE.exists() else None)
        assert eng.weapon_classes == {"gun", "knife"}       # design change: only gun/knife are weapons
        assert eng.person_model is not None or not Path(aie.DEFAULT_PERSON_WEIGHTS).exists()
        for p in TEST_IMAGES:
            frame = cv2.imread(str(p))
            out, meta = eng.process_frame(frame)
            assert out.shape == frame.shape and out is not frame
            assert 0 <= meta["threat_score"] <= 100
            assert all(d["name"] in ("gun", "knife") for d in meta["weapons"])
            print(f"  tile={tile} {p.name[:30]}: score={meta['threat_score']} flags={meta['flags']} "
                  f"dets={meta['raw_counts']}")
    # real person model on a real image with a synthetic 'weapon' box placed on a detected person
    if eng.person_model is not None:
        frame = cv2.imread(str(TEST_IMAGES[0]))
        persons = eng.detect_persons(frame)
        print(f"  real person detector: {len(persons)} person(s)")
        if persons:
            px1, py1, px2, py2 = persons[0]
            wbox = [px1 + 0.4 * (px2 - px1), py1 + 0.4 * (py2 - py1), px1 + 0.5 * (px2 - px1), py1 + 0.5 * (py2 - py1)]
            assert aie.derive_armed_persons(persons, [wbox])[0] == persons[0]


def test_predict_cli():
    import predict
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "src"
        src.mkdir()
        for p in TEST_IMAGES:
            shutil.copy(p, src / p.name)
        out = Path(td) / "out"
        predict.main(["--weights", str(COCO), "--source", str(src), "--out", str(out),
                      "--device", "cpu", "--imgsz", "640", "--allow-any-model", "--tta"])
        with open(out / "detections.csv") as f:
            rows = list(csv.reader(f))
        assert rows[0] == predict.CSV_HEADER, rows[0]
        assert len(list(out.glob("*.jpg"))) == len(TEST_IMAGES)
        predict.main(["--weights", str(COCO), "--source", str(src), "--out", str(out), "--device", "cpu",
                      "--imgsz", "640", "--allow-any-model", "--person-weights", "none", "--mode", "video"])
        # .txt list source (relative paths resolved against the list) + --limit
        lst = Path(td) / "list.txt"
        lst.write_text("\n".join(f"src/{p.name}" for p in TEST_IMAGES) + "\n")
        out2 = Path(td) / "out2"
        predict.main(["--weights", str(COCO), "--source", str(lst), "--out", str(out2), "--device", "cpu",
                      "--imgsz", "640", "--allow-any-model", "--person-weights", "none", "--no-tta", "--limit", "1"])
        assert len(list(out2.glob("*.jpg"))) == 1
        try:
            predict.main(["--weights", str(COCO), "--source", str(src), "--out", str(out)])
        except SystemExit as e:
            assert "not a weapon model" in str(e.code)
        else:
            raise AssertionError("predict accepted COCO weights")


def test_predict_csv_rows():
    import io
    import predict
    meta = {"threat_score": 70.0, "flags": ["GUN_DETECTED", "ARMED_PERSON"],
            "weapons": [{"name": "gun", "det_conf": 0.6, "verifier_p": 0.5, "conf": 0.3, "box": [1, 2, 3, 4]},
                        {"name": "knife", "det_conf": 0.4, "verifier_p": None, "conf": 0.4, "box": [5, 6, 7, 8]}],
            "armed_persons": [[0, 0, 10, 10]]}
    buf = io.StringIO()
    predict.write_rows(csv.writer(buf), "a.jpg", 0, meta)
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    assert rows[0][:6] == ["a.jpg", "0", "gun", "0.6000", "0.5000", "0.3000"], rows[0]
    assert rows[1][3:6] == ["0.4000", "", "0.4000"], rows[1]
    assert rows[2][2] == "armed person" and rows[2][3:6] == ["", "", ""]
    assert all(len(r) == len(predict.CSV_HEADER) for r in rows)


def test_dashboard_syntax():
    src = (HERE / "dashboard.py").read_text()
    ast.parse(src)
    compile(src, "dashboard.py", "exec")
    assert "Antigravity" not in src and "C:\\" not in src
    for needle in ("st.tabs", "verifier", "TTA", 'mode="image"', 'mode="video"'):
        assert needle in src, needle


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                traceback.print_exc()
                print(f"FAIL {name}: {e}")
    print("ALL PASSED" if not failed else f"{failed} FAILED")
    sys.exit(1 if failed else 0)
