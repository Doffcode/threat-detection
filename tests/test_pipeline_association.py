"""
tests/test_pipeline_association.py
==================================
Unit tests on synthetic bounding boxes & keypoints:
1. Weapon inside one person
2. Weapon between two people (nearest wrist wins)
3. Weapon far from every person (no link)
4. Two weapons one person
5. Occluded wrist (fallback to containment)
6. Track gap (temporal confirmation and decay)
"""

from pathlib import Path
import numpy as np
import pytest

from src.pipeline.associator import WeaponPersonAssociator, WeaponAssociation
from src.pipeline.detector import WeaponDetection
from src.pipeline.pose import WristKeypoint
from src.pipeline.temporal import TemporalAlertManager
from src.pipeline.tracker import PersonTrack


@pytest.fixture
def associator():
    cfg = {
        "assignment_method": "max_score",
        "min_link_score": 0.25,
        "min_containment_ioa": 0.20,
        "max_wrist_distance_rel": 0.40,
        "weight_wrist": 0.45,
        "weight_containment": 0.40,
        "weight_center_in_box": 0.15,
    }
    return WeaponPersonAssociator(cfg)


def test_weapon_inside_one_person(associator):
    # Person box: [100, 100, 300, 500] (H=400)
    person = PersonTrack(track_id=1, box=[100.0, 100.0, 300.0, 500.0], conf=0.90)
    # Weapon box: completely inside
    weapon = WeaponDetection(
        box=[150.0, 200.0, 200.0, 250.0],
        conf=0.85,
        cls_id=0,
        cls_name="handgun",
        det_conf=0.85,
    )
    wrists = {1: [WristKeypoint(wrist_name="right_wrist", x=180.0, y=220.0, conf=0.80)]}

    assocs = associator.associate([weapon], [person], wrists)
    assert len(assocs) == 1
    a = assocs[0]
    assert a.is_associated is True
    assert a.track_id == 1
    assert a.containment_ioa == 1.0
    assert a.center_inside is True
    assert a.link_score >= 0.25


def test_weapon_between_two_people_nearest_wrist_wins(associator):
    # Person 1 at x=[50, 200], wrist at (190, 250)
    p1 = PersonTrack(track_id=1, box=[50.0, 100.0, 200.0, 500.0], conf=0.90)
    # Person 2 at x=[250, 400], wrist at (260, 250)
    p2 = PersonTrack(track_id=2, box=[250.0, 100.0, 400.0, 500.0], conf=0.90)

    # Weapon between them: center at (215, 250)
    # Dist to p1 wrist: 25px; Dist to p2 wrist: 45px
    weapon = WeaponDetection(
        box=[200.0, 240.0, 230.0, 260.0],
        conf=0.80,
        cls_id=2,
        cls_name="knife",
        det_conf=0.80,
    )
    wrists = {
        1: [WristKeypoint(wrist_name="right_wrist", x=190.0, y=250.0, conf=0.85)],
        2: [WristKeypoint(wrist_name="left_wrist", x=260.0, y=250.0, conf=0.85)],
    }

    assocs = associator.associate([weapon], [p1, p2], wrists)
    assert len(assocs) == 1
    a = assocs[0]
    assert a.is_associated is True
    # Nearest wrist belongs to Person 1
    assert a.track_id == 1
    assert a.wrist_dist_px == pytest.approx(25.0, abs=1.0)


def test_weapon_far_from_every_person_no_link(associator):
    person = PersonTrack(track_id=1, box=[100.0, 100.0, 300.0, 500.0], conf=0.90)
    weapon = WeaponDetection(
        box=[800.0, 800.0, 850.0, 850.0],
        conf=0.75,
        cls_id=0,
        cls_name="handgun",
        det_conf=0.75,
    )
    wrists = {1: [WristKeypoint(wrist_name="right_wrist", x=150.0, y=300.0, conf=0.80)]}

    assocs = associator.associate([weapon], [person], wrists)
    assert len(assocs) == 1
    a = assocs[0]
    assert a.is_associated is False
    assert a.track_id is None
    assert a.link_score == 0.0


def test_two_weapons_one_person(associator):
    person = PersonTrack(track_id=1, box=[100.0, 100.0, 400.0, 600.0], conf=0.95)
    w1 = WeaponDetection(
        box=[120.0, 230.0, 150.0, 260.0],
        conf=0.82,
        cls_id=0,
        cls_name="handgun",
        det_conf=0.82,
    )
    w2 = WeaponDetection(
        box=[350.0, 230.0, 380.0, 260.0],
        conf=0.78,
        cls_id=2,
        cls_name="knife",
        det_conf=0.78,
    )
    wrists = {
        1: [
            WristKeypoint(wrist_name="left_wrist", x=135.0, y=245.0, conf=0.90),
            WristKeypoint(wrist_name="right_wrist", x=365.0, y=245.0, conf=0.90),
        ]
    }

    assocs = associator.associate([w1, w2], [person], wrists)
    assert len(assocs) == 2
    assert assocs[0].is_associated is True
    assert assocs[0].track_id == 1
    assert assocs[1].is_associated is True
    assert assocs[1].track_id == 1


def test_occluded_wrist_fallback_to_containment(associator):
    person = PersonTrack(track_id=1, box=[100.0, 100.0, 300.0, 500.0], conf=0.90)
    # Weapon inside person body, but wrist keypoints are completely absent/occluded
    weapon = WeaponDetection(
        box=[140.0, 200.0, 190.0, 260.0],
        conf=0.75,
        cls_id=0,
        cls_name="handgun",
        det_conf=0.75,
    )
    wrists = {1: []}  # Occluded wrists

    assocs = associator.associate([weapon], [person], wrists)
    assert len(assocs) == 1
    a = assocs[0]
    assert a.is_associated is True
    assert a.track_id == 1
    assert a.containment_ioa == 1.0
    assert a.center_inside is True
    assert a.wrist_dist_px is None
    assert a.link_score >= 0.25


def test_track_gap(tmp_path):
    cfg = {
        "confirmation_frames": 3,
        "max_gap_frames": 2,
        "cooldown_frames": 20,
        "crop_padding": 0.20,
    }
    mgr = TemporalAlertManager(cfg, output_dir=tmp_path)
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    w = WeaponDetection([150, 200, 200, 250], 0.85, 0, "handgun", 0.85)

    def make_assoc():
        return [
            WeaponAssociation(
                weapon=w,
                track_id=1,
                person_box=[100, 100, 300, 500],
                containment_ioa=1.0,
                center_inside=True,
                wrist_dist_px=20.0,
                wrist_dist_rel_height=0.05,
                link_score=0.85,
                is_associated=True,
            )
        ]

    # Frame 1: Hit 1
    al1 = mgr.update(1, 100.0, "seq01", dummy_img, make_assoc())
    assert len(al1) == 0

    # Frame 2: Hit 2
    al2 = mgr.update(2, 103.0, "seq01", dummy_img, make_assoc())
    assert len(al2) == 0

    # Frame 3: Gap (e.g. dropped frame or occlusion, no association)
    al3 = mgr.update(3, 106.0, "seq01", dummy_img, [])
    assert len(al3) == 0

    # Frame 4: Gap <= max_gap (2 frames), Hit 3 -> CONFIRMATION THRESHOLD REACHED!
    al4 = mgr.update(4, 109.0, "seq01", dummy_img, make_assoc())
    assert len(al4) == 1
    assert al4[0].alert_id == "ALERT-0001"
    assert al4[0].track_id == 1
    assert al4[0].consecutive_hits >= 3

    # Frame 15: Huge gap (> max_gap) with no detections -> Alert finalized
    mgr.update(15, 140.0, "seq01", dummy_img, [])
    assert mgr.track_histories[1].is_alerted is False
    all_final = mgr.finalize_all()
    assert len(all_final) == 1
    assert all_final[0]["alert_id"] == "ALERT-0001"


def test_identical_input_determinism(associator):
    """GATE 1: Identical inputs must yield bit-for-bit identical association outputs."""
    p1 = PersonTrack(track_id=1, box=[100.0, 100.0, 300.0, 500.0], conf=0.92)
    p2 = PersonTrack(track_id=2, box=[350.0, 120.0, 550.0, 520.0], conf=0.88)
    w1 = WeaponDetection(box=[220.0, 250.0, 260.0, 290.0], conf=0.83, cls_id=0, cls_name="handgun", det_conf=0.83)
    w2 = WeaponDetection(box=[400.0, 240.0, 440.0, 280.0], conf=0.79, cls_id=2, cls_name="knife", det_conf=0.79)
    wrists = {
        1: [WristKeypoint("right_wrist", 230.0, 260.0, 0.85)],
        2: [WristKeypoint("left_wrist", 390.0, 250.0, 0.81)],
    }

    # Run pass 1
    res1 = associator.associate([w1, w2], [p1, p2], wrists)
    # Run pass 2
    res2 = associator.associate([w1, w2], [p1, p2], wrists)

    assert len(res1) == len(res2) == 2
    for r1, r2 in zip(res1, res2):
        assert r1.is_associated == r2.is_associated
        assert r1.track_id == r2.track_id
        assert r1.link_score == pytest.approx(r2.link_score, abs=1e-7)
        assert r1.containment_ioa == pytest.approx(r2.containment_ioa, abs=1e-7)
        assert r1.center_inside == r2.center_inside
        if r1.wrist_dist_px is not None:
            assert r1.wrist_dist_px == pytest.approx(r2.wrist_dist_px, abs=1e-7)
        else:
            assert r2.wrist_dist_px is None

