"""
tests/test_behavior_rules.py
============================
Deterministic synthetic trajectory tests for Stage 2 behavioral kinematics & rules:
1. Normal walking (baseline control): NO alerts must fire.
2. Loitering: wandering inside zone for > dwell limit with low net transit -> LOITERING fires.
3. Running: rapid locomotion > speed threshold -> RUNNING fires.
4. Abrupt reversal: 180 deg turn while in motion -> REVERSAL fires.
5. Moving against dominant flow: 3 tracks moving east, 1 moving west -> AGAINST FLOW fires.
"""

import math
from pathlib import Path
import pytest
import yaml

from src.behavior.kinematics import PersonKinematicsTracker
from src.behavior.rules import BehaviorRuleEngine


@pytest.fixture
def behavior_system():
    cfg_p = Path("/home/shiva/Desktop/projects/akriti/weapon-detection/src/behavior/config.yaml")
    with open(cfg_p) as f:
        cfg = yaml.safe_load(f)

    tracker = PersonKinematicsTracker(zones_cfg=cfg["zones"], default_fps=cfg["pipeline"]["default_fps"])
    engine = BehaviorRuleEngine(thresholds_cfg=cfg["thresholds"])
    return tracker, engine


def test_normal_walking_no_alerts(behavior_system):
    tracker, engine = behavior_system
    # Simulate person walking straight east at 0.04 heights/frame (normal walking pace)
    # H = 200px, so 8px/frame
    img_w, img_h = 800, 600
    p_box = [100.0, 300.0, 160.0, 500.0]  # w=60, h=200

    all_alerts = []
    for f_idx in range(1, 15):
        # Move +8px in X each frame
        box = [p_box[0] + (f_idx - 1) * 8.0, p_box[1], p_box[2] + (f_idx - 1) * 8.0, p_box[3]]
        prof = tracker.update_track(track_id=1, box=box, frame_idx=f_idx, frame_number=float(f_idx), img_w=img_w, img_h=img_h)
        alerts = engine.evaluate_frame([prof], frame_idx=f_idx, frame_number=float(f_idx))
        all_alerts.extend(alerts)

    # Normal linear walking must NOT trigger any alert
    assert len(all_alerts) == 0


def test_loitering_rule_fires(behavior_system):
    tracker, engine = behavior_system
    # Person stays inside central_hallway_zone [0.20-0.85 X, 0.25-0.85 Y]
    # In 800x600: X=[160, 680], Y=[150, 510]
    # Person walks in a small circle (r=15px) inside zone for 16 frames
    img_w, img_h = 800, 600
    center_x, center_y = 350.0, 350.0

    all_alerts = []
    for f_idx in range(1, 20):
        theta = (f_idx * 45.0) * (math.pi / 180.0)
        foot_x = center_x + 15.0 * math.cos(theta)
        foot_y = center_y + 15.0 * math.sin(theta)
        box = [foot_x - 30.0, foot_y - 180.0, foot_x + 30.0, foot_y]

        prof = tracker.update_track(track_id=2, box=box, frame_idx=f_idx, frame_number=float(f_idx), img_w=img_w, img_h=img_h)
        alerts = engine.evaluate_frame([prof], frame_idx=f_idx, frame_number=float(f_idx))
        all_alerts.extend(alerts)

    loiter_alerts = [a for a in all_alerts if a.rule_name == "loitering"]
    assert len(loiter_alerts) >= 1
    al = loiter_alerts[0]
    assert al.track_id == 2
    assert al.status == "FOR_HUMAN_REVIEW_ONLY"
    assert "dwell_frames" in al.trigger_numbers
    assert al.trigger_numbers["displacement_ratio"] <= 0.25


def test_running_rule_fires(behavior_system):
    tracker, engine = behavior_system
    img_w, img_h = 800, 600
    # H = 200px. Speed threshold is 0.10 heights/frame (20 px/frame).
    # Simulate sprinting at 30 px/frame (0.15 heights/frame)
    all_alerts = []
    for f_idx in range(1, 6):
        box = [100.0 + (f_idx - 1) * 30.0, 200.0, 160.0 + (f_idx - 1) * 30.0, 400.0]
        prof = tracker.update_track(track_id=3, box=box, frame_idx=f_idx, frame_number=float(f_idx), img_w=img_w, img_h=img_h)
        alerts = engine.evaluate_frame([prof], frame_idx=f_idx, frame_number=float(f_idx))
        all_alerts.extend(alerts)

    run_alerts = [a for a in all_alerts if a.rule_name == "running_or_sudden_speed"]
    assert len(run_alerts) >= 1
    al = run_alerts[0]
    assert al.track_id == 3
    assert al.status == "FOR_HUMAN_REVIEW_ONLY"
    assert al.trigger_numbers["speed_h_per_frame"] >= 0.10


def test_abrupt_reversal_rule_fires(behavior_system):
    tracker, engine = behavior_system
    img_w, img_h = 800, 600
    # Person walks right for 3 frames, then abruptly turns 180 deg and runs left
    all_alerts = []
    # Moving east: +15px/frame
    p1 = tracker.update_track(track_id=4, box=[100.0, 200.0, 160.0, 400.0], frame_idx=1, frame_number=1.0, img_w=img_w, img_h=img_h)
    p2 = tracker.update_track(track_id=4, box=[115.0, 200.0, 175.0, 400.0], frame_idx=2, frame_number=2.0, img_w=img_w, img_h=img_h)
    engine.evaluate_frame([p1], 1, 1.0)
    engine.evaluate_frame([p2], 2, 2.0)

    # Reversal: jumps -20px left
    p3 = tracker.update_track(track_id=4, box=[95.0, 200.0, 155.0, 400.0], frame_idx=3, frame_number=3.0, img_w=img_w, img_h=img_h)
    alerts = engine.evaluate_frame([p3], 3, 3.0)

    rev_alerts = [a for a in alerts if a.rule_name == "abrupt_stop_or_reversal"]
    assert len(rev_alerts) >= 1
    al = rev_alerts[0]
    assert al.track_id == 4
    assert al.status == "FOR_HUMAN_REVIEW_ONLY"
    assert al.trigger_numbers["heading_change_deg"] >= 120.0


def test_moving_against_dominant_flow_fires(behavior_system):
    tracker, engine = behavior_system
    img_w, img_h = 800, 600

    # 3 people moving East (+X, heading 0 deg)
    # 1 person moving West (-X, heading 180 deg)
    def step_tracks(dx_crowd, dx_opp):
        boxes = {
            1: [100.0 + dx_crowd, 200.0, 160.0 + dx_crowd, 400.0],
            2: [100.0 + dx_crowd, 250.0, 160.0 + dx_crowd, 450.0],
            3: [100.0 + dx_crowd, 300.0, 160.0 + dx_crowd, 500.0],
            4: [400.0 + dx_opp, 200.0, 460.0 + dx_opp, 400.0],  # Opposing
        }
        profs = []
        for tid, b in boxes.items():
            p = tracker.update_track(track_id=tid, box=b, frame_idx=step, frame_number=float(step), img_w=img_w, img_h=img_h)
            profs.append(p)
        return profs

    all_alerts = []
    # Step 1
    step = 1
    p_step1 = step_tracks(0.0, 0.0)
    engine.evaluate_frame(p_step1, 1, 1.0)

    # Step 2: crowd moves +15px east, track 4 moves -15px west
    step = 2
    p_step2 = step_tracks(15.0, -15.0)
    alerts = engine.evaluate_frame(p_step2, 2, 2.0)
    all_alerts.extend(alerts)

    flow_alerts = [a for a in all_alerts if a.rule_name == "against_dominant_flow"]
    assert len(flow_alerts) >= 1
    al = flow_alerts[0]
    assert al.track_id == 4  # Opposing track
    assert al.status == "FOR_HUMAN_REVIEW_ONLY"
    assert al.trigger_numbers["opposing_angle_deg"] >= 120.0


def test_slow_walking_no_alerts(behavior_system):
    """Edge Case 1: Slow walking (creeping pace, 2px/frame) must NOT trigger any alert."""
    tracker, engine = behavior_system
    img_w, img_h = 800, 600
    p_box = [100.0, 300.0, 160.0, 500.0]  # H=200px

    all_alerts = []
    for f_idx in range(1, 15):
        # Move +2px in X each frame (0.01 heights/frame, well below walking threshold)
        box = [p_box[0] + (f_idx - 1) * 2.0, p_box[1], p_box[2] + (f_idx - 1) * 2.0, p_box[3]]
        prof = tracker.update_track(track_id=5, box=box, frame_idx=f_idx, frame_number=float(f_idx), img_w=img_w, img_h=img_h)
        alerts = engine.evaluate_frame([prof], frame_idx=f_idx, frame_number=float(f_idx))
        all_alerts.extend(alerts)

    assert len(all_alerts) == 0


def test_track_starting_mid_frame_no_false_alerts(behavior_system):
    """Edge Case 2: Track starting mid-sequence (frame 50) must initialize cleanly without spurious alerts."""
    tracker, engine = behavior_system
    img_w, img_h = 800, 600

    # First frame of new track at frame 50
    p_box = [200.0, 250.0, 270.0, 450.0]
    prof1 = tracker.update_track(track_id=10, box=p_box, frame_idx=50, frame_number=50.0, img_w=img_w, img_h=img_h)
    alerts1 = engine.evaluate_frame([prof1], frame_idx=50, frame_number=50.0)
    assert len(alerts1) == 0
    assert prof1.speed_h_per_frame == 0.0
    assert prof1.heading_deg is None

    # Second frame: regular step
    p_box2 = [206.0, 250.0, 276.0, 450.0]
    prof2 = tracker.update_track(track_id=10, box=p_box2, frame_idx=51, frame_number=51.0, img_w=img_w, img_h=img_h)
    alerts2 = engine.evaluate_frame([prof2], frame_idx=51, frame_number=51.0)
    assert len(alerts2) == 0


def test_short_tracks_no_spurious_alerts(behavior_system):
    """Edge Case 3: Transient short track lasting only 1 or 2 frames before disappearing."""
    tracker, engine = behavior_system
    img_w, img_h = 800, 600

    # Frame 1
    prof1 = tracker.update_track(track_id=99, box=[300.0, 200.0, 350.0, 380.0], frame_idx=1, frame_number=1.0, img_w=img_w, img_h=img_h)
    alerts1 = engine.evaluate_frame([prof1], frame_idx=1, frame_number=1.0)
    assert len(alerts1) == 0

    # Frame 2
    prof2 = tracker.update_track(track_id=99, box=[305.0, 202.0, 355.0, 382.0], frame_idx=2, frame_number=2.0, img_w=img_w, img_h=img_h)
    alerts2 = engine.evaluate_frame([prof2], frame_idx=2, frame_number=2.0)
    assert len(alerts2) == 0

