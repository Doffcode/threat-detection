"""
src/pipeline/
=============
Stage 1 Real-Time Surveillance Pipeline:
- Person Detection & Tracking (YOLO11 + ByteTrack)
- Pose / Hand Keypoint Estimation (YOLO11-pose)
- Weapon Detection (YOLO11m baseline_m960_v2)
- Spatial & Pose-Aware Weapon Association
- Temporal Confirmation & Alert Management
"""

from .detector import WeaponDetector
from .tracker import PersonTracker
from .pose import PoseEstimator
from .associator import WeaponPersonAssociator
from .temporal import TemporalAlertManager
from .engine import Stage1PipelineEngine

__all__ = [
    "WeaponDetector",
    "PersonTracker",
    "PoseEstimator",
    "WeaponPersonAssociator",
    "TemporalAlertManager",
    "Stage1PipelineEngine",
]
