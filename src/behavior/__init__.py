"""
src/behavior/
=============
Stage 2 Person Movement and Kinematic Behavior Analysis:
- Speed (normalized by person height per frame and per second)
- Heading and Direction Reversals
- Zone Dwell Time (Point-in-Polygon)
- Total Path Length & Displacement / Path-Length Ratio
- Rule Alerts (Loitering, Running, Abrupt Stop/Reversal, Opposing Flow)
"""

from .kinematics import KinematicProfile, PersonKinematicsTracker
from .rules import BehaviorAlert, BehaviorRuleEngine

__all__ = [
    "KinematicProfile",
    "PersonKinematicsTracker",
    "BehaviorAlert",
    "BehaviorRuleEngine",
]
