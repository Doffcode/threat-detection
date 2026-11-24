# Objectives and scope

**Phase 3 final review**  
**Author:** Akriti Jaiswal  
**Review date:** [Panel review date]

## Executive summary

The completed dashboard analyzes a selected or uploaded video using three independent branches. It presents their findings on one timeline with event-specific evidence and downloads. The strongest result is **91.0% clip-level fight accuracy and 90.4% fight F1 score on 300 Surveillance Camera Fight Dataset (SCFD) clips**. The run produced 127 correct fight detections, 146 correct non-fight decisions, 23 missed fights, and 4 false positives. The F1 score is the harmonic mean of precision and recall. Each branch has its own evaluation unit and ground truth.

## Objectives

1. Localize visible weapons in closed-circuit television (CCTV) footage and expose predicted classes, confidence scores, and bounding boxes.
2. Classify short video clips for visible fighting using a frozen temporal model and surface flagged windows in uploaded videos.
3. Provide timestamped evidence for a defined approach-and-retreat movement cue with annotated replay.
4. Provide a unified review workflow and expose the saved benchmark data required to verify reported numbers.

The implemented system covers these deliverables for recorded clips. Its measured tasks are weapon localization, fight versus non-fight classification, and the defined movement cue. The evidence supports a review prototype for a human operator. Real-time incident-level accuracy, prediction of future violence, and operational performance on continuous camera networks are outside the evaluated scope.
