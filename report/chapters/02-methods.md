# Architecture and methodology

@fig-architecture shows the three branches that process the same selected clip. The dashboard validates the input and runs the branches independently. A low fight score has no effect on weapon detections or movement cues. The saved benchmark view is available through **Evaluation**, and the original tracking and association inspection tools are available through **Advanced**.

![Independent analysis branches and shared evidence timeline.](figures/architecture.png){#fig-architecture width=85%}

## Weapon localization

The unified dashboard uses Ultralytics YOLO11m, checkpoint `baseline_m960_v2`, at an image size of 960 pixels, confidence threshold 0.30, and non-maximum suppression (NMS) intersection over union (IoU) threshold 0.50. Its classes are handgun, rifle, and knife. The unified branch runs the detector directly. The separate Advanced inspector can use person association and an optional crop verifier. For display, the unified timeline retains the highest-confidence detection for each weapon class within each second, with its exact annotated frame. This display grouping does not enter the benchmark calculation.

A distinct YOLO11m checkpoint, `v5_hybrid_m640`, was evaluated at 640 pixels on a tabletop split with augmentation disabled. The reported 0.919 mean average precision at IoU 0.50 (mAP@0.50) comes from the direct detector export. The separate two-stage tabletop export identifies `verifier_v5` and score fusion under a different procedure.

## Fight classification

The pretrained X3D-M checkpoint `final_x3d_realtime.pt` takes 16 uniformly sampled red, green, and blue (RGB) frames resized to 224 × 224. The input channels use mean 0.45 and standard deviation 0.225 after scaling pixel values to [0, 1]. The positive-class softmax score is thresholded at 0.40. There was no threshold tuning on SCFD. The benchmark produces one decision per whole clip. In the dashboard, consecutive five-second windows and a final partial window are scored separately. The video-level display uses the maximum window score, and the timeline records the start and end of each flagged window.

## Movement evidence and timeline

The movement branch uses YOLO11m person detections and ByteTrack identities. It compares tracked bounding-box footpoints over a nominal 0.5-second window. Both tracks must be visible throughout the window. Projected approach must be at least 0.50 box heights, projected retreat at least 0.15, reduction in separation at least 0.30, and final separation at most 2.00. Distances are normalized using the mean bounding-box height at the window endpoints. Each continuous trigger for a track pair produces one event. The annotated replay covers approximately one second on either side of the trigger within the source bounds.

Each timeline entry links to its supporting frame, score interval, or replay. @fig-movement illustrates the movement feature. These are image-plane measurements and depend on camera perspective and track stability.
