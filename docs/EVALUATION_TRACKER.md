# Evaluation tracker

Work on one item at a time, in this order. **Next item: paper results.**

- [x] **Weapon metrics display.** Remove mAP@0.50:0.95 from the dashboard headline and main presentation table. Show mAP@0.50, precision, recall, F1, and sample counts with clear split labels. Preserve the full evaluation export. Verified the table, model label, threshold, and download control with Streamlit AppTest. The source JSON retains mAP@0.50:0.95. All 9 dashboard regression tests passed.
- [x] **Pretrained behavior baseline.** X3D-M selected provisionally from two measured models on 40 RWF-2000 validation clips. See `outputs/eval/behavior_baseline/` for runs, reproduction instructions, and the deferred CUE-Net candidate. All nine dashboard tests passed and the weapon table and JSON download control were verified.
- [x] **Behavior evaluation.** Frozen X3D-M at 0.40 scored all 300 SCFD clips with 91.0% accuracy and 90.4% fight F1. The manifest, predictions, metrics, and overlap limits are in `outputs/eval/behavior_scfd/`. Four V100 workers reproduced all 300 local decisions at 16.43 clips/s. Metrics regenerated from the CSV. All nine dashboard tests passed and the weapon table and download control were verified.
- [ ] **Movement evidence.** Show timestamped cues such as rapid approach, following, or retreat for review. Keep weapon detection independent of the behavior score. Any early escalation claim needs separately labeled cues and incident onset times.
- [x] **Dashboard walkthrough.** Verified a fresh browser flow with the weapon example, live fight and non-fight classifications, real movement replay, and SCFD results. The captioned H.264 backup was decoded and played in a separate browser player.
- [ ] **Paper and recording.** Populate the paper from measured exports, test the dashboard in a fresh session, and verify playback of the recorded H.264 MP4.

## Carry completed work forward

When closing an item, record what changed and how it was verified here, with a commit or output path when available. Before closing each later item, rerun the relevant checks for completed items that the change touches. If a check fails, fix it before marking the new item complete. Preserve existing local edits.

- **Movement implementation progress.** Video classification and independent movement evidence are available in Kinematic Behavior Studio. Both SCFD samples passed real Streamlit inference. Controlled cue checks and 17 existing tests passed. Timestamped movement playback uses the input video and an annotated evidence frame. Human-labeled cue benchmarking is still pending.
- **Movement demo display.** Replay appears first, followed by the gap measurement and a two-track annotated still. The full path plots are under View all tracks with a marker key. Verified the evidence path with a controlled video in Streamlit AppTest and reran all 17 tests.
- **Movement replay.** Selecting a cue produces a downloadable H.264 clip with the same two tracks, clipped to the source bounds. Verified decode, matching playback and download bytes, caching, fallback, and all 17 regression tests.
- **Movement review queue.** The 24-clip blind review app and scoring/report commands are ready. Verified clip hashes, save/resume, interval validation, and matching. Human labeling is pending before the actual benchmark can be scored.
- **Presentation decision.** Show movement as a visual review example. Human labeling remains optional for a separate measured movement benchmark.
- **Revised presentation recording.** The two-minute annotated 1080p walkthrough is at `/home/shiva/Desktop/Threat_Detection_Annotated_Demo.mp4`. Includes weapon freeze frame, both fight scores, slow movement replay with track callouts, and SCFD results. Decoded the export and verified normal-speed playback through the end in Chrome.
- **Unified dashboard.** Analyze Video runs all three independent branches on one clip with event evidence and downloads. Evaluation reads saved results and Advanced retains the original tools. Verified four examples, a separate upload, corrupt input, model failure isolation, replay decoding, and benchmark downloads. All 17 dashboard and movement tests passed.
- **Phase 3 report.** Complete Report Kit PDF and Word report in `report/out/`, including the evidence audit, formulas, figures, opening statement, and panel questions. Final build passed with zero warnings. Institutional template uses the Akriti profile and leaves the review date open.
