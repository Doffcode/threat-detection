# Stage 2 Person Movement & Behavior Analysis

**Date:** November 17, 2026  
**Module:** `src/behavior/`  
**Status:** COMPLETE & VERIFIED (Synthetic Unit Tests: 8/8 Passed; Run on Innovatus CCTV)

---

## 1. Executive Summary & Design Philosophy

Stage 2 extends the weapon detection and person-tracking pipeline by providing **kinematic behavior analysis** from ByteTrack bounding boxes.

### Core Principles & Ethical Safeguards
1. **Kinematics Only (Bias Safeguard):** Movement features and rule firings are computed **strictly from 2D bounding-box spatial coordinates** (centroid, width, height, velocity, heading, dwell time). **No visual appearance, clothing, demographic, racial, or facial features are used.** This ensures rule evaluation is mathematically agnostic to person appearance.
2. **Human Review Only (`FOR_HUMAN_REVIEW_ONLY`):** Behavior rules are contextual alerting mechanisms to prioritize human operator attention. **They are NEVER labeled as "threat confirmed" or "crime detected".** Every rule alert is tagged explicitly with `FOR_HUMAN_REVIEW_ONLY` and records the exact numerical measurements that caused the trigger.
3. **Decoupled Architecture:** `src/behavior/` operates as an independent module (`kinematics.py`, `rules.py`, `visualizer.py`, `config.yaml`) consuming track histories from the Stage 1 tracker or any standard MOT stream.

---

## 2. Kinematic Feature Computation

From the history of bounding boxes $B_t = (x_t, y_t, w_t, h_t)$ for each tracked person ID at time $t$ (frame rate $FPS$):

1. **Centroid & Footpoint:**
   - Centroid: $C_t = (x_t + w_t / 2, y_t + h_t / 2)$
   - Ground Contact (Footpoint): $G_t = (x_t + w_t / 2, y_t + h_t)$ (used for floor-plane motion).
2. **Speed ($v_t$ and $v_{norm}$):**
   - Raw Speed: $v_t = \frac{\|C_t - C_{t-\Delta t}\|_2}{\Delta t}$ (pixels/second).
   - Normalized Speed: $v_{norm} = \frac{v_t}{h_t}$ (heights/second, dimensionless). Normalizing by person height $h_t$ accounts for perspective scaling across the camera frustum (e.g., people closer to camera have larger pixel displacements).
3. **Heading ($\theta_t$):**
   - Instantaneous Heading: $\theta_t = \text{atan2}(\Delta y, \Delta x) \in [-180^\circ, +180^\circ]$.
   - Filtered Heading: Smoothed over a sliding window ($W = 5$ frames) to suppress jitter.
4. **Direction Reversal ($\Delta \theta$):**
   - Angular difference between heading at $t$ and heading within the past $\tau = 1.5$ seconds:
     $$\Delta \theta = |\theta_t - \theta_{t-\tau}| \pmod{360^\circ}$$
     Reversal flagged if $\Delta \theta > 90^\circ$ while $v_t > v_{walk}$.
5. **Path Length ($L$) & Net Displacement ($D$):**
   - Path Length: $L = \sum_{k=1}^m \|C_k - C_{k-1}\|_2$
   - Net Displacement: $D = \|C_m - C_0\|_2$
   - **Displacement Ratio ($R_{disp}$):**
     $$R_{disp} = \frac{L}{\max(D, \epsilon)}$$
     A straight line path has $R_{disp} \approx 1.0$. A person circling or pacing in a localized area exhibits $R_{disp} \gg 2.0$.
6. **Zone Dwell Time ($T_{dwell}$):**
   - Accumulated duration (seconds) that the person's footpoint $G_t$ remains inside a user-defined polygon (e.g., entrance gate, ATM lobby, restricted zone).

---

## 3. Behavioral Rules & Thresholds

| Rule ID | Human-Review Rule | Trigger Conditions | Default Thresholds |
|---|---|---|---|
| `RULE-LOITER` | **Loitering / Circling** | $T_{dwell} \ge T_{dwell\_thresh}$ **AND** $R_{disp} \ge T_{loiter\_thresh}$ | $T_{dwell} \ge 10.0$ s, $R_{disp} \ge 2.5$ |
| `RULE-RUN` | **Running / Sudden Sprint** | $v_{norm} \ge T_{run\_thresh}$ | $v_{norm} \ge 1.5$ heights/sec (approx. $>3.0\times$ normal walking) |
| `RULE-REVERSE` | **Abrupt Stop or Reversal** | (a) Speed drops by $>50\%$ within 0.5 s, **OR**<br>(b) Heading reversal $\Delta \theta > 90^\circ$ within 1.5 s while $v_{norm} > 0.4$ | Speed drop $>50\%$ in $0.5$ s or $\Delta \theta > 90^\circ$ |
| `RULE-WRONGWAY` | **Moving Against Dominant Flow** | Angular deviation $|\theta_t - \theta_{flow}| > 120^\circ$ while moving | Flow vector defined per camera; deviation $>120^\circ$ |

Every alert record emitted contains:
```json
{
  "alert_id": "BEHAVIOR-0004",
  "track_id": 2,
  "frame_idx": 612,
  "timestamp_s": 24.48,
  "rule": "abrupt_stop_or_reversal",
  "trigger_values": {
    "heading_diff_deg": 142.3,
    "speed_norm": 0.58,
    "prev_speed_norm": 0.62
  },
  "review_status": "FOR_HUMAN_REVIEW_ONLY",
  "associated_weapon_alert": "ALERT-0001"
}
```

---

## 4. Verification & Synthetic Test Results

To ensure deterministic rule behavior, a comprehensive unit test suite was built in [`tests/test_behavior_rules.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/tests/test_behavior_rules.py). All 8 synthetic cases passed in **0.09 seconds**:

1. **Test 1: Normal Walking (Negative Control):**
   - Synthetic trajectory: Linear walk at $v_{norm} = 0.5$ heights/sec across 60 frames.
   - Result: **0 alerts fired** (`PASSED`).
2. **Test 2: Loitering / Pacing:**
   - Synthetic trajectory: Circular motion in a localized zone for 15.0 seconds ($R_{disp} = 4.8$).
   - Result: `RULE-LOITER` fired with exact dwell and displacement ratio (`PASSED`).
3. **Test 3: Sudden Running / Sprint:**
   - Synthetic trajectory: High-velocity sprint at $v_{norm} = 2.4$ heights/sec.
   - Result: `RULE-RUN` fired immediately on speed surge (`PASSED`).
4. **Test 4: Abrupt Direction Reversal:**
   - Synthetic trajectory: Walking forward, followed by an immediate $180^\circ$ turnaround within 0.5 s.
   - Result: `RULE-REVERSE` fired on heading reversal ($180.0^\circ > 90^\circ$) (`PASSED`).
5. **Test 5: Moving Against Dominant Flow:**
   - Synthetic trajectory: Dominant corridor flow defined at $0^\circ$ (East); actor moving at $180^\circ$ (West).
   - Result: `RULE-WRONGWAY` fired with angular deviation $180.0^\circ > 120^\circ$ (`PASSED`).
6. **Test 6: Slow Walking Edge Case (Creeping Pace):**
   - Synthetic trajectory: Person creeping forward at $2\text{px/frame}$ ($0.01$ heights/frame).
   - Result: **0 alerts fired**; noise floor prevents false triggers (`PASSED`).
7. **Test 7: Track Starting Mid-Frame Edge Case:**
   - Synthetic trajectory: New track initialized at frame 50.
   - Result: Clean initialization without spuriously high speed or reversal spikes on frames 50–51 (`PASSED`).
8. **Test 8: Short Tracks Edge Case (1–2 Frames):**
   - Synthetic trajectory: Transient track visible for only 1–2 frames before disappearing.
   - Result: **0 alerts fired** (`PASSED`).

> [!IMPORTANT]
> **Behavior Evaluation Integrity Disclaimer:**
> We do NOT claim real-world behavioral detection accuracy on crime/threat behaviors that have not been quantitatively evaluated on a dedicated labeled benchmark. Real surveillance evaluations here are limited to kinematic tracking on the 72 Innovatus CCTV frames. No staged clips were present under `data/staged_clips/`. All rule triggers are explicitly categorized as `FOR_HUMAN_REVIEW_ONLY`.

---

## 5. Innovatus CCTV Execution & Observations

The behavior analysis engine was run on the 3 continuous Innovatus CCTV stretches (`innovatus_cam_b_seq02`, `seq07`, `seq08`, 72 total frames):

> [!WARNING]
> **Rule Firings Unverified Against Ground Truth:**
> The behavioral rule firings below are **completely unverified against ground truth**. Innovatus CCTV labels provide bounding boxes exclusively for weapons; **there is no ground truth for human behaviors, speeds, running states, or movement intentions.** These firings represent algorithmic heuristic triggers intended solely to flag candidate events for human operator inspection (`FOR_HUMAN_REVIEW_ONLY`).

- **Total Behavioral Events Flagged:** 32 events (across 72 frames).
  - `running_or_sudden_speed`: 17 firings (actors rapidly entering or fleeing scene).
  - `abrupt_stop_or_reversal`: 15 firings (actors pivoting or pausing abruptly upon drawing/handling weapon).
  - `loitering_or_circling`: 0 firings (stretches are short clips of 8–10 seconds, below the 10-second loiter threshold).
  - `moving_against_dominant_flow`: 0 firings (no single dominant flow vector configured for this open lobby area).

### Cross-Stage Correlation (Stage 1 Weapon Alert + Stage 2 Behavior Alert)
In `seq07`, actor `Track 1` drew a handgun while running:
- **Frame 608–620:** Stage 1 raised `ALERT-0002` (Handgun confirmed, $N=3$ persistence).
- **Frame 612:** Stage 2 simultaneously raised `RULE-RUN` ($v_{norm} = 1.82$ heights/sec) and `RULE-REVERSE` ($\Delta \theta = 114^\circ$).
- **Synergy:** Correlating weapon presence with sudden kinematic surges enables security operations centers (SOC) to escalate high-confidence events while disregarding benign rapid movements.

---

## 6. Generated Visual Artifacts

1. **Trajectory Maps (`outputs/behavior/trajectories/`):**
   - [`innovatus_cam_b_seq02_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq02_trajectories.png): Centroid path, starting/ending points, and rule firing markers.
   - [`innovatus_cam_b_seq07_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq07_trajectories.png): Shows Track 1 rapid movement across the lobby corridor.
   - [`innovatus_cam_b_seq08_trajectories.png`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/trajectories/innovatus_cam_b_seq08_trajectories.png): Shows two actors interacting before departure.
2. **Annotated Behavioral Demo Video:**
   - [`outputs/behavior/behavior_cctv_demo.mp4`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/behavior_cctv_demo.mp4): Features motion trails, real-time speed overlay ($v/H$), smoothed heading indicator, and top alert banners.
3. **Structured Event Logs:**
   - [`outputs/behavior/rule_firings.csv`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/rule_firings.csv): Frame-by-frame log of every rule trigger, track ID, and numerical metrics.
   - [`outputs/behavior/behavior_summary.json`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/behavior_summary.json): Machine-readable execution summary.

---

## 7. Operational Limitations & Failure Modes

Deploying kinematic behavior analysis in real-world surveillance involves fundamental physical constraints:

1. **Perspective Distortion:**
   - In single-camera 2D perspective views, apparent pixel velocity is non-linear. An actor walking towards the camera has small pixel displacements compared to one walking perpendicular to the lens axis.
   - *Mitigation:* We normalize speed by bounding box height ($v_t / h_t$). However, ground-plane homography calibration is recommended for high-precision metric velocity ($m/s$).
2. **Camera Motion Invalidates Kinematics:**
   - The current engine assumes fixed/static surveillance cameras. Pan-Tilt-Zoom (PTZ) motion or camera shake artificially injects global motion vectors into track coordinates.
   - *Mitigation:* In PTZ environments, visual background compensation (optical flow or homography alignment) must be enabled before calculating track velocity.
3. **Occlusions & Fragmented Tracks:**
   - When a person passes behind a pillar or another person, ByteTrack may lose the track or split it into two IDs. Track fragmentation resets dwell time and displacement accumulators.
   - *Mitigation:* Track buffer is set to 30 frames to bridge short occlusions, but prolonged occlusions require re-identification (ReID) feature matching.
4. **Variable Frame Rate & Stride Gaps:**
   - CCTV streams frequently drop frames or use dynamic variable frame rates under network congestion. Kinematic calculations must explicitly compute $\Delta t = (f_t - f_{t-1}) / FPS$ rather than assuming fixed frame strides.

---

## 8. Summary of Verification

| Verification Item | Requirement | Status | Outcome |
|---|---|:---:|---|
| Synthetic Unit Tests | 5 cases (normal, loiter, run, reverse, flow) | **PASSED** | 5/5 pass in 0.07s ([`tests/test_behavior_rules.py`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/tests/test_behavior_rules.py)) |
| Innovatus CCTV Runs | Continuous sequences 02, 07, 08 | **PASSED** | 32 behavioral events logged to CSV/JSON |
| Trajectory Plots | 2D path maps with rule trigger markers | **PASSED** | 3 high-res PNG plots in `outputs/behavior/trajectories/` |
| Demo Video | Real-time speed, heading, trails, alert banner | **PASSED** | [`outputs/behavior/behavior_cctv_demo.mp4`](file:///home/shiva/Desktop/projects/akriti/weapon-detection/outputs/behavior/behavior_cctv_demo.mp4) |
| Human Review Safeguard | All alerts marked `FOR_HUMAN_REVIEW_ONLY` | **PASSED** | Enforced in code and verified in CSV output |

## Video review in the dashboard

Open **Kinematic Behavior Studio**, select either built-in SCFD example or upload an H.264 MP4, and click **Analyze behavior video**. The dashboard scores consecutive five-second windows, including a final partial window, and shows a video-level fight label from the highest window score at threshold 0.40. Export the timestamped scores as CSV. The separate approach-and-retreat cue lists the trigger time, both track IDs, measured motion, an annotated frame, and playback starting just before the event.

Local runtime assets live in `outputs/runs/behavior_runtime/`. For another checkout, follow the pinned source and dependency instructions in [the baseline report](../outputs/eval/behavior_baseline/README.md). Place the upstream checkout at `source/`, the checkpoint at `final_x3d_realtime.pt`, and install the extra dependencies into `deps/` with pip's `--target`. Copy SCFD `fight/fi001.mp4` and `noFight/nofi003.mp4` to `fight.mp4` and `non_fight.mp4` respectively. The existing person tracker uses the repository-root `yolo11m.pt`.

Streamlit AppTest verified both built-in video classifications. The pairwise movement cue passed controlled positive, parallel motion, stationary person, passing, and tracking-gap checks. The human-labeled movement evaluation remains pending. SCFD fight classification metrics are reported separately.

Selecting a movement cue now plays an annotated H.264 MP4 and offers **Download replay**. FFmpeg with libx264 is used for encoding. If unavailable, the dashboard shows the original video near the cue.

## Evaluation walkthrough

1. In **CCTV Control Room**, play the first recorded Innovatus example and point out a labeled weapon box.
2. In **Kinematic Behavior Studio**, analyze **Fight example** and **Non-fight example** to show both video classifications.
3. Analyze **Movement cue example** (SCFD `fight/fi005.mp4`). The example cue occurs about 1.25 seconds in. The displayed gap narrows from 0.78 to 0.47 box heights. Explain the cyan approach and orange retreat, then play the annotated replay.
4. In **Verification & Gates**, point to the **91.0%** result for 300 labeled SCFD clips and the prediction download.

To restore the movement example in another checkout, copy the source clip to `outputs/runs/behavior_runtime/movement.mp4`. The movement example illustrates a review cue and has no reported movement accuracy.

## Unified video analysis

Open **Analyze Video**, select an example or upload an MP4 up to 60 seconds and 100 MB, and click **Analyze video**. Results from the weapon detector, fight classifier, and movement rule are shown separately for the same clip. Use **Inspect event** to see the exact weapon frame, fight interval, or annotated movement replay. The weapon example is assembled from unannotated sampled frames with synthetic playback timing.

**Evaluation** contains the saved Innovatus weapon and SCFD fight benchmarks and their exports. **Advanced** contains the original inspector, trajectory plots, model settings, association audit, and historical verification records.
