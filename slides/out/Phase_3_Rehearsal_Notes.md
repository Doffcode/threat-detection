# Phase 3 rehearsal notes

## Five likely faculty questions

1. **How do you reproduce 91.0% accuracy and 90.4% fight F1?**  
   Apply the 0.40 threshold to the 300 saved SCFD scores. Accuracy is (146 + 127) / 300 = 91.0%. Fight F1 is (2 × 127) / (2 × 127 + 4 + 23) = 90.4%. Both results regenerate from `local_predictions.csv`.

2. **Does that benchmark establish reliability on live CCTV?**  
   It measures one decision per SCFD clip, with 150 clips per class. The dashboard scores five-second windows. Continuous-video performance requires event labels and measurements of missed incidents, false alerts per hour, and onset-to-alert delay. The overlap check covered exact file hashes for 40 accessible RWF-2000 clips, so full training-set overlap remains possible.

3. **Why is tabletop mAP much higher than CCTV mAP?**  
   The 0.919 tabletop result uses `v5_hybrid_m640` without a crop verifier on 35 tabletop validation images. The 0.252 CCTV result uses `baseline_m960_v2` on 72 Innovatus images. Different checkpoints, datasets, and protocols mean these numbers cannot quantify an improvement.

4. **Why show a movement replay if zero intervals matched?**  
   The replay is a separate example of the evidence feature. The 24-clip review produced zero alerts and zero matches. All 15 positive intervals were recorded at 0.00–0.02 seconds, so valid cue recall remains unverified until the reference timing is checked.

5. **What exactly is complete for Phase 3?**  
   One recorded-video workflow runs weapon detection, fight scoring, and movement evidence generation. The dashboard aligns their outputs on a timeline and supports playback and downloads. Each branch has its own evaluation artifacts and measured outcome.
