# Presentation notes

## Two-minute opening statement

Good morning. I am Akriti Jaiswal. This is my Phase 3 review of the Surveillance Threat Detection System. The system takes one recorded CCTV clip and produces three kinds of evidence in the same dashboard. It can show a weapon box, a fight score for a short interval, and an annotated replay when a defined movement cue fires. Each branch runs independently, so one branch's score does not control the others.

The strongest result is from the frozen X3D-M classifier. It correctly classified 273 of 300 SCFD clips, giving 91.0 percent accuracy and 90.4 percent fight F1. There were 150 clips in each class. It missed 23 fights and falsely flagged four non-fight clips. The benchmark scored each whole clip at a threshold of 0.40. The dashboard applies the same threshold to five-second windows, which is a separate use of the model.

For weapons, the dashboard checkpoint achieved 0.252 mAP at an overlap of 0.50 on 72 Innovatus CCTV images, detecting 17 of 75 annotated weapons at the selected confidence. A separate tabletop validation achieved 0.919 with the v5 detector alone. That result applies to tabletop images.

The movement replay demonstrates how the interface explains an approach-and-retreat cue. The separate review recorded zero alerts, and its positive timestamps require rechecking. I will show the live results, click through an event, and open the evaluation downloads so the panel can trace the reported numbers.

## Likely panel questions

1. **How did you calculate 91.0%, and can I reproduce it?**  
   Add 146 correct non-fight decisions and 127 correct fight decisions, then divide by 300. Apply the 0.40 threshold to each saved score, or run the reporting command in this report. The confusion matrix and per-class metrics regenerate from the prediction CSV.

2. **How independent was the fight evaluation, and does it validate live alerts?**  
   SCFD is a separate dataset. The exact-hash check covered 40 accessible RWF-2000 selection files, so training-set overlap cannot be fully ruled out. The measured unit was a whole clip. Continuous alert performance requires an event-level evaluation.

3. **Why are there two very different weapon mAP values? Which uses the verifier?**  
   The 0.919 value uses `v5_hybrid_m640` alone on tabletop validation. The 0.252 value uses the dashboard's `baseline_m960_v2` on Innovatus CCTV. The crop-verifier experiment has its own export. The two reported rows differ in dataset and protocol.

4. **Can I cite the movement replay as evidence of detector accuracy?**  
   It demonstrates a functioning event and replay in a separate clip. The 24-clip review produced zero alerts and zero matches. Every positive label used the same 0.00–0.02-second window, so valid cue recall is still unverified.

5. **In what sense is Phase 3 complete?**  
   The three branches, unified timeline, evidence viewer, replay, and exports are implemented. The results support clip-level fight classification on SCFD, show a weak recall operating point for weapon detection on Innovatus, and document an open validation issue for movement cues.
