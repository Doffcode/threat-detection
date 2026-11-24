# Movement cue review

Start the review app from the project root.

```bash
work/.venv/bin/python -m streamlit run scripts/behavior/movement_review.py --server.port 8511
```

The queue contains 24 SCFD clips selected with seed 42 from the frozen 300-clip manifest and shuffled before any movement scoring. The app shows the original clips by review number. Click **No cue** or **Unclear** to save and advance. For a visible approach-and-retreat cue, open **Cue present**, enter approximate start and end in seconds, and save. The app resumes from the next unlabeled clip. Set `SCFD_ROOT` to the pinned SCFD checkout if the default temporary path is unavailable.

When all clips are reviewed, click **Score reviewed clips**. The app uses the existing person tracker and frozen approach-and-retreat rule. Alternatively, run the same script with the positional command `score` to score or `report` to regenerate the summary from the saved labels, alerts, and processing errors. Alerts are matched by timestamp to one interval per match. This review measures temporal agreement for the movement cue.
