---
name: parallel-subagents-ok
description: User wants many subagents run in parallel for any task in this project to save time
metadata:
  node_type: memory
  type: feedback
  originSessionId: 7f6b6edd-de3a-4078-aa7f-b4e33f358931
  modified: 2026-10-14T00:22:52.439Z
---

User (2026-10-14, working remotely) said to use lots of subagents in parallel, "as many as you want", not just for one task.

**Why:** tight deadline (7 AM); they want wall-clock time saved.

**How to apply:** fan out independent non-GPU work (label audits, app integration, reports, data prep) to parallel agents. The single RTX A2000 GPU and ~15.7 GB RAM are the real limits — only one GPU job at a time, keep agents off the GPU and light on RAM. Related: [[data-source-policy]]
