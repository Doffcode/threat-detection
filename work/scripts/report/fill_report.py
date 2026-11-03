#!/usr/bin/env python
"""Refresh the auto-generated parts of work/REPORT.md from the metrics files in work/runs.

REPORT.md contains blocks of the form

    <!-- AUTO:name -->
    ...generated text...
    <!-- /AUTO:name -->

Everything between the two markers is regenerated; the rest of the report (hand-written text) is
left untouched. A value whose metrics file does not exist yet is written as "TBD". Rows marked
optional below are hidden while their file is missing.

Usage (stdlib only, no GPU, safe to run while training is going on):

    cd /home/shiva/Desktop/projects/akriti/weapon-detection/work
    python scripts/report/fill_report.py              # refresh, keep TBD for unfinished runs
    python scripts/report/fill_report.py --final      # end of night: TBD -> "not run"
    python scripts/report/fill_report.py --dry-run    # print the new report instead of writing it
    python scripts/report/fill_report.py --best quick_tta   # force the "after" row of the headline

Blocks: status, headline, experiments, pipeline, verifier, audit, final_config, final_compare.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path

WORK = Path(__file__).resolve().parents[2]
RUNS = WORK / "runs"
REPORT = WORK / "REPORT.md"

GT_GUN, GT_KNIFE = 150, 61  # Indian domain test (144 images), current labels

# key, label, candidate metrics files (first existing wins), optional (hide when missing), note
DET_ROWS = [
    ("baseline", "Old 3-class model, leak-free split (knife_nosyn_m960)",
     ["knife_nosyn_m960/metrics_test.json"], False, "older labels of the same 144 images"),
    ("2cls", "Indian-only, gun + knife (indian_2cls_m960)",
     ["indian_2cls_m960/metrics_test.json"], False, ""),
    ("2cls_nms5", "  same + TTA + NMS 0.5",
     ["indian_2cls_m960/metrics_test_domain_tta_nms5.json"], True, ""),
    ("strict", "Indian-strict, no web knives (indian_strict_m960)",
     ["indian_strict_m960/metrics_test_domain.json", "indian_strict_m960/metrics_test.json"], False, ""),
    ("strict_tta", "  same + test-time augmentation (TTA)",
     ["indian_strict_m960/metrics_test_domain_tta.json"], False, ""),
    ("strict_nms5", "  same + TTA + NMS 0.5",
     ["indian_strict_m960/metrics_test_domain_tta_nms5.json"], False, ""),
    ("quick", "Quick fine-tune + public data, 800 px (quick_ft_m800)",
     ["quick_ft_m800/metrics_test_domain.json", "quick_ft_m800/metrics_test.json"], False, ""),
    ("quick_tta", "  same + TTA",
     ["quick_ft_m800/metrics_test_domain_tta.json"], False, ""),
    ("quick_nms5", "  same + TTA + NMS 0.5",
     ["quick_ft_m800/metrics_test_domain_tta_nms5.json"], False, ""),
    ("big_s1", "Big data stage 1, 640 px (big_s1_m640)",
     ["big_s1_m640/metrics_test_domain.json", "big_s1_m640/metrics_test.json"], False, ""),
    ("big_s1_nms5", "  same + TTA + NMS 0.5",
     ["big_s1_m640/metrics_test_domain_tta_nms5.json"], True, ""),
    ("big_s2", "Big data stage 2, 960 px (big_s2_m960)",
     ["big_s2_m960/metrics_test_domain.json", "big_s2_m960/metrics_test.json"], False, ""),
    ("big_s2_tta", "  same + TTA",
     ["big_s2_m960/metrics_test_domain_tta.json"], False, ""),
    ("big_s2_nms5", "  same + TTA + NMS 0.5",
     ["big_s2_m960/metrics_test_domain_tta_nms5.json"], False, ""),
]

# key, label, pipeline json, verifier run name, optional
PIPE_ROWS = [
    ("strict_pipe", "indian_strict_m960 + verifier_quick (TTA)",
     "indian_strict_m960/pipeline_tta/pipeline_test_domain.json", "verifier_quick", False),
    ("quick_pipe", "quick_ft_m800 + verifier_quick (TTA)",
     "quick_ft_m800/pipeline_tta/pipeline_test_domain.json", "verifier_quick", False),
    ("big_pipe", "big_s2_m960 + verifier_big (TTA)",
     "big_s2_m960/pipeline_tta/pipeline_test_domain.json", "verifier_big", False),
]

VERIFIERS = [("verifier_quick", False), ("verifier_big", False)]
AUDITS = [("indian_strict_m960", False), ("quick_ft_m800", True), ("big_s2_m960", False)]

TBD = "TBD"


# ---------------------------------------------------------------- helpers
def load(rel: str):
    p = RUNS / rel
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())  # python's json accepts the NaN written by eval.py
    except Exception as e:  # half-written file while a job is running
        print(f"[warn] cannot parse {p}: {e}")
        return None


def first(cands):
    for c in cands:
        d = load(c)
        if d is not None:
            return c, d
    return None, None


def f3(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:.3f}"


def det_metrics(d: dict) -> dict:
    """eval.py json -> {cls: {p, r, map50, f1, n}}"""
    out = {}
    for r in d.get("per_class", []):
        if r["class"] in ("gun", "knife"):
            out[r["class"]] = {"p": r["precision"], "r": r["recall"], "map50": r["map50"],
                               "f1": r.get("best_f1"), "n": r["instances"]}
    return out


def pipe_metrics(d: dict, mode: str) -> dict:
    res = d["results"][mode]
    return {c: {"p": res[c]["precision"], "r": res[c]["recall"], "map50": res[c]["ap50"],
                "f1": res[c]["f1"], "n": res[c]["n_gt"]} for c in ("gun", "knife") if c in res}


def cells(m: dict | None) -> list[str]:
    out = []
    for c in ("gun", "knife"):
        if m is None or c not in m:
            out += [TBD] * 4
        else:
            out += [f3(m[c]["p"]), f3(m[c]["r"]), f3(m[c]["map50"]), f3(m[c]["f1"])]
    return out


def mean_map(m: dict) -> float:
    v = [m[c]["map50"] for c in ("gun", "knife") if c in m]
    return sum(v) / len(v) if v else -1


def gt_note(m: dict) -> str:
    g, k = m.get("gun", {}).get("n"), m.get("knife", {}).get("n")
    if g is not None and k is not None and (g, k) != (GT_GUN, GT_KNIFE):
        return f"GT {g} guns / {k} knives"
    return ""


# ---------------------------------------------------------------- collect
def collect():
    rows = []  # dicts: key, label, full (label with parent run), kind, metrics, file, note
    parent = ""
    for key, label, cands, optional, note in DET_ROWS:
        if not label.startswith("  "):
            parent = label
        full = parent + label.strip().removeprefix("same") if label.startswith("  ") else label
        f, d = first(cands)
        if d is None and optional:
            continue
        m = det_metrics(d) if d else None
        n = "; ".join(x for x in (note, gt_note(m) if m else "") if x)
        rows.append({"key": key, "label": label, "full": full, "kind": "det", "m": m, "file": f, "note": n})
    for key, label, rel, ver, optional in PIPE_ROWS:
        d = load(rel)
        if d is None and optional:
            continue
        m, n = None, ""
        if d:
            mode = d.get("selected_mode", "det")
            m = pipe_metrics(d, mode)
            n = f"score mode `{mode}`, thresholds tuned on val"
            det_only = pipe_metrics(d, "det") if "det" in d["results"] else None
            if det_only:
                lab = label.split(" + ")[0] + " alone (same matcher, val thresholds)"
                rows.append({"key": key + "_det", "label": lab, "full": lab,
                             "kind": "pipe_det", "m": det_only, "file": rel, "note": "reference for the row below"})
        rows.append({"key": key, "label": label, "full": label, "kind": "pipe", "m": m, "file": rel if d else None, "note": n,
                     "verifier": ver})
    return rows


# ---------------------------------------------------------------- blocks
def table(rows) -> str:
    h = ["| # | model / setting | gun P | gun R | gun mAP50 | gun F1 | knife P | knife R | knife mAP50 | knife F1 | note |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        label = r["label"].replace("  ", "&nbsp;&nbsp;&nbsp;", 1) if r["label"].startswith("  ") else r["label"]
        h.append(f"| {i} | {label} | " + " | ".join(cells(r["m"])) + f" | {r['note']} |")
    return "\n".join(h)


def block_experiments(rows):
    det = [r for r in rows if r["kind"] == "det"]
    return (table(det) + "\n\nP and R are Ultralytics' values at the confidence that maximises the mean F1; F1 is each class's best "
            "F1 over all confidences (so slightly optimistic). mAP50 = average precision at IoU 0.5, independent of any threshold.")


def block_pipeline(rows):
    pr = [r for r in rows if r["kind"] in ("pipe", "pipe_det")]
    return (table(pr) + "\n\nHere P, R and F1 are measured at the per-class thresholds chosen on the validation split "
            "(an honest deployment setting, nothing tuned on test); AP50 is computed from the final score. Matching: IoU >= 0.5, one-to-one.")


def pick_best(rows, force: str | None):
    cand = [r for r in rows if r["m"] and r["key"] != "baseline" and r["kind"] in ("det", "pipe")]
    if force:
        f = [r for r in cand if r["key"] == force]
        if f:
            return f[0]
        print(f"[warn] --best {force}: no such row with metrics; choosing automatically")
    return max(cand, key=lambda r: mean_map(r["m"])) if cand else None


def block_headline(rows, best, forced=False):
    base = next((r for r in rows if r["key"] == "baseline"), None)
    lines = ["| class | metric | before (old model) | after (best so far) | change |", "|---|---|---|---|---|"]
    for c in ("knife", "gun"):
        for k, name in (("map50", "mAP50"), ("p", "precision"), ("r", "recall"), ("f1", "F1")):
            b = base["m"][c][k] if base and base["m"] else None
            a = best["m"][c][k] if best else None
            ch = f"{a - b:+.3f}" if (a is not None and b is not None) else TBD
            lines.append(f"| {c} | {name} | {f3(b) if b is not None else TBD} | {f3(a) if a is not None else TBD} | {ch} |")
    who = f"\"after\" = **{best['full']}** ({'chosen by hand' if forced else 'row with the highest mean gun + knife mAP50 on the domain test'})" if best \
        else "\"after\" = TBD"
    return ("\n".join(lines) + f"\n\n\"before\" = old 3-class model on the same 144 test images (older labels: 141 guns / 55 knives). {who}.")


def block_verifier():
    out = []
    for name, optional in VERIFIERS:
        d = load(f"{name}/verifier_val.json")
        if d is None:
            if not optional:
                out.append(f"- **{name}**: {TBD}")
            continue
        pc = {r["class"]: r for r in d["per_class"]}
        parts = ", ".join(f"{c} P {pc[c]['precision']:.2f} / R {pc[c]['recall']:.2f} (n={pc[c]['n_val']})" for c in pc)
        out.append(f"- **{name}**: top-1 accuracy {d['top1']:.3f} on validation crops; {parts}.")
        cm, names = d.get("confusion_matrix_pred_x_true"), d.get("names")
        if cm and names and "gun" in names and "knife" in names:
            gi, ki = names.index("gun"), names.index("knife")
            out.append(f"  Gun crops called knife: {cm[ki][gi]}; knife crops called gun: {cm[gi][ki]}.")
    return "\n".join(out)


def block_audit():
    out = []
    for run, optional in AUDITS:
        d = load(f"{run}/audit/summary.json")
        if d is None:
            if not optional:
                out.append(f"- **{run}**: {TBD}")
            continue
        fc = d.get("fn_by_cause", {})
        out.append(f"- **{run}** (knife threshold {d['knife_threshold']:.3f}, {d['gt_knives']} knives): "
                   f"found {d['tp']}, missed {d['fn']}, false alarms {d['fp']} (recall {d['recall']:.2f}, precision {d['precision']:.2f}). "
                   f"Missed knives: {fc.get('missed', 0)} not detected as anything, {fc.get('misclassified-as-gun', 0)} called gun, "
                   f"{fc.get('localisation', 0)} box in the wrong place; {d.get('fn_missed_with_subthreshold_detection', 0)} of the misses "
                   f"had a correct knife box just below the threshold (under-confidence).")
    return "\n".join(out)


def block_final_config(best):
    if not best:
        return f"Final model: {TBD}."
    lines = [f"Chosen configuration: **{best['full']}** (source `runs/{best['file']}`)."]
    if best["kind"] == "pipe":
        th = load(str(Path(best["file"]).parent / "pipeline_thresholds.json"))
        if th:
            sel = th["selected_mode"]
            thr = ", ".join(f"{c} {v['score_thr']:.3f}" for c, v in th["classes"].items())
            lines.append(f"Score = `{th['formula'].get(sel, sel)}`; per-class thresholds (tuned on validation): {thr}. "
                         f"Detector weights `{th['det_weights']}`, verifier `{th['verifier']}`, TTA {th['tta']}.")
    else:
        run = best["file"].split("/")[0]
        th = load(f"{run}/thresholds.json")
        if th:
            thr = ", ".join(f"{c} {v['conf']:.3f}" for c, v in th["classes"].items())
            tta = " with TTA (thresholds were tuned without TTA; re-tune if TTA is used in the app)" if "tta" in best["file"] else ""
            lines.append(f"Detector only (no verifier){tta}; per-class confidence thresholds from validation: {thr}.")
    return "\n".join(lines)


def block_final_compare():
    p = RUNS / "FINAL_COMPARE.md"
    if not p.is_file():
        return f"{TBD} (written by `runs/post.sh` after the last training run)."
    txt = p.read_text().strip()
    txt = re.sub(r"^(#+)", r"\1##", txt, flags=re.M)  # demote headings below the report's own
    return txt


def block_status(rows):
    have = [r for r in rows if r["m"]]
    miss = [r for r in rows if not r["m"]]
    s = f"Tables last refreshed {dt.datetime.now():%Y-%m-%d %H:%M} by `scripts/report/fill_report.py`. "
    s += f"{len(have)} result rows available"
    if miss:
        s += ", still waiting for: " + ", ".join(r["key"] for r in miss)
    return s + "."


# ---------------------------------------------------------------- main
def replace_block(text: str, name: str, body: str) -> str:
    pat = re.compile(rf"(<!-- AUTO:{name} -->\n).*?(\n<!-- /AUTO:{name} -->)", re.S)
    if not pat.search(text):
        print(f"[warn] block {name} not found in report")
        return text
    return pat.sub(lambda m: m.group(1) + body + m.group(2), text)


def main():
    global TBD, RUNS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--final", action="store_true", help='write "not run" instead of TBD')
    ap.add_argument("--best", default=None, help="row key to use as the 'after' / final model")
    ap.add_argument("--runs", default=str(RUNS), help="runs directory (default work/runs)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--blocks-only", action="store_true", help="print the generated blocks, do not touch the report")
    a = ap.parse_args()

    RUNS = Path(a.runs)
    if a.final:
        TBD = "not run"
    rows = collect()
    best = pick_best(rows, a.best)
    blocks = {
        "status": block_status(rows),
        "headline": block_headline(rows, best, forced=bool(a.best) and best is not None and best["key"] == a.best),
        "experiments": block_experiments(rows),
        "pipeline": block_pipeline(rows),
        "verifier": block_verifier(),
        "audit": block_audit(),
        "final_config": block_final_config(best),
        "final_compare": block_final_compare(),
    }
    if a.blocks_only:
        for k, v in blocks.items():
            print(f"<!-- AUTO:{k} -->\n{v}\n<!-- /AUTO:{k} -->\n")
        return
    path = Path(a.report)
    if not path.is_file():
        ap.error(f"{path} does not exist: create it first (it needs the <!-- AUTO:name --> markers)")
    text = path.read_text()
    for k, v in blocks.items():
        text = replace_block(text, k, v)
    if a.dry_run:
        print(text)
    else:
        path.write_text(text)
        print(f"[report] updated {path}; best row = {best['key'] if best else None}")


if __name__ == "__main__":
    main()
