"""RQ1: how often does a chat template change after release, silently, and does the change matter?

Three things are computed here and they are deliberately different questions:

  DRIFT          the template string changed at least once after the first one appeared
  WEIGHT-SILENT  it changed in an interval where no model-weight file moved, so a consumer who
                 pinned the model by name and checked its weights would have seen nothing
  BEHAVIOURAL    the change actually alters what the model is shown, decided by RENDERING the fixed
                 probe set through both templates, never by reading the Jinja source

The third is the one that stops this being a study of diff noise. A 600-character rewrite that
renders byte-identically is cosmetic; a four-character edit that drops the system message is not.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HIST = os.path.join(DATA, "history")
SETTLED_DAYS = 30  # a change this long after the first template is not initial publication churn


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def load_frame(target: int) -> tuple[list, dict]:
    """The frame is the first `target` INCLUDED repositories in download-rank order.

    Inclusion and the ordering are fixed in notes/preregistration.md. Everything that qualified
    beyond `target` is kept separately and reported as an out-of-frame extension, never merged in.
    """
    recs = []
    for p in glob.glob(os.path.join(HIST, "*.json")):
        with open(p, encoding="utf-8") as fh:
            recs.append(json.load(fh))
    recs.sort(key=lambda r: (r.get("rank") if r.get("rank") is not None else 10**9))
    status = Counter(r["status"] for r in recs)
    included = [r for r in recs if r["status"] == "ok"]
    frame, extra = included[:target], included[target:]
    return frame, {"status": dict(status), "n_candidates": len(recs),
                   "n_included_total": len(included), "n_frame": len(frame),
                   "n_out_of_frame": len(extra), "extra": extra}


def classify_pair(a: dict, b: dict) -> dict:
    """Render both templates through the probe set and say what, if anything, differs."""
    ra = render.render_probes(a["template"], a.get("bos") or "", a.get("eos") or "")
    rb = render.render_probes(b["template"], b.get("bos") or "", b.get("eos") or "")
    cmp = render.compare_probes(ra, rb)
    return {
        "textual_change": a["template"] != b["template"],
        "behavioural": cmp["behavioural"],
        "decidable": cmp["decidable"],
        "probes_differing": cmp["differing"],
        "probes_undecidable": cmp["undecidable"],
        "n_comparable": cmp["comparable"],
        "len_before": len(a["template"]), "len_after": len(b["template"]),
        "render_errors_before": sorted(k for k, v in ra.items() if not v["ok"]),
        "render_errors_after": sorted(k for k, v in rb.items() if not v["ok"]),
    }


def analyse_repo(rec: dict) -> dict:
    tpls = rec.get("_templates") or []
    out = {"mid": rec["mid"], "rank": rec.get("rank"), "downloads": rec.get("downloads"),
           "n_distinct": rec.get("n_distinct_templates", 0),
           "n_commits": rec.get("n_commits"), "pairs": []}
    if len(tpls) < 2:
        out["drifted"] = False
        return out
    out["drifted"] = True
    first = _dt(tpls[0]["date"])
    intervals = rec.get("intervals") or []
    for i in range(len(tpls) - 1):
        c = classify_pair(tpls[i], tpls[i + 1])
        iv = intervals[i] if i < len(intervals) else {}
        age_days = (_dt(tpls[i + 1]["date"]) - first).total_seconds() / 86400.0
        c.update({"index": i, "date": tpls[i + 1]["date"], "age_days": round(age_days, 2),
                  "settled": age_days >= SETTLED_DAYS,
                  "weight_silent": iv.get("weight_silent"),
                  "files_changed": iv.get("files_changed"),
                  "subject": tpls[i + 1].get("subject") or iv.get("subject", "")})
        out["pairs"].append(c)
    out["any_behavioural"] = any(p["behavioural"] for p in out["pairs"])
    out["any_behavioural_settled"] = any(p["behavioural"] and p["settled"] for p in out["pairs"])
    out["any_behavioural_weight_silent"] = any(
        p["behavioural"] and p["weight_silent"] for p in out["pairs"])
    out["days_to_first_change"] = min(p["age_days"] for p in out["pairs"])
    return out


def summarise(rows: list) -> dict:
    n = len(rows)
    drifted = [r for r in rows if r["drifted"]]
    beh = [r for r in drifted if r["any_behavioural"]]
    pairs = [p for r in drifted for p in r["pairs"]]
    decidable = [p for p in pairs if p["decidable"]]
    undecidable = [p for p in pairs if not p["decidable"]]
    bpairs = [p for p in decidable if p["behavioural"]]
    probe_counts = Counter(pid for p in bpairs for pid in p["probes_differing"])

    def pct(a, b):
        return round(100.0 * a / b, 1) if b else None

    return {
        "n_frame": n,
        "n_drifted": len(drifted), "pct_drifted": pct(len(drifted), n),
        "n_behavioural": len(beh), "pct_behavioural_of_frame": pct(len(beh), n),
        "pct_behavioural_of_drifted": pct(len(beh), len(drifted)),
        "n_behavioural_settled": sum(1 for r in drifted if r["any_behavioural_settled"]),
        "pct_behavioural_settled_of_frame": pct(
            sum(1 for r in drifted if r["any_behavioural_settled"]), n),
        "n_behavioural_weight_silent": sum(
            1 for r in drifted if r["any_behavioural_weight_silent"]),
        "pct_behavioural_weight_silent_of_behavioural": pct(
            sum(1 for r in drifted if r["any_behavioural_weight_silent"]), len(beh)),
        "n_pairs": len(pairs),
        "n_pairs_decidable": len(decidable), "n_pairs_undecidable": len(undecidable),
        "n_pairs_behavioural": len(bpairs),
        "pct_pairs_behavioural": pct(len(bpairs), len(decidable)),
        "n_pairs_weight_silent": sum(1 for p in pairs if p["weight_silent"]),
        "pct_pairs_weight_silent": pct(sum(1 for p in pairs if p["weight_silent"]), len(pairs)),
        "n_pairs_cosmetic": len(decidable) - len(bpairs),
        "probes_differing": dict(probe_counts.most_common()),
        "distinct_template_hist": dict(sorted(Counter(r["n_distinct"] for r in rows).items())),
        "median_days_to_first_change": (
            round(sorted(r["days_to_first_change"] for r in drifted)[len(drifted) // 2], 1)
            if drifted else None),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(DATA, "rq1.json"))
    a = ap.parse_args()

    frame, meta = load_frame(a.target)
    rows = [analyse_repo(r) for r in frame]
    extra_rows = [analyse_repo(r) for r in meta.pop("extra")]

    res = {"meta": meta, "summary": summarise(rows),
           "out_of_frame_summary": summarise(extra_rows) if extra_rows else None,
           "repos": rows}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    s = res["summary"]
    # An instrument that cannot render anything must not be able to report a clean null.
    if s["n_pairs"] and s["n_pairs_decidable"] == 0:
        print("ABORT: every template pair was undecidable; the renderer is broken, not the data.")
        return 3
    print("candidates drawn      : %d" % meta["n_candidates"])
    print("status                : %s" % json.dumps(meta["status"]))
    print("included (has template): %d ; FRAME = first %d by download rank"
          % (meta["n_included_total"], s["n_frame"]))
    print()
    print("drifted                        : %d/%d = %s%%"
          % (s["n_drifted"], s["n_frame"], s["pct_drifted"]))
    print("  of which BEHAVIOURAL         : %d = %s%% of frame, %s%% of drifted"
          % (s["n_behavioural"], s["pct_behavioural_of_frame"], s["pct_behavioural_of_drifted"]))
    print("  behavioural AND settled(>%dd): %d = %s%% of frame"
          % (SETTLED_DAYS, s["n_behavioural_settled"], s["pct_behavioural_settled_of_frame"]))
    print("  behavioural AND weight-silent: %d = %s%% of behavioural"
          % (s["n_behavioural_weight_silent"], s["pct_behavioural_weight_silent_of_behavioural"]))
    print()
    print("template-change events         : %d" % s["n_pairs"])
    print("  decidable / undecidable      : %d / %d"
          % (s["n_pairs_decidable"], s["n_pairs_undecidable"]))
    print("  behavioural (of decidable)   : %d = %s%%"
          % (s["n_pairs_behavioural"], s["pct_pairs_behavioural"]))
    print("  cosmetic (renders identical) : %d" % s["n_pairs_cosmetic"])
    print("  weight-silent                : %d = %s%%"
          % (s["n_pairs_weight_silent"], s["pct_pairs_weight_silent"]))
    print("  median days to first change  : %s" % s["median_days_to_first_change"])
    print()
    print("which probe conversations change: %s" % json.dumps(s["probes_differing"]))
    if res["out_of_frame_summary"]:
        o = res["out_of_frame_summary"]
        print("\n[out of frame, %d repos] drifted %s%%, behavioural %s%% of frame"
              % (o["n_frame"], o["pct_drifted"], o["pct_behavioural_of_frame"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
