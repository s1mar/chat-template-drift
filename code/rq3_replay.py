"""RQ3: replay the frame's histories through the check, and price it.

Two questions a practitioner will actually ask before adopting anything:

  1. How noisy is it? A hash of the template fires on every edit, and about half of real edits render
     identically. We count how many alerts each check raises over the same histories, so the
     suppression rate is measured rather than asserted.
  2. What does it miss and what does it cost? The check is a hash and six renders; the cost is
     reported as wall time per model.

We also replay the consumer's position: a pipeline that pinned at a model's first template and never
re-pinned. For each repository we report whether it would have ended on a different rendering from
the one it validated against, and how long it sat there.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
import templatelock  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1", default=os.path.join(DATA, "rq1.json"))
    ap.add_argument("--out", default=os.path.join(DATA, "rq3.json"))
    ap.add_argument("--no-bench", action="store_true",
                    help="skip the wall-clock timing and carry the existing cost fields through. "
                         "Every other field is deterministic, so this reproduces the published "
                         "rq3.json exactly on any machine.")
    a = ap.parse_args()

    with open(a.rq1, encoding="utf-8") as fh:
        rq1 = json.load(fh)

    naive_alerts = 0        # every textual change
    render_alerts = 0       # only changes that alter a rendering
    suppressed = 0          # textual changes correctly held back as cosmetic
    undecid = 0
    # Stranded = the repo's FINAL template renders differently from its FIRST, decided by
    # rendering both, exactly as the paper's sentence claims. The original implementation
    # counted "any behavioural pair", which is a different construct: a repository that broke
    # and then repaired back to a byte-identical rendering is not stranded, and four such
    # repositories exist in the frame. Caught by the cr2 review panel (2026-08-29); every gate
    # was green because the gates trace macros to this file's output, and this file computed
    # the wrong thing.
    hist_dir = os.path.join(DATA, "history")

    def first_last_differ(mid: str) -> bool | None:
        p = os.path.join(hist_dir, mid.replace("/", "__") + ".json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as fh:
            tp = json.load(fh).get("_templates") or []
        if len(tp) < 2:
            return None
        ra = render.render_probes(tp[0]["template"], tp[0].get("bos") or "", tp[0].get("eos") or "")
        rb = render.render_probes(tp[-1]["template"], tp[-1].get("bos") or "", tp[-1].get("eos") or "")
        comparable = [k for k in ra if ra[k]["ok"] or rb[k]["ok"]]
        if not comparable:
            return None  # neither end renders anything: undecidable, not counted as stranded
        return any(not (ra[k]["ok"] and rb[k]["ok"]) or ra[k]["text"] != rb[k]["text"]
                   for k in comparable)

    stranded = []
    for r in rq1["repos"]:
        if not r["drifted"]:
            continue
        first_behavioural_days = None
        any_behavioural = False
        for p in r["pairs"]:
            if not p["textual_change"]:
                continue
            naive_alerts += 1
            if not p["decidable"]:
                undecid += 1
                continue
            if p["behavioural"]:
                render_alerts += 1
                any_behavioural = True
                if first_behavioural_days is None:
                    first_behavioural_days = p["age_days"]
            else:
                suppressed += 1
        # Only a repo with a behavioural pair can end on a different rendering (a chain of
        # byte-identical renderings is transitively identical), so the render check is only
        # paid there; it then decides, rather than assumes, whether the repo stayed different.
        if any_behavioural and first_last_differ(r["mid"]):
            stranded.append({"mid": r["mid"], "days": first_behavioural_days,
                             "downloads": r.get("downloads")})

    # Cost, split into its two very different halves. Reporting only the end-to-end figure would
    # overstate the check by two orders of magnitude: almost all of it is fetching the tokenizer
    # over the network, which a build already does, and which is cached after the first call.
    #
    # These two fields are the only wall-clock measurements in the study, so they are the only ones
    # that do not reproduce exactly. Re-running on different hardware, or on the same machine under
    # load, moves them by several times; every other field in this file is deterministic. Pass
    # --no-bench to regenerate the deterministic fields and carry the published timings through
    # unchanged, which is what a reviewer checking the alert counts wants.
    fixture = "microsoft/Phi-3-mini-4k-instruct"
    cost_s, compute_ms = None, None
    if a.no_bench:
        prev = {}
        if os.path.exists(a.out):
            with open(a.out, encoding="utf-8") as fh:
                prev = json.load(fh)
        cost_s = prev.get("cost_seconds_per_model")
        compute_ms = prev.get("cost_ms_compute_only")
        cost_ok = compute_ms is not None
        if not cost_ok:
            raise SystemExit("--no-bench needs an existing %s to carry timings from" % a.out)
    else:
        t0 = time.time()
        try:
            entry = templatelock.entry_for(fixture)
            cost_s = time.time() - t0
            cost_ok = True
        except SystemExit:
            entry, cost_s, cost_ok = None, None, False

        if cost_ok:
            tpl, bos, eos = templatelock.load_template(fixture)
            reps = 20
            t1 = time.time()
            for _ in range(reps):
                templatelock.render_signature(tpl, bos, eos)
            compute_ms = round(1000.0 * (time.time() - t1) / reps, 1)

    # Functional self-test of the tool itself, on templates we already know disagree.
    selftest = {"ok": None}
    hist = os.path.join(DATA, "history", "NousResearch__Meta-Llama-3-8B-Instruct.json")
    if os.path.exists(hist):
        with open(hist, encoding="utf-8") as fh:
            h = json.load(fh)
        tp = h["_templates"]
        s0, _ = templatelock.render_signature(tp[0]["template"], tp[0]["bos"] or "", tp[0]["eos"] or "")
        s1, _ = templatelock.render_signature(tp[1]["template"], tp[1]["bos"] or "", tp[1]["eos"] or "")
        same_text_sig, _ = templatelock.render_signature(tp[0]["template"], tp[0]["bos"] or "",
                                                         tp[0]["eos"] or "")
        selftest = {"ok": (s0 != s1) and (s0 == same_text_sig),
                    "distinct_renderings_detected": s0 != s1,
                    "identical_template_stable": s0 == same_text_sig}

    total_dec = naive_alerts - undecid
    res = {
        "naive_alerts": naive_alerts,
        "render_alerts": render_alerts,
        "suppressed_as_cosmetic": suppressed,
        "undecidable": undecid,
        "pct_suppressed": round(100.0 * suppressed / total_dec, 1) if total_dec else None,
        "n_stranded_repos": len(stranded),
        "median_days_to_first_behavioural": (
            round(sorted(s["days"] for s in stranded)[len(stranded) // 2], 1) if stranded else None),
        "cost_seconds_per_model": round(cost_s, 3) if cost_ok else None,
        "cost_ms_compute_only": compute_ms,
        "cost_fixture": fixture,
        "n_probes": len(render.PROBES),
        "selftest": selftest,
    }
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    print("alerts a plain template hash would raise      : %d" % res["naive_alerts"])
    print("alerts the render-aware check raises          : %d" % res["render_alerts"])
    print("held back as cosmetic (identical rendering)   : %d = %s%% of decidable"
          % (res["suppressed_as_cosmetic"], res["pct_suppressed"]))
    print("undecidable (neither version renders)         : %d" % res["undecidable"])
    print("repositories where a pipeline pinned at the first template would have ended on a")
    print("  DIFFERENT rendering than it validated against: %d" % res["n_stranded_repos"])
    print("  median days until that first happened       : %s"
          % res["median_days_to_first_behavioural"])
    print("cost end to end incl. network fetch           : %ss" % res["cost_seconds_per_model"])
    print("cost of the check itself (%d renders + hash)   : %s ms" % (res["n_probes"],
                                                                     res["cost_ms_compute_only"]))
    print("tool self-test                                : %s" % json.dumps(res["selftest"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
