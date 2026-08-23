"""Validate the renderer on the HISTORICAL templates the study actually uses, not just on HEAD.

`validate_renderer.py` compares our renderer against `transformers` on each repository's CURRENT
template. That leaves a gap: every behavioural claim in RQ2 is made about templates from earlier
revisions, and the gate never touched them. The gap is closable, because the Hub serves any revision:
`AutoTokenizer.from_pretrained(repo, revision=<sha>)` loads the tokenizer as it was, and
`apply_chat_template` then uses that revision's template. If our rendering of a historical template
equals the reference implementation's rendering of the same historical template, byte for byte, the
gap is closed rather than bounded.

This is run over exactly the render-distinct templates used as RQ2 arms, so it validates the
instrument on the inputs the results depend on.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402
import run_exec  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    from transformers import AutoTokenizer
    rows = []
    for model, repo, _mirror in run_exec.CHECKPOINTS:
        for vi, v in enumerate(run_exec.render_versions(repo)):
            rec = {"model": model, "repo": repo, "version_index": vi, "sha": v["sha"],
                   "date": v["date"][:10]}
            try:
                tok = AutoTokenizer.from_pretrained(repo, revision=v["sha"])
            except Exception as exc:  # noqa: BLE001
                rec["skipped"] = "tokenizer at revision unavailable: %s" % str(exc)[:110]
                rows.append(rec)
                continue
            ref_tpl = getattr(tok, "chat_template", None)
            if not isinstance(ref_tpl, str) or not ref_tpl:
                rec["skipped"] = "no chat_template on that revision's tokenizer"
                rows.append(rec)
                continue
            # The template the reference loads at this revision must be the one we extracted from
            # the commit graph. If they differ, our extraction is wrong and nothing downstream holds.
            rec["template_matches_extracted"] = (ref_tpl.strip() == v["template"].strip())

            matched, mismatched, errs = 0, [], 0
            for p in render.PROBES:
                try:
                    ref = tok.apply_chat_template(p["messages"], tokenize=False,
                                                  add_generation_prompt=p["add_generation_prompt"])
                    ref_err = None
                except Exception as exc:  # noqa: BLE001
                    ref, ref_err = None, str(exc)[:80]
                try:
                    ours = render.render(v["template"], p["messages"], p["add_generation_prompt"],
                                         tok.bos_token or "", tok.eos_token or "")
                    our_err = None
                except render.TemplateError as exc:
                    ours, our_err = None, str(exc)[:80]
                if ref_err is not None and our_err is not None:
                    errs += 1          # both refuse: agreement, and a refusal is itself an outcome
                    matched += 1
                elif ref == ours:
                    matched += 1
                else:
                    mismatched.append(p["id"])
            rec.update({"probes": len(render.PROBES), "matched": matched,
                        "mismatched": mismatched, "both_refused": errs,
                        "ok": not mismatched and rec["template_matches_extracted"]})
            rows.append(rec)

    checked = [r for r in rows if "skipped" not in r]
    bad = [r for r in checked if not r["ok"]]
    res = {"rows": rows, "n_versions": len(rows), "n_checked": len(checked),
           "n_mismatched": len(bad),
           "n_skipped": len(rows) - len(checked),
           "verdict": "PASSED" if (checked and not bad) else
                      ("INCONCLUSIVE" if not checked else "FAILED")}
    with open(os.path.join(DATA, "historical_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    for r in rows:
        if "skipped" in r:
            print("  SKIP  %-30s v%d %s  %s" % (r["model"], r["version_index"], r["date"],
                                                r["skipped"]))
        elif r["ok"]:
            print("  OK    %-30s v%d %s  %d/%d probes identical (extraction matches reference)"
                  % (r["model"], r["version_index"], r["date"], r["matched"], r["probes"]))
        else:
            print("  FAIL  %-30s v%d %s  mismatched=%s extraction_matches=%s"
                  % (r["model"], r["version_index"], r["date"], r["mismatched"],
                     r["template_matches_extracted"]))
    print("\n%d historical templates, %d checked, %d mismatched, %d skipped -> %s"
          % (res["n_versions"], res["n_checked"], res["n_mismatched"], res["n_skipped"],
             res["verdict"]))
    return 0 if res["verdict"] == "PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
