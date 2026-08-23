"""Gate: our renderer must equal `transformers.apply_chat_template` byte for byte, or nothing runs.

This is the study's load-bearing correctness check. Every behavioural claim is of the form "these two
templates render this conversation differently", so a renderer that differs from the reference
implementation manufactures exactly the finding we are looking for. The gate runs against real
templates pulled from the Hub and fails loudly.

Usage:  python validate_renderer.py --repos <n>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

DEFAULT_REPOS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "NousResearch/Meta-Llama-3-8B-Instruct",
    "teknium/OpenHermes-2.5-Mistral-7B",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


def check_repo(repo: str) -> dict:
    from transformers import AutoTokenizer
    rec = {"repo": repo, "probes": {}, "ok": True}
    try:
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        return {"repo": repo, "skipped": "tokenizer load failed: %s" % str(exc)[:160]}
    tpl = getattr(tok, "chat_template", None)
    if not tpl:
        return {"repo": repo, "skipped": "no chat_template on HEAD"}
    if not isinstance(tpl, str):
        return {"repo": repo, "skipped": "chat_template is not a single string"}

    for p in render.PROBES:
        try:
            ref = tok.apply_chat_template(p["messages"], tokenize=False,
                                          add_generation_prompt=p["add_generation_prompt"])
            ref_err = None
        except Exception as exc:  # noqa: BLE001
            ref, ref_err = None, str(exc)[:200]
        try:
            ours = render.render(tpl, p["messages"], p["add_generation_prompt"],
                                 bos_token=tok.bos_token or "", eos_token=tok.eos_token or "")
            our_err = None
        except render.TemplateError as exc:
            ours, our_err = None, str(exc)[:200]

        same = (ref == ours) if (ref_err is None and our_err is None) else (
            ref_err is not None and our_err is not None)
        rec["probes"][p["id"]] = {"match": same,
                                  "ref_error": ref_err, "our_error": our_err,
                                  "ref_len": len(ref) if ref else None,
                                  "our_len": len(ours) if ours else None}
        if not same:
            rec["ok"] = False
            rec["probes"][p["id"]]["ref_head"] = (ref or "")[:200]
            rec["probes"][p["id"]]["our_head"] = (ours or "")[:200]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--out", default=os.path.join(DATA, "renderer_validation.json"))
    a = ap.parse_args()

    recs, failed, skipped = [], [], []
    for r in a.repos:
        rec = check_repo(r)
        recs.append(rec)
        if "skipped" in rec:
            skipped.append(r)
            print("SKIP  %-45s %s" % (r, rec["skipped"]), flush=True)
        elif rec["ok"]:
            print("OK    %-45s all %d probes byte-identical" % (r, len(rec["probes"])), flush=True)
        else:
            failed.append(r)
            bad = [k for k, v in rec["probes"].items() if not v["match"]]
            print("FAIL  %-45s mismatched probes: %s" % (r, ",".join(bad)), flush=True)

    os.makedirs(DATA, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(recs, fh, indent=1)

    checked = len(recs) - len(skipped)
    print("\n%d checked, %d byte-identical, %d MISMATCHED, %d skipped"
          % (checked, checked - len(failed), len(failed), len(skipped)))
    if failed:
        print("RENDERER GATE FAILED: %s" % ", ".join(failed))
        return 1
    if checked == 0:
        print("RENDERER GATE INCONCLUSIVE: nothing could be checked")
        return 2
    print("RENDERER GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
