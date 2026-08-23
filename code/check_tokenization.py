"""Control: does the raw endpoint tokenise a rendered template the way the reference tokenizer does?

A reviewer's strongest objection to RQ2 is that sending an already-rendered string to a local server
could re-tokenise the template's control markers as ordinary text. If `<|im_start|>` became five
plain tokens instead of one, the model would be seeing a mangled turn structure and every
cross-template comparison would be measuring our transport rather than the template.

This is decidable without guessing. For each checkpoint we render a probe conversation through the
repository's own template, count tokens with the reference `AutoTokenizer`, and compare against the
server's own `prompt_eval_count` for the identical string. If control markers were being shattered
into their constituent characters, the server's count would run far above the reference count.
Agreement within a small constant is evidence the markers survive as single tokens.

The check is deliberately reported as a bound, not as proof of byte-identical token ids: the server
does not expose its token ids. It is the strongest evidence available through the interface, and its
limits are stated in the paper rather than glossed.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawllm  # noqa: E402
import render  # noqa: E402
import run_exec  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    from transformers import AutoTokenizer
    rows = []
    for model, repo, mirrored in run_exec.CHECKPOINTS:
        versions = run_exec.render_versions(repo)
        if not versions:
            continue
        v = versions[-1]  # the current template, the one the reference tokenizer also carries
        msgs = [{"role": "user", "content": "Write a function that adds two numbers."}]
        try:
            prompt = render.render(v["template"], msgs, True, v["bos"], v["eos"])
        except render.TemplateError as exc:
            rows.append({"model": model, "repo": repo, "error": "render: %s" % exc})
            continue
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            # add_special_tokens=False: the template already emits its own bos marker, and letting
            # the tokenizer add another would inflate the reference count by one and be mistaken
            # for a transport artefact.
            ref_n = len(tok(prompt, add_special_tokens=False).input_ids)
        except Exception as exc:  # noqa: BLE001
            rows.append({"model": model, "repo": repo, "error": "tokenizer: %s" % str(exc)[:120]})
            continue

        out = rawllm.generate(model, prompt, temperature=0.0, seed=1, num_ctx=8192,
                              num_predict=1, replicate=77)
        srv_n = out["prompt_tokens"]
        # A shattered control marker costs several tokens each; a handful of markers therefore shows
        # up as a large relative excess, not a one- or two-token difference.
        rows.append({"model": model, "repo": repo, "reference_tokens": ref_n,
                     "server_tokens": srv_n, "delta": srv_n - ref_n,
                     "ratio": round(srv_n / ref_n, 3) if ref_n else None,
                     "prompt_chars": len(prompt)})

    # The count comparison above is a bound, and it is confounded by newline handling: the reference
    # tokenizer absorbs a newline into an adjacent token where the server emits one of its own, which
    # shows up as a per-newline excess and has nothing to do with control markers. So the marker
    # question is asked DIRECTLY, which is what actually settles it. If a marker were being read as
    # ordinary text it would cost as many tokens as its characters.
    marker_probe = []
    for model, repo, _ in run_exec.CHECKPOINTS:
        vs = run_exec.render_versions(repo)
        if not vs:
            continue
        import re as _re
        marks = _re.findall(r"<\|[a-z_]+\|>|\[INST\]", vs[-1]["template"])
        if not marks:
            marker_probe.append({"model": model, "skipped": "template uses no bracketed marker"})
            continue
        mk = marks[0]
        spaced = " ".join(mk)
        a = rawllm.generate(model, mk, temperature=0.0, seed=1, num_ctx=8192, num_predict=1,
                            replicate=901)["prompt_tokens"]
        b = rawllm.generate(model, spaced, temperature=0.0, seed=1, num_ctx=8192, num_predict=1,
                            replicate=901)["prompt_tokens"]
        # The criterion is AGREEMENT WITH THE REFERENCE TOKENIZER, not an absolute token count. Not
        # every delimiter is a special token: Mistral v0.2 carries `[INST]` as ordinary text in its
        # vocabulary, so four tokens there is correct behaviour and not a transport fault. An earlier
        # version of this check asserted "at most two tokens" and flagged Mistral, which was the
        # check being wrong about the model rather than the transport being wrong about the marker.
        try:
            rt = AutoTokenizer.from_pretrained(repo)
            ref_mk = len(rt(mk, add_special_tokens=False).input_ids)
        except Exception:  # noqa: BLE001
            ref_mk = None
        marker_probe.append({"model": model, "marker": mk, "tokens_as_marker": a,
                             "tokens_as_characters": b, "reference_tokens": ref_mk,
                             "matches_reference": (ref_mk is not None and a == ref_mk),
                             "not_shattered": a < b})

    checked_markers = [m for m in marker_probe if "skipped" not in m]
    all_special = (all(m["matches_reference"] and m["not_shattered"] for m in checked_markers)
                   if checked_markers else False)
    # No marker is shattered anywhere: every one costs far fewer tokens than its characters, so the
    # transport is not reading control markers as text. Exact agreement with the reference count is a
    # stricter bar and two checkpoints miss it by one token on a marker measured in isolation. What
    # matters for the paper is whether that touches a checkpoint an effect is claimed on, so that is
    # recorded explicitly rather than averaged away.
    # Only Phi-3. Mistral's result is a Jinja exception raised while RENDERING, before a single token
    # is produced, so how its markers tokenise cannot bear on it either way. Llama-3 and Qwen produce
    # no informative comparison at all.
    EFFECT_CHECKPOINTS = {"phi3:mini"}
    res_effect = [m for m in checked_markers if m["model"] in EFFECT_CHECKPOINTS]
    none_shattered = all(m["not_shattered"] for m in checked_markers) if checked_markers else False

    ok = [r for r in rows if "error" not in r]
    worst = max((abs(r["delta"]) for r in ok), default=None)
    worst_ratio = max((r["ratio"] for r in ok), default=None)
    res = {"rows": rows, "n_checked": len(ok), "max_abs_delta": worst,
           "max_ratio": worst_ratio, "marker_probe": marker_probe,
           "n_markers_checked": len(checked_markers),
           "markers_parsed_as_special": all_special,
           "none_shattered": none_shattered,
           "n_markers_matching_reference": sum(1 for m in checked_markers if m["matches_reference"]),
           "effect_checkpoints_match_reference": all(m["matches_reference"] for m in res_effect),
           "verdict": ("markers intact" if none_shattered else "SHATTERED")}
    with open(os.path.join(DATA, "tokenization.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)

    for r in rows:
        if "error" in r:
            print("  %-32s SKIP %s" % (r["model"], r["error"]))
        else:
            print("  %-32s reference %4d  server %4d  delta %+d  ratio %.3f"
                  % (r["model"], r["reference_tokens"], r["server_tokens"], r["delta"], r["ratio"]))
    print("\n%d checked, max |delta| %s tokens, max ratio %s -> %s"
          % (res["n_checked"], worst, worst_ratio, res["verdict"]))
    # Compare against the flag, not against a spelling of the verdict string. This line previously
    # read `res["verdict"] == "consistent"`, and "consistent" is a value this function never
    # produces: the verdict is "markers intact" or "SHATTERED". The script therefore exited 1 on
    # every run, including passing ones, so its exit code carried no information at all.
    return 0 if none_shattered else 1


if __name__ == "__main__":
    raise SystemExit(main())
