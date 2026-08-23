"""RQ2: does a template edit change what the model does, once the noise floor is subtracted?

Everything here is paired over tasks, because every arm sees every task. Three quantities matter and
they answer different objections:

  REPLICATION DIVERGENCE  the same template, the same prompt, the same settings, generated twice.
                          This is the floor. A cross-template divergence at or below it is not
                          evidence of anything, and reporting one without this number would be
                          reporting GPU nondeterminism as a finding.
  RESPONSE DIVERGENCE     fraction of tasks whose reply differs between two templates.
  RESOLUTION DIFFERENCE   the outcome a user cares about: does the returned program pass its test.

Significance uses McNemar's exact test on the discordant pairs, which is the right test for a paired
binary outcome, and Holm within each family. Both families are corrected, not just the one that
suits the argument.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
from collections import defaultdict


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant counts b and c (binomial with p=0.5)."""
    from scipy.stats import binomtest
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5, alternative="two-sided").pvalue)


def holm(pvals: list) -> list:
    """Holm-Bonferroni adjusted p-values, order preserved."""
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(idx):
        v = (m - rank) * pvals[i]
        running = max(running, v)
        adj[i] = min(1.0, running)
    return adj


def boot_diff(pairs: list, reps: int = 10000, seed: int = 1) -> tuple:
    """Percentile bootstrap CI for the paired difference in rates, resampling TASKS."""
    rng = random.Random(seed)
    n = len(pairs)
    if n == 0:
        return (None, None)
    diffs = []
    for _ in range(reps):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        diffs.append(sum(x for x, _ in s) / n - sum(y for _, y in s) / n)
    diffs.sort()
    return (round(100 * diffs[int(0.025 * reps)], 1), round(100 * diffs[int(0.975 * reps)], 1))


def load(path: str) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exec", dest="execp", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    execp = a.execp or os.path.join(data, "exec.jsonl")
    outp = a.out or os.path.join(data, "rq2.json")

    rows = load(execp)
    # An episode log must never mix conversation shapes. The A7 control run wrote system-free arms
    # alongside a replication arm that was still system-message, and pooling them reported a 100%
    # noise floor that was really two different conversations. Refuse to analyse a mixed file.
    shapes = {bool(r.get("system_free", False)) for r in rows}
    if len(shapes) > 1:
        raise SystemExit(
            "ABORT: %s mixes system_free=True and system_free=False episodes (%d/%d). "
            "These are different conversations and must be analysed separately."
            % (execp, sum(1 for r in rows if r.get("system_free")), len(rows)))
    # index[model][render_sig][tid] = record   (replicate 0 only; replicate 1 held separately)
    index: dict = defaultdict(lambda: defaultdict(dict))
    repl: dict = defaultdict(dict)
    order: dict = defaultdict(list)
    for r in rows:
        if r.get("replicate", 0) == 1:
            repl[r["model"]][r["tid"]] = r
            continue
        index[r["model"]][r["render_sig"]][r["tid"]] = r
        if r["render_sig"] not in order[r["model"]]:
            order[r["model"]].append(r["render_sig"])

    result = {"models": {}, "truncated": sum(1 for r in rows if r.get("truncated")),
              "render_errors": sum(1 for r in rows if r.get("render_error")),
              "n_episodes": len(rows)}

    primary, consecutive = [], []
    for model in sorted(index):
        sigs = order[model]
        vers = {s: index[model][s] for s in sigs}
        tids = sorted(set.intersection(*[set(v) for v in vers.values()])) if vers else []
        m = {"n_versions": len(sigs), "n_tasks": len(tids), "versions": [], "pairs": []}

        # Noise floor: same template, same prompt, generated twice.
        rep = repl.get(model, {})
        base = sigs[0]
        common = [t for t in tids if t in rep]
        nd = sum(1 for t in common
                 if rep[t].get("reply_sha") != vers[base][t].get("reply_sha"))
        rr = sum(1 for t in common if rep[t].get("resolved"))
        r0 = sum(1 for t in common if vers[base][t].get("resolved"))
        m["replication"] = {
            "n": len(common),
            "response_divergence_pct": round(100.0 * nd / len(common), 1) if common else None,
            "resolved_a": r0, "resolved_b": rr,
            "resolution_diff_pts": round(100.0 * (r0 - rr) / len(common), 1) if common else None,
        }

        for s in sigs:
            v = vers[s]
            m["versions"].append({
                "render_sig": s,
                "date": next((v[t]["date"] for t in tids if t in v), None)[:10] if tids else None,
                "resolved_pct": round(100.0 * sum(1 for t in tids if v[t]["resolved"]) / len(tids), 1),
                "contract_pct": round(100.0 * sum(1 for t in tids if v[t]["contract_ok"]) / len(tids), 1),
                "render_errors": sum(1 for t in tids if v[t].get("render_error")),
            })

        for i, j in itertools.combinations(range(len(sigs)), 2):
            va, vb = vers[sigs[i]], vers[sigs[j]]
            # Two templates can differ on a probe conversation and still render THIS task's
            # conversation identically, for instance when the edit only touches the branch taken
            # when no system message is supplied. Then the model receives byte-identical input and
            # any comparison is uninformative by construction, not evidence that the edit was
            # harmless. Detect it from the recorded prompt hashes and label it, rather than
            # reporting a zero that means "we did not vary anything".
            same_prompt = sum(1 for t in tids
                              if va[t].get("prompt_sha") == vb[t].get("prompt_sha"))
            div = sum(1 for t in tids if va[t].get("reply_sha") != vb[t].get("reply_sha"))
            b = sum(1 for t in tids if va[t]["resolved"] and not vb[t]["resolved"])
            c = sum(1 for t in tids if not va[t]["resolved"] and vb[t]["resolved"])
            p = mcnemar_exact(b, c)
            # O2, output-contract compliance, is a pre-registered outcome in its own right and is
            # tested with the same machinery. It is the measure a prompt suite would actually assert
            # on, so a template edit that moves it is one that a prompt suite would have caught had
            # it been re-run, and did not catch because nothing told it to re-run.
            cb = sum(1 for t in tids if va[t]["contract_ok"] and not vb[t]["contract_ok"])
            cc = sum(1 for t in tids if not va[t]["contract_ok"] and vb[t]["contract_ok"])
            cp = mcnemar_exact(cb, cc)
            lo, hi = boot_diff([(1 if va[t]["resolved"] else 0, 1 if vb[t]["resolved"] else 0)
                                for t in tids])
            rec = {"model": model, "i": i, "j": j, "a": sigs[i], "b": sigs[j],
                   "n_prompt_identical": same_prompt,
                   "prompts_identical": same_prompt == len(tids) and len(tids) > 0,
                   "informative": same_prompt < len(tids),
                   "response_divergence_pct": round(100.0 * div / len(tids), 1) if tids else None,
                   "resolved_a_pct": round(100.0 * sum(1 for t in tids if va[t]["resolved"]) / len(tids), 1),
                   "resolved_b_pct": round(100.0 * sum(1 for t in tids if vb[t]["resolved"]) / len(tids), 1),
                   # p-values are stored at full double precision: Holm multiplies them by the
                   # family size, so rounding BEFORE correction propagated a 33% error into a
                   # printed corrected p (0.0004 for a true 0.0003) and collapsed a 1.7e-18
                   # contract p to a stored literal zero. Display rounding belongs to
                   # make_macros.py's fmtp, not here. Caught by the cr3 review panel (2026-08-30).
                   "discordant_b": b, "discordant_c": c, "p": p,
                   "ci_lo": lo, "ci_hi": hi,
                   "contract_a_pct": round(100.0 * sum(1 for t in tids if va[t]["contract_ok"]) / len(tids), 1),
                   "contract_b_pct": round(100.0 * sum(1 for t in tids if vb[t]["contract_ok"]) / len(tids), 1),
                   "contract_discordant_b": cb, "contract_discordant_c": cc,
                   "contract_p": cp}
            m["pairs"].append(rec)
            if j == i + 1:
                consecutive.append(rec)
            if i == 0 and j == len(sigs) - 1:
                primary.append(rec)
        result["models"][model] = m

    # The multiplicity family is EVERY informative within-checkpoint pair (amendment A6). Pairs whose
    # rendered prompt is identical in both arms are excluded: nothing was varied, so they are not
    # tests and must not consume family budget or be reported as nulls.
    informative = [r for m in result["models"].values() for r in m["pairs"] if r["informative"]]
    adj = holm([r["p"] for r in informative]) if informative else []
    for r, q in zip(informative, adj):
        r["p_holm_family"] = round(q, 5)
    # BOTH outcome families are corrected, each within itself. Correcting only the one that carries
    # the argument is a free win this project has been caught taking before.
    adjc = holm([r["contract_p"] for r in informative]) if informative else []
    for r, q in zip(informative, adjc):
        r["contract_p_holm_family"] = round(q, 6)
    result["family"] = informative
    result["n_family_sig_resolution"] = sum(1 for r in informative if r["p_holm_family"] < 0.05)
    result["n_family_sig_contract"] = sum(
        1 for r in informative if r["contract_p_holm_family"] < 0.05)

    # ROBUSTNESS: correct over EVERY pair, including the uninformative ones. A reviewer objected that
    # excluding them is self-serving. It is in fact the LESS conservative choice, since each excluded
    # pair has p=1 by construction and would only enlarge the family and inflate every other p, so
    # including them can only make survival harder. Rather than argue the direction, we report it.
    allpairs = [r for m in result["models"].values() for r in m["pairs"]]
    adj_all = holm([r["p"] for r in allpairs])
    adj_all_c = holm([r["contract_p"] for r in allpairs])
    for r, q, qc in zip(allpairs, adj_all, adj_all_c):
        r["p_holm_allpairs"] = round(q, 6)
        r["contract_p_holm_allpairs"] = round(qc, 6)
    result["n_allpairs"] = len(allpairs)
    result["n_allpairs_sig_resolution"] = sum(1 for r in allpairs if r["p_holm_allpairs"] < 0.05)
    result["n_allpairs_sig_contract"] = sum(
        1 for r in allpairs if r["contract_p_holm_allpairs"] < 0.05)

    # NOISE FLOOR as an explicit comparator. McNemar tests each pair against zero discordance, but
    # replaying an identical prompt already produces some. For each checkpoint we record the
    # replication arm's own resolution discordance, so a cross-template discordance can be read
    # against what the same checkpoint produces when nothing changed at all.
    for model, mm in result["models"].items():
        rep = repl.get(model, {})
        sigs = order[model]
        vers = {s: index[model][s] for s in sigs}
        tids = sorted(set.intersection(*[set(v) for v in vers.values()])) if vers else []
        common = [t for t in tids if t in rep]
        base = vers[sigs[0]]
        rb = sum(1 for t in common if base[t]["resolved"] and not rep[t]["resolved"])
        rc = sum(1 for t in common if not base[t]["resolved"] and rep[t]["resolved"])
        mm["replication"]["discordant_b"] = rb
        mm["replication"]["discordant_c"] = rc
        mm["replication"]["discordant_total"] = rb + rc
        mm["replication"]["p"] = round(mcnemar_exact(rb, rc), 5)
        for r in mm["pairs"]:
            r["discordant_total"] = r["discordant_b"] + r["discordant_c"]
            r["exceeds_floor"] = r["discordant_total"] > (rb + rc)
    result["n_uninformative_pairs"] = sum(
        1 for m in result["models"].values() for r in m["pairs"] if not r["informative"])

    # First-versus-last is reported as an explicitly secondary observation, corrected within itself:
    # it is what a "then versus now" check would see, and part of the argument is that it can be
    # near zero while the repository passed through a window in which the system was much worse.
    for name, fam in (("primary_first_vs_last", primary), ("consecutive", consecutive)):
        f = [r for r in fam if r["informative"]]
        adj = holm([r["p"] for r in f]) if f else []
        for r, q in zip(f, adj):
            r["p_holm_%s" % name] = round(q, 5)
        result[name] = fam

    with open(outp, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1)

    print("episodes %d | truncated %d | render errors %d"
          % (result["n_episodes"], result["truncated"], result["render_errors"]))
    for model, m in result["models"].items():
        rp = m["replication"]
        print("\n%s  (%d render-distinct templates, %d tasks)"
              % (model, m["n_versions"], m["n_tasks"]))
        print("   NOISE FLOOR same template twice: response divergence %s%%  resolution diff %s pts (n=%d)"
              % (rp["response_divergence_pct"], rp["resolution_diff_pts"], rp["n"]))
        for v in m["versions"]:
            print("   v %s %s  resolved %5s%%  contract %5s%%  render_err %d"
                  % (v["date"], v["render_sig"], v["resolved_pct"], v["contract_pct"],
                     v["render_errors"]))
        for r in m["pairs"]:
            flag = "" if r["informative"] else "  [SAME RENDERED PROMPT: uninformative]"
            print("   v%d vs v%d: divergence %5s%%  resolved %5s -> %5s  b/c %d/%d  p=%.4f  CI[%s,%s]%s"
                  % (r["i"], r["j"], r["response_divergence_pct"], r["resolved_a_pct"],
                     r["resolved_b_pct"], r["discordant_b"], r["discordant_c"], r["p"],
                     r["ci_lo"], r["ci_hi"], flag))
    fam = result.get("family", [])
    print("\nFAMILY: every informative pair (%d tests; %d pairs excluded as uninformative), Holm:"
          % (len(fam), result.get("n_uninformative_pairs", 0)))
    print("  RESOLUTION: %d of %d survive Holm" % (result["n_family_sig_resolution"], len(fam)))
    for r in sorted(fam, key=lambda x: x["p_holm_family"])[:6]:
        star = " <-- survives" if r["p_holm_family"] < 0.05 else ""
        print("   %-30s v%d->v%d  div %5s%%  %5s -> %5s  p=%.4f  p_holm=%.4f  CI[%s,%s]%s"
              % (r["model"], r["i"], r["j"], r["response_divergence_pct"], r["resolved_a_pct"],
                 r["resolved_b_pct"], r["p"], r["p_holm_family"], r["ci_lo"], r["ci_hi"], star))
    print("  CONTRACT COMPLIANCE: %d of %d survive Holm" % (result["n_family_sig_contract"], len(fam)))
    for r in sorted(fam, key=lambda x: x["contract_p_holm_family"])[:6]:
        star = " <-- survives" if r["contract_p_holm_family"] < 0.05 else ""
        print("   %-30s v%d->v%d  contract %5s -> %5s  b/c %d/%d  p=%.6f  p_holm=%.6f%s"
              % (r["model"], r["i"], r["j"], r["contract_a_pct"], r["contract_b_pct"],
                 r["contract_discordant_b"], r["contract_discordant_c"], r["contract_p"],
                 r["contract_p_holm_family"], star))
    print("\nsecondary, first vs last per checkpoint (what a then-versus-now check would see):")
    for r in result.get("primary_first_vs_last", []):
        if not r["informative"]:
            print("   %-30s [uninformative: identical rendered prompt]" % r["model"])
            continue
        print("   %-30s div %5s%%  %5s -> %5s  p_holm=%.4f"
              % (r["model"], r["response_divergence_pct"], r["resolved_a_pct"],
                 r["resolved_b_pct"], r["p_holm_primary_first_vs_last"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
