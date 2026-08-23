"""Figure 1: why you cannot eyeball a template diff, and when the changes land.

Panel (a): the absolute change in template length, cosmetic edits against behavioural ones. The two
distributions sit on top of each other, which is the argument for deciding this by rendering rather
than by reading a diff.

Panel (b): days from a repository's first template to each change, as a cumulative share. This is the
honest panel. It shows how much of the raw drift rate is launch-week churn, and how much is not.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
# TrueType (Type-42) fonts, not Type-3: publisher upload checkers reject Type-3, and matplotlib
# embeds Type-3 by default. Fixed at the generator so every regeneration is clean.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "paper", "fig1.pdf")


def main() -> int:
    with open(os.path.join(DATA, "rq1.json"), encoding="utf-8") as fh:
        rq1 = json.load(fh)

    cos, beh, days_c, days_b = [], [], [], []
    for r in rq1["repos"]:
        for p in r.get("pairs", []):
            if not p["decidable"]:
                continue
            d = abs(p["len_after"] - p["len_before"])
            if p["behavioural"]:
                beh.append(max(d, 0.5))
                days_b.append(max(p["age_days"], 0.01))
            else:
                cos.append(max(d, 0.5))
                days_c.append(max(p["age_days"], 0.01))

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 2.5))

    bins = [0.4, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    ax[0].hist([cos, beh], bins=bins, label=["cosmetic", "behavioural"],
               color=["#9ecae1", "#e6550d"], edgecolor="white", linewidth=0.4)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("absolute change in template length (characters)")
    ax[0].set_ylabel("change events")
    ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title("(a) diff size does not separate them", fontsize=9)

    for series, lab, col in ((days_c, "cosmetic", "#9ecae1"), (days_b, "behavioural", "#e6550d")):
        s = sorted(series)
        if not s:
            continue
        y = [(i + 1) / len(s) for i in range(len(s))]
        ax[1].step(s, y, where="post", label=lab, color=col, linewidth=1.6)
    ax[1].set_xscale("log")
    ax[1].axvline(30, color="0.4", linestyle=":", linewidth=1)
    ax[1].text(33, 0.08, "30 days", fontsize=7, color="0.35")
    ax[1].set_xlabel("days from first template to the change")
    ax[1].set_ylabel("cumulative share")
    ax[1].set_ylim(0, 1)
    ax[1].legend(frameon=False, fontsize=8, loc="lower right")
    ax[1].set_title("(b) most changes land early, not all", fontsize=9)

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight")
    print("wrote %s  (cosmetic %d, behavioural %d)" % (OUT, len(cos), len(beh)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
