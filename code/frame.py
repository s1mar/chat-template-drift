"""Draw the candidate frame: text-generation models on the Hub, in descending all-time downloads.

The sampling rule is in notes/preregistration.md. The frame is written once, with its collection
date, and not re-drawn.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="candidates to draw, before inclusion")
    ap.add_argument("--out", default=os.path.join(DATA, "candidates.json"))
    a = ap.parse_args()

    if os.path.exists(a.out):
        print("frame already exists at %s; refusing to re-draw it (see preregistration)" % a.out)
        return 0

    from huggingface_hub import HfApi
    api = HfApi()
    rows = []
    for i, m in enumerate(api.list_models(filter="text-generation", sort="downloads", limit=a.n)):
        rows.append({"rank": i, "id": m.id, "downloads": getattr(m, "downloads", None),
                     "likes": getattr(m, "likes", None)})

    os.makedirs(DATA, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    with open(os.path.join(DATA, "frame_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({"drawn_utc": datetime.now(timezone.utc).isoformat(),
                   "n_candidates": len(rows),
                   "rule": "pipeline_tag=text-generation, sort=downloads desc, take in order"},
                  fh, indent=1)
    print("drew %d candidates -> %s" % (len(rows), a.out))
    print("top 10:", ", ".join(r["id"] for r in rows[:10]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
