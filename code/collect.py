"""Reconstruct the chat-template history of every repository in the frame.

One blob-filtered clone per repository (`--filter=blob:limit=1m --no-checkout`, LFS smudging off)
brings down the complete commit graph and every small blob in a single network operation; the whole
history is then read locally with no further requests. Model weights are LFS objects and are never
fetched, so the largest repository costs a fraction of a megabyte.

Failures are never persisted as results: a repository whose collection raises is left without an
output record and is retried on the next run. Refusals that are themselves data (gated, deleted) are
recorded as a terminal status with their reason, because "we could not look" and "there was nothing
there" are different facts and must not be merged.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HIST = os.path.join(DATA, "history")
SEP = "\x1f"  # NOT \x1e: str.splitlines() treats \x1c/\x1d/\x1e as line terminators.

ENV = dict(os.environ, GIT_LFS_SKIP_SMUDGE="1", GIT_TERMINAL_PROMPT="0",
           GIT_ASKPASS="echo", GCM_INTERACTIVE="never")

TEMPLATE_FILES = ["tokenizer_config.json", "chat_template.jinja", "chat_template.json"]
# `.ckpt` (TensorFlow checkpoints) and `.npz` (numpy archives) were missing from the first version of
# this list. The omission mattered in one direction only: a weight change shipped under either
# extension would have been classified as a weight-SILENT interval, inflating the headline figure in
# the direction that flatters the paper. Added, and the collection re-run.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".pt", ".pth", ".h5", ".msgpack", ".onnx",
                   ".ckpt", ".npz", ".tflite", ".mlmodel", ".param", ".weights")
WEIGHT_INDEX = ("model.safetensors.index.json", "pytorch_model.bin.index.json")

_print_lock = threading.Lock()


def git(args, cwd=None, timeout=600):
    p = subprocess.run(["git"] + args, cwd=cwd, capture_output=True, text=True,
                       timeout=timeout, encoding="utf-8", errors="replace", env=ENV)
    return p.returncode, p.stdout, p.stderr


def safe_name(mid: str) -> str:
    return mid.replace("/", "__")


def is_weight_path(path: str) -> bool:
    base = path.rsplit("/", 1)[-1]
    return path.endswith(WEIGHT_SUFFIXES) or base in WEIGHT_INDEX


def classify_clone_failure(stderr: str) -> str:
    """git cannot tell gated from deleted: the Hub answers an unauthenticated request for either
    with a credential challenge, deliberately, so that a 404 does not leak a private repository."""
    s = (stderr or "").lower()
    if ("could not read username" in s or "authentication" in s or "cannot prompt" in s
            or "401" in s or "terminal prompts disabled" in s):
        return "gated_or_missing"
    if "not found" in s or "404" in s:
        return "not_found"
    return "transport_error"


def template_from_blob(blob: str, fname: str):
    """Return (template, bos, eos) for a revision, or (None, ..) if this file carries no template."""
    if fname.endswith(".jinja"):
        t = blob.strip()
        return (t or None, None, None)
    try:
        obj = json.loads(blob)
    except Exception:
        return (None, None, None)
    if not isinstance(obj, dict):
        return (None, None, None)
    t = obj.get("chat_template")
    if isinstance(t, list):
        t = json.dumps(t, sort_keys=True)
    if t is not None and not isinstance(t, str):
        t = None

    def tok(v):
        if isinstance(v, dict):
            return v.get("content")
        return v if isinstance(v, str) else None

    return (t, tok(obj.get("bos_token")), tok(obj.get("eos_token")))


def collect_one(mid: str, keep_clone: bool = False) -> dict:
    tmp = tempfile.mkdtemp(prefix="tpl_")
    d = os.path.join(tmp, "r")
    try:
        rc, _, err = git(["clone", "--filter=blob:limit=1m", "--no-checkout",
                          "--single-branch", "https://huggingface.co/" + mid, d], timeout=900)
        if rc != 0 and "packfile" in (err or ""):
            # Fallback for repositories that abort pack negotiation with
            # "fatal: expected 'packfile'". The cause is not a refusal and is not random: every
            # affected repository publishes hundreds of training-checkpoint BRANCHES (Pythia's
            # step-N, OLMo's revisions), and cloning all refs is what overwhelms the negotiation.
            # `--single-branch` is both the fix and the correct scope, since every definition in
            # this study is about the DEFAULT BRANCH. Dropping these would have carved a
            # non-random hole in the frame shaped like two organisations.
            shutil.rmtree(d, ignore_errors=True)
            rc, _, err = git(["clone", "--no-checkout", "--single-branch",
                              "https://huggingface.co/" + mid, d], timeout=1800)
        if rc != 0:
            return {"mid": mid, "status": classify_clone_failure(err),
                    "detail": (err or "").strip().splitlines()[-1][:200] if err else ""}

        # Every commit on the default branch, oldest first, with its date.
        rc, log, _ = git(["log", "--reverse", "--format=%H" + SEP + "%aI" + SEP + "%s"], cwd=d)
        if rc != 0:
            return {"mid": mid, "status": "log_failed"}
        commits = []
        for line in log.splitlines():
            parts = line.split(SEP)
            if len(parts) >= 2:
                commits.append({"sha": parts[0], "date": parts[1],
                                "subject": (parts[2] if len(parts) > 2 else "")[:120]})
        if not commits:
            return {"mid": mid, "status": "empty_history"}

        # Which commits touched a template-bearing file, and what the template was afterwards.
        touch = {}
        for fname in TEMPLATE_FILES:
            rc, out, _ = git(["log", "--reverse", "--format=%H", "--", fname], cwd=d)
            if rc == 0 and out.strip():
                for sha in out.split():
                    touch.setdefault(sha, []).append(fname)
        if not touch:
            return {"mid": mid, "status": "no_template_file", "n_commits": len(commits)}

        revisions = []
        for c in commits:
            if c["sha"] not in touch:
                continue
            for fname in touch[c["sha"]]:
                rc, blob, _ = git(["show", "%s:%s" % (c["sha"], fname)], cwd=d)
                if rc != 0:
                    continue
                tpl, bos, eos = template_from_blob(blob, fname)
                if tpl is None:
                    continue
                revisions.append({"sha": c["sha"], "date": c["date"], "subject": c["subject"],
                                  "file": fname, "template": tpl, "bos": bos, "eos": eos})
        if not revisions:
            return {"mid": mid, "status": "no_template_in_any_revision", "n_commits": len(commits)}

        # Collapse to changes only: consecutive revisions carrying an identical string are not events.
        distinct = []
        for r in revisions:
            if not distinct or distinct[-1]["template"] != r["template"]:
                distinct.append(r)

        # For each interval between consecutive distinct templates, did any weight file move?
        intervals = []
        for i in range(len(distinct) - 1):
            a, b = distinct[i], distinct[i + 1]
            rc, names, _ = git(["log", "--name-only", "--format=", "%s..%s" % (a["sha"], b["sha"])],
                               cwd=d)
            paths = sorted({p.strip() for p in names.splitlines() if p.strip()}) if rc == 0 else []
            wpaths = [p for p in paths if is_weight_path(p)]
            intervals.append({
                "from_sha": a["sha"][:12], "to_sha": b["sha"][:12],
                "from_date": a["date"], "to_date": b["date"],
                "subject": b["subject"],
                "files_changed": len(paths),
                "weight_files_changed": len(wpaths),
                "weight_silent": len(wpaths) == 0,
                # Record the extensions actually seen, so a future question about whether some
                # weight-bearing format escaped the suffix list can be answered from the stored
                # data instead of by re-cloning a thousand repositories.
                "exts": sorted({os.path.splitext(p)[1].lower() for p in paths if os.path.splitext(p)[1]}),
            })

        return {
            "mid": mid, "status": "ok",
            "n_commits": len(commits),
            "first_commit": commits[0]["date"], "last_commit": commits[-1]["date"],
            "n_template_revisions": len(revisions),
            "n_distinct_templates": len(distinct),
            "template_files": sorted({r["file"] for r in revisions}),
            "distinct": [{k: v for k, v in r.items() if k != "template"} |
                         {"len": len(r["template"])} for r in distinct],
            "intervals": intervals,
            "_templates": [{"sha": r["sha"][:12], "date": r["date"], "file": r["file"],
                            "template": r["template"], "bos": r["bos"], "eos": r["eos"]}
                           for r in distinct],
        }
    finally:
        if not keep_clone:
            shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=os.path.join(DATA, "candidates.json"))
    ap.add_argument("--target", type=int, default=400)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    with open(a.frame, encoding="utf-8") as fh:
        cands = json.load(fh)
    os.makedirs(HIST, exist_ok=True)

    todo = [c for c in cands if not os.path.exists(os.path.join(HIST, safe_name(c["id"]) + ".json"))]
    print("%d candidates, %d already collected, %d to do"
          % (len(cands), len(cands) - len(todo), len(todo)), flush=True)

    done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(collect_one, c["id"]): c for c in todo}
        for fut in as_completed(futs):
            c = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - never persist a failure as a result
                with _print_lock:
                    print("  RETRYABLE %s: %r" % (c["id"], exc), flush=True)
                continue
            rec["downloads"] = c.get("downloads")
            rec["rank"] = c.get("rank")
            with open(os.path.join(HIST, safe_name(c["id"]) + ".json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh)
            done += 1
            if done % 25 == 0:
                with _print_lock:
                    print("  ... %d collected" % done, flush=True)
    print("collected %d this run" % done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
