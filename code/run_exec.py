"""RQ2: hold the checkpoint and the conversation fixed, vary ONLY the historical chat template.

For each checkpoint we take the distinct chat templates its own repository has shipped over time,
render the identical repair conversation through each one, and send the rendered string to the same
local weights through Ollama's raw endpoint. Decoding is greedy with a fixed seed and the stop
tokens are the checkpoint's own defaults, identical in every arm, so the rendered prompt string is
the only thing that differs.

Two controls run alongside, and the study is uninterpretable without them:

  REPLICATION   the first template is run a second time, byte-identical prompt and settings. This is
                the noise floor. Any cross-template divergence has to be read against it, because
                greedy decoding on a GPU is not perfectly deterministic and a divergence rate below
                the replication rate means nothing.
  TRUNCATION    every episode records whether it hit the output cap, so a reply cut off by the cap
                is identifiable rather than being scored as a difference in content.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rawllm  # noqa: E402
import render  # noqa: E402
import sandbox  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
HIST = os.path.join(DATA, "history")

# Each entry pairs a LOCAL checkpoint with the Hub repository whose template history it uses.
# Four are the model's own repository. Llama-3 is the one exception: the official
# meta-llama repository is gated and cannot be read anonymously, so the NousResearch mirror of the
# same checkpoint is used. That substitution is disclosed in the paper rather than hidden here.
CHECKPOINTS = [
    ("qwen2.5:7b-instruct", "Qwen/Qwen2.5-7B-Instruct", False),
    ("phi3:mini", "microsoft/Phi-3-mini-4k-instruct", False),
    ("deepseek-coder:6.7b-instruct", "deepseek-ai/deepseek-coder-6.7b-instruct", False),
    ("mistral:7b-instruct-v0.2-q4_0", "mistralai/Mistral-7B-Instruct-v0.2", False),
    ("llama3:latest", "NousResearch/Meta-Llama-3-8B-Instruct", True),
]

SYSTEM = ("You are a helpful coding assistant. Fix the bug in the function you are given. "
          "Reply with exactly one fenced python code block and no other text.")

USER = """This function is failing its tests.

```python
{code}
```

The failure was:

{feedback}

Return the corrected function."""

FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)(?:```|\Z)", re.DOTALL | re.IGNORECASE)


def extract_code(reply: str, entry_point: str):
    """Return (code, contract_ok). The contract is exactly one fenced block defining the entry point.

    Extraction stays permissive even when the contract is violated, because that is what a real
    harness does: the contract measure records the violation, and the task measure still records
    whether the code worked. Conflating the two would make a formatting slip look like a wrong fix.
    """
    blocks = [m.group(1) for m in FENCE_RE.finditer(reply or "")]
    good = [b for b in blocks if re.search(r"^\s*def\s+%s\s*\(" % re.escape(entry_point), b, re.M)]
    contract_ok = len(blocks) == 1 and len(good) == 1
    if good:
        return good[0], contract_ok
    if blocks:
        return max(blocks, key=len), contract_ok
    return (reply or ""), False


def render_versions(repo: str):
    """Distinct chat templates for a repository, deduplicated by what they RENDER, oldest first."""
    path = os.path.join(HIST, repo.replace("/", "__") + ".json")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        rec = json.load(fh)
    out = []
    for t in rec.get("_templates") or []:
        probes = render.render_probes(t["template"], t.get("bos") or "", t.get("eos") or "")
        if not any(v["ok"] for v in probes.values()):
            continue  # a template we cannot render at all is not evidence about the model
        sig = json.dumps({k: (v["text"] if v["ok"] else "!ERR") for k, v in probes.items()},
                         sort_keys=True)
        h = hashlib.sha256(sig.encode("utf-8")).hexdigest()[:12]
        if any(o["render_sig"] == h for o in out):
            continue
        out.append({"sha": t["sha"], "date": t["date"], "template": t["template"],
                    "bos": t.get("bos") or "", "eos": t.get("eos") or "", "render_sig": h})
    return out


_FEEDBACK: dict = {}


def feedback_for(task: dict) -> str:
    """The buggy program's failure text, computed once per task and reused across every arm.

    It cannot differ between arms: it is a property of the task, not of the template. Recomputing it
    per episode ran the sandbox 1,260 times to produce 60 distinct strings and was the single
    largest cost in the run.
    """
    tid = task["tid"]
    if tid not in _FEEDBACK:
        res = sandbox.run_program(sandbox.build_program(task["prelude"], task["buggy_code"],
                                                        task["visible_test"]))
        _FEEDBACK[tid] = sandbox.feedback_from(res, task["buggy_code"])
    return _FEEDBACK[tid]


def build_prompt_system_free(version: dict, task: dict) -> str | None:
    """The A7 control: identical conversation with the contract moved OUT of the system message.

    Both RQ2 effects involve how a template handles the system role. If the contract lives in the
    user turn instead, a template that raises on a system role never raises, and a template that has
    deleted its system branch has nothing to discard. This is the arm that can refute the paper's own
    mechanism, so it exists.
    """
    feedback = feedback_for(task)
    msgs = [{"role": "user", "content": SYSTEM + "\n\n" + USER.format(
        code=task["buggy_code"], feedback=feedback)}]
    try:
        return render.render(version["template"], msgs, True, version["bos"], version["eos"])
    except render.TemplateError:
        return None


def build_prompt(version: dict, task: dict) -> str | None:
    feedback = feedback_for(task)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER.format(code=task["buggy_code"], feedback=feedback)}]
    try:
        return render.render(version["template"], msgs, True, version["bos"], version["eos"])
    except render.TemplateError:
        return None


def run(model: str, version: dict, task: dict, seed: int, num_ctx: int, num_predict: int,
        replicate: int = 0, system_free: bool = False) -> dict:
    prompt = (build_prompt_system_free if system_free else build_prompt)(version, task)
    rec = {"model": model, "sha": version["sha"], "render_sig": version["render_sig"],
           "date": version["date"], "tid": task["tid"], "replicate": replicate,
           "system_free": system_free}
    if prompt is None:
        # The template refused the conversation. That is an outcome, not a missing value: a pipeline
        # on this revision could not have sent this request at all.
        rec.update({"render_error": True, "resolved": False, "contract_ok": False,
                    "reply_sha": None, "truncated": False})
        return rec
    out = rawllm.generate(model, prompt, temperature=0.0, seed=seed, num_ctx=num_ctx,
                          num_predict=num_predict, replicate=replicate)
    code, contract_ok = extract_code(out["text"], task["entry_point"])
    prog = sandbox.build_program(task["prelude"], code, task["visible_test"])
    status = sandbox.run_program(prog)["status"]
    rec.update({
        "render_error": False,
        "resolved": status == sandbox.PASS,
        "contract_ok": contract_ok,
        "reply_sha": hashlib.sha256((out["text"] or "").encode("utf-8")).hexdigest()[:16],
        "reply_len": len(out["text"] or ""),
        "truncated": out.get("truncated", False),
        "prompt_tokens": out.get("prompt_tokens"),
        "completion_tokens": out.get("completion_tokens"),
        "prompt_sha": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
    })
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=os.path.join(DATA, "tasks.json"))
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--num-ctx", type=int, default=8192)
    # The published run used 1536 (the README command passes it explicitly); the default now
    # matches, because num_predict enters the generation-cache key and a replicator running the
    # bare script with the old 1024 default would silently miss the shipped cache and regenerate
    # every episode. Caught by the cr3 review panel (2026-08-30).
    ap.add_argument("--num-predict", type=int, default=1536)
    ap.add_argument("--only", default="")
    ap.add_argument("--system-free", action="store_true",
                    help="A7 control: move the output contract out of the system message")
    ap.add_argument("--out", default=os.path.join(DATA, "exec.jsonl"))
    a = ap.parse_args()

    # Two copies of this script appending to the same log wrote 89 duplicated episodes before it was
    # noticed. They happened to agree, because the generation cache absorbed the second request, but
    # that was luck: two concurrent processes can both miss the cache and write two different replies
    # under one key. The append-only log has no primary key, so nothing downstream would have caught
    # it. A single-instance guard is cheaper than trusting that the previous run really died.
    lock = os.path.join(DATA, "run_exec.pid")
    if os.path.exists(lock):
        try:
            with open(lock, encoding="utf-8") as fh:
                other = int(fh.read().strip())
        except (OSError, ValueError):
            other = None
        if other and other != os.getpid():
            try:
                os.kill(other, 0)
                print("REFUSING TO START: run_exec.py is already running as pid %d "
                      "(delete %s if that is stale)" % (other, lock))
                return 4
            except OSError:
                pass  # stale pid, the process is gone
    os.makedirs(DATA, exist_ok=True)
    with open(lock, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))

    with open(a.tasks, encoding="utf-8") as fh:
        all_tasks = json.load(fh)
    pool = sorted([t for t in all_tasks if t["corpus"] == "humanevalfix"], key=lambda t: t["tid"])
    rng = random.Random(a.seed)
    tasks = rng.sample(pool, min(a.n_tasks, len(pool)))
    print("%d tasks sampled from %d humanevalfix tasks (seed %d)" % (len(tasks), len(pool), a.seed))

    done = set()
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["model"], r["render_sig"], r["tid"], r.get("replicate", 0),
                          bool(r.get("system_free", False))))
        print("resuming: %d episodes already recorded" % len(done))

    fh_out = open(a.out, "a", encoding="utf-8")
    for model, repo, mirrored in CHECKPOINTS:
        if a.only and a.only not in model:
            continue
        versions = render_versions(repo)
        print("\n=== %s  <-  %s%s : %d render-distinct templates ==="
              % (model, repo, " (MIRROR)" if mirrored else "", len(versions)), flush=True)
        if len(versions) < 2:
            print("   skipped: fewer than two render-distinct templates", flush=True)
            continue
        for vi, v in enumerate(versions):
            for t in tasks:
                key = (model, v["render_sig"], t["tid"], 0, a.system_free)
                if key in done:
                    continue
                r = run(model, v, t, a.seed, a.num_ctx, a.num_predict, replicate=0,
                        system_free=a.system_free)
                r["version_index"] = vi
                fh_out.write(json.dumps(r) + "\n")
                fh_out.flush()
            print("   v%d %s %s done" % (vi, v["date"][:10], v["render_sig"]), flush=True)
        # Replication control: the FIRST template, run a second time, identical in every respect.
        for t in tasks:
            key = (model, versions[0]["render_sig"], t["tid"], 1, a.system_free)
            if key in done:
                continue
            # system_free MUST be threaded here too. Without it the replication arm silently built
            # the system-message prompt while every other arm in the run was system-free, so the
            # "same template twice" floor was actually "two different conversations" and read 100%.
            r = run(model, versions[0], t, a.seed, a.num_ctx, a.num_predict, replicate=1,
                    system_free=a.system_free)
            r["version_index"] = 0
            fh_out.write(json.dumps(r) + "\n")
            fh_out.flush()
        print("   replication control done", flush=True)
    fh_out.close()
    print("\nwrote %s" % a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
