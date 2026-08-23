"""templatelock: pin a model's chat template, and tell a build whether a change actually matters.

The problem this solves. A prompt pipeline versions its prompt text and pins its model by name, and
still has an unversioned executable dependency: the chat template that rewrites every conversation
before the model sees it. It is edited in place, on the default branch, with no version of its own.

The naive fix is to hash the template and fail the build when the hash moves. That works and it cries
wolf, because roughly half of real template edits are reformatting that renders identically. An alert
channel that is mostly noise gets muted, and then it is not a control any more.

So this tool reports two levels:

  CHANGED       the template string moved. Informational.
  BEHAVIOURAL   the template renders at least one probe conversation differently, so what the model
                is shown has changed. This is the one that should fail a build.

Usage:
  python templatelock.py pin    --model <repo-or-path> --out prompts.lock
  python templatelock.py verify --model <repo-or-path> --lock prompts.lock

Exit codes: 0 unchanged or cosmetic; 1 BEHAVIOURAL change; 2 could not read a template.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import render  # noqa: E402


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def render_signature(template: str, bos: str = "", eos: str = "") -> tuple[str, dict]:
    """A hash of what the template PRODUCES, not of what it says.

    Two templates with the same render signature cannot make the model behave differently, however
    different their source looks. This is the whole basis for separating a cosmetic edit from one
    that should stop a release.
    """
    probes = render.render_probes(template, bos, eos)
    payload = {k: (v["text"] if v["ok"] else "!ERR:" + (v.get("error") or "")[:80])
               for k, v in probes.items()}
    return sha(json.dumps(payload, sort_keys=True)), probes


def load_template(model: str) -> tuple[str, str, str]:
    """Read a chat template from a local tokenizer_config.json or from the Hub."""
    if os.path.isdir(model):
        path = os.path.join(model, "tokenizer_config.json")
    elif os.path.isfile(model):
        path = model
    else:
        path = None
    if path:
        with open(path, encoding="utf-8") as fh:
            obj = json.load(fh)
        t = obj.get("chat_template")
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        t = getattr(tok, "chat_template", None)
        return (t or "", tok.bos_token or "", tok.eos_token or "")
    if isinstance(t, list):
        t = json.dumps(t, sort_keys=True)

    def tokv(v):
        return v.get("content") if isinstance(v, dict) else (v if isinstance(v, str) else "")

    return (t or "", tokv(obj.get("bos_token")), tokv(obj.get("eos_token")))


def entry_for(model: str) -> dict:
    t, bos, eos = load_template(model)
    if not t:
        raise SystemExit("no chat template found for %s" % model)
    rsig, _ = render_signature(t, bos, eos)
    return {"model": model, "template_sha256": sha(t), "render_signature": rsig,
            "probe_set": [p["id"] for p in render.PROBES]}


def cmd_pin(a) -> int:
    lock = {}
    if os.path.exists(a.out):
        with open(a.out, encoding="utf-8") as fh:
            lock = json.load(fh)
    lock[a.model] = entry_for(a.model)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(lock, fh, indent=1, sort_keys=True)
    print("pinned %s\n  template  %s\n  renders as %s"
          % (a.model, lock[a.model]["template_sha256"][:16],
             lock[a.model]["render_signature"][:16]))
    return 0


def cmd_verify(a) -> int:
    with open(a.lock, encoding="utf-8") as fh:
        lock = json.load(fh)
    if a.model not in lock:
        print("NOT PINNED: %s is not in %s" % (a.model, a.lock))
        return 2
    want, got = lock[a.model], entry_for(a.model)
    if want["template_sha256"] == got["template_sha256"]:
        print("OK        %s: chat template unchanged" % a.model)
        return 0
    if want["render_signature"] == got["render_signature"]:
        print("COSMETIC  %s: the chat template text changed but renders identically on all %d "
              "probe conversations.\n          Pinned %s -> now %s. No behavioural change; update "
              "the lock when convenient."
              % (a.model, len(render.PROBES), want["template_sha256"][:16],
                 got["template_sha256"][:16]))
        return 0
    print("BEHAVIOURAL %s: the chat template now renders DIFFERENTLY.\n"
          "            Pinned template %s -> %s\n"
          "            Pinned rendering %s -> %s\n"
          "            What the model is shown has changed. Re-run your prompt evaluations before "
          "shipping." % (a.model, want["template_sha256"][:16], got["template_sha256"][:16],
                         want["render_signature"][:16], got["render_signature"][:16]))
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pin"); p.add_argument("--model", required=True)
    p.add_argument("--out", default="prompts.lock"); p.set_defaults(fn=cmd_pin)
    v = sub.add_parser("verify"); v.add_argument("--model", required=True)
    v.add_argument("--lock", default="prompts.lock"); v.set_defaults(fn=cmd_verify)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
