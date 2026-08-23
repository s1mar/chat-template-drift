"""Render a conversation through a historical chat template, the way `transformers` would.

Why this file is the riskiest one in the study. Every behavioural claim rests on "template A and
template B render this conversation differently". If our renderer differs from the reference
implementation in any detail (block trimming, the sandbox, a missing filter), it will report
divergence that no real user would ever see, and it will do so in the direction that flatters the
hypothesis. So this module deliberately mirrors `transformers._compile_jinja_template` rather than
being written from scratch, and `validate.py` refuses to let the study proceed unless our output is
byte-identical to `apply_chat_template` on every probe conversation of every repository we can load.

The probe set is fixed in notes/preregistration.md and must not be extended after seeing results.
"""
from __future__ import annotations

import json
from typing import Any

import jinja2
import jinja2.ext  # MUST be explicit: `import jinja2` alone does not bind the `ext` submodule.
from jinja2.sandbox import ImmutableSandboxedEnvironment

# The six probe conversations, frozen. Index order is stable and is used as an id everywhere.
PROBES: list[dict] = [
    {"id": "user_only",
     "messages": [{"role": "user", "content": "Write a function that adds two numbers."}],
     "add_generation_prompt": True},
    {"id": "system_user",
     "messages": [{"role": "system", "content": "You are a terse assistant."},
                  {"role": "user", "content": "Write a function that adds two numbers."}],
     "add_generation_prompt": True},
    {"id": "multi_turn",
     "messages": [{"role": "system", "content": "You are a terse assistant."},
                  {"role": "user", "content": "Write a function that adds two numbers."},
                  {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
                  {"role": "user", "content": "Now make it handle strings."}],
     "add_generation_prompt": True},
    {"id": "genprompt_true",
     "messages": [{"role": "user", "content": "Reply with the single word ready."}],
     "add_generation_prompt": True},
    {"id": "genprompt_false",
     "messages": [{"role": "user", "content": "Reply with the single word ready."}],
     "add_generation_prompt": False},
    {"id": "contract",
     "messages": [{"role": "system",
                   "content": "Always reply with exactly one fenced python code block and no prose."},
                  {"role": "user", "content": "Write a function that reverses a list."}],
     "add_generation_prompt": True},
]


class TemplateError(RuntimeError):
    pass


def _raise_exception(message):
    raise jinja2.exceptions.TemplateError(message)


def _tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
    return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators,
                      sort_keys=sort_keys)


def _strftime_now(fmt: str) -> str:
    """Templates that stamp a date would make every render non-reproducible.

    A fixed instant is returned instead of the wall clock, so that two renders of the SAME template
    can never differ for a reason unrelated to the template. Any pair that differs only by this
    value would otherwise be scored as drift.
    """
    import datetime
    return datetime.datetime(2026, 1, 1, 0, 0, 0).strftime(fmt)


def _env() -> ImmutableSandboxedEnvironment:
    # These options mirror transformers' own compiler. They are not a stylistic choice: with
    # trim_blocks/lstrip_blocks off, almost every template renders with extra newlines and the whole
    # frame would look render-divergent.
    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = _raise_exception
    env.globals["strftime_now"] = _strftime_now
    return env


def render(template_src: str, messages: list, add_generation_prompt: bool,
           bos_token: str = "", eos_token: str = "", extra: dict[str, Any] | None = None) -> str:
    if not template_src:
        raise TemplateError("empty template")
    try:
        tpl = _env().from_string(template_src)
    except Exception as exc:  # noqa: BLE001
        raise TemplateError("compile: %s" % exc) from exc
    kw = {"messages": messages, "add_generation_prompt": add_generation_prompt,
          "bos_token": bos_token or "", "eos_token": eos_token or ""}
    if extra:
        kw.update(extra)
    try:
        return tpl.render(**kw)
    except Exception as exc:  # noqa: BLE001
        raise TemplateError("render: %s" % exc) from exc


def render_probes(template_src: str, bos_token: str = "", eos_token: str = "") -> dict:
    """Render every probe. A probe the template refuses is recorded as its error, not skipped.

    A template that raises on a system message is not 'missing data'; it is a template that has
    stopped accepting system prompts, which is one of the behavioural changes we are looking for.
    """
    out = {}
    for p in PROBES:
        try:
            out[p["id"]] = {"ok": True,
                            "text": render(template_src, p["messages"], p["add_generation_prompt"],
                                           bos_token, eos_token)}
        except TemplateError as exc:
            out[p["id"]] = {"ok": False, "text": None, "error": str(exc)[:300]}
    return out


def compare_probes(a: dict, b: dict) -> dict:
    """Classify a template pair probe by probe.

    Three outcomes per probe, and keeping them apart is the whole point:

      same        both rendered, byte-identical
      differing   both rendered and differ, OR one rendered and the other refused. A template that
                  stops accepting a conversation shape has changed behaviour in the strongest way
                  available, so that asymmetry is a difference, not missing data.
      undecidable BOTH refused to render. This is absence of evidence and is reported as such.

    An earlier version folded `undecidable` into `same`, on the reasoning that it was the
    conservative choice. It was not conservative, it was blinding: a missing import made every
    render fail, every probe became both-failed, and the analysis reported 0 behavioural changes out
    of 289 with no error anywhere. A category that can absorb total instrument failure and still
    look like a clean null must not exist.
    """
    same, differing, undecidable = [], [], []
    for p in PROBES:
        ra, rb = a.get(p["id"], {}), b.get(p["id"], {})
        oka, okb = bool(ra.get("ok")), bool(rb.get("ok"))
        if not oka and not okb:
            undecidable.append(p["id"])
        elif oka != okb:
            differing.append(p["id"])
        elif ra.get("text") != rb.get("text"):
            differing.append(p["id"])
        else:
            same.append(p["id"])
    return {"same": same, "differing": differing, "undecidable": undecidable,
            "comparable": len(same) + len(differing),
            "behavioural": len(differing) > 0,
            "decidable": (len(same) + len(differing)) > 0}
