"""Validate the build gate by INJECTING defects and confirming it catches every one.

Reading a gate tells you what it was meant to check. Injection tells you what it does check. On
earlier papers in this portfolio, 16 of 43 injected defects passed a gate that had been read and
believed, and every miss failed OPEN: a clean run with a real defect present.

Two rules this harness follows, both learned the hard way:

  * It works on a COPY of the tree, made fresh, so a live file can never be damaged and so a stale
    copy cannot silently test yesterday's manuscript.
  * An injection that does not APPLY is a harness failure, not a curiosity. If the anchor text has
    moved, the case is reported as ERROR, never as "caught" and never as "missed".
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _apply(path: str, old: str, new: str) -> bool:
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    if old not in s:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s.replace(old, new, 1))
    return True


def _apply_all(path: str, old: str, new: str) -> bool:
    # For defects that must hit EVERY site: the artifact URL appears in a page-1 footnote and in
    # Data availability, and losing one of the two is not the defect the gate guards against.
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    if old not in s:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s.replace(old, new))
    return True


def _append(path: str, text: str) -> bool:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("\n" + text + "\n")
    return True


def _regex_sub(path: str, pattern: str, repl: str, flags=0) -> bool:
    with open(path, encoding="utf-8") as fh:
        s = fh.read()
    s2, n = re.subn(pattern, repl, s, count=1, flags=flags)
    if n == 0:
        return False
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(s2)
    return True


# Each case: (id, what it simulates, mutate(tmpdir) -> bool applied, substring expected in output)
CASES = [
    ("stale_macro",
     "a number in the paper no longer matches the analysis output after a re-run",
     lambda d: _regex_sub(os.path.join(d, "paper", "macros.tex"),
                          r"\\newcommand\{\\PctDrifted\}\{[\d.]+",
                          r"\\newcommand{\\PctDrifted}{99.9"),
     "macro.value.PctDrifted"),

    ("hand_typed_number",
     "a number typed into the prose instead of coming from the analysis",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "In our frame, 37.4\\% of repositories did something."),
     "no_hand_typed_numbers"),

    ("undefined_macro",
     "prose referring to a macro the pipeline never generated",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "The value was \\PctNeverGenerated{} overall."),
     "macro.defined.PctNeverGenerated"),

    # RETIRED 2026-08-31 with the clause it validated. It injected "International Workshop on" and
    # expected venue.not_international_workshop to fire; ACM's rights block then supplied a official
    # proceedings title that CONTAINS that phrase, so the clause was retired and
    # this case with it. Its replacement is venue_title_altered below, which pins the ACM strings.
    ("venue_title_altered",
     "the ACM-assigned proceedings title reworded by hand",
     lambda d: _apply(os.path.join(d, "paper", "main.tex"),
                      "Proceedings of the 1st International Workshop on PromptOps and Vibe Coding "
                      "(POVC '26)",
                      "Proceedings of the 1st Intl. Workshop on PromptOps and Vibe Coding "
                      "(POVC '26)"),
     "venue.acm_booktitle"),

    ("venue_doi_altered",
     "one digit changed in the ACM-assigned DOI, which is unrecoverable after publication",
     lambda d: _apply(os.path.join(d, "paper", "main.tex"),
                      r"\acmDOI{10.1145/3843779.3844636}",
                      r"\acmDOI{10.1145/3843779.3844616}"),
     "venue.doi"),

    # Inject into the LIVE \acmConference line, not the first textual match. The first occurrence of
    # "Munich, Germany" is inside the preamble comment that documents the fetched venue facts, and
    # mutating a comment changes nothing in the rendered paper, so the gate was right to stay silent
    # and the harness was wrong to expect a failure.
    ("wrong_city",
     "a venue city carried over from a different paper",
     lambda d: _regex_sub(os.path.join(d, "paper", "main.tex"),
                          r"(\\acmConference.*?)\{Munich, Germany\}",
                          r"\1{Seoul, South Korea}", re.DOTALL),
     "venue.city"),

    ("overclaim_all_edits",
     "claiming every template edit changes behaviour, which the data refutes",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "We conclude that every template edit changes behaviour."),
     "negative.no_all_edits_matter"),

    ("blame_maintainers",
     "blaming maintainers, which the paper explicitly disclaims",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "This shows maintainers are careless with their templates."),
     "negative.no_maintainer_blame"),

    ("hosted_api_claim",
     "extending the claim to hosted APIs, which this method cannot observe",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "The same is true where hosted commercial APIs change their templates too."),
     "negative.no_hosted_api_claim"),

    ("placeholder_survives",
     "a scaffolding marker left in the manuscript",
     lambda d: _append(os.path.join(d, "paper", "body.tex"), "PLACEHOLDER for the ablation."),
     "placeholder.PLACEHOLDER"),

    # The two cases below reproduce the corruption that a shell one-liner actually caused in this
    # project: "\S\ref{...}" became "\S", a carriage return, and "ef{...}". LaTeX reported zero
    # errors and every other gate passed, because \S is valid and "ef" is just letters.
    ("eaten_backslash_r",
     "a shell escape turning a macro into a control character plus bare letters",
     lambda d: _regex_sub(os.path.join(d, "paper", "body.tex"),
                          r"Section~\\ref\{sec:mechanism\}", "Section~\\\\S\ref{sec:mechanism}"),
     "integrity.no_control_chars"),

    ("debackslashed_macro",
     "a macro that lost its leading backslash and now renders as literal text",
     lambda d: _append(os.path.join(d, "paper", "body.tex"),
                       "See Section~ef{sec:mechanism} for the mechanism."),
     "integrity.no_debackslashed"),

    # Final-version inversion (accepted 2026-08-23): the submission-era injection dropped
    # [review,anonymous] and expected venue.anonymous to fire. The final version is the
    # opposite: restoring those class options is the defect. Two further defects are injected:
    # losing the author block, and losing the artifact URL (an availability claim with no
    # reachable link is a documented rejection ground).
    ("reintroduced_anonymity",
     "the camera-ready regaining review/anonymous options",
     lambda d: _apply(os.path.join(d, "paper", "main.tex"),
                      "[sigconf,screen]", "[sigconf,review,anonymous]"),
     "venue.cr_not_anonymous"),

    ("dropped_author_block",
     "the camera-ready losing its author block",
     lambda d: _apply(os.path.join(d, "paper", "main.tex"),
                      r"\author{Simarjot Khanna}", r"\author{Anonymous Author}"),
     "venue.cr_author_block"),

    ("dropped_artifact_url",
     "the camera-ready losing the live artifact URL",
     lambda d: _apply_all(os.path.join(d, "paper", "body.tex"),
                          "github.com/s1mar/chat-template-drift", "example.invalid/gone"),
     "artifact.url_present"),
]


def run_gate(d: str) -> tuple[int, str]:
    p = subprocess.run([PY, os.path.join(d, "code", "consistency.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def fresh_copy() -> str:
    d = tempfile.mkdtemp(prefix="povc_selftest_")
    for sub in ("paper", "code", "data"):
        shutil.copytree(os.path.join(ROOT, sub), os.path.join(d, sub),
                        ignore=shutil.ignore_patterns("__pycache__", "*.sqlite", "history"))
    # history/ is large and the gate does not read it; rq1.json and rq3.json are what matter.
    os.makedirs(os.path.join(d, "notes"), exist_ok=True)
    for f in ("retired_claims.json",):
        src = os.path.join(ROOT, "notes", f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(d, "notes", f))
    return d


def main() -> int:
    base = fresh_copy()
    rc, out = run_gate(base)
    baseline_fails = [l.strip() for l in out.splitlines() if "FAIL" in l]
    print("baseline on an unmodified copy: exit=%d, %d failing checks" % (rc, len(baseline_fails)))
    for f in baseline_fails:
        print("   (pre-existing) %s" % f)
    shutil.rmtree(base, ignore_errors=True)

    caught, missed, errors = [], [], []
    for cid, what, mutate, expect in CASES:
        d = fresh_copy()
        try:
            applied = mutate(d)
            if not applied:
                errors.append((cid, "INJECTION DID NOT APPLY: anchor text has moved"))
                continue
            rc, out = run_gate(d)
            hit = expect in out and "FAIL" in out
            if hit:
                caught.append(cid)
            else:
                missed.append((cid, what, expect))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    print("\n%d injected defects: %d caught, %d MISSED, %d harness errors"
          % (len(CASES), len(caught), len(missed), len(errors)))
    for cid in caught:
        print("   caught  %s" % cid)
    for cid, what, expect in missed:
        print("   MISSED  %-22s (%s) expected check %s to fire" % (cid, what, expect))
    for cid, why in errors:
        print("   ERROR   %-22s %s" % (cid, why))

    return 1 if (missed or errors) else 0


if __name__ == "__main__":
    raise SystemExit(main())
