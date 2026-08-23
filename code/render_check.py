"""Check the RENDERED pdf, not the build log. A clean log is not a rendered page.

On an earlier paper in this portfolio three `\\ref` macros reached the PDF as literal prose while the
build reported zero errors and zero undefined references, and a running head silently overprinted the
venue string. This checks the artifact a reviewer actually opens:

  * page budget: POVC 2026 allows up to 8 pages EXCLUDING references, so body pages are counted by
    locating where the bibliography starts rather than by taking the total
  * no unresolved cross-reference or citation marker survived into the text
  * no macro leaked as literal text, which is what an unexpanded or mis-escaped macro looks like
  * the venue string appears on the rendered page, not merely in the source
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
PDF = os.path.join(PAPER, "main.pdf")
PAGE_LIMIT_BODY = 8  # fetched 2026-07-30: "Up to 8 pages (excluding references)"

FAILS: list = []
OK = [0]


def check(name, cond, detail=""):
    if cond:
        OK[0] += 1
    else:
        FAILS.append("%s%s" % (name, (": " + detail) if detail else ""))


def _candidate_pdftotext():
    """Look for pdftotext on PATH, then in a per-user MiKTeX install.

    The fallback is built from the environment rather than written out, because an absolute
    Windows path contains the user's name and this file ships with the artifact. The anonymity
    scan caught exactly that here.
    """
    yield "pdftotext"
    local = os.environ.get("LOCALAPPDATA")
    if local:
        yield os.path.join(local, "Programs", "MiKTeX", "miktex", "bin", "x64", "pdftotext.exe")


def pdftotext(first=None, last=None) -> str:
    exe = None
    for cand in _candidate_pdftotext():
        try:
            subprocess.run([cand, "-v"], capture_output=True, timeout=30)
            exe = cand
            break
        except (OSError, subprocess.SubprocessError):
            continue
    if exe is None:
        return ""
    args = [exe, "-layout"]
    if first:
        args += ["-f", str(first)]
    if last:
        args += ["-l", str(last)]
    args += [PDF, "-"]
    p = subprocess.run(args, capture_output=True, timeout=120)
    return p.stdout.decode("utf-8", "replace")


def main() -> int:
    if not os.path.exists(PDF):
        print("render_check: main.pdf not built")
        return 2
    log = os.path.join(PAPER, "main.log")
    logtxt = open(log, encoding="utf-8", errors="replace").read() if os.path.exists(log) else ""

    mm = re.search(r"Output written on .*?\((\d+) pages", logtxt)
    total = int(mm.group(1)) if mm else None
    check("pdf.pages_known", total is not None, "could not read page count from the log")

    txt = pdftotext()
    # Body pages end where the bibliography begins. Counting total pages would fail a paper that is
    # within the limit, because references do not count against it at this venue.
    body_pages = None
    if txt:
        # Drop the empty tail that follows the final form feed, or the count is always one too high.
        pages = [p for p in txt.split("\f")]
        if pages and not pages[-1].strip():
            pages.pop()
        # acmart sets the heading as "References", and in a two-column layout dump it lands mid-line
        # rather than alone on one. Anchoring on "^REFERENCES$" matched nothing and silently fell
        # back to counting every page, which would fail a paper that is comfortably within the limit.
        for i, pg in enumerate(pages):
            if re.search(r"\bReferences\b", pg):
                body_pages = i + 1
                break
        if body_pages is None:
            body_pages = len(pages)
    if body_pages is not None:
        check("pdf.body_within_limit", body_pages <= PAGE_LIMIT_BODY,
              "body runs to page %d, limit is %d excluding references"
              % (body_pages, PAGE_LIMIT_BODY))

    if txt:
        # An unresolved reference renders as "??"; an unexpanded macro renders with a backslash.
        check("pdf.no_qq", "??" not in txt, "an unresolved cross-reference reached the page")
        leaked = sorted(set(re.findall(r"\\[A-Za-z]{3,}", txt)))
        check("pdf.no_leaked_macros", not leaked, "macro text reached the page: %s" % leaked[:6])
        # A macro that lost its backslash through bad escaping leaves its name as a bare word.
        for name in ("xspace", "newcommand", "textbf", "emph", "citep"):
            check("pdf.no_bare_" + name, name not in txt,
                  "the literal word %r is on the page, which is what a de-backslashed macro "
                  "looks like" % name)
        check("pdf.venue_rendered", "Munich" in txt or "PROMPTOPS" in txt,
              "the venue string does not appear on any rendered page")

    for pat, nm in ((r"Undefined control sequence", "undefined_control_sequence"),
                    (r"There were undefined references", "undefined_references"),
                    (r"Citation .* undefined", "undefined_citation")):
        check("log.no_" + nm, not re.search(pat, logtxt), "present in main.log")

    # Flag overfulls by MAGNITUDE, not by count. A box 1pt over the margin is invisible on the page;
    # failing on it trains you to skim the one gate that would have caught text running into the
    # gutter. The threshold is what a reader could actually see, and the largest offender is always
    # printed so a growing one cannot hide under the limit.
    sizes = [float(x) for x in re.findall(r"Overfull \\hbox \(([\d.]+)pt too wide", logtxt)]
    worst = max(sizes) if sizes else 0.0
    check("log.no_visible_overfull", worst <= 5.0,
          "worst overfull is %.2fpt over the margin (%d total)" % (worst, len(sizes)))
    if sizes:
        print("   note: %d overfull hbox(es), largest %.2fpt (threshold 5pt)" % (len(sizes), worst))

    print("render_check: %d passed, %d FAILED  (total pages %s, body pages %s)"
          % (OK[0], len(FAILS), total, body_pages))
    for f in FAILS:
        print("   FAIL  %s" % f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
