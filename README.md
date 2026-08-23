# Silent chat-template drift in open-weight model repositories

Replication package. Every number in the paper is produced by `code/make_macros.py` from the
analysis JSON in `data/`, written into `paper/macros.tex`, and used in the manuscript as a macro.
No number is typed by hand, and `code/consistency.py` fails if one in the text stops matching the
analysis output.

## Quick start

On Windows, clone with `git clone -c core.longpaths=true <url>`: two files under `data/history/`
carry repository names long enough to exceed the default 260-character path limit, and a plain
clone fails at checkout with "Filename too long".

If more than one Python is on `PATH`, install and run with the same interpreter: the smoke test
looks for `python3` before `python`, so if `pip` installed into `python`, run
`PYTHON=python bash scripts/smoke_test.sh`.

    pip install -r requirements.txt
    bash scripts/smoke_test.sh

The script runs an offline part and a network part, and reports which of the two ran. The offline
part regenerates every macro and both tables from the analysis JSON and runs the build gate. The
network part checks the renderer against `transformers.apply_chat_template`; it needs the
`transformers` package and downloads tokenizers. If `transformers` is missing, that part is skipped
and the summary says so.

No GPU is needed at any point.

## Reproducing each result

| result in the paper | command | inputs |
|---|---|---|
| RQ1: drift rates, behavioural split, weight silence | `python code/analyze_rq1.py` | `data/history/` |
| RQ2: execution outcomes, McNemar, bootstrap, Holm | `python code/analyze_rq2.py` | `data/exec.jsonl` |
| RQ2 system-free control | `python code/analyze_rq2.py --exec data/exec_systemfree.jsonl --out data/rq2_systemfree.json` | `data/exec_systemfree.jsonl` |
| RQ3: alert volume of the check | `python code/rq3_replay.py --no-bench` | `data/rq1.json` |
| RQ3 including a fresh cost measurement | `python code/rq3_replay.py` | `data/rq1.json`, network, `transformers` |
| transport check (tokenization) | `python code/check_tokenization.py` | needs Ollama and network; **writes** `data/tokenization.json` |
| renderer equals the reference implementation | `python code/validate_renderer.py` | network |
| the same on historical revisions | `python code/validate_historical.py` | network |
| all macros, tables, figure | `python code/make_macros.py && python code/tables.py && python code/figures.py` | the JSON above |
| the PDF | `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main` | LaTeX |
| the gates | `python code/consistency.py`, then `selftest.py`, then `render_check.py` | build the PDF first |

Commands are written for a POSIX shell. Windows PowerShell 5.1 has no `&&`, so run chained commands
one at a time there.

`render_check.py` reads the built PDF and `paper/main.log`. Neither ships in this package, so build
the PDF before running it or it reports that it cannot find the page count.

Build the PDF before `consistency.py`, or its staleness check fires on macros you just regenerated.

## What reproduces exactly, and what does not

`analyze_rq1.py`, `analyze_rq2.py` and `rq3_replay.py --no-bench` regenerate `data/rq1.json`,
`data/rq2.json`, `data/rq2_systemfree.json` and `data/rq3.json` **byte-identically** to the copies
shipped here. That has been checked by re-running all four in a clean copy of this package and
diffing the results.

The exception is the two cost fields in `data/rq3.json`, `cost_seconds_per_model` and
`cost_ms_compute_only`. They are wall-clock timings, so they move with hardware and with load: on a
loaded machine the compute figure measured four times the published one. Every other field in that
file is deterministic. `--no-bench` skips the timing and carries the published values through, which
is why the reproduce table above uses it. Run without the flag to take your own measurement.

`check_tokenization.py` is the one step with no offline path. It queries a running Ollama server for
the token count of a string it also tokenizes locally, so it cannot replay from a cache the way the
execution study does. Its result ships as `data/tokenization.json`, and `make_macros.py` reads the
two values the paper uses from that file. Re-running it needs Ollama and the checkpoints in
THIRD_PARTY.md.

### Re-running the execution study

    python code/run_exec.py --n-tasks 60 --num-predict 1536

This replays from `data/raw_cache.sqlite`, which holds every generation, so it needs no GPU and no
Ollama. Delete the cache to run the models for real; that needs Ollama and the checkpoints listed in
THIRD_PARTY.md.

### Re-running the mining

    python code/frame.py --n 1000
    python code/collect.py --workers 6

This re-draws the frame from the Hub and re-clones; it needs network access, and `frame.py`
refuses to overwrite an existing `data/candidates.json` (delete it first to force a fresh draw).
Download counts change over time, so a fresh
draw will not reproduce the exact frame; `data/candidates.json` and `data/history/` are the frame
this paper used.

## Layout

    code/     collection, rendering, both analyses, the tool, and the gates
    data/     the frame, 1000 template histories, analysis outputs, episode logs, the replay cache
    notes/    the pre-registration and its amendments; the citation record
    paper/    manuscript sources
    scripts/  smoke_test.sh
    MANIFEST.sha256

## Gates

`consistency.py` traces every number from the analysis output to each place it appears in the
manuscript, and also checks retired claims, venue facts, and the sources for control characters.
`selftest.py` injects every defect class listed in its `CASES` table into a copy of the tree and
fails unless the gate catches each one; the table is the authoritative count, so a class added
later is exercised without this sentence going stale.
`render_check.py` reads the built PDF rather than the build log, checking the
page budget, unresolved cross-references, and macros that leaked as literal text.

## Scope

Template histories are reconstructed from public default-branch commit graphs. Model weights are Git
LFS objects and were never fetched. Nothing here observes a hosted commercial API. The paper makes
no claim that any maintainer acted wrongly.

## Paper

This is the replication package for:

> Simarjot Khanna. Your Prompt Is Only Half the Prompt: Silent Chat-Template Drift in Open-Weight
> Model Repositories. PROMPTOPS 2026 (workshop at ASE 2026), Munich, Germany.

The package contains no home-directory path or session identifier. Sandbox tracebacks are redacted
where they are captured, in `code/sandbox.py`.
