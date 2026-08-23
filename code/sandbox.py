"""Run a candidate program against a test block in a child process with a wall-clock cap.

The child is started in a scratch directory so that a program which writes files does not touch the
repository. Model-generated code is executed, which is the standard practice for HumanEval-family
benchmarks, and the same caveat applies here: run this only on a machine you are willing to have
untrusted Python execute on.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import uuid

TIMEOUT_VISIBLE = 30
TIMEOUT_HIDDEN = 90
SCRATCH = os.environ.get("POVC_SCRATCH") or os.path.join(tempfile.gettempdir(), "povc_exec")

PASS = "pass"
FAIL = "fail"
TIMEOUT = "timeout"

_FRAME_RE = re.compile(r'File "([^"]+)", line (\d+)')


def run_program(program: str, timeout: int = TIMEOUT_VISIBLE) -> dict:
    """Execute `program`. Returns status, the child's stderr, and the exit code."""
    os.makedirs(SCRATCH, exist_ok=True)
    path = os.path.join(SCRATCH, "cand_%s.py" % uuid.uuid4().hex)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(program)
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        proc = subprocess.run(
            [sys.executable, path], cwd=SCRATCH, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.TimeoutExpired:
        return {"status": TIMEOUT, "stderr": "", "returncode": None}
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    err = redact_paths(proc.stderr.decode("utf-8", "replace"))
    status = PASS if proc.returncode == 0 else FAIL
    return {"status": status, "stderr": err, "returncode": proc.returncode}


_PATH_RE = re.compile(r'(?:[A-Za-z]:\\\\?|/)[^\s"\']*?(cand_[0-9a-f]+\.py)')


def redact_paths(text: str) -> str:
    """Strip the absolute scratch path out of a child traceback.

    Every traceback names the temporary file it executed, and on Windows that path contains the
    user's home directory. Those tracebacks are stored verbatim in the episode logs, all of which
    ship with the artifact. Redacting at the point of capture is the only place that fixes every
    downstream consumer at once.
    """
    return _PATH_RE.sub(r"<scratch>/\1", text or "")


def feedback_from(result: dict, program: str, max_chars: int = 900) -> str:
    """Turn an execution result into the diagnostic text the harness shows the model.

    This is deliberately the ordinary output of the toolchain: the exception type, its message, and
    the source line the traceback blames. It is identical for every prompt variant, so the
    diagnostic channel is held fixed and only the prompt varies.
    """
    if result["status"] == PASS:
        return "All tests passed."
    if result["status"] == TIMEOUT:
        return "The test run did not finish within the time limit; the program appears to loop forever."
    err = result["stderr"].strip()
    if not err:
        return "The test run exited with a non-zero status and produced no output."
    lines = [ln for ln in err.splitlines() if ln.strip()]
    exc_line = lines[-1] if lines else ""
    # Locate the deepest frame inside the candidate file and quote that source line.
    blamed = ""
    src = program.splitlines()
    for m in _FRAME_RE.finditer(err):
        try:
            ln = int(m.group(2))
        except ValueError:
            continue
        if 1 <= ln <= len(src):
            blamed = src[ln - 1].strip()
    parts = ["The test run failed."]
    if blamed:
        parts.append("Failing line: %s" % blamed)
    parts.append(exc_line)
    tail = "\n".join(parts)
    return tail[:max_chars]


def build_program(prelude: str, candidate: str, test_block: str) -> str:
    chunks = [c for c in (prelude, candidate, test_block) if c and c.strip()]
    return "\n\n".join(chunks) + "\n"
