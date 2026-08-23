#!/usr/bin/env bash
# Offline checks first, then the checks that need transformers and the network.
#
# pipefail is set because a pipeline returns its last command's status: `cmd | tail` exits 0
# whatever cmd did.
set -euo pipefail

# Quoted throughout: an unquoted expansion splits on a Python path containing spaces.
# `python` is not present on every system, so fall back rather than dying on the first line.
if [ -n "${PYTHON:-}" ]; then
  PY="$PYTHON"
else
  PY=""
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
  if [ -z "$PY" ]; then
    echo "No python found. Set PYTHON to your interpreter, for example:" >&2
    echo "  PYTHON=/usr/bin/python3 bash scripts/smoke_test.sh" >&2
    exit 1
  fi
fi
echo "python: $("$PY" --version 2>&1)  [$PY]"
echo

echo "== offline 1/3: regenerate every macro from the analysis JSON =="
"$PY" code/make_macros.py

echo "== offline 2/3: regenerate Table 1 and Table 2 =="
"$PY" code/tables.py

echo "== offline 3/3: build gate =="
"$PY" code/consistency.py

echo
RENDERER_RAN=no
if "$PY" -c "import transformers" >/dev/null 2>&1; then
  echo "== network 1/2: renderer equals transformers.apply_chat_template =="
  "$PY" code/validate_renderer.py
  echo "== network 2/2: the same on the historical revisions =="
  "$PY" code/validate_historical.py
  RENDERER_RAN=yes
else
  echo "SKIPPED: transformers is not importable, so the renderer checks did not run."
  echo "These are the checks the behavioural claims rest on."
  echo "  pip install -r requirements.txt"
fi

echo
echo "----------------------------------------------------------"
echo "offline checks passed."
if [ "$RENDERER_RAN" = yes ]; then
  echo "renderer checks passed, current and historical templates."
else
  echo "renderer checks DID NOT RUN. This run does not validate the renderer."
fi
echo "----------------------------------------------------------"
