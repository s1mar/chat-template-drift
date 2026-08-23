"""Ollama client that sends an ALREADY-RENDERED prompt string, bypassing Ollama's own template.

This is what makes the experiment clean. Ollama ships its own chat template per model and would
apply it to a message list, which is precisely the variable under study. `/api/generate` with
`"raw": true` sends the string through untouched, so the conversation, the checkpoint, the decoding
parameters and the seed can be held identical while the historical chat template is the only thing
that differs between arms.

Every generation is cached under a key derived from (model, prompt, options), so a re-run is free and
the analysis is reproducible from the cache alone. Token counts are recorded on every call, so an
episode that hit the output cap is identifiable rather than being scored as a difference in content.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request

OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CACHE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "raw_cache.sqlite")

_lock = threading.Lock()
_conn = None


def _db():
    global _conn
    if _conn is None:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        _conn = sqlite3.connect(CACHE_PATH, check_same_thread=False)
        _conn.execute(
            "CREATE TABLE IF NOT EXISTS gen ("
            " k TEXT PRIMARY KEY, model TEXT, prompt TEXT, response TEXT,"
            " prompt_tokens INTEGER, completion_tokens INTEGER, done_reason TEXT,"
            " wall_s REAL, ts REAL)")
        _conn.commit()
    return _conn


def cache_key(model: str, prompt: str, options: dict, stop: list, replicate: int = 0) -> str:
    """`replicate` enters the CACHE key but is never sent to the model.

    It exists so that the same prompt can be generated a second time under identical settings, which
    is how the study measures its own noise floor. Without it the cache would return the first
    reply and the replication arm would report zero divergence by construction: a negative control
    that can only pass is not a control.
    """
    blob = json.dumps({"m": model, "p": prompt, "o": options, "s": stop, "r": replicate},
                      sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Truncation(RuntimeError):
    pass


def generate(model: str, prompt: str, temperature: float = 0.0, seed: int = 1,
             num_ctx: int = 8192, num_predict: int = 1024, stop: list | None = None,
             replicate: int = 0, use_cache: bool = True, retries: int = 3) -> dict:
    stop = list(stop or [])
    options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx,
               "num_predict": num_predict}
    k = cache_key(model, prompt, options, stop, replicate)
    if use_cache:
        with _lock:
            row = _db().execute(
                "SELECT response, prompt_tokens, completion_tokens, done_reason, wall_s"
                " FROM gen WHERE k=?", (k,)).fetchone()
        if row is not None:
            return {"text": row[0], "prompt_tokens": row[1], "completion_tokens": row[2],
                    "done_reason": row[3], "wall_s": row[4], "cached": True,
                    "truncated": row[2] is not None and row[2] >= num_predict}

    payload = {"model": model, "prompt": prompt, "raw": True, "stream": False,
               "options": dict(options, stop=stop) if stop else options}
    body = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            req = urllib.request.Request(OLLAMA + "/api/generate", data=body,
                                         headers={"Content-Type": "application/json"},
                                         method="POST")
            with urllib.request.urlopen(req, timeout=1800) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            break
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last = exc
            time.sleep(5 * (attempt + 1))
    else:
        raise RuntimeError("ollama raw failed after %d attempts: %r" % (retries, last))

    wall = time.time() - t0
    text = obj.get("response", "") or ""
    pt = int(obj.get("prompt_eval_count") or 0)
    ct = int(obj.get("eval_count") or 0)
    if pt >= num_ctx - 8:
        raise Truncation("prompt_eval_count=%d filled num_ctx=%d for %s" % (pt, num_ctx, model))
    with _lock:
        _db().execute("INSERT OR REPLACE INTO gen VALUES (?,?,?,?,?,?,?,?,?)",
                      (k, model, prompt, text, pt, ct, obj.get("done_reason"), wall, time.time()))
        _db().commit()
    return {"text": text, "prompt_tokens": pt, "completion_tokens": ct,
            "done_reason": obj.get("done_reason"), "wall_s": wall, "cached": False,
            "truncated": ct >= num_predict}
