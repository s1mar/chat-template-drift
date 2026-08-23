# Third-party material

## Checkpoints used in RQ2

Run locally through Ollama. None are redistributed here.

Ollama tags are mutable, so the digest column records the exact builds the study ran (read from
the serving instance that produced the episode log); a replicator whose pull shows a different
digest is running different bytes.

| checkpoint | Ollama tag | digest (first 12) | Hub repository |
|---|---|---|---|
| Qwen2.5-7B-Instruct | `qwen2.5:7b-instruct` | `845dbda0ea48` | Qwen/Qwen2.5-7B-Instruct |
| Phi-3-mini-4k-instruct | `phi3:mini` | `4f2222927938` | microsoft/Phi-3-mini-4k-instruct |
| deepseek-coder-6.7b-instruct | `deepseek-coder:6.7b-instruct` | `ce298d984115` | deepseek-ai/deepseek-coder-6.7b-instruct |
| Mistral-7B-Instruct-v0.2 | `mistral:7b-instruct-v0.2-q4_0` | `61e88e884507` | mistralai/Mistral-7B-Instruct-v0.2 |
| Meta-Llama-3-8B-Instruct | `llama3:latest` | `365c0bd3c000` | NousResearch mirror, see below |

The official Llama-3 repository is gated and cannot be read anonymously, so its template history
comes from a public mirror of the same checkpoint. The paper states this in its threats section.

The local checkpoints are quantised rather than the Hub's full-precision weights. Quantisation is
held fixed across arms, so it cannot confound a within-checkpoint comparison, but the size of an
effect on full-precision weights may differ.

## Repository histories

`data/history/` holds one reconstructed default-branch chat-template history per candidate
repository, built with one blob-filtered single-branch clone each. Only text blobs from the commit
tree were read.

## Task corpus

`data/tasks.json` is drawn from the code-repair split of HumanEvalPack (Muennighoff et al.,
OctoPack, ICLR 2024). Each episode is graded on the task's own visible test.

## Reference implementation

The renderer is checked against `transformers.apply_chat_template`. See `code/validate_renderer.py`
and `code/validate_historical.py`.
