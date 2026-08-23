# Pre-registration: silent chat-template drift in open-weight model repositories

**Frozen 2026-07-30, before the study frame was collected and before any model was run under a
historical template.** Amendments are appended at the end with a reason and a timestamp; nothing
above the amendment section is edited afterwards. `code/crosscheck.py` compares every threshold that
appears in the manuscript against this file and fails the build on a mismatch.

## The question

A chat template is executable code. It lives inside the model repository, usually as the
`chat_template` field of `tokenizer_config.json`, and it rewrites every conversation into the exact
token string the model sees. A prompt pipeline that versions its prompt text, pins a model name, and
records temperature and seed still has this dependency, and does not version it.

The workshop's own topic list asks for prompt versioning, traceability, drift detection and
regression prevention. This study asks whether the drift is there to detect:

- **RQ1.** How often is a chat template edited after the model is released, on the default branch,
  and how often does that happen in an interval where no model-weight artifact changed at all?
- **RQ2.** Are those edits behaviour-affecting? Holding the checkpoint and the conversation fixed and
  varying ONLY the template, does the model's output change, and does a task outcome change?
- **RQ3.** What must a prompt pipeline record or check to catch this, and what does that cost?

## Frame and sampling rule

Fixed in advance so the frame cannot be tuned to the result.

- **Population.** Models on the Hugging Face Hub with `pipeline_tag = text-generation`, sorted by
  all-time downloads, descending.
- **Inclusion.** Public and ungated (an anonymous blob-filtered clone succeeds); the default branch
  carries at least one revision in which a chat template is present in any of
  `tokenizer_config.json`, `chat_template.jinja`, `chat_template.json`.
- **Exclusion, recorded with a reason, never silently dropped.** Gated or deleted repositories;
  repositories with no chat template in any revision (base models, not chat models); repositories
  whose clone fails for a transport reason, which are retried and only then recorded as failures.
- **Target size.** The first **400** repositories in download order that satisfy inclusion. If fewer
  than 400 qualify within the first 1,000 candidates, the frame is whatever qualified, and the
  realised size is reported.
- **Collection.** One blob-filtered clone per repository
  (`--filter=blob:limit=1m --no-checkout`, `GIT_LFS_SKIP_SMUDGE=1`), then the whole history is read
  locally. Weights are LFS objects and are never fetched.

**Frame freeze.** The frame is written once to `data/frame.jsonl` with the collection date and is
not re-drawn after any result is seen.

## Definitions

- **Template revision.** A default-branch commit at which the extracted chat-template string differs
  from the one at the immediately preceding commit that carried a template.
- **Distinct templates.** The number of distinct chat-template strings a repository exhibits over its
  observable default-branch history, in commit order.
- **Drifted repository.** One with at least two distinct templates, i.e. the template a user gets
  depends on when they pulled.
- **Post-release drift.** A template revision occurring strictly after the first commit in which a
  template appears.
- **Weight-silent interval.** An interval between consecutive template revisions in which no tracked
  model-artifact file (`*.safetensors`, `*.bin`, `*.gguf`, `*.pt`, `*.h5`, `*.msgpack`, and their
  index files) changed on the default branch. In such an interval the weights a user pulls are
  unchanged and only the prompt wrapper moved. This is the harm case, because a team that pinned the
  model by name and verified its weights would see nothing.
- **Cosmetic vs behavioural.** Decided by RENDERING, never by reading the Jinja source. Two templates
  are **render-equivalent** if, for every probe conversation in the fixed probe set below, they
  produce byte-identical rendered strings. Otherwise they are **render-divergent**. A template pair
  that is render-equivalent cannot change behaviour and is counted as cosmetic regardless of how
  large its textual diff is.

## Probe conversation set (fixed here, used for every pair)

1. single user turn
2. system + user
3. system + user + assistant + user (multi-turn)
4. user turn, `add_generation_prompt=True`
5. user turn, `add_generation_prompt=False`
6. system + user with a system message containing an output contract sentence

Rendering uses a sandboxed Jinja environment configured to match `transformers`' own
`apply_chat_template`: `ImmutableSandboxedEnvironment`, `trim_blocks` and `lstrip_blocks` off, a
`raise_exception` global, a `tojson` filter, and `bos_token`/`eos_token` supplied from the same
revision's tokenizer configuration.

**Renderer validation gate, run before any measurement is believed.** For every repository in the
frame whose HEAD template can be loaded by `transformers`, our renderer's output on the probe set
must equal `transformers.AutoTokenizer.apply_chat_template` byte for byte. Repositories where they
disagree are reported and excluded from render-based claims. A renderer that silently differs from
the reference implementation would manufacture divergence, which is the direction that favours our
hypothesis, so this gate is mandatory.

## RQ2: the execution-grounded half

- **Checkpoints.** Local Ollama checkpoints chosen because each corresponds to a mined repository:
  `llama3` (Meta-Llama-3-8B-Instruct), `qwen2.5:7b-instruct`, `phi3:mini`,
  `mistral:7b-instruct-v0.2`, `deepseek-coder:6.7b-instruct`. Quantisation differs from the Hub
  fp16 weights; it is held FIXED across arms, so it cannot confound a within-checkpoint comparison,
  and it is disclosed.
- **Arms.** For a checkpoint whose repository has T distinct render-divergent templates, one arm per
  template. The conversation, the decoding parameters, the seed and the checkpoint are identical
  across arms; the template is the only thing that varies. Prompts are delivered through Ollama's
  raw endpoint so that Ollama's own bundled template is bypassed.
- **Tasks.** A seeded sample of **60** single-function repair tasks drawn from the validated
  HumanEvalFix task set already built for this rig, plus the fixed probe set for contract measures.
- **Outcomes.**
  - **O1 (primary)** task resolution: the returned program passes the task's executable test.
  - **O2** output-contract compliance: the reply contains exactly one fenced Python block defining
    the expected entry point.
  - **O3** response divergence: the fraction of tasks whose reply differs between two arms.
- **Analysis.** Paired over tasks. McNemar's exact test for the paired binary outcomes, paired
  bootstrap (10,000 resamples, tasks are the resampling unit) for interval estimates. Multiplicity
  corrected with Holm across the family of template-pair comparisons, reported for BOTH the null
  family and the alternative family.

## Falsification criterion, committed in advance

The paper's claim is that chat-template drift is real, silent, and behaviour-affecting. It is
**falsified**, and must be reported as falsified, if either:

- **RQ1 fails:** fewer than 10% of frame repositories show post-release drift, OR post-release drift
  is essentially never weight-silent (under 10% of drifted repositories have a weight-silent
  interval). Then the dependency exists on paper but is not moving underneath anyone.
- **RQ2 fails:** among render-divergent template pairs, no comparison shows a task-resolution
  difference whose interval excludes zero after Holm correction, AND response divergence is under
  5%. Then templates are edited but the model does not care, and the honest headline is that this
  dependency is unversioned and it does not matter.

If RQ1 holds and RQ2 fails, the paper is a measurement paper with a bounded null on consequence, and
must say so in the abstract rather than implying harm.

## What will NOT be claimed

- No claim that any maintainer acted wrongly. Many template edits are corrections, and correcting a
  template is good practice; the point is that consumers cannot see it happen.
- No claim about hosted commercial APIs, which are not observable by this method.
- No claim that a behaviour change observed on a quantised local checkpoint has the same magnitude on
  the full-precision Hub weights.
- No causal claim about download counts or popularity driving drift.

---

# Amendments

Each amendment records what changed, why, and when relative to the data it could affect.

**A1 (2026-07-30, before any repository was analysed). Jinja environment options.** The design above
states the renderer runs with `trim_blocks` and `lstrip_blocks` off. That was a factual error about
the reference implementation: `transformers` compiles chat templates with both ON. The renderer is
implemented ON. The renderer-validation gate, not this document, is the arbiter: with both ON it
passes 7/7 byte-identical against `apply_chat_template`.

**A2 (2026-07-30, before RQ2 was run). A third comparison outcome, `undecidable`.** The design said a
pair is render-equivalent if every probe matches, counting "both templates failed to render" as a
match. A missing import made every render fail and the analysis reported 0 behavioural changes out of
289 with no error. Pairs where NEITHER template renders are now a separate `undecidable` category,
excluded from the cosmetic/behavioural split and reported in their own right. The analysis aborts if
no pair is decidable. This amendment can only shrink the paper's claim.

**A4 (2026-07-30, before RQ2 was run). Render-distinct deduplication of arms.** The design said one
arm per distinct template. Two textually different templates that render identically on every probe
cannot produce a behavioural difference, so running both would spend compute on a comparison whose
answer is known by construction and would dilute the family-wise correction. Arms are therefore
deduplicated by their rendered signature, oldest kept, and the count of arms per checkpoint is
reported.

**A3 (2026-07-30, during collection). Clones are `--single-branch`.** 28 repositories aborted pack
negotiation with `fatal: expected 'packfile'`; all of them publish hundreds of training-checkpoint
branches. `--single-branch` resolves it and matches this study's scope, which is the default branch
throughout.

**A5 (2026-07-30, made AFTER seeing interim RQ2 data for two of five checkpoints, and disclosed as
such). Comparisons whose rendered prompt is identical are labelled uninformative, not null.**

Two templates can be render-divergent on the fixed probe set and still render a *particular*
conversation identically, because the edit only affects a branch that conversation does not take.
When that happens the model receives byte-identical input in both arms, so the outcome is identical
by construction. Reporting that as "this template change had no effect" would be false: nothing was
varied.

Each pair therefore records how many of its tasks had identical rendered prompts, and a pair where
all of them did is marked `informative = false` and excluded from effect estimates and from the
multiplicity family, while still being reported and counted.

**Why this is disclosed rather than folded in silently.** It was written after the interim analysis
showed Qwen2.5-7B-Instruct's two templates producing 0.0% response divergence, which looked like a
clean null and was in fact a pair of identical prompts. The change is a correctness fix, not a
choice between outcomes: it cannot make an effect larger, it only refuses to count a non-comparison
as evidence either way. It removes one of five checkpoints from the effect estimate, which moves the
paper's claim toward weaker, not stronger.

**A6 (2026-07-30, same disclosure). The multiplicity family is every informative within-checkpoint
pair, not first-versus-last.** The pre-registration says Holm is applied "across the family of
template-pair comparisons" without naming a primary contrast; an early implementation chose
first-versus-last. Interim data showed why that is the wrong choice on its own: Phi-3-mini's second
template moves the outcome sharply and its third moves it back, so first-versus-last is close to
zero while the repository passed through a window in which the system was materially worse. A
consumer lives through the transitions, not the endpoints.

The family is therefore all informative pairs, Holm-corrected together. First-versus-last is still
reported, as a separate and explicitly secondary observation about what a "then versus now" check
would have seen, which is part of the paper's argument rather than a competing headline.

**A7 (2026-07-30, written BEFORE the run, with the prediction fixed in advance). A system-free
control.**

Both review rounds independently named the same missing experiment, and it is the one that decides
whether this paper has a mechanism or only a correlation.

**The question.** Both behavioural effects in RQ2 involve the system message. \ShortMistral's earlier
template raises on any role that is not user or assistant. \ShortPhi's middle template deletes the
`system` arm of its role dispatch, so a system message matches no branch and is discarded. Our
conversation puts the output contract in a system message. If the contract is moved into the user
turn instead, does the damage disappear?

**Design.** Identical in every respect to the RQ2 run except that the conversation carries no system
message: the contract sentence is prepended to the user turn instead. Same tasks, same seed, same
decoding, same checkpoints, same historical templates. Only the message list changes. Run on the two
checkpoints that produced an effect, \ShortMistral{} and \ShortPhi{}, across all their
render-distinct templates.

**Predictions, committed now so the interpretation cannot be chosen afterwards.**

1. \ShortMistral{} v0 render errors fall from all tasks to zero, because the exception fires only on
   a non-user role. Resolution becomes non-zero and the gap between its two templates shrinks
   substantially.
2. \ShortPhi's contract-compliance collapse at its middle template shrinks substantially, because the
   deleted branch no longer has anything to discard.
3. If BOTH hold, the mechanism is established causally: the harm is not "the template changed" in
   general, it is "the template changed how it handles the role your contract lives in". That yields
   a concrete mitigation, which is worth more to a practitioner than the measurement alone.
4. **If either fails**, the mechanism story in Section 4 is wrong or incomplete and must be weakened
   to a correlation. In particular, if \ShortPhi's collapse persists without a system message, then
   something other than the deleted system branch is responsible and the paper must say so.

**This control can only cost us.** It is capable of refuting the paper's own explanation, which is
why it is worth running rather than asserting. Whatever it returns is reported.

**A8 (2026-08-05, after all results were known). Two statements in the frozen text above are wrong,
and this amendment records that rather than editing them.** The rule at the top of this document is
that nothing above the amendment section is edited afterwards, so the errors stay where they are.
Both were found while assembling the release package, by a check that requires every file named in
the artifact to exist in it.

1. **The header claims `code/crosscheck.py` compares every threshold in the manuscript against this
   file and fails the build on a mismatch. No such script was written.** That automated
   prereg-to-manuscript threshold comparison does not exist in this project. What does exist is
   `code/consistency.py`, which traces every number in the manuscript back to the analysis output
   that produced it, checks retired claims, venue facts and source integrity, and is validated by
   `code/selftest.py` injecting twelve defect classes. It does **not** read this document, so no
   automated check ties the manuscript's thresholds to the pre-registered ones; that comparison was
   done by hand. The sentence describes a control that was intended and never built, and it should
   be read as an error, not as a description of the instrument.

2. **The frame-freeze section names `data/frame.jsonl`. The file is `data/candidates.json`.** The
   substance is unaffected: the frame was drawn once, before any result was seen, and the file in
   the artifact is the frame this paper used.

Neither correction changes a design decision, a threshold, or a reported number. They are recorded
because a pre-registration that names an instrument which does not exist is exactly the kind of
claim this document was written to make checkable.
