# Citation verification record

Every entry in `paper/refs.bib` was verified by FETCHING the source during this build, not from
memory and not from a search-result summary. This file is the audit record; the `.bib` itself carries
no `note` fields, because ACM-Reference-Format prints them into the rendered bibliography.

**A note in this file is not verification of a quotation.** No claim in the manuscript quotes a
sentence from any of these sources. Every number attributed to a source is marked `\lit{}` in the
manuscript and appears in the row below. If a future round adds a quoted phrase, that source's URL
must be re-opened in that round: a recorded verification is not a substitute for re-fetching.

| key | verified | what was checked |
|---|---|---|
| `tafreshipour2024prompting` | 2026-07-30 | arXiv 2412.17298 abstract page fetched. Title, six authors, submitted 23 Dec 2024. Numbers used: **1,262** prompt changes, **243** repositories, **21.9%** documented in commit messages. All three appear verbatim in the fetched abstract. |
| `fawzy2026vibepractice` | 2026-07-30 | Publisher PDF first page read directly (page image). Title, three authors, ICSE-SEIP '26, 12-18 April 2026, Rio de Janeiro, DOI 10.1145/3786583.3786866, ACM reference block. Number used: **101** practitioner sources (also 518 firsthand accounts, not cited). |
| `fawzy2026verification` | 2026-07-30 | arXiv 2605.24521 abstract page fetched. Title, three authors, submitted 23 May 2026. Number used: **162** survey participants. |
| `muennighoff2024octopack` | 2026-07-30 | arXiv 2308.07124 abstract page fetched. Title, ten authors, submitted 14 Aug 2023, ICLR 2024. **Caution recorded:** the abstract introduces *HumanEvalPack* and does NOT use the string "HumanEvalFix". The manuscript therefore cites it as "the code-repair split of HumanEvalPack" and never as a paper that names HumanEvalFix. |
| `liu2023evalplus` | 2026-07-30 | arXiv 2305.01210 abstract page fetched. Title, four authors, submitted 2 May 2023. EvalPlus / HumanEval+ confirmed. Cited for the task set's provenance only; no number quoted. |
| `mitchell2019modelcards` | 2026-07-30 | arXiv 1810.03993 abstract page fetched. Title, nine authors, v1 5 Oct 2018, v2 Jan 2019. No number quoted. |
| `venturini2023depended` | 2026-07-30 | arXiv 2301.04563 abstract page fetched. Title, five authors, submitted 11 Jan 2023. Numbers used: **~12%** of dependent packages, **44%** of manifesting breaking changes in minor and patch releases. Both verbatim in the fetched abstract. |
| `olausson2024selfrepair` | 2026-07-30 | arXiv 2306.09896 abstract page fetched. Title, five authors, submitted 16 Jun 2023, venue ICLR 2024 confirmed. No number quoted. |
| `ray2026structured` | 2026-07-30 | arXiv 2607.14167 abstract page fetched. Title, two authors (Ray, Goyal), submitted 15 Jul 2026. No number quoted. |
| `iscan2026falsification` | 2026-07-30 | arXiv 2606.31511 abstract page fetched. Title, single author, submitted 30 Jun 2026. No number quoted. |
| `hf-chat-templating` | 2026-07-30 | Transformers chat-templating documentation page fetched. Documents `chat_template` and `apply_chat_template`, the reference implementation the renderer is validated against. No number quoted. |

## Numbers attributed to sources in the manuscript

Each is wrapped in `\lit{}` so the build gate can tell it apart from a measurement of ours.

- `\lit{101}` practitioner sources, `fawzy2026vibepractice`
- `\lit{162}` participants, `fawzy2026verification`
- `\lit{1{,}262}` prompt changes and `\lit{243}` projects and `\lit{21.9}\%` documented,
  `tafreshipour2024prompting`
- `\lit{12}\%` of dependent packages and `\lit{44}\%` of breaking changes in minor/patch releases,
  `venturini2023depended`

## Two claims deliberately NOT cited to a paper

- That "vibe coding" was coined by Karpathy in 2025. The ICSE-SEIP paper attributes it, but the
  primary source is a social-media post that cannot be fetched and verified here, so the manuscript
  does not make the attribution at all.
- Any statement about hosted commercial APIs changing their templates. Not observable by this
  method, and the threats section says so.

## Added in review round r1 (2026-07-30), both fetched

| key | verified | what was checked |
|---|---|---|
| `sclar2024formatspread` | 2026-07-30 | arXiv 2310.11324 abstract fetched. Title, four authors (Sclar, Choi, Tsvetkov, Suhr), v1 17 Oct 2023, ICLR 2024. Number used: **76** accuracy points on LLaMA-2-13B, verbatim in the abstract. |
| `chen2023chatgptdrift` | 2026-07-30 | arXiv 2307.09009 abstract fetched. Title, three authors (Chen, Zaharia, Zou), 18 Jul 2023. No number quoted. |

Added because a hostile review correctly noted that presenting template sensitivity as unstudied
would misposition the paper. Prompt-format sensitivity is established; the contribution is that on
open weights the formatting artifact is an unversioned file whose edits and consequences are
directly measurable.

## Round r6 (2026-08-05): a SEARCH, not a confirmation fetch

**The method used in r5 was wrong and this section records why.** r5 re-opened each URL recorded in
`refs.bib` and confirmed the page matched what the entry said. That detects an entry drifting from
its own source. It is structurally blind to the defect that actually existed: *a published version
exists and we are citing the preprint*. Confirming the arXiv page for `venturini2023depended`
returned exactly the metadata `refs.bib` claimed, so the check passed, three times, across r3 and r5.

r6 therefore SEARCHED for a published version of every preprint-typed entry rather than re-fetching
the recorded URL.

| key | searched | result |
|---|---|---|
| `venturini2023depended` | 2026-08-05 | **WAS WRONG.** Published in ACM TOSEM **32(4), 1-26, 2023**, doi `10.1145/3576037`. Corrected. Journal, volume, issue, pages and year taken from the OpenAlex record for that DOI (DBLP was down; the ACM DL 403s). |
| `fawzy2026verification` | 2026-08-05 | arXiv only. Correctly cited as a preprint. |
| `ray2026structured` | 2026-08-05 | arXiv only. Correctly cited as a preprint. |
| `iscan2026falsification` | 2026-08-05 | arXiv only. Correctly cited as a preprint. |

**A trap found while doing this, recorded so no future round falls into it.** The author-hosted PDF
at `ime.usp.br/~gerosa/papers/TOSEM-BreakingChanges.pdf` carries an ACM Reference Format block
reading *"Proc. ACM Meas. Anal. Comput. Syst. 37, 4, Article 111 (October 2021), 26 pages"*. That is
the **`acmart` template placeholder**, not this paper's record: wrong journal, wrong year, dummy
article number. It looks exactly like an authoritative citation block. Metadata was taken from
OpenAlex instead. The 12% / 14% / 44% figures WERE read verbatim from that PDF's abstract and are
correct.

### The r5 gap is now closed

| key | verified | what was checked |
|---|---|---|
| `fawzy2026vibepractice` | 2026-08-05 | r5 could not re-fetch this (ACM DL returned 403). The author-hosted copy at `kblincoe.github.io/publications/2026_ICSE_SEIP_vibe-coding.pdf` was read as a page image. ACM Reference Format confirms *ICSE-SEIP '26, April 12-18 2026, Rio de Janeiro, Brazil*, doi `10.1145/3786583.3786866`. **101** practitioner sources verbatim in the abstract, as is "QA practices are frequently overlooked, with many reporting skipping testing". |

### Added in r6, fetched

| key | verified | what was checked |
|---|---|---|
| `carrigan2023chattemplates` | 2026-08-05 | Page fetched. Title, byline (Matthew Carrigan) and date (3 October 2023) read off the post. It introduced the `chat_template` attribute and named the hazard: "Using a format different from the format a model was trained with will usually cause severe, silent performance degradation." **Not quoted in the manuscript**; the prose paraphrases. Added because our title uses "silent" and a Hub-literate reviewer knows this post. |
