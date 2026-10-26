# Exploratory AI fidelity comparisons

The frozen eight-article run is complete: **524 source units, seven methods**.
**Variant B leads** in the conversational AI assessment, with mean fidelity
**3.560/4** and **388/524** outputs retained by the strict AI-plus-technical rule.
These are AI opinions about source-text fidelity, not human-reference scores.

| Method | Returned / 524 | Mean fidelity / 4 | AI strict retained / 524 | AI strict yield |
| --- | ---: | ---: | ---: | ---: |
| B: read both, then translate | 524 | 3.560 | 388 | 74.05% |
| A: translate, then refine | 524 | 3.476 | 370 | 70.61% |
| C: search-assisted refinement | 524 | 3.461 | 372 | 70.99% |
| Blind translation | 524 | 3.109 | 284 | 54.20% |
| Reranking without rewriting | 170 | 1.759 | 6 | 1.15% |
| BGE-M3 rank 1 | 524 | 0.518 | 4 | 0.76% |
| LASER-1 rank 1 | 524 | 0.298 | 4 | 0.76% |

Means exclude uncertain cases and missing outputs; every yield denominator is
524. A residual template source is uncertain for six methods. Reranking has
354 abstentions and its mean is conditional on 170 returned spans.
Extraction methods return existing English text; their grades include extra
or missing source claims and are not gold retrieval precision/recall.

B improves 153 sources, worsens 29 and ties 341 against blind translation,
with one uncertain source. It also leads on the equal-article-weight mean
(3.691/4), leading or tying on seven of eight article means. A/C lead on the EU
article. The source pool is uneven: 270/524 units come from the Bavarian-language
article. This is descriptive evidence, without a statistical significance claim.

All **2,561 distinct judgments** were recorded and validated before opening the
method key; no grades were changed afterward. Method names were hidden in the
new assessment inputs, but the assistant had prior project context and no
validated Bavarian reviewer expertise. The eight articles had already been
inspected in an older pilot. No human review, gold alignments or Tatoeba
reference evaluation was performed; actual cost per retained pair is unavailable.

| File | Purpose |
| --- | --- |
| [Run README](data/evaluation_runs/frozen_eight_v2/README.md) | Interpretation, protocol and complete file reference |
| [results.csv](data/evaluation_runs/frozen_eight_v2/assessment/results.csv) | Overall seven-method table |
| [comparison.csv](data/evaluation_runs/frozen_eight_v2/assessment/comparison.csv) | Every source, English output, grade, issues and note |
| [per_article.csv](data/evaluation_runs/frozen_eight_v2/assessment/per_article.csv) | All eight articles separately |
| [summary.json](data/evaluation_runs/frozen_eight_v2/assessment/summary.json) | Paired counts, distributions and issue counts |
| [Supervisor email draft](data/evaluation_runs/frozen_eight_v2/SUPERVISOR_EMAIL.txt) | Simple explanation, code names and results |
| [Handoff archive](data/evaluation_runs/frozen_eight_v2_handoff.tar.gz) | Code, prompts, inputs, snapshots and complete results |

The completed development comparison remains separately documented below.

## Fixed eight-article evaluation

`evaluation_pipeline.py` and `modal_evaluation.py` reuse the existing prompts,
selection grammar, validators, encoder execution, technical filtering and AI
rubric. The development runners and their output fingerprints are unchanged.
The separate experiment has 524 eligible Bavarian sources, and records its
settings and code/input/prompt hashes in `experiments/evaluation_eight_v2.json`.
Outputs go to `data/evaluation_runs/frozen_eight_v2/`. Version 1 is retained as
a failed CPU import attempt before any generation. Version 2 packages the
missing helper module and records its predecessor; model, prompts and
scientific settings are unchanged. An isolated import regression check tests
the uploaded modules without access to the checkout.

Mining remains **unthresholded** (ratio margin, four neighbours, five saved
suggestions). Its first-ranked candidate is assessed. No acceptance threshold
will be selected using evaluation outputs. AI strict yield means score 4 and a
technical pass; it is not gold alignment precision. Missing reranking outputs
remain in the yield denominator and its mean is conditional on returned spans.
The rubric, deduplication, uncertainty handling and method-hidden inputs match
development. Per-article results and paired comparisons with blind translation
are also exported. Previously inspected evaluation pages and conversational
AI assessment retain the limitations documented below.

The sequence, from this directory, is:

```sh
# Offline: validate the committed settings and frozen source pool.
.venv/bin/python evaluation_pipeline.py

# Local compute: both original encoders, with the existing mining environment.
.venv-mining/bin/python evaluation_pipeline.py --mine

# Paid Modal compute: run only after approval of the workspace and cost.
# Each invocation has a one-hour worker timeout and commits every valid stage.
.venv/bin/python modal_evaluation.py --method blind --execute --max-new 2 --detach
.venv/bin/python modal_evaluation.py --method blind --execute --max-new 524 --detach
.venv/bin/python modal_evaluation.py --method a --execute --max-new 2 --detach
.venv/bin/python modal_evaluation.py --method a --execute --max-new 524 --detach
.venv/bin/python modal_evaluation.py --method b --execute --max-new 2 --detach
.venv/bin/python modal_evaluation.py --method b --execute --max-new 524 --detach

# C: collect the fixed queries with the connected web tool, saving every query,
# result list, attempted URL, excerpt/snippet and failure in search/.
.venv/bin/python evaluation_pipeline.py --prepare-search
# Complete sources/ and audit/ for all 524 records before executing C.
.venv/bin/python modal_evaluation.py --method c --execute --max-new 2 --detach
.venv/bin/python modal_evaluation.py --method c --execute --max-new 524 --detach

# Recover committed responses after interruption, without requesting a GPU.
.venv/bin/python modal_evaluation.py --method b --sync

# All seven methods must be complete before preparing method-hidden assessment.
# Reranking is exported from B's selections; it incurs no new inference.
.venv/bin/python evaluation_pipeline.py --prepare-judge
# Record all AI decisions in assessment/judgments.json before unmasking.
.venv/bin/python evaluation_pipeline.py --score
```

The CPU preflight checks every first-stage prompt and each page's selection
grammar before requesting a GPU; every later stage also checks its context
budget. Inputs are never silently truncated. A run exceeding context or
returning invalid output stops, retaining completed stages and validation
failures. Resuming reuses valid stages. Use one local process/workspace at a
time; the local lock does not coordinate independent computers. An invocation
that reaches its timeout may require sync and resume; do not automatically
repeat paid jobs. Token counts and per-generation timings are saved, but cost
per retained pair requires actual billing including startup and shared-call
accounting. Reranking shares B's selection calls and A/C reuse blind calls.

The `--freeze` action is for initial creation before generation. The committed
freeze rejects subsequent settings, code or input changes; experimental
amendments require a newly named protocol and explicit reporting.

### Source quality in the fixed evaluation corpus

A source-only AI inspection, before method-output scoring, found residual
template/editor text, abbreviation-related sentence splits and merges,
standalone lexical examples and a Latin quotation in the Bavarian articles.
Representative source IDs and observations are recorded in
`data/evaluation_runs/frozen_eight_v2/source_review_notes.json`.
These are qualitative AI observations, not human annotations or an exhaustive
source-quality reference. All 524 frozen units are retained. Faithfulness to
an interpretable text segment does not certify that it is a complete, clean
parallel sentence. Apply the existing uncertainty option where interpretation
is insufficient; do not silently remove cases or repair boundaries mid-run.
Any preprocessing repair belongs to a separately versioned corpus and new runs.

## Completed development results

On the existing 53 development sources, **B leads narrowly** in this
conversational AI fidelity assessment. A is close; this is not a decisive or
held-out result. No human reviewers were used or requested.

| Method | English outputs | Mean fidelity / 4 | Fully faithful (4) / 53 | Mostly or fully faithful (3–4) / 53 |
| --- | ---: | ---: | ---: | ---: |
| B: read both, select, translate | 53 | 3.528 | 38 | 43 |
| A: blind draft, select, refine | 53 | 3.491 | 36 | 46 |
| C: search-assisted refinement | 53 | 3.453 | 35 | 45 |
| Blind translation | 53 | 3.245 | 32 | 39 |
| Reranking without rewriting | 19 | 1.895 | 0 | 1 |
| BGE-M3 rank 1 | 53 | 0.717 | 0 | 1 |
| LASER-1 rank 1 | 53 | 0.302 | 0 | 1 |

Reranking abstained on 34 sources; its mean is conditional on the 19 returned
spans. All 53 sources remain in its yield denominator. Mining uses the saved,
unthresholded rank-1 suggestions, fixed before assessment. These scores judge
the English spans **as faithful translations**, not their usefulness as
partial evidence or retrieval precision. A useful partial match can still
omit source claims and add English claims, making it unsuitable as an unchanged
parallel sentence pair. Zero fully faithful spans does not mean zero useful
retrieval results.

B's mean advantage over A is only 2 total rating points across 53 sources
(0.038 per source). B scored higher on 7 sources, A on 3, and 43 tied. A had
more outputs in the combined mostly/fully faithful category. A also scored
slightly higher on Bozn (3.538 vs B's 3.513), while B scored higher on
Kristndum (3.571 vs A's 3.357). The preferred method therefore depends on the
quality criterion; B is the current leader for mean score and strict fidelity
yield, rather than a generally established winner.

Compared with blind translation, A improved 9 scores, tied 43 and worsened 1;
B improved 12, tied 39 and worsened 2; C improved 8, tied 44 and worsened 1.
These are paired AI rating comparisons, not significance tests or gold labels.

## Assessment protocol

- Assess the same 39 Bozn and 14 Kristndum source sentences for every method.
- Use one final output per method/source; for each embedding baseline use its
  saved rank-1 candidate. The original top-five suggestions remain unchanged.
- Deduplicate identical source/English pairs within the shared context, so
  the same text receives the same judgment across methods. This gives 265
  unique judgments for 337 nonempty outputs, plus 34 reranking abstentions:
  371 method/source records in total.
- Show the assessor the original source, numbered Bavarian context and opaque
  English candidate IDs. Hide method names, model identities, original model
  labels, mining ranks/scores and C's self-checks in the assessment inputs.
- Fix candidate IDs using seed 20261026. Present candidates grouped by source
  for context, with opaque-ID order within each source.
- Record all 265 judgments and explanations before reading the private method
  mapping and calculating results. No judgments were changed after unmasking.
- Score fidelity from 0 to 4: 4 fully faithful; 3 minor local errors; 2
  substantial meaning with major errors; 1 limited overlap; 0 no substantive
  source meaning or the only central claim reversed. Use null/uncertain if
  interpretation is insufficient. No cases were assigned uncertain in this run.
- Record additions, omissions, mistranslations, contradictions and minor
  issues. The original source controls facts, negation and uncertainty, even
  when a current encyclopedia or other outside source would disagree.
- Keep readability separate from fidelity: understandable awkward English
  does not automatically lose points. Specific local meaning errors do.
- Strict AI retention requires score 4 and a technical pass. This is a declared
  development criterion, not calibrated human acceptance or corpus gold.

The assessor was **the Codex assistant in this conversation**, distinct from
the Gemma 4 generator. Its Bavarian expertise is not independently validated.
It had prior project context and had previously seen some examples: hiding
method names in the current inputs does not establish perfect blinding.
This is conversational assessment, not a controlled inference API run; exact
assessor model ID, sampling settings, token billing and a full model replay
are unavailable. Candidate inputs, rubric and judgment decisions are saved.

These limitations, the coarse ordinal scale and the small two-article
development set make small differences uncertain. Do not present these scores
as Tatoeba benchmark results, a held-out comparison, human review or proof
that B will win on other articles/languages. The assigned 524 evaluation
sources have not been generated by the clean pipeline. The earlier pilot's
evaluation-article exposure remains documented in the preprocessing freeze.

## Results and code

| File | Purpose |
| --- | --- |
| [results.csv](data/evaluation/ai_fidelity_development_v1/results.csv) | Compact cross-method results table |
| [comparison.csv](data/evaluation/ai_fidelity_development_v1/comparison.csv) | All 371 method/source rows with original texts, scores and reasons |
| [summary.json](data/evaluation/ai_fidelity_development_v1/summary.json) | Metrics, issue counts and assessment scope |
| [judgments.json](data/evaluation/ai_fidelity_development_v1/judgments.json) | All 265 recorded unique judgments |
| [blinded_items.json](data/evaluation/ai_fidelity_development_v1/blinded_items.json) | Method-hidden source/candidate groups |
| [bavarian_contexts.json](data/evaluation/ai_fidelity_development_v1/bavarian_contexts.json) | Shared numbered source context |
| [per_output.json](data/evaluation/ai_fidelity_development_v1/per_output.json) | Method mapping joined to recorded judgments, including abstentions |
| [paired_vs_blind.json](data/evaluation/ai_fidelity_development_v1/paired_vs_blind.json) | Paired higher/tied/lower rating counts |
| `private/method_key.json` | Method key withheld until judgments were complete |
| `judgment_parts/` | Incrementally saved pre-unmasking assessment batches |
| [semantic_evaluation.py](semantic_evaluation.py) | Prepare hidden items, validate judgments, compute summary/results and per-output JSON |
| [tests/test_semantic_evaluation.py](tests/test_semantic_evaluation.py) | Deduplication, judgment validation and missing-output denominator tests |

Run fingerprint:
`5a7a6cc66373c8dcdec4da0fdd8db449dd6bec8958c8c5b852d65b17765fee5d`.
Recorded-judgment fingerprint:
`639db3f21f91936f4acf64db8de163e093c8dd2789182fde88c4fdb8790aa64e`.

Three new offline tests passed. The artifact audit checked 53-source coverage
for all seven methods, all 265 unique judgments, 371 method mappings, unchanged
assessment inputs, summary arithmetic, abstention handling and the detailed
CSV contents. Earlier outputs, technical-filter artifacts and prompts remain
the source experiment records.

```sh
.venv/bin/python semantic_evaluation.py --prepare
.venv/bin/python semantic_evaluation.py --score
.venv/bin/python -m unittest discover -s tests -p test_semantic_evaluation.py -v
```

The score command requires complete recorded judgments before unmasking the
method key. The preparation command does not run a judge. Existing conflicting
artifacts are rejected rather than silently overwritten. In this session the
existing temporary recovery environment was used for these offline commands.

No new Modal jobs or inference API calls were made for this assessment.
Cost per retained pair remains unavailable without actual experiment billing;
it must not be inferred from token counts alone. No gold reference alignments
exist for these Wikipedia pages, so gold match precision/recall is unreported.
Tatoeba can supply a separate reference-based experiment; it cannot provide
reference translations for these different Wikipedia source sentences.
