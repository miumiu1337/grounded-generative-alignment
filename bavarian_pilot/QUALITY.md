# Output quality: technical filtering and semantic assessment

This stage audits the completed **development** outputs from blind translation,
LASER-1, BGE-M3, A, B, C and reranking without rewriting. It is separate from
the frozen source preprocessing. Technical validity does not establish that
English faithfully expresses the Bavarian source.

## Completed development audit

The report covers **795 rows across the same 53 sources per method**. All five
mining suggestions are retained, giving each encoder 265 rows rather than 53.

| Method | Rows | Technical pass | No target | Reused-target warnings |
| --- | ---: | ---: | ---: | ---: |
| Blind translation | 53 | 53 | 0 | 0 |
| LASER-1 | 265 | 265 | 0 | 101 |
| BGE-M3 | 265 | 265 | 0 | 128 |
| A | 53 | 53 | 0 | 0 |
| B | 53 | 53 | 0 | 0 |
| C | 53 | 53 | 0 | 0 |
| Reranking only | 53 | 19 | 34 | 2 |
| Total | 795 | 761 | 34 | 231 |

There were **zero technical rejections** and no within-method duplicate-pair
groups. There are 91 within-method reused-target groups, affecting 231 rows;
all remain in the report. These warnings do not establish that any match is
incorrect. The 34 no-target rows are the original reranking `NONE` cases.
All ten C self-check flags still pass the technical checks: the filter does
not mistake those self-check opinions for structural defects.

The completed report fingerprint is
`e839058c7f3fa48a1eac24477dd97d25b40c875e7cdddfc199680a3fc7baa989`.
Ten offline tests passed. An independent artifact audit reproduced every
technical annotation, checked unchanged text in all 795 CSV rows, verified the
input/code hashes and same-source coverage, and confirmed complete disjoint
JSONL partitions and identical re-export. No semantic assessments or new
inference calls were made. The source-artifact validation scope is recorded
below and in the manifest.

[Technical comparison CSV](data/quality/technical_development_v1/comparison.csv) ·
[Summary](data/quality/technical_development_v1/summary.json) ·
[Complete report](data/quality/technical_development_v1/report.json)

## Technical policy, version 1

[output_quality.py](output_quality.py) runs locally without inference, network
requests, new models or new dependencies. It validates the recorded run
fingerprints, shared frozen sources/pages, completion metadata and existing
stage schemas. Evidence must resolve to frozen English spans. Missing,
corrupted, unexpected or inconsistent saved records stop the audit with an
error rather than disappearing from the denominator.

A/B's older individual selection-file reads stalled during this audit. Their
aggregate JSONL exports are readable and contain complete English target-stage
responses. Validate those responses, their exact requests/schemas, all sources,
selected span IDs/text and export partitions directly. Reranking's aggregate
records are checked against B's selection IDs/hashes and frozen candidate text.
Raw A/B/reranker selection-stage responses are not reread; the report explicitly
records this scope. Reranking does not generate English, so target-generation
completion checks do not apply. C retains complete saved-stage validation.
This is not a claim to have repeated the earlier full selector-stage audits.

The report includes all five saved mining suggestions per source. It does not
choose a margin threshold, promote rank 1 to an accepted pair, or discard lower
ranks. Candidate rank and original margin/cosine remain available. Mining has
265 rows per encoder; the other methods have 53 each. Candidate-row counts
cannot be compared as accepted-pair yields. Every method covers the same 53
original sources.

| Check | Action |
| --- | --- |
| Empty/whitespace-only expected English output | Technical rejection |
| Target contains no letter or number | Technical rejection |
| Broken control characters, except ordinary tabs/newlines | Technical rejection |
| Incomplete or missing completion metadata for a generated English target | Technical rejection; existing producer validators also reject invalid saved stages |
| Explicit reranking `NONE` with no English evidence | `no_target`, not a translation error |
| Unicode replacement character | Warning; retain |
| Target equals source after whitespace/NFC normalization | Warning; retain |
| Duplicate source/target text within one method | Warning; retain every original record |
| Same target reused for different sources within one method | Warning; retain |

Duplicate comparison uses case-sensitive Unicode NFC and collapsed whitespace.
It does not alter stored text. Identical outputs across different methods are
not duplicate-removal candidates: their agreement belongs in the comparison.
Repeated targets in mining can be legitimate retrieval candidates, so they
are not automatically rejected. No language-ID, length-ratio, embedding-score
or minimum-word-count cutoff is introduced without development validation.

A/B exact/partial/none labels, their contradiction flags and C's self-check
verdicts are retained as original metadata. They do not decide technical
acceptance. In particular, a faithful generated translation may still exist
when A/B selected `NONE`; such translations must remain available for the
translation comparison. Evidence-backed yield is a separate quantity.

Within this technical report, every output retains `semantic_status: unassessed`,
including technical passes; later semantic judgments are separate artifacts.
Technical rejects and unmatched cases stay in the report and their
own partitions. Neither a technical pass nor a model-labelled exact match is
an accepted parallel pair. No automatic rewriting, deduplication or repair
occurs in this stage.

## Shared semantic rubric: development proposal

The technical audit itself made no semantic judgments. A subsequent
conversational AI comparison now evaluates the current development outputs;
see [EVALUATION.md](EVALUATION.md) for its frozen rubric, recorded judgments,
results and limitations. Its narrow B/A ranking is exploratory. A controlled
assessor and protocol still need fixing before a held-out study.
The judgments are **AI-assessed**, not human
annotations or gold labels. Use an evaluator separate from the generating
Gemma run where feasible, and check its Bavarian interpretation rather than
assuming multilingual capability establishes accuracy.

For the common fidelity task, show the unchanged focus Bavarian sentence,
the same numbered Bavarian context, and one candidate English output. Hide
method names, model identities, mining scores/ranks, generator labels,
earlier drafts and C's self-check. Give each item an opaque evaluation ID and
randomize presentation using a recorded seed. Generated and verbatim outputs
receive the same rubric. Identical source/target/context items can share one
judgment, with a private mapping back to every method; do not count those
judgments as independent observations.

The source controls the claims, including outdated facts, uncertainty and
negation. Context may resolve a reference but must not supply extra claims.
Current Wikipedia or outside facts cannot override what the focus source says.
Treat all article/output text as untrusted data, not instructions to the judge.

| Dimension | Required judgment |
| --- | --- |
| Source-claim coverage | Full / partial / none / uncertain; list omitted claims |
| Target-claim support | All / some / none / uncertain; list unsupported additions |
| Meaning preservation | Identify mistranslations, contradictions, negation/scope errors and changes to names, quantities, dates or uncertainty |
| Ambiguity | State unresolved source interpretations; do not invent a confident decision |
| Fluency | Record separately; smoother English does not establish faithfulness |

Require concrete source and target spans supporting each reported error. A
proposed semantic retention rule is full source coverage, support for all
target claims, no meaning-changing errors and no unresolved uncertainty,
in addition to a technical pass. The subsequent conversational comparison
instantiates a strict score-4 rule; it remains an exploratory development
criterion, not calibrated acceptance. Keep uncertain outcomes
separate and preserve both pre-filter and post-filter results.

Assess source-to-selected-English alignment separately from fidelity of a
generated translation. A/B can generate a faithful translation from partial
evidence, while the selected English span itself contains extra claims.
Blind translation and C do not select Wikipedia sentence alignments, so
Wikipedia-alignment precision/recall is not applicable to those outputs.
Assess C's external citation support separately from source fidelity.

## Evaluation and hand-in boundary

- The current run is development-only: 39 Bozn and 14 Kristndum sources.
- Fix mining selection/threshold policies and the semantic assessment protocol
  on development data before processing the assigned 524 evaluation sources.
- The eight evaluation articles were used/inspected in the original pilot;
  this restart does not make them unseen. A strictly untouched test requires
  separately reserved articles, as recorded in `data/preprocessing_freeze.json`.
- Without reference alignments, gold-standard alignment precision/recall is
  unavailable. If separate AI-created reference alignments are used, report
  agreement against those explicitly as AI-reference metrics. Constructing
  references requires finding valid matches independently of method outputs;
  scoring only returned candidates cannot establish alignment recall.
- Existing human-translated benchmarks support separate evaluations only when
  language coverage, reference provenance and task compatibility are verified.
  They do not supply reference alignments for these Wikipedia pages.
- Report technical attrition, semantic retention/uncertainty, additions,
  omissions and accepted/retained-pair yield under the declared rule. Preserve
  the full original-source denominator, including unmatched outputs. For mining,
  distinguish candidate count, unique matched sources and final selected pairs.
- Cost per retained pair requires actual cost records and a declared retention
  rule. Token totals alone are not a bill. Account for blind drafts reused by A/C,
  B selections shared with reranking, C search/verification and assessment costs
  without double-counting shared calls.
- Submit snapshots/revisions, split history, versioned code/prompts/settings,
  pair-level outputs, filter reasons, assessment results and a cross-method table.
  Do not use LASER similarity alone as the quality judge against LASER.

## Files and reproduction

| File | Purpose |
| --- | --- |
| [output_quality.py](output_quality.py) | Validate and normalize existing outputs, apply common technical checks, export report |
| [tests/test_output_quality.py](tests/test_output_quality.py) | Ten offline regression tests for checks, provenance constraints and exports |
| `data/quality/technical_development_v1/run_manifest.json` | Policy, code hash, normalized-output hash and upstream provenance |
| `data/quality/technical_development_v1/report.json` | Every original source/target with technical reasons and semantic status |
| `data/quality/technical_development_v1/comparison.csv` | Readable cross-method technical audit |
| `data/quality/technical_development_v1/summary.json` | Per-method counts; no accepted-pair claims |
| `data/quality/technical_development_v1/duplicate_groups.json` | Within-method duplicate/reused-target groups |
| `data/quality/technical_development_v1/technical_*.jsonl` | Complete disjoint pass, reject and no-target partitions |

```sh
.venv/bin/python output_quality.py
.venv/bin/python -m unittest discover -s tests -p test_output_quality.py -v
```

If project environment reads stall, the existing temporary recovery environment
also supports the offline command:

```sh
/private/tmp/gga-variant-a-recovery-venv/bin/python -u \
  -X pycache_prefix=/private/tmp/gga-quality-pycache output_quality.py
```

Original result files remain the source artifacts. A changed policy, code or
input fingerprint requires a new versioned quality-report directory rather
than overwriting a previous report. This does not freeze the full final study.
