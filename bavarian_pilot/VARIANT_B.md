# Variant B: read both pages, select evidence, then translate

Variant B uses the same 53 frozen Bavarian development sentences as Variant A
(39 Bozn, 14 Kristndum), the same numbered pages, and the same complete English
candidate pools (327 and 1,012 spans). It does not use blind translations,
Variant A's selected evidence, or mining rankings in either model request.
The eight evaluation article pairs are not processed by this experiment.

## Status

Completed **53/53 development sources** on 2026-10-25, with 53 saved selections
and 53 translations. The two-source pilot passed structural validation; the
remaining 51 reused those two results with the same fingerprint:
`443db3a01d50baa81e8e55ed5069096c207e2eb64ad50cf7bcca890a0b3f9b64`.

| Article | Sources | Model-labelled exact | Partial | None |
| --- | ---: | ---: | ---: | ---: |
| Bozn | 39 | 3 | 10 | 26 |
| Kristndum | 14 | 0 | 6 | 8 |
| Total | 53 | 3 | 16 | 34 |

There are **19 grounded candidates**, **34 unmatched translations**, and no
flagged contradictions. These are model labels, not accepted parallel pairs.
For context, A's saved summary reports 1 exact, 18 partial and 34 none, also
19 grounded candidates. Equal candidate counts do not establish equal matches
or translation quality. Pair-by-pair A/B agreement has not been computed.

[Comparison CSV](data/translations/grounded_b_gemma4_modal_development_v1/comparison.csv) ·
[Summary JSON](data/translations/grounded_b_gemma4_modal_development_v1/summary.json)

The saved B stages contain **1,201,189 prompt tokens** and **6,079 output
tokens**. Both GPU workers shut down normally. These totals describe saved
inference stages, not Modal billing or model/container startup costs.

Fresh local validation passed for the preprocessing hashes, equality with A's
fingerprinted source/page/candidate snapshot, all 106 B stage records, fixed
evidence and stage order, draft exclusion, all 53 CSV rows, complete disjoint
JSONL partitions, and summary token totals. An optional fresh per-pair A/B
comparison was stopped when older A result-file reads stalled on local I/O;
this occurred after B's validation completed. No pairwise agreement metrics
are reported from that interrupted comparison.

All 53 applicable offline tests passed, including eight new B tests; six
mining tests were skipped in this environment. This is a development protocol,
not a final experiment freeze or translation-quality evaluation.

[Two-source pilot](https://modal.com/apps/denise-obrein/main/ap-cN13GzpUL18J8wpE4HLFL1) ·
[Remaining-source run](https://modal.com/apps/denise-obrein/main/ap-XJLhPVB7ahOGFLvaNgEurP)

### Preliminary quality observation

The saved pilot translation for `bar_59:v1.0.1:S0001` says Christianity is the
largest religion "after Islam and Hinduism". The source's "vuam Islam und am
Hinduismus" appears to mean ahead of those religions, so this may reverse a
source claim. This is a tentative AI-assisted observation, not an independently
assessed label. The raw output and protocol are preserved without correction.
Passing the structural checks does not establish translation fidelity or the
correctness of the model's match labels.

## Protocol

1. **Select evidence:** show the focus Bavarian source and both eligible
   numbered pages. Select an English span or `NONE`, with match status
   (`exact`, `partial`, `none`) and a contradiction flag. No draft is supplied
   and this call cannot return a translation.
2. Save and commit the selection before starting the translation call.
3. **Translate:** show the unchanged focus source, Bavarian page context,
   fixed selection, and only its verbatim English evidence. Generate a faithful
   English translation and check it against the source before returning it.
   This call cannot change evidence or see the complete English page.

There are two calls per source. Translation and its final wording check occur
within the second call; there is no separately saved third refinement call.
This matches A's two new stages, with A refining an existing blind draft and B
generating its translation after selection. A also has the earlier blind call;
include that earlier compute when comparing complete pipeline costs.

Evidence is one eligible English sentence or two consecutive eligible sentences
within one paragraph. Selection uses the same complete-span EBNF grammar and
status/contradiction rules as A. English evidence can clarify wording but cannot
add source claims or replace source facts. Missing or contradictory evidence
does not prevent translation; unmatched and contradictory results are exported
separately. All outputs remain `quality_status: unassessed`.

## Model and execution

The model and settings match A: `google/gemma-4-26B-A4B-it`, revision
`47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, vLLM 0.21.0, Transformers 5.5.4,
H200, bfloat16, temperature 0, seed 42, thinking disabled, 1,024 output tokens
per call, and a 65,536-token context. Selection uses a constrained grammar;
translation uses a JSON schema. Preflight checks all selection prompt lengths
and both span grammars on CPU before GPU allocation. Neither stage truncates
inputs. Each GPU job uses one single-use worker with a one-hour timeout and
no automatic retries.

From this workspace, use the existing Python 3.12 environment:

```sh
.venv/bin/python modal_variant_b.py                             # Offline only
.venv/bin/python modal_variant_b.py --execute                   # Two sources
.venv/bin/python modal_variant_b.py --execute --max-new 53 --detach
.venv/bin/python modal_variant_b.py --sync                      # CPU recovery
```

Execution is billable and requires the intended Modal profile. The default
two-source job covers one source from each article. After validation, a resumed
job skips completed sources and reuses committed selections. `--detach` lets
the remote worker continue if the local connection closes; use `--sync` to
recover outputs afterward. The equivalent `modal run modal_variant_b.py`
commands also work; `--detach` above is a direct Python entry-point option.

If project environment file reads stall, the existing temporary recovery
environment used for A also supports these commands:

```sh
/private/tmp/gga-variant-a-recovery-venv/bin/python \
  -X pycache_prefix=/private/tmp/gga-variant-b-pycache modal_variant_b.py
```

That environment is temporary and may not survive a restart. No dependencies
were added or changed for B. The existing Modal results and model-cache Volumes
are reused, with B outputs saved under a separate run directory. Failed model
responses are preserved remotely under `failures/<stage>/` and errors propagate.
Changes to fingerprinted code, prompts, settings or inputs require a new run.

## Files

| File | Purpose |
| --- | --- |
| [modal_variant_b.py](modal_variant_b.py) | Modal preflight, stage execution, commit/resume, offline CLI |
| [variant_b.py](variant_b.py) | Draft-free requests, input/record validation, local sync and exports |
| [variant_a.py](variant_a.py) | Unchanged shared span grammar, selection rules and atomic-file helpers |
| [Experiment configuration](experiments/grounded_b_gemma4_modal_development_v1.json) | Versioned model/settings and A comparison binding |
| [Selection prompt](prompts/grounded_b_selection_v1.txt) | Direct source-to-English evidence selection |
| [Translation prompt](prompts/grounded_b_translation_v1.txt) | Source-faithful generation using fixed evidence |
| [tests/test_variant_b.py](tests/test_variant_b.py) | Offline regression checks |

Outputs are saved in
`data/translations/grounded_b_gemma4_modal_development_v1/`:
`run_manifest.json`, `inputs.json`, `selections/`, `results/`, `comparison.csv`,
`summary.json`, `grounded_candidates.jsonl`, `unmatched.jsonl`, and
`contradictory.jsonl`. Each stage records its request, constrained output,
raw response, token usage, runtime versions and timestamp. B's inputs and
results have no blind draft field. Token totals describe validated saved stages,
not Modal billing; failed-stage usage remains in the remote failure records.

Match labels and grounded-candidate counts are model decisions, not measured
precision, accepted parallel data or evidence of improvement over A. A and B
both see Bavarian page context; the blind baseline saw only the focus sentence.

```sh
.venv/bin/python -m unittest discover -s tests -p test_variant_b.py -v
```
