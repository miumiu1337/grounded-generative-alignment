# Variant A: blind translation, evidence selection, refinement

This development experiment reuses all 53 saved Gemma 4 blind translations.
It uses frozen preprocessing 1.0.1, with the same 39 Bozn and 14 Kristndum
source sentences and the same English span pools as the mining baselines.
No evaluation sources are selected. Baseline candidate rankings are available;
acceptance thresholds and the final evaluation remain later work.

## Completed development run: 2026-10-24

Version 2 completed **53/53 sources**, with 53 saved evidence selections and
53 saved refinements. Its run fingerprint is
`8f8b50509e883ed6cedc0a98496c14ae8a3b66d2b843c6426c1f361c4d6d4d09`.

| Article | Sources | Model-labelled exact | Partial | None |
| --- | ---: | ---: | ---: | ---: |
| Bozn | 39 | 1 | 13 | 25 |
| Kristndum | 14 | 0 | 5 | 9 |
| Total | 53 | 1 | 18 | 34 |

There are **19 grounded candidates**, **34 unmatched translations**, and no
flagged contradictions. **31 translations changed** from the saved blind
drafts. These counts describe model outputs; changes are not measured quality
improvements, and grounded candidates are not independently accepted pairs.

[Side-by-side comparison CSV](data/translations/grounded_a_gemma4_modal_development_v2/comparison.csv) ·
[Summary JSON](data/translations/grounded_a_gemma4_modal_development_v2/summary.json)

The saved new stages contain 1,207,085 prompt tokens and 6,131 output tokens.
The reused blind stage contains 6,540 prompt tokens and 1,541 output tokens.
These are validated saved-stage totals, not GPU billing or a count of any
unfinished work during the interrupted session.

Final checks passed for all source/draft snapshots, legal evidence spans,
selection-before-refinement timestamps, selected-evidence-only refinement
inputs, normal stops, runtime pins, context limits, 53 CSV rows and complete
disjoint JSONL partitions. Final export validation used the committed input
snapshot and the recorded run fingerprint; repeated fresh reads of older
baseline files were stalled by local I/O. Full source/baseline verification
had succeeded before generation and again before resuming. All 45 applicable
offline tests passed; the six mining-specific tests were skipped in this
environment. The 524 evaluation sources were not processed.

The [initial v2 session](https://modal.com/apps/denise-obrein/main/ap-cvjAS1Kr0MRQBgKRR64EX5)
was recovered after 22 records. The
[detached continuation](https://modal.com/apps/denise-obrein/main/ap-AEv58kiql6DPlE6lVjE2mM)
completed the remaining 31, and the GPU worker shut down normally.

## Fixed protocol

For each source, the order is explicit:

1. Load its existing blind translation and verify the source, baseline run,
   model revision, prompt/settings and file hash. Do not generate a new draft.
2. **Evidence-selection call:** supply the focus Bavarian sentence, blind draft,
   and the eligible numbered Bavarian and English pages, with paragraph and
   section information. Select one frozen English span or `NONE`. A span is
   one sentence or two consecutive sentences within one paragraph.
3. Save and commit the selection before the next call.
4. **Refinement call:** supply the source, blind draft, Bavarian page context,
   fixed selection and only its verbatim English evidence. Correct the draft
   when needed, preserving all source claims. This call cannot choose other
   evidence. Save and commit the refinement.

The English candidate pool is the complete linked page's eligible span pool,
not a mined top-five shortlist: 327 spans for Bozn and 1,012 for Kristndum.
Excluded sentences are not available as evidence. The original articles are
comparable, so a source can legitimately have no corresponding English span.

Selection returns `source_id`, `evidence_ids`, `match_status` and
`contradiction`. Exact means equivalent substantive claims with no extras or
conflicts. Partial means at least one supported source claim but missing,
additional or conflicting claims. None means no supported source claim.
Explicitly contradictory evidence may be recorded even for none. These are
model decisions, not independently assessed labels.

The source is the authority on which claims belong in the translation. English
evidence can clarify wording but cannot add claims or replace source facts.
When evidence is absent or contradictory, still produce a faithful translation
and keep it outside the grounded-candidate output. A correct draft can remain
unchanged. Neither translation edits nor a model's match label establish
quality improvement.

## Model and prompts

- Gemma: `google/gemma-4-26B-A4B-it`
- Checkpoint: `47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`
- Same Modal image/dependency pins as the blind baseline: vLLM 0.21.0,
  Transformers 5.5.4, bfloat16, H200.
- Temperature 0, sampling seed 42, thinking disabled, up to 1,024 output
  tokens per new stage. Selection uses an EBNF grammar enumerating complete
  frozen spans and consistent status/contradiction combinations; refinement
  uses a JSON schema. Both stages return JSON.
- Context capacity is **65,536**, increased from the blind runner's 8,192 to
  fit full numbered pages. This is a recorded inference-setting difference;
  the model checkpoint is unchanged. A CPU preflight counts every selection
  prompt before GPU allocation, and each inference checks its context budget.
  There is no silent truncation.

Prompts are [selection](prompts/grounded_a_selection_v1.txt) and
[refinement](prompts/grounded_a_refinement_v1.txt). The protocol is recorded in
[the experiment configuration](experiments/grounded_a_gemma4_modal_development_v2.json).
For a later A/B comparison, hold the full-page candidate pool, model,
generation settings, evidence constraints and stage definitions fixed; Variant
B omits the supplied blind draft; see [VARIANT_B.md](VARIANT_B.md) for its
implementation, offline checks and execution status.

The blind baseline saw only the focus Bavarian sentence. Variant A also sees
the Bavarian page, so an A-versus-blind comparison combines additional source
context with English evidence and refinement. It cannot attribute any gain
exclusively to English grounding without a source-context-matched comparison.

## Run and resume

Use the existing `.venv` and Modal account; no new dependency installation or
Gemini API key is required.

```sh
cd /Users/adriannahampelska/Desktop/tum/thesis/grounded-generative-alignment-clean
.venv/bin/python modal_variant_a.py
.venv/bin/python modal_variant_a.py --execute
.venv/bin/python modal_variant_a.py --execute --max-new 53
```

The offline command validates the corpus and all blind results. The default
execution tests one source from each development article, including the
longer English page. `--max-new` counts source sentences, each requiring up to
two new calls. Previously committed selections and completed results are
reused after validation. The worker loads the model once per job, uses a
single-use GPU container, has a one-hour timeout, and has no automatic retries.

Recover validated completed stages without a GPU:

```sh
.venv/bin/python modal_variant_a.py --sync
```

Results use the existing `gga-clean-gemma4-dev-results` Volume; weights reuse
`gga-gemma4-model-cache`. Each stage is committed explicitly. Invalid model
responses are saved under `failures/<stage>/` in the remote run directory and
the error propagates. Completed validated stages can still be synced. Raw
failed responses can be downloaded separately from that Modal Volume. Do not
change versioned prompts, code or settings to resume the same run; use a new
run name/protocol when those change.

The direct Python commands use Modal's normal `app.run()` SDK interface. The
equivalent `modal run modal_variant_a.py` commands are also supported. Direct
execution avoids loading the CLI's unrelated login-server dependencies, whose
local file reads stalled during setup. Modal, Rich and Markdown-it were
refreshed at their existing versions (1.6.0, 15.0.0 and 4.2.0 respectively);
the frozen preprocessing dependencies were not changed.

Later, an attached v2 run lost its local network connection after 22 completed
records. A CPU-only sync recovered those records with the same run fingerprint.
Repeated dependency-file reads under the project directory also stalled, so
recovery used a temporary Python 3.12.15 environment outside the project,
installed from the existing `requirements.lock` and `requirements-modal.txt`.
The remaining sources were resumed with the same code, prompts, model and
input fingerprint in a detached app. No recovered records were regenerated.

The working recovery environment for this session is
`/private/tmp/gga-variant-a-recovery-venv`; its bytecode cache is also outside
the project. It is temporary and may need recreating after a restart. A sync
through that environment uses:

```sh
/private/tmp/gga-variant-a-recovery-venv/bin/python \
  -X pycache_prefix=/private/tmp/gga-variant-a-recovery-pycache \
  modal_variant_a.py --sync
```

For network-resilient execution, Modal's SDK supports a detached app context:

```python
import modal
import modal_variant_a as runner

with modal.enable_output(), runner.app.run(detach=True):
    runner.main(execute=True, max_new=53)
```

The worker still has one H200 container, no automatic generation retries,
per-stage Volume commits and a one-hour deadline. Detaching changes the
connection lifecycle, not the experimental protocol.

## Files

`data/translations/grounded_a_gemma4_modal_development_v2/` contains:

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Protocol, model, prompt/code/input fingerprints |
| `inputs.json` | Frozen source/draft references, numbered pages, valid English spans |
| `selections/bar_*.json` | Selection JSON, full request, schema/grammar, raw response, tokens, runtime, timing |
| `results/bar_*.json` | Source, original draft, fixed selection, verbatim evidence and refinement record |
| `comparison.csv` | Source, blind/refined translations, evidence, match flags and per-stage token counts |
| `grounded_candidates.jsonl` | Exact/partial model matches without flagged contradiction |
| `unmatched.jsonl` | No supporting English evidence; translations retained separately |
| `contradictory.jsonl` | Outputs with explicitly flagged source/evidence conflicts |
| `summary.json` | Completion, model-label/partition counts, edits and token totals |

All outputs have `quality_status: unassessed`. Grounded candidates are not
accepted pairs or gold alignments. The CSV and partitions are regenerated from
validated saved records, so partial runs can also be inspected.

New-token totals include validated selections and refinements, including a
selection saved before an interrupted refinement. Reused blind-stage tokens
are reported separately. Raw failed-stage token counts remain in the remote
failure records. Stage duration excludes model/container startup. Modal bills
GPU/CPU time rather than these token totals; use its run dashboard for costs.

The protocol/validation code is [variant_a.py](variant_a.py); the Modal adapter
is [modal_variant_a.py](modal_variant_a.py). Regression tests are in
[tests/test_variant_a.py](tests/test_variant_a.py).

```sh
.venv/bin/python -m unittest discover -s tests -p test_variant_a.py -v
```

The order and both-page logic adapt the original workspace's
`grounded_a_dev_batch.py` and `grounded_b_dev_sample.py`, with separate stages,
Gemma 4 hosting, frozen-input verification and strict contiguous-span validation.

Version 1 completed two pilot sources, then its validator stopped at Bozn
S0002 because the model selected nonadjacent English S0013 and S0020. Its
per-ID JSON schema did not enforce adjacency during decoding. The two results,
raw failed selection and code snapshot remain in the v1 output directory.
Version 2 enforces complete spans directly in the generation grammar; its CPU
preflight parses both article grammars before GPU allocation. The prompts,
checkpoint, blind drafts, source/candidate pools and generation limits are
unchanged. Version 2 generates fresh results rather than mixing in v1 outputs.

References: [vLLM structured outputs](https://docs.vllm.ai/en/v0.21.0/features/structured_outputs/),
[Modal Volumes](https://modal.com/docs/guide/volumes).
