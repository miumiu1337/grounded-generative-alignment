# Reranking without rewriting

For each unchanged Bavarian source sentence, read its numbered Bavarian and
English articles and select the best existing English evidence, or `NONE`.
Resolve the selected IDs to verbatim frozen English text. There is no English
translation, refinement, external search or other rewriting in this condition.
The LLM produces selection metadata; "no generation" here means no generated
English sentence, not zero output tokens from the selector.

## Completed development results

All **53 sources** were exported on 2026-10-25, using the already completed
draft-free selection stage of Variant B. Those requests saw both pages and
could return only source ID, evidence IDs, match status and a contradiction
flag. They were saved before B's translation stage. The exporter never reads
B's generated translations or calls an inference API.

This is a **paired ablation sharing B's selection decisions**, not a fresh
independent model run. Keeping the same selections lets us compare verbatim
English evidence with B's generated English for the same source and evidence.
Results are therefore identical to B's match labels by construction.

| Article | Sources | Model-labelled exact | Partial | None |
| --- | ---: | ---: | ---: | ---: |
| Bozn | 39 | 3 | 10 | 26 |
| Kristndum | 14 | 0 | 6 | 8 |
| Total | 53 | 3 | 16 | 34 |

There are no model-flagged contradictions. The 19 selected spans are
**candidates, not confirmed parallel pairs**. Partial matches may omit source
claims or add claims that cannot be removed in this condition. Even an `exact`
label is unassessed: Bozn S0013 selects an English sentence with dates and
wartime events absent from the focus source. Preserve these outputs for
evaluation rather than silently cropping or rewriting them.

New inference calls and new inference tokens: **0**. The reused selection
stages account for **1,062,890 prompt and 2,916 output tokens**. A standalone
reranking run would incur selection cost. Count shared selection compute only
once when reporting actual study spending; do not assign B's translation
tokens to this ablation. Tokens do not represent full Modal billing.

All 53 input/selection records passed validation against the frozen B run,
including request contents, schemas, span constraints, completion status,
runtime versions and source IDs. Five new offline tests cover verbatim spans,
no translation-stage dependency, separate result groups, invalid/incomplete
records and artifact tampering. A final artifact audit checks CSV/JSONL
coverage, verbatim evidence, provenance hashes and token totals.

## Selection protocol

- Source: the same 39 Bozn and 14 Kristndum development sentences as A–C.
  The eight evaluation article pairs are untouched.
- Context: both complete eligible numbered article texts, with paragraph and
  section context. No blind draft, mining rankings or external evidence.
- Candidate pool: all frozen English one- or two-sentence spans in the linked
  article (327 Bozn and 1,012 Kristndum spans). Two-sentence spans must be
  consecutive within one paragraph. There is no top-N mining shortlist.
- Decision: reuse the existing B selection prompt and constrained grammar;
  choose one span or `NONE`, and retain exact/partial/none and contradiction
  labels without postprocessing them.
- Pair output: original Bavarian sentence and the selected candidate's exact
  stored `text`, sentence IDs, paragraph/section metadata and revision IDs.
  Unmatched cases have no English text. Exact and partial candidates are
  exported separately; contradictions have their own partition.

The selector was Gemma 4 `google/gemma-4-26B-A4B-it`, revision
`47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, with temperature 0, seed 42,
thinking disabled and the same 65,536-token context as B. See the archived
source manifest for all settings. Reuse introduces no new GPU run or dependency.

## Files and reproduction

| File | Purpose |
| --- | --- |
| [rerank_only.py](rerank_only.py) | Validate saved selections and export original English spans; no inference path |
| [Selection prompt](prompts/grounded_b_selection_v1.txt) | Existing draft-free both-page selection instructions, reused unchanged |
| [variant_b.py](variant_b.py) | Frozen request and record validation |
| [variant_a.py](variant_a.py) | Frozen candidate resolution, span grammar and selection rules |
| [tests/test_rerank_only.py](tests/test_rerank_only.py) | Offline regression checks |

Outputs are under `data/reranking/rerank_only_gemma4_development_v1/`:

- [comparison.csv](data/reranking/rerank_only_gemma4_development_v1/comparison.csv): all 53 sources, original English text, match labels and selection tokens.
- [summary.json](data/reranking/rerank_only_gemma4_development_v1/summary.json): counts and shared/new usage.
- [exact_candidates.jsonl](data/reranking/rerank_only_gemma4_development_v1/exact_candidates.jsonl): 3 model-labelled exact candidates.
- [partial_candidates.jsonl](data/reranking/rerank_only_gemma4_development_v1/partial_candidates.jsonl): 16 partial candidates.
- [unmatched.jsonl](data/reranking/rerank_only_gemma4_development_v1/unmatched.jsonl): 34 sources without a selected span.
- `contradictory.jsonl`: empty for this run.
- `results/`: source, verbatim evidence and selection provenance per source.
- `selections/`: original selection requests, schemas, grammar, raw responses,
  runtime versions, timestamps and tokens, copied unchanged from B.
- `inputs.json`, `source_run_manifest.json`, `run_manifest.json`: frozen
  source/page/candidate snapshot, original B manifest and derived-run manifest.
  The original manifest includes B's historical translation configuration for
  provenance; no translation-stage result is used by this export.

Source B run fingerprint:
`443db3a01d50baa81e8e55ed5069096c207e2eb64ad50cf7bcca890a0b3f9b64`.
Reranking export fingerprint:
`77aa7864c232a8158e816ca2fe99f8c067882aa2e9904ef2de9af7f3f2d2b296`.
The derived manifest binds the source run, all selection records, source/page
inputs and exporter code. Modified existing artifacts are rejected, not
overwritten; protocol changes require a new versioned output directory.

From this workspace, using the existing Python 3.12 environment:

```sh
.venv/bin/python rerank_only.py
.venv/bin/python -m unittest discover -s tests -p test_rerank_only.py -v
```

The temporary recovery environment used for A–C also supports this offline
export, if project environment reads stall:

```sh
/private/tmp/gga-variant-a-recovery-venv/bin/python -u \
  -X pycache_prefix=/private/tmp/gga-rerank-pycache rerank_only.py
```

Matching precision/recall, fidelity and comparisons with LASER/BGE-M3 remain
evaluation work. Agreement with B or an embedding score does not establish
correctness. This exporter is scoped to the completed Bavarian development
pilot; new sources need their own selection calls and versioned protocol.
