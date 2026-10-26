# Variant C: search-assisted English refinement

C reuses the 53 saved Gemma 4 blind Bavarian-to-English translations, searches
for additional English evidence beyond Wikipedia, refines the English, then
checks the result against the unchanged Bavarian source. Reranking without
rewriting is a separate ablation and is not implemented by C.

This follows `../first-step.docx` and `../next_steps_v2.md`, plus the clarified
translate → external search → refine → verify sequence. The notes leave search
scope/budget and verification design open; the choices below are development
settings, not a frozen final study design.

## Status: 2026-10-25

**All 53 development sources completed refinement and verification on Modal.**
One source from each article completed first; structural validation passed,
then the remaining 51 ran under the same frozen protocol. Both GPU workers
shut down normally. No evaluation article pairs were processed.

All 61 applicable offline tests passed; six mining-specific tests were skipped
in this environment. The eight C tests cover input/citation validation,
tracking-URL duplicates, edit reconstruction and interrupted verification.
The completed run fingerprint is
`0b4b35285423b98cb54e0f4c8274a10da0e5b135c24b8b80bc5089f3d3bdd742`.
It binds the code, prompts, settings, original drafts, numbered pages, cached
evidence, search plan, collection recipe and all 53 search-audit files.
The cache covers the same 39 Bozn and 14 Kristndum development sources as A/B.
There are no requests for the eight evaluation article pairs.

| Article | Sources | Self-check pass | Self-check flag | Self-check uncertain |
| --- | ---: | ---: | ---: | ---: |
| Bozn | 39 | 33 | 6 | 0 |
| Kristndum | 14 | 10 | 4 | 0 |
| Total | 53 | 43 | 10 | 0 |

C changed 31 blind drafts and reported using external references in only
3 outputs: Bozn S0002 (`WEB1`), Bozn S0035 (`WEB2`) and Kristndum S0004
(`WEB1`, `WEB2`). Bozn S0002 cited evidence without changing its draft.
Evidence was supplied for 49 sources; citation counts describe reported use,
not whether the evidence improved a translation. All four no-evidence sources
also completed and remain in the exports.

The verifier reported eight mistranslation issues and two omission issues.
These are model opinions, **not measured errors or precision**. In particular,
Kristndum S0009 and S0011 have `flag` labels whose issue descriptions then say
the translation is faithful or that no substantive error was found. Their
original outputs are retained without relabelling. The pilot's Bozn S0001
also retains the awkward phrase "a Boazn" despite a self-check pass. These
observations show why the self-check cannot establish translation quality;
neither the 43 passes nor the 10 flags are accepted/gold labels.

The final local audit validated all 106 saved stages, unchanged source/draft/
search inputs, schemas, allowed citations, normal completion, stage order,
runtime versions, edit reconstruction, CSV contents, token totals, and
disjoint/exhaustive self-check JSONL partitions. The fingerprint remained
unchanged throughout both runs. CPU preflight checked all 106 schemas and
53 refinement prompts (14,428–36,754 tokens); no inputs were truncated.

- [Two-source Modal run](https://modal.com/apps/denise-obrein/main/ap-SsOvkg7DCSahZALjfUNwJQ)
- [Remaining-51 Modal run](https://modal.com/apps/denise-obrein/main/ap-JIyKTy7IrLwkEwE6Orz3VQ)
- [Results comparison CSV](data/translations/grounded_c_gemma4_modal_development_v1/comparison.csv)
- [Machine-readable summary](data/translations/grounded_c_gemma4_modal_development_v1/summary.json)
- [Run manifest](data/translations/grounded_c_gemma4_modal_development_v1/run_manifest.json)
- [Self-check flags with original issue descriptions](data/translations/grounded_c_gemma4_modal_development_v1/selfcheck_flag.jsonl)

New refinement/verification usage was **1,241,186 prompt tokens and 6,927
output tokens** across 106 calls. Reused blind translations account separately
for 6,540 prompt and 1,541 output tokens. These totals do not include web-tool
usage and are not a Modal invoice; GPU startup and CPU/memory are additional.

| Search-cache property | Count |
| --- | ---: |
| Sources with a cached search record | 53 |
| Sources with external references | 49 |
| Sources without eligible English evidence | 4 |
| Cached references across source records | 92 |
| Distinct reference pages after tracking-URL normalization | 70 |
| Opened-page excerpts | 52 |
| Explicitly labelled search snippets | 40 |
| Logged page-access failures | 30 |

`ok` means reference text was collected, not that it supports a source claim.
Some inaccessible pages have a cached search snippet; the failed page access
remains logged and the snippet is not labelled as opened-page evidence.
Four queries returned no eligible English evidence: Bozn S0003, S0007, S0024
and S0038. These sources remain in the experiment.

The complete cache is under `data/search/grounded_c_web_development_v1/`:

- `plan.json`: deterministic queries and budgets, bound to frozen sources/drafts.
- `sources/bar_*.json`: query, search status, URLs, titles, UTC capture times,
  evidence excerpts/types, provider references and access failures.
- `audit/bar_*.json`: returned-result URL metadata and attempted pages.
- `collection_protocol.json`: tool collection recipe and limitations.
- `summary.json`: counts and input/search-plan fingerprints.
- `collection_debug/` and `pre_freeze_normalization.json`: pre-inference
  collection corrections and removal of one tracking-URL duplicate.

## Search protocol

The user selected collection through the connected Codex web tool. Gemma does
not browse. No separate search API key or dependency was added. Live collection
is performed through that tool; the Python pipeline replays saved evidence
and does not issue fresh web searches.

Each source has one query: the English article title, up to 26 word tokens of
its saved blind draft, and Wikipedia/Wikimedia exclusion operators. This can
inherit draft errors; queries are not corrected after seeing model outcomes.
The budget is at most two distinct attempted external pages per source and
25 words per cached excerpt. The short evidence windows may miss necessary
context and cannot establish full claim coverage.

The first three queries were submitted together and returned a shared source
list; their selection was manual and AI-assisted. Subsequent queries were
submitted individually, with up to three sources collected concurrently.
English-text and primary-domain heuristics preferred English primary sources
in returned order. Candidate URLs, selected URLs and the collection recipe are
retained. This is AI-assisted evidence collection, not a controlled comparison
of web-search engines or a guarantee that references are relevant/independent.

Opened HTML uses extractive text windows. PDF references and unavailable page
bodies use explicitly labelled search snippets. Reused URL excerpts stay
fixed. Capture timestamps describe the cached tool evidence; a provider's
crawl or a previously opened page may be older.

An initial tool-adapter URL parsing error produced three false empty records;
these were corrected from the original tool responses before any model run.
PDF line-format handling and snippet labelling were also corrected during
collection. The diagnostic entries are retained. One duplicate Pew URL that
differed only by a tracking parameter was removed before freezing. These are
pre-inference cache corrections, not translation edits or post-hoc exclusions.

## Model protocol

1. Validate preprocessing 1.0.1, the complete original blind run and all 53
   search records. Missing cache records block execution. A completed search
   with no eligible evidence is an explicit condition, not a missing record.
2. **Refinement call:** provide the focus source, original blind draft, eligible
   numbered Bavarian and English Wikipedia pages, and the cached external
   search record. Return a faithful translation and the external IDs actually
   used, or an empty citation list. Preserve the source's claims even when
   current references disagree. No live search occurs in this call.
3. Save and commit the refinement before verification. Derive a character-level
   edit log directly from the unchanged draft and the model output.
4. **Verification call:** provide the focus Bavarian source, Bavarian context,
   refined translation, cited IDs and external excerpts. Omit the blind draft
   and English Wikipedia page. Return `pass`, `flag` or `uncertain` and issues
   for additions, omissions, mistranslations, contradictions or citations.
   This call cannot rewrite the translation.
5. Retain every output in the main CSV. Export self-check verdicts separately;
   do not automatically repair failures or treat passes as accepted pairs.

The verifier is a fresh call to the **same model**, not an independent assessor.
Its confidence and flags may be wrong. All results use
`quality_status: ai_self_checked_not_gold`. Separate method-blind quality
evaluation remains future work. Preserve C as a separate condition; search
collection effort and the verification stage belong in its cost accounting.

Model/settings match A/B: `google/gemma-4-26B-A4B-it`, checkpoint
`47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`, vLLM 0.21.0, Transformers 5.5.4,
H200, bfloat16, temperature 0, seed 42, thinking disabled, up to 1,024 output
tokens per call, and a 65,536-token context. Both calls use JSON schemas.
CPU preflight checks all refinement prompt lengths and both JSON schemas for
every source before GPU allocation, including sources with no external IDs;
each stage checks its own context budget without truncation. The worker uses
one single-use GPU container, a one-hour deadline and zero automatic retries.

## Code and run commands

| File | Purpose |
| --- | --- |
| [variant_c.py](variant_c.py) | Search-plan/cache validation, requests, evidence IDs, edit logs and exports |
| [modal_variant_c.py](modal_variant_c.py) | Gemma refinement/verification, per-stage commits and recovery |
| [Experiment configuration](experiments/grounded_c_gemma4_modal_development_v1.json) | Model, search budget, prompts and frozen-input binding |
| [Refinement prompt](prompts/grounded_c_refinement_v1.txt) | Source-faithful English refinement with logged evidence |
| [Verification prompt](prompts/grounded_c_verification_v1.txt) | Source fidelity and citation self-check without rewriting |
| [tests/test_variant_c.py](tests/test_variant_c.py) | Offline validation, edit-log and interruption/resume tests |

From this workspace, use the existing Python 3.12 environment:

```sh
.venv/bin/python variant_c.py --prepare-search  # Validate/reuse existing query plan
.venv/bin/python modal_variant_c.py             # Offline input/cache validation
.venv/bin/python modal_variant_c.py --execute   # Two sources, one per article
.venv/bin/python modal_variant_c.py --execute --max-new 53 --detach
.venv/bin/python modal_variant_c.py --sync      # Recover committed stages
```

Paid execution requires the intended Modal profile and approval. The existing
temporary recovery environment also works if project dependency reads stall:

```sh
/private/tmp/gga-variant-a-recovery-venv/bin/python \
  -X pycache_prefix=/private/tmp/gga-variant-c-pycache modal_variant_c.py
```

Outputs are saved separately under
`data/translations/grounded_c_gemma4_modal_development_v1/`: `run_manifest.json`,
`inputs.json`, `refinements/`, `results/`, `comparison.csv`, `summary.json`, and
`selfcheck_pass.jsonl`, `selfcheck_flag.jsonl`, `selfcheck_uncertain.jsonl`.
Requests, schemas, raw responses, tokens, runtime versions and timestamps are
saved for each stage. Failed responses are preserved remotely under `failures/`.
Resume checks code, settings, sources, drafts, evidence and saved stages.
Changes after inference starts require a new versioned run, not edited caches.

Schema preflight uses the documented
[XGrammar JSON-schema interface](https://xgrammar.mlc.ai/docs/latest/api/python/grammar.html).

Reused blind-stage tokens and new-stage tokens are separate. Neither token
totals nor inference-stage durations represent complete Modal billing: startup,
CPU preflight and web collection are additional operational costs.
