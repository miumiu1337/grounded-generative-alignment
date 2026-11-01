# Grounded Generative Alignment

Master's thesis on reference-grounded parallel-data generation for low-resource machine translation.

## Structure

```text
language_selection.py       Candidate language selection
preprocessing/             FineWiki text, segmentation and eligibility
parallel_data_generation/  Blind translation, mining, grounded A/B/C and reranking
bavarian_pilot/
  config/                  Model settings, procedure and reviewed decisions
  data/article_split.json  Two development and eight training article pairs
  prompts/                 Translation and evidence prompts
  results/run_index.json   Method definitions and development counts
```

## Procedure

Use a pinned FineWiki snapshot, pair articles by Wikidata ID, segment both languages, review boundaries and eligibility, then construct adjacent English evidence spans. Original article text and offsets are preserved.

All parallel-data generation methods live together in `parallel_data_generation/`:

- Blind translation without English article evidence.
- Mining with LASER-1 and BGE-M3: rank existing English spans by ratio margin; no acceptance threshold is currently applied.
- Grounded A: blind translation, evidence selection and refinement.
- Grounded B: read both numbered articles, select evidence and translate.
- Grounded C: blind translation, external English evidence, refinement and self-verification.
- Pure reranking: reuse B's evidence selection and export existing English verbatim without generating a translation.

## Bavarian pilot

Development v1 used 57 Bavarian units. V2 uses 54 shared inputs and 837 eligible English units after recorded context-only exclusions. Six v2 subsets have 54 candidate pairs each; reranking has 21, with 30 unmatched and three rejected cases retained in coverage.

The procedure is approved for generating training datasets from the remaining eight article pairs. Those eight are training-data sources, not the final translation benchmark. Evaluation will use independent reference translations. Outputs remain unassessed candidates; model self-verification is diagnostic.

[Pilot details](bavarian_pilot/README.md) · [Reviewed decisions](bavarian_pilot/config/finewiki_revision_plan.json) · [Frozen procedure](bavarian_pilot/config/protocol_freeze_v2.json)

Execution infrastructure and account configuration are maintained outside this repository. No end-to-end speedup or gold segmentation-accuracy comparison has been established.
