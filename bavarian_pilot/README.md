# Bavarian–English pilot

This experiment tests whether using a linked English Wikipedia article helps
an LLM translate Bavarian more faithfully. Gemma 4 was self-hosted on Modal.

The dataset contains 10 article pairs: 2 for development (53 Bavarian
segments) and 8 for evaluation (524 segments). The articles cover the
same topics but are not necessarily translations of each other.

## Code

- `download_articles.py`, `prepare_articles.py`, `source_quality.py`: download,
  clean and segment the articles, check source quality and create candidate spans.
- `translate_blind.py`: translation without the English article.
- `mine_baselines.py`: LASER-1 and BGE-M3 mining.
- `variant_a.py`: start with a blind translation, select English evidence and refine.
- `variant_b.py`: select English evidence before translating.
- `variant_c.py`: refine with additional English evidence found through web search.
- `rerank_only.py`: select existing English text without rewriting it.
- `output_quality.py`, `semantic_evaluation.py`: technical checks and AI assessment.
- `evaluation_pipeline.py`: inspect the completed run and recompute its scores.

Prompts are in `prompts/`, settings in `experiments/`, and article snapshots,
prepared inputs and results in `data/`. Optional GPU runners are in
`runners/modal/` but offline analysis does not require Modal.

## Results

On the eight evaluation articles, variant B had the highest mean fidelity
score: **3.560/4**.

[Results table](data/evaluation_runs/frozen_eight_v2/assessment/results.csv)
and [individual outputs and scores](data/evaluation_runs/frozen_eight_v2/assessment/comparison.csv).

