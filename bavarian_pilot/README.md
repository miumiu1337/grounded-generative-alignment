# Bavarian–English grounded alignment pilot

This folder contains the code, prompts, frozen inputs and results for an
exploratory Wikipedia parallel-data pilot. Ten comparable article pairs were
prepared: two for development (53 Bavarian segments) and eight for evaluation
(524 segments). Linked articles cover the same topics but are not assumed to
be translations.

**Variant B leads in the recorded AI assessment**, with mean fidelity 3.560/4
and 388/524 fully faithful outputs passing technical checks. These are
conversational AI judgments, not human-reference benchmark results.

## Start here

- [Overall results](data/evaluation_runs/frozen_eight_v2/assessment/results.csv)
- [Every method/source output, score and assessment note](data/evaluation_runs/frozen_eight_v2/assessment/comparison.csv)
- [Results by article](data/evaluation_runs/frozen_eight_v2/assessment/per_article.csv)
- [Evaluation protocol and limitations](EVALUATION.md)
- [Frozen experiment settings and input/code hashes](experiments/evaluation_eight_v2.json)

## Methods and code

| Component | Main code | Purpose |
|---|---|---|
| Download | `download_articles.py` | Download revision-pinned MediaWiki responses and metadata |
| Preprocessing | `prepare_articles.py`, `source_quality.py` | Parse rendered HTML, preserve sections/paragraphs, segment text, flag quality/duplicate issues and form candidate spans |
| Blind translation | `translate_blind.py` | Translate Bavarian without the English article |
| Embedding mining | `mine_baselines.py` | Rank existing English spans with LASER-1 and BGE-M3 |
| Grounded A | `variant_a.py` | Start with a blind draft, select English evidence, then refine |
| Grounded B | `variant_b.py` | Select evidence from both articles before translating |
| Grounded C | `variant_c.py` | Refine using cached external English evidence, record edits and perform a separate model self-check |
| Reranking | `rerank_only.py` | Export B's selected English evidence verbatim, without rewriting or additional inference |
| Technical checks | `output_quality.py` | Check output structure, source preservation and evidence constraints |
| AI assessment | `semantic_evaluation.py` | Define the rubric, method-hidden inputs, validation and aggregation |
| Frozen evaluation | `evaluation_pipeline.py` | Validate the fixed corpus/settings and calculate evaluation reports |

`modal_evaluation.py` runs the frozen Gemma conditions on Modal. The
`modal_translate_blind.py` and `modal_variant_*.py` files are the development
cloud runners. Their filenames and scientific code are preserved because
imports and experiment hashes depend on them. `prompts/` contains the prompts,
`experiments/` the configurations, and `tests/` the offline regression tests.

## Results on the eight evaluation articles

| Method | Mean fidelity / 4 | Fully faithful + technical pass / 524 |
|---|---:|---:|
| **Grounded B** | **3.560** | **388 (74.05%)** |
| Grounded A | 3.476 | 370 (70.61%) |
| Grounded C | 3.461 | 372 (70.99%) |
| Blind translation | 3.109 | 284 (54.20%) |
| Reranking | 1.759 | 6 (1.15%) |
| BGE-M3 | 0.518 | 4 (0.76%) |
| LASER-1 | 0.298 | 4 (0.76%) |

Means cover returned, confidently assessed outputs. Six methods have one
uncertain source. Reranking returns 170 outputs and abstains on 354 sources;
all 524 sources remain in every yield denominator. Mining scores assess the
first-ranked proposal; no mining acceptance threshold was tuned on these
articles. Existing English spans are assessed for fidelity to the complete
source, including extra or missing claims, so these scores do not measure
gold alignment precision or recall.

Assessment used the Codex assistant in the project conversation, separately
from Gemma 4. Method names were hidden in the assessment inputs, but prior
project context prevents perfect blinding. Exact assessor model/settings and
validated Bavarian expertise are unavailable. No human review or Tatoeba
reference evaluation was performed. The eight articles had been inspected in
an earlier pilot and are not pristine unseen test data. Some frozen source
units are fragments or lexical examples. Actual cloud billing and cost per
accepted pair are not available.

## Included data and reproducibility

- `data/articles.csv`, `article_split.json`, `preprocessing_freeze.json` and
  `provenance.json`: article mapping, split, preprocessing hashes and provenance.
- `data/raw/` and `data/html/`: pinned API responses and rendered HTML snapshots.
  Page URLs and revision IDs are retained for attribution and traceability.
- `data/prepared/`: the same ten prepared article pairs used in the experiments.
- `data/translations/blind_gemma4_modal_development_v3/`: the 53 saved blind
  development translations and their manifest, required by Variant A's
  offline tests.
- `data/evaluation_runs/frozen_eight_v2/assessment/`: result tables, recorded AI
  judgments, assessment inputs and the subsequently unmasked method mapping.
  The `private/` subfolder is the original experiment name for this mapping;
  it contains method labels, not credentials.
- `data/evaluation_runs/frozen_eight_v2/quality/report.json`: the saved technical
  report needed to recompute the assessment tables.

This Git folder is a selected code/input/assessment handoff. The complete
104 MB handoff archive is kept separately; it also contains raw generation
requests/responses, per-stage outputs, mining embeddings and the evaluation
search cache. References to development outputs or complete-run artifacts in
the method documentation describe those separately retained files. Model
weights, virtual environments, credentials, cloud logs and email drafts are
not included here. Toolforge discovery and model fine-tuning are not implemented.

## Local setup and offline checks

From this folder, use Python 3.12 for the core code and Modal SDK:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-modal.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python evaluation_pipeline.py
```

The final command validates the frozen inputs and prints a plan; it does not
start GPU inference. Mining has a separate legacy dependency environment;
see [MINING.md](MINING.md) and `requirements-mining.txt`. Mining-dependent
tests skip when those optional dependencies are unavailable.

To recompute the saved assessment from its recorded judgments, without new
model calls:

```sh
.venv/bin/python evaluation_pipeline.py --score
```

This reproduces the arithmetic, not the conversational AI judgments. Live
inference requires Modal authentication and billable GPU resources; use the
explicit execution instructions in [EVALUATION.md](EVALUATION.md) and
[MODAL.md](MODAL.md). Variant C needs the separately retained search cache
before replay. Do not re-freeze or retune this named experiment after
inspecting evaluation results.
