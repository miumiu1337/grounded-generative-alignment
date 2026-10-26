# Grounded Generative Alignment

Master's thesis on reference-grounded generation of parallel data for low-resource machine translation.

## Bavarian–English pilot

The [Bavarian pilot](bavarian_pilot/README.md) contains preprocessing and
parallel-data generation code, prompts, frozen experiment settings, page
snapshots and the exploratory eight-article results. Start with its
[results table](bavarian_pilot/data/evaluation_runs/frozen_eight_v2/assessment/results.csv)
or [evaluation notes](bavarian_pilot/EVALUATION.md).

## Scripts

- `language_selection.py` - identifies candidate Wikipedia languages based on article counts, Google Translate support and article links in English
