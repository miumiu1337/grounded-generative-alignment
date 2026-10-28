# Grounded Generative Alignment

Master's thesis on reference-grounded generation of parallel data for low-resource machine translation.

## Bavarian–English pilot

[Code and data](bavarian_pilot/) · [Results](bavarian_pilot/data/evaluation_runs/frozen_eight_v2/assessment/results.csv)

## Scripts

- `language_selection.py` - identifies candidate Wikipedia languages by article counts and Google Translate support, or collects linked English articles through Toolforge.

## Toolforge article collector

I developed `toolforge_articles.py` to collect source-language/English Wikipedia article pairs with revision-pinned text and resumable downloads.

With the project in a Toolforge tool account and `requirements.txt` installed in a Python 3.13 `.venv`, run from the project directory:

```bash
toolforge jobs run collect-bar --image python3.13 --command \
  "$PWD/.venv/bin/python $PWD/language_selection.py --x-lang bar --limit 10 --output-dir $HOME/corpora/bar-en"
```

Replace `bar` with another Wikipedia language code. For the full corpus, replace `--limit 10` with `--all` and choose a new output directory. Rerun the same collection command to resume.
