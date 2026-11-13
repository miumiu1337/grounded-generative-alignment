# Grounded generative alignment

FineWiki preprocessing and five methods for generating English–X parallel data.

- `Cleaning/`: article structure, SaT boundaries, DataTrove quality filters and deduplication.
- `preprocessing/prepare_articles.py`: Wikidata article matching and pinned article retrieval.
- `preprocessing/context.py`: previous sentence, focus sentence, next sentence, within one paragraph and section.
- `parallel_data_generation/pipeline.py`: Blind, mining, A, B and C.
- `parallel_data_generation/prompts/`: editable system prompts.
- `configs/asturian.json`: language, tokenizer proxy, models and retrieval settings.

Change `source_language`, `source_language_name`, `quality_tokenizer_language` and `reference_headings` for another language. Review the tokenizer proxy and heading names for that language. Set `mining.encoder` to `bge_m3` or `laser`; these are backends of one method. LASER also needs a tokenizer language and its original model assets.

## Preprocessing

Use Python 3.12 in a separate environment from the original NLLB training environment:

```bash
python -m pip install -r requirements-preprocessing.txt
python -m Cleaning.prepare_segmenter --destination models/sat
python -m preprocessing.prepare_articles --config configs/asturian.json --output work/articles --maximum-articles 100
python -m Cleaning.run --config configs/asturian.json --articles work/articles --model-directory models/sat --output work/cleaned
```

The first retrieval indexes the pinned English metadata and can take a while. Completed metadata and article files are reused. Omit `--maximum-articles` for all matched articles; capped article selection uses seed 42.

Cleaning preserves original text and offsets and records exclusion reasons. SaT replaces the older sentence-boundary step; DataTrove quality filters follow it. No capitalization, SVO or finite-verb rules are applied. See [Cleaning](Cleaning/README.md) for the order and limitations. SaT has no text prompt to edit.

## Generation

Run on a machine with the required models. The LLM methods connect to an already running llama.cpp server; this code does not allocate a cloud GPU. Verify the checkpoint SHA256 and chat template on the server against the experiment configuration. A matching filename alone does not verify model weights. The Asturian deployment used Gemma 4-26B-A4B-it, UD-IQ4_XS, llama.cpp b12110, 4096 context tokens, temperature 0 and seed 42.

```bash
python -m parallel_data_generation.prompts --config configs/asturian.json
python -m parallel_data_generation.pipeline --config configs/asturian.json --cleaned work/cleaned --output work/generated --method blind --condition sentence_only --maximum-sources 5000 --endpoint http://localhost:8089
```

Choose `blind`, `mining`, `variant_a`, `variant_b` or `variant_c`. Use `--condition paragraph_context` for context runs. Both conditions use the same ordered source cohort and English pool: only focuses with full three-sentence windows enter this paired comparison. Sentences at paragraph edges remain in the cleaning audit but have no full window. `--maximum-sources` is a cap, not a promise that 5000 eligible sources exist.

Training pairs contain the original X-language focus and one English translation or mined focus. Context is separate. B receives no Blind draft. A keeps Blind exactly when no evidence was selected. C keeps Blind unless a quoted correction passes support, fidelity and completeness checks; self-verification remains a model judgment.

A/B evidence may be one sentence or two originally adjacent sentences. This differs from the three-sentence context window. Long articles use overlapping selection requests with full candidate coverage. There is no separate reranking method.

Raw requests and per-source results are saved and reused on rerun. Invalid responses and unmatched sources are recorded as deferred; network failures propagate. Use durable output storage. Install `requirements-generation.txt` for BGE/LLM generation. LASER-1 uses the separate Linux/Python 3.10 environment in `requirements-mining-laser.txt`.

## Variant C with FineWeb

C retrieves from a local FineWeb archive index, without a search API key. Obtain the exact configured shard; the indexer checks its SHA256:

```bash
hf download HuggingFaceFW/fineweb sample/10BT/000_00000.parquet --repo-type dataset --revision 9bb295ddab0e05d785b879661af7260fed5140fc --local-dir work/fineweb
python -m parallel_data_generation.fineweb --config configs/asturian.json --cleaned work/cleaned --shard work/fineweb/sample/10BT/000_00000.parquet --output work/fineweb.sqlite
```

Run `variant_c` with `--fineweb-index work/fineweb.sqlite`. Records retain URLs, crawl dates, shard identity, document hashes and passage offsets. Wikimedia and known mirrors are excluded. Literal topic matching and BM25 retrieve candidates; this one sample shard has limited coverage and is not a live or exhaustive web crawl.

## Checks and earlier experiments

Run `python -m unittest discover -s tests` with preprocessing dependencies and `sacrebleu==2.5.1` installed. Tests use local fixtures and mocked teacher calls. Translation-model inference and full-corpus throughput require a separate hosted run.

The older 20k Bavarian preparation, NLLB training and Tatoeba evaluation remain under `runners/` and `experiments/`, with dependencies in `requirements.txt`. `snapshot.json` records the older snapshot, not hashes of this refactor. Existing scores describe those earlier experiments. [Results and datasets](https://huggingface.co/adahmpl/datasets).
