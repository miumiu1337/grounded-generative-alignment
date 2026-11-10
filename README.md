# Bavarian experiments

Tests on preprocessing, parallel-pair construction, fine-tuning and Tatoeba evaluation.

- `runners/prepare_nllb_*.py`: matched training pairs and overlap checks
- `runners/nllb_blind_training.py`: shared training worker; `nllb_modal_execution.py` adds persistent storage
- `runners/nllb_tatoeba_eval.py`: decoding and chrF++/BLEU scoring
- `parallel_data_generation/`: prompts, evidence validation and mining
- `experiments/`: original training manifests and generation settings

Each condition uses 20k sentence pairs, the same 500 validation pairs and the original base initialization.

Run checks with `python -m unittest discover -s tests`. Data-preparation scripts retain the original relative archive paths; they require the frozen source artifacts. The preparation script expects its enclosing checkout to be named `grounded-generative-alignment`. Training and evaluation require a hosted CUDA GPU.

`snapshot.json` records file hashes. 

For results and parallel datasets see [Hugging Face](https://huggingface.co/adahmpl/datasets).

Current data curation: `python -m runners.curate_datatrove_parallel`. Install `requirements-quality.txt` in a separate environment; the original training dependencies stay in `requirements.txt`. Quality checks use `python -m unittest discover -s tests -p test_datatrove_sentences.py` in that environment.

The current Hugging Face tables contain only `source_lang`, `target_lang`, `source_text`, `target_text`, `source_article` and `target_article`. DataTrove filtering and deduplication produce a smaller matched cohort; earlier preparation passes remain for tracing historical inputs. Existing model scores refer to the original 20k data.
