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



Revised sentence preparation is in `preprocessing/quality_prose.py`; `runners/refine_preprocessing.py` applies it to the frozen articles. `runners/export_quality_context.py` adds full article context to the datasets and exports the revised source inventory. The original 20k training inputs and reported scores remain historical; structural screening does not establish translation correctness.


`preprocessing/retained_cleanup.py` and `runners/retain_source_cleanup.py` preserve all 225,700 original source records and record non-generative repairs. Unresolved clauses remain marked for review. `runners/export_retained_cleanup.py` exports the complete retained inventory with article context and retraining readiness; repaired context requires new aligned English outputs before fitting new models.
