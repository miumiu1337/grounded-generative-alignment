# Bavarian experiments

Tests on preprocessing, parallel-pair construction, fine-tuning and Tatoeba evaluation.

- `runners/prepare_nllb_*.py`: matched training pairs and overlap checks
- `runners/nllb_blind_training.py`: shared training worker; `nllb_modal_execution.py` adds persistent storage
- `runners/nllb_tatoeba_eval.py`: decoding and chrF++/BLEU scoring
- `parallel_data_generation/`: prompts, evidence validation and mining
- `experiments/`: original training manifests and generation settings

Each condition uses 20k sentence pairs, the same 500 validation pairs and the original base initialization.

Run checks with `python -m unittest discover -s tests`. Data-preparation scripts retain the original relative archive paths; they require the frozen source artifacts. The pilot exclusion file is included at `bavarian_pilot/data/article_split.json`. The preparation script expects its enclosing checkout to be named `grounded-generative-alignment`. Training and evaluation require a hosted CUDA GPU.

Source files are unchanged; `snapshot.json` records hashes. 

For results and parallel datasets see [Hugging Face](https://huggingface.co/adahmpl/datasets).
