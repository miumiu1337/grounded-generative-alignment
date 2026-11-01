# Bavarian–English FineWiki pilot

`prompts/` contains translation/evidence prompts. `config/` records model settings, reviewed eligibility decisions and the approved procedure. `data/article_split.json` retains the original topic labels: two development pairs and eight pairs now designated for training-data generation.

All methods are in [parallel_data_generation/](../parallel_data_generation/): blind translation, LASER/BGE-M3 mining, grounded A/B/C and extractive reranking. [results/run_index.json](results/run_index.json) records method definitions, model versions and v1/v2 counts. Dataset artifacts are maintained separately from this source repository.

V2 has 54 shared inputs and 21 reranking targets. The procedure is approved for the remaining eight training article pairs. Translation quality remains unassessed; independent reference benchmarks will evaluate the trained translation models.
