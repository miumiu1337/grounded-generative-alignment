# Development mining: LASER-1 and BGE-M3

Both baselines use the 53 original Bavarian development sentences used by the
blind translation baseline. They do not read Gemma translations. Candidates
are the frozen English one/two-sentence spans within each linked article:

| Article | Bavarian sources | English candidate spans |
| --- | ---: | ---: |
| Bozn / Bolzano | 39 | 327 |
| Kristndum / Christianity | 14 | 1,012 |

No evaluation articles are encoded. Candidate texts, sentence IDs, revisions,
paragraphs and sections are retained. Neither baseline rewrites any text.

## Completed run: 2026-10-24

Both models completed all 53 sources and saved five candidates each (265
candidate records per model). The run fingerprint is
`ce3019d872a7656223716872e408762e48ba36b6a04980991ebad4b12f83818a`.

| Article | Sources | Exact same top span | Top spans sharing any sentence |
| --- | ---: | ---: | ---: |
| Bozn | 39 | 4 | 5 |
| Kristndum | 14 | 0 | 0 |
| Total | 53 | 4 (7.5%) | 5 (9.4%) |

Mean top-five span Jaccard overlap is 0.0738. LASER chose two-sentence spans
for 24 sources, BGE-M3 for 13. This is a disagreement analysis, not a quality
ranking. Sources may have partial or no English counterparts, and overlapping
spans can express related evidence without identical IDs.

[Readable comparison](data/mining/laser_bge_development_v1/comparison.md) ·
[Side-by-side CSV](data/mining/laser_bge_development_v1/comparison.csv) ·
[Full comparison JSON](data/mining/laser_bge_development_v1/comparison.json)

Validation passed: six mining regression tests, all 35 existing tests, complete
input/cache fingerprints, and reproduction of every saved score matrix and
ranking. Re-encoding the first and last source/candidate of each article
confirmed vector order for both models (cosine with saved vectors at least
0.9999999). All sources also match the blind-translation run. The preprocessing
freeze is unchanged. Thresholds and translation/alignment quality remain
unassessed; the 524 evaluation sources were not processed.

## Models and scoring

- **LASER-1**, the original `bilstm.93langs.2018-12-26.pt` checkpoint through
  `laserembeddings==1.1.2`, running on CPU. Bavarian uses German Moses
  tokenization as an explicit approximation; English uses English tokenization.
  This is a close reproduction using a third-party implementation, not an
  exact reproduction of the original LASER toolkit or a LASER-2/3 experiment.
- **BGE-M3 dense embeddings** through the existing local Ollama service, F16,
  1,024 dimensions, context 8,192, truncation disabled. The actual Ollama model
  digest is pinned; the mutable `latest` tag is checked before and after every
  batch. This is the Ollama build, not a new Hugging Face inference run.

LASER's assets are checked against the original workspace's combined model
fingerprint. The large original file could not be read and timed out, so a
fresh copy was downloaded from the official LASER asset URLs. The resulting
fingerprint matches. Nothing in the original workspace is changed.

Bavarian-specific performance must be tested rather than inferred from either
model's multilingual description. LASER lacks a dedicated Bavarian model in
its published language list. BGE-M3's model card does not establish Bavarian
alignment accuracy.

Both models use L2 normalization, cosine similarity, and the same ratio margin:

```text
margin(x,y) = cosine(x,y) / ((mean(top4 cosine(x, English candidates))
                          + mean(top4 cosine(Bavarian sources, y))) / 2)
```

Neighborhoods are computed separately within each development article pair,
using all 53 source sentences in their respective articles (39 and 14), and
all English spans. Save the five highest margin candidates per source; ties
follow the frozen candidate order. Also retain the highest-cosine choice and
whether the top margin choice is reciprocal. No one-to-one assignment or
automatic duplicate removal is applied. Overlapping one/two-sentence spans
can occupy multiple neighborhood positions; this affects both models.

**There is no acceptance threshold yet.** The score grid in the experiment
file is descriptive sensitivity analysis, not calibrated precision or accepted
parallel pairs. The old experiment's cutoffs are not adopted here. Scores
from different encoders are not interchangeable quality measurements.

## Run

Use a separate Python 3.9 or 3.10 environment because the legacy LASER package
requires PyTorch 1.13. The installed environment is `.venv-mining`; the main
`.venv`, `requirements.lock` and frozen preprocessing files are unchanged.

For a fresh setup, create the environment with your Python 3.9/3.10 interpreter:

```sh
python3.10 -m venv .venv-mining
.venv-mining/bin/python -m pip install -r requirements-mining.txt
.venv-mining/bin/python mine_baselines.py --download-models
```

The local Ollama server must be running with the exact BGE-M3 build in
`experiments/mining_development_v1.json`. A different digest causes an error;
do not silently replace it. Model loading/checkpoint validation uses no Gemini
API key and no Modal GPU.

```sh
cd /Users/adriannahampelska/Desktop/tum/thesis/grounded-generative-alignment-clean
.venv-mining/bin/python mine_baselines.py             # Offline input plan
.venv-mining/bin/python mine_baselines.py --execute   # Run both/resume
.venv-mining/bin/python -m unittest discover -s tests -p test_mining.py -v
```

Completed embedding batches are saved atomically. Running again uses those
batches after checking the ordered inputs, code/settings, models and runtime
fingerprint. Incompatible artifacts fail; use a new versioned run for protocol
changes. A local file lock prevents simultaneous invocations. Remote calls
have bounded timeouts and no automatic retries; failures propagate with prior
batches preserved. The model downloader has per-file time/size limits.

## Files and interpretation

Outputs are in `data/mining/laser_bge_development_v1/`:

| File | Purpose |
| --- | --- |
| `run_manifest.json` | Protocol, model hashes/URLs, code/input hashes, runtime versions |
| `inputs.json` | Exact ordered sources and English candidate pools |
| `laser_1/rankings.json`, `bge_m3/rankings.json` | Five candidates and scores for each of 53 sources |
| Per-model `bar_<page>_<bar/en>.npz` | Resumable embedding caches with input/run fingerprints |
| Per-model `bar_<page>_scores.npz` | Full cosine and ratio-margin matrices |
| `comparison.csv` | All 53 sources and both models' top choices side by side |
| `comparison.json` | Exact/overlapping top-choice agreement, top-five overlap, score diagnostics |
| `comparison.md` | Short readable comparison |
| `run_log.jsonl` | Completion times, including cache reads when resuming |

Timing is operational context, not a controlled speed benchmark: LASER uses
CPU and Ollama can use different local acceleration. No Modal inference was
used for this stage. There are no human or automatic quality labels in these
outputs; agreement is not accuracy. Keep unmatched decisions and accepted-pair
yield for a later explicitly specified acceptance rule. Do not report Wikipedia
precision/recall without applicable gold alignments.

References:

- [Original margin-based mining method](https://aclanthology.org/P19-1309/)
- [LASER implementation and language coverage](https://github.com/facebookresearch/LASER)
- [laserembeddings implementation](https://github.com/yannvgn/laserembeddings)
- [BGE-M3 model card](https://huggingface.co/BAAI/bge-m3)
- [Ollama embedding API, including truncation behavior](https://docs.ollama.com/api/embed)
