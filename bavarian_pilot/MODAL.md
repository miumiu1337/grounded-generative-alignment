# Blind Bavarian → English translations on Modal

Run these commands in your IDE's terminal. The Python code and frozen inputs
stay in the clean workspace; Gemma inference runs on a Modal GPU. No Gemini
API key is used.

The development run completed on **2026-10-24**: **53/53 translations** (39 from
Bozn, 14 from Kristndum), all with normal stops and unchanged frozen sources.
There were 6,540 prompt tokens and 1,541 output tokens across these records.
These counts are not GPU billing measurements. Translation quality remains
unassessed; the 524 evaluation sentences were not processed.

The [two-sentence test](https://modal.com/apps/denise-obrein/main/ap-OowugDIe572PViymjv4YvP)
and [remaining development run](https://modal.com/apps/denise-obrein/main/ap-t8ADgBgxtzpl1wqpg2dCEQ)
both completed. Their shared run fingerprint is
`5a6e2f8b0916e5a31e3fac50b08e26a716e287c3db980326286b9b16c82c8a9d`.

```sh
cd /Users/adriannahampelska/Desktop/tum/thesis/grounded-generative-alignment-clean
.venv/bin/python -m pip install -r requirements-modal.txt
```

Authenticate once if your Modal account is not already configured:

```sh
.venv/bin/modal setup
```

You need a Modal account with GPU billing enabled. Do not paste Modal tokens
into scripts or chat; `modal setup` stores the credentials locally.

## Plan and run

Check the 53 development inputs offline, without contacting Modal:

```sh
.venv/bin/python modal_translate_blind.py
```

Start with at most two new translations:

```sh
.venv/bin/modal run modal_translate_blind.py --execute
```

Then complete the development set:

```sh
.venv/bin/modal run modal_translate_blind.py --execute --max-new 53
```

Completed IDs are skipped, so the second command adds only missing translations.
The source set always contains exactly the 53 eligible development sentences;
evaluation articles are not selected. Each inference receives the fixed
translation instruction and one Bavarian sentence, with no English article,
mined candidates, search or chat history.

The first invocation may take several minutes to build the container and load
the model. `--max-new` bounds the number of new translations, not GPU startup
time or cost. The worker uses one H200 GPU and has a one-hour timeout. It loads
the model once per job, caches downloaded weights, and shuts down its
single-use GPU container after the job. There are no automatic function retries.

## Outputs and recovery

Results are committed after every completed translation to the persistent
Modal Volume `gga-clean-gemma4-dev-results`. Model weights use the cache Volume
`gga-gemma4-model-cache`. Results are copied to:

```text
data/translations/blind_gemma4_modal_development_v3/
  run_manifest.json
  bar_<page_id>_v1.0.1_S<sentence_number>.json
```

Each record preserves source text/ID, model revision, prompt messages, decoding
settings, translation, token counts, runtime versions and generation duration.
`quality_status: unassessed` means generation succeeded; it is not a translation
quality label. Model loading and container startup are not included in the
per-sentence duration, and token counts do not represent Modal GPU charges.

If the terminal disconnects or a later sentence fails, download committed
results without allocating a GPU:

```sh
.venv/bin/modal run modal_translate_blind.py --sync
```

Run the execution command again to resume. Saved inputs, code fingerprints,
model revision and settings are checked; incompatible results cause an error.
Local concurrent invocations are prevented by a file lock. Use a single
workspace/process for this pilot; the file lock does not coordinate separate
computers running independent Modal apps.

## Reproducibility

The protocol is in `experiments/blind_gemma4_modal_development_v3.json`:

- Gemma: `google/gemma-4-26B-A4B-it`
- Checkpoint: `47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301`
- vLLM: `0.21.0`; Transformers: `5.5.4`; inference dtype: `bfloat16`
- Temperature: `0`; seed: `42`; output limit: `1024` tokens; thinking disabled
- Preprocessing: the exact version `1.0.1` freeze hash
- Prompt: `prompts/blind_translation_v1.txt`

Changes to this protocol require a new run name. The original workspace and
frozen preprocessing artifacts are not changed by translation runs.

The initial Modal v1 attempt stopped at a Python bytecode import error. The v2
attempt cleared those caches but exposed an incompatible Transformers v4
dependency. Both stopped before generating translations. The v3 image removes
dependency bytecode caches and pins Transformers 5.5.4, which supports Gemma 4.
A CPU preflight checks the model configuration and chat template before GPU
inference. The checkpoint, prompt, source set and decoding settings are unchanged.

The existing Google API adapter remains unused. This Modal route loads Gemma's
weights directly under the model license and Modal's service terms; Wikipedia's
source-content licensing remains separate.

- [Modal Gemma/vLLM example](https://modal.com/docs/examples/llm_inference)
- [Persistent Modal Volumes](https://modal.com/docs/guide/volumes)
- [Modal terms](https://modal.com/legal/terms)
- [Gemma 4 model card/license](https://ai.google.dev/gemma/docs/core/model_card_4)
