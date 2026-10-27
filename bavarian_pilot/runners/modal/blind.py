"""Gemma 4 blind development translations on Modal.

python -m runners.modal.blind                       Offline plan; no Modal connection
modal run -m runners.modal.blind --execute          Two new sentences, then sync
modal run -m runners.modal.blind --execute --max-new 53
modal run -m runners.modal.blind --sync             Download committed results; no GPU
"""

import fcntl
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal


from blind_records import (ROOT, EXPERIMENT, MODEL, REVISION, atomic_json, digest,
                           local_run, messages_for, read_saved, record_path,
                           require_original_runner, validate_record, validate_run, write_local)
REMOTE_ROOT = Path("/results")
RESULTS_VOLUME_NAME = "gga-clean-gemma4-dev-results"

app = modal.App("gga-clean-gemma4-blind-dev")
results_volume = modal.Volume.from_name(RESULTS_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name("gga-gemma4-model-cache", create_if_missing=True)
gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    # Gemma 4 needs Transformers 5; the vLLM dependency resolver selected v4.
    .uv_pip_install("transformers==5.5.4")
    # A cached image contained an unreadable .pyc; compile from the installed sources.
    .run_commands("find /usr/local/lib/python3.12/site-packages -type f -name '*.pyc' -delete")
    .env({"HF_HOME": "/cache/huggingface", "VLLM_CACHE_ROOT": "/cache/vllm", "PYTHONDONTWRITEBYTECODE": "1"})
    .add_local_python_source("blind_records")
)


@app.function(image=modal.Image.debian_slim(python_version="3.12").add_local_python_source("blind_records"),
              volumes={"/results": results_volume}, timeout=120, retries=0)
def saved_results(manifest, sources):
    results_volume.reload()
    directory = REMOTE_ROOT / manifest["experiment"]["run_name"]
    return read_saved(directory, manifest, sources)


@app.function(image=gpu_image, volumes={"/cache": cache_volume}, timeout=180, retries=0)
def check_model_support(manifest, sources):
    validate_run(manifest, sources)
    from transformers import AutoConfig, AutoTokenizer

    if version("transformers") != manifest["experiment"]["inference"]["transformers"]:
        raise RuntimeError("The container does not have the pinned Transformers version")
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    if config.model_type != "gemma4":
        raise RuntimeError("The selected checkpoint is not a Gemma 4 model")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    formatted = tokenizer.apply_chat_template(messages_for(sources[0], manifest["prompt"]),
                                              tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if not formatted:
        raise RuntimeError("The Gemma chat template returned an empty prompt")
    cache_volume.commit()
    return {"model_type": config.model_type, "transformers": version("transformers"), "chat_template_checked": True}


@app.function(image=gpu_image, gpu="H200", timeout=3600, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/results": results_volume, "/cache": cache_volume})
def translate(manifest, sources, max_new):
    validate_run(manifest, sources)
    if type(max_new) is not int or not 1 <= max_new <= len(sources):
        raise ValueError("--max-new must be between 1 and 53")
    results_volume.reload()
    directory = REMOTE_ROOT / manifest["experiment"]["run_name"]
    records = read_saved(directory, manifest, sources)
    pending = [s for s in sources if s["source_id"] not in records][:max_new]
    if not pending:
        return records
    atomic_json(directory / "run_manifest.json", manifest)
    results_volume.commit()

    from vllm import LLM, SamplingParams

    config = manifest["experiment"]
    llm = LLM(model=MODEL, revision=REVISION, max_model_len=config["inference"]["max_model_len"],
              dtype=config["inference"]["dtype"],
              gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
              enforce_eager=True, limit_mm_per_prompt={"image": 0})
    cache_volume.commit()
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=config["generation"]["temperature"],
                              seed=config["generation"]["seed"], max_tokens=config["generation"]["max_tokens"])
    runtime = {name: version(name) for name in ("vllm", "torch", "transformers")}
    runtime["python"] = platform.python_version()
    for source in pending:
        source_id = source["source_id"]
        messages = messages_for(source, manifest["prompt"])
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                  enable_thinking=config["generation"]["enable_thinking"])
        started = time.monotonic()
        outputs = llm.generate([formatted], sampling, use_tqdm=False)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError(f"No single model response for {source_id}; prior results are committed")
        response = outputs[0]
        result = response.outputs[0]
        translation = result.text.strip()
        if result.finish_reason != "stop" or not translation:
            raise RuntimeError(f"Incomplete response for {source_id}; prior results are committed")
        record = {
            "condition": "blind_translation", "provider": "modal_self_hosted",
            "model": MODEL, "model_revision": REVISION, "run_id": digest(manifest),
            "source": source, "messages": messages, "generation": config["generation"],
            "translation": translation, "finish_reason": result.finish_reason,
            "prompt_tokens": len(response.prompt_token_ids), "output_tokens": len(result.token_ids),
            "runtime": runtime, "quality_status": "unassessed",
            "duration_seconds": round(time.monotonic() - started, 3),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json(record_path(directory, source_id), record)
        results_volume.commit()
        records[source_id] = record
        print(f"Committed {source_id} ({len(records)}/53)", flush=True)
    return records


@app.local_entrypoint()
def main(execute: bool = False, sync: bool = False, max_new: int = 2):
    if execute and sync or not 1 <= max_new <= 53:
        raise ValueError("Choose --execute or --sync; --max-new must be between 1 and 53")
    manifest, sources = local_run()
    if execute:
        require_original_runner(manifest, Path(__file__))
    directory = ROOT / "data/translations" / manifest["experiment"]["run_name"]
    records = read_saved(directory, manifest, sources)
    print(f"Gemma 4, frozen development set: {len(records)}/53 local results; H200; max new: {max_new}")
    print(f"Checkpoint: {REVISION}; run fingerprint: {digest(manifest)}", flush=True)
    if not execute and not sync:
        print("Offline plan only. No Modal connection or GPU call.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local Modal translation process is already running") from None
        # Sync before a GPU call, including results from earlier interrupted local sessions.
        records = saved_results.remote(manifest, sources)
        write_local(records, manifest, sources)
        if execute and len(records) < len(sources):
            print(f"CPU model preflight: {check_model_support.remote(manifest, sources)}", flush=True)
            records = translate.remote(manifest, sources, max_new)
            write_local(records, manifest, sources)


if __name__ == "__main__":
    manifest, sources = local_run()
    print(f"Offline plan: {len(sources)} development sentences; {MODEL}; checkpoint {REVISION}")
    print("Start two translations: .venv/bin/modal run -m runners.modal.blind --execute")
    print("Complete development: .venv/bin/modal run -m runners.modal.blind --execute --max-new 53")
