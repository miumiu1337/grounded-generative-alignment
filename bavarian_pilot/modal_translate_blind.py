"""Gemma 4 blind development translations on Modal; adapted from modal_gemma_dev.py.

python modal_translate_blind.py                       Offline plan; no Modal connection
modal run modal_translate_blind.py --execute          Two new sentences, then sync
modal run modal_translate_blind.py --execute --max-new 53
modal run modal_translate_blind.py --sync             Download committed results; no GPU
"""

import fcntl
import hashlib
import json
import os
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/blind_gemma4_modal_development_v3.json"
MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"
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
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def messages_for(source, prompt):
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": f"Bavarian text:\n{source['bar_text']}"}]


def atomic_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Saved result differs from this run: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def validate_run(manifest, sources):
    config = manifest["experiment"]
    if (config["provider"] != "modal_self_hosted" or config["model"] != MODEL
            or config["model_revision"] != REVISION or config["split"] != "development"
            or config["source_language"] != "bar" or config["target_language"] != "en"
            or config["source_unit"] != "single_sentence"
            or config["generation"] != {"temperature": 0.0, "seed": 42, "max_tokens": 1024, "enable_thinking": False}
            or config["inference"] != {"vllm": "0.21.0", "transformers": "5.5.4", "gpu": "H200", "dtype": "bfloat16", "max_model_len": 8192, "gpu_memory_utilization": 0.9}
            or not re.fullmatch(r"[a-z0-9_]+", config["run_name"])):
        raise ValueError("Unsupported or changed Gemma Modal development protocol")
    if len(sources) != 53 or {s["bar_page_id"] for s in sources} != {59, 2544}:
        raise ValueError("Expected the 53 frozen development sentences")
    if manifest["sources_sha256"] != digest(sources) or len({s["source_id"] for s in sources}) != len(sources):
        raise ValueError("Development source fingerprint changed or contains duplicate IDs")
    if any(not re.fullmatch(r"bar_(?:59|2544):v1\.0\.1:S\d+", s["source_id"]) for s in sources):
        raise ValueError("Unexpected source ID in development run")


def record_path(directory, source_id):
    return directory / (source_id.replace(":", "_") + ".json")


def validate_record(record, source_id, manifest, expected):
    if (source_id not in expected or not isinstance(record, dict)
            or record.get("source") != expected[source_id]):
        raise ValueError(f"Unexpected or changed Modal source: {source_id}")
    if (record.get("run_id") != digest(manifest) or record.get("condition") != "blind_translation"
            or record.get("provider") != "modal_self_hosted" or record.get("model") != MODEL
            or record.get("model_revision") != REVISION
            or record.get("messages") != messages_for(expected[source_id], manifest["prompt"])
            or record.get("generation") != manifest["experiment"]["generation"]
            or record.get("finish_reason") != "stop"
            or not isinstance(record.get("translation"), str) or not record["translation"].strip()
            or any(type(record.get(k)) is not int or record[k] < 0 for k in ("prompt_tokens", "output_tokens"))):
        raise ValueError(f"Saved Modal translation does not match the protocol: {source_id}")


def read_saved(directory, manifest, sources):
    validate_run(manifest, sources)
    path = directory / "run_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Saved Modal run has different inputs, code or settings; use a new run name")
    expected = {s["source_id"]: s for s in sources}
    records = {}
    for path in sorted(directory.glob("bar_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = record.get("source") if isinstance(record, dict) else None
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if source_id in records:
            raise ValueError(f"Unexpected, duplicate or changed Modal result: {path.name}")
        validate_record(record, source_id, manifest, expected)
        if path != record_path(directory, source_id):
            raise ValueError(f"Unexpected saved result filename: {path.name}")
        records[source_id] = record
    return records


@app.function(image=modal.Image.debian_slim(python_version="3.12"),
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


def local_run():
    from download_articles import sha256
    from translate_blind import load_sources, local_path

    config = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    prompt_path = local_path(config["prompt_file"])
    sources = load_sources(config)
    manifest = {
        "condition": "blind_translation", "experiment": config,
        "experiment_sha256": sha256(EXPERIMENT), "prompt": prompt_path.read_text(encoding="utf-8").strip(),
        "prompt_sha256": sha256(prompt_path), "runner_sha256": sha256(Path(__file__)),
        "source_loader_sha256": sha256(ROOT / "translate_blind.py"),
        "sources_sha256": digest(sources), "modal_sdk": version("modal"),
        "python": platform.python_version(),
    }
    validate_run(manifest, sources)
    return manifest, sources


def write_local(records, manifest, sources):
    directory = ROOT / "data/translations" / manifest["experiment"]["run_name"]
    read_saved(directory, manifest, sources)
    expected = {s["source_id"]: s for s in sources}
    if not set(records).issubset(expected):
        raise ValueError("Remote results contain unexpected source IDs")
    for source_id, record in records.items():
        validate_record(record, source_id, manifest, expected)
    atomic_json(directory / "run_manifest.json", manifest)
    for source_id, record in records.items():
        atomic_json(record_path(directory, source_id), record)
    verified = read_saved(directory, manifest, sources)
    print(f"Saved locally: {len(verified)}/53 in {directory}", flush=True)


@app.local_entrypoint()
def main(execute: bool = False, sync: bool = False, max_new: int = 2):
    if execute and sync or not 1 <= max_new <= 53:
        raise ValueError("Choose --execute or --sync; --max-new must be between 1 and 53")
    manifest, sources = local_run()
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
    print("Start two translations: .venv/bin/modal run modal_translate_blind.py --execute")
    print("Complete development: .venv/bin/modal run modal_translate_blind.py --execute --max-new 53")
