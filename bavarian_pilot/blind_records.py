"""Blind-translation inputs and saved-record validation, without cloud imports."""

import hashlib
import json
import os
import platform
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/blind_gemma4_modal_development_v3.json"
MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"


def optional_version(package):
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def recorded_manifest(directory, expected):
    """Read a historical manifest while checking the inputs used by this reader."""
    path = directory / "run_manifest.json"
    if not path.exists():
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Saved run inputs or settings changed: {directory.name}")
    return manifest


def require_original_runner(manifest, runner_path):
    """Prevent new code from adding inference to a completed historical run."""
    actual = hashlib.sha256(Path(runner_path).read_bytes()).hexdigest()
    if manifest.get("runner_sha256") != actual:
        raise ValueError(
            "This saved run uses the original experiment code. "
            "Restore commit 94e719c to resume it, or use a new run name for new inference."
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



def local_run():
    from download_articles import sha256
    from translate_blind import load_sources, local_path

    config = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    prompt_path = local_path(config["prompt_file"])
    sources = load_sources(config)
    inputs = {
        "condition": "blind_translation", "experiment": config,
        "experiment_sha256": sha256(EXPERIMENT), "prompt": prompt_path.read_text(encoding="utf-8").strip(),
        "prompt_sha256": sha256(prompt_path),
        "source_loader_sha256": sha256(ROOT / "translate_blind.py"),
        "sources_sha256": digest(sources),
    }
    directory = ROOT / "data/translations" / config["run_name"]
    manifest = recorded_manifest(directory, inputs) or {
        **inputs, "runner_sha256": sha256(ROOT / "runners/modal/blind.py"),
        "modal_sdk": optional_version("modal"),
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
