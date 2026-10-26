"""Run resumable Gemma 4 blind translations on the frozen development sentences.

Adapted from the original workspace's translate_gemma_dev.py. Default: plan only.
"""

import argparse
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import requests

from download_articles import ROOT, save_json, sha256


EXPERIMENT = ROOT / "experiments/blind_gemma4_development_v1.json"
MODEL = "gemma-4-26b-a4b-it"
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def local_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Experiment paths must stay within this workspace")
    return path


def verified_json(path, expected_hash):
    if sha256(path) != expected_hash:
        raise ValueError(f"Frozen file changed: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_experiment():
    config = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    expected = {"provider": "google_gemini_api", "model": MODEL, "split": "development",
                "source_language": "bar", "target_language": "en", "source_unit": "single_sentence"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("This runner requires Gemma 4 and single Bavarian development sentences")
    if config.get("protocol_version") != 1 or not re.fullmatch(r"[a-z0-9_]+", config.get("run_name", "")):
        raise ValueError("Invalid translation protocol version or run name")
    generation = config.get("generation_config", {})
    temperature = generation.get("temperature")
    maximum = generation.get("maxOutputTokens")
    if (set(generation) != {"temperature", "candidateCount", "maxOutputTokens", "thinkingConfig"}
            or type(temperature) not in (int, float) or not math.isfinite(temperature)
            or not 0 <= temperature <= 2 or type(maximum) is not int or not 1 <= maximum <= 8192
            or generation["candidateCount"] != 1
            or generation["thinkingConfig"] != {"thinkingLevel": "minimal"}):
        raise ValueError("Invalid Gemma generation settings")
    prompt_path = local_path(config["prompt_file"])
    prompt = prompt_path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("The translation prompt is empty")
    return config, prompt


def load_sources(config):
    freeze_path = ROOT / "data/preprocessing_freeze.json"
    freeze = verified_json(freeze_path, config["preprocessing_freeze_sha256"])
    manifest = verified_json(local_path(freeze["manifest_file"]), freeze["manifest_sha256"])
    for name, expected in manifest["preparation"]["code_sha256"].items():
        if sha256(local_path(name)) != expected:
            raise ValueError(f"Frozen preprocessing code changed: {name}")
    for relative, expected in (
        ("data/article_split.json", manifest["split_sha256"]),
        ("data/articles.csv", manifest["selection_sha256"]),
        ("data/provenance.json", manifest["provenance_sha256"]),
        ("requirements.lock", manifest["requirements_lock_sha256"]),
        (freeze["sentence_boundary_corrections_file"], freeze["sentence_boundary_corrections_sha256"]),
    ):
        if sha256(local_path(relative)) != expected:
            raise ValueError(f"Frozen preprocessing input changed: {relative}")
    prepared_dir = local_path(freeze["prepared_directory"])
    verified_json(prepared_dir / "quality_report.json", freeze["quality_report_sha256"])
    split = json.loads((ROOT / "data/article_split.json").read_text(encoding="utf-8"))
    development = {pair["bar_page_id"]: pair for pair in split["development"]}
    sources = []
    seen_pages = set()
    for entry in manifest["articles"]:
        if Path(entry["file"]).name != entry["file"]:
            raise ValueError("Invalid prepared article filename")
        path = prepared_dir / entry["file"]
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"Prepared article changed: {path}")
        # Inspect the split before extracting any source; English content never enters a request.
        prepared = json.loads(path.read_text(encoding="utf-8"))
        if prepared["split"] != "development":
            continue
        bar = prepared["bar"]
        page_id = bar["page_id"]
        if page_id not in development or page_id in seen_pages or bar["title"] != development[page_id]["bar_title"]:
            raise ValueError("Prepared development articles differ from the frozen split")
        seen_pages.add(page_id)
        singles = {span["sentence_ids"][0]: span["text"] for span in bar["candidate_spans"]
                   if len(span["sentence_ids"]) == 1}
        eligible = [sentence for sentence in bar["sentences"] if sentence["eligible_for_matching"]]
        if set(singles) != {sentence["sentence_id"] for sentence in eligible}:
            raise ValueError(f"Single-sentence candidates differ from eligibility: {page_id}")
        for sentence in eligible:
            source_id, text = sentence["sentence_id"], sentence["text"]
            if not re.fullmatch(r"bar_\d+:v[0-9.]+:S\d+", source_id) or singles[source_id] != text:
                raise ValueError(f"Invalid source sentence or candidate: {source_id}")
            sources.append({"source_id": source_id, "bar_text": text,
                            "bar_title": bar["title"], "bar_page_id": page_id,
                            "bar_revision_id": bar["revision_id"]})
    if (seen_pages != set(development) or len(sources) != freeze["counts"]["bar"]["development_eligible_sentences"]
            or len({s["source_id"] for s in sources}) != len(sources)):
        raise ValueError("Development source set differs from the frozen preprocessing counts")
    return sources


def body_for(source, config, prompt):
    return {
        "systemInstruction": {"parts": [{"text": prompt}]},
        "contents": [{"role": "user", "parts": [{"text": f"Bavarian text:\n{source['bar_text']}"}]}],
        "generationConfig": config["generation_config"],
    }


def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        path = ROOT / ".env"
        values = []
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                name, separator, value = line.partition("=")
                if separator and name.strip() == "GEMINI_API_KEY":
                    value = value.strip()
                    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                        value = value[1:-1]
                    values.append(value)
        if len(values) > 1:
            raise ValueError("Repeated GEMINI_API_KEY entries in .env")
        key = values[0] if values else ""
    if not key or key == "replace_with_your_google_ai_studio_key":
        raise ValueError("Add GEMINI_API_KEY to the environment or the workspace .env file; no request sent")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", key):
        raise ValueError("GEMINI_API_KEY contains invalid characters; no request sent")
    return key


def validate_result(result):
    if not isinstance(result, dict) or not isinstance(result.get("translation"), str) or not result["translation"].strip():
        raise ValueError("Gemma returned no translation text")
    model = result.get("response_model")
    if not isinstance(model, str) or not (model == MODEL or model.startswith(MODEL + "-")):
        raise ValueError("Gemma returned an unexpected model version")
    if result.get("finish_reason") != "STOP":
        raise ValueError("Gemma did not finish normally; no translation accepted")
    counts = [result.get(name) for name in ("prompt_tokens", "output_tokens", "total_tokens", "thought_tokens")]
    if any(type(count) is not int or count < 0 for count in counts) or counts[2] < counts[0] + counts[1]:
        raise ValueError("Gemma returned incomplete or inconsistent token usage")


def parse_response(response):
    candidates = response.get("candidates") if isinstance(response, dict) else None
    if not isinstance(candidates, list) or len(candidates) != 1 or not isinstance(candidates[0], dict):
        raise ValueError("Gemma returned no single candidate; the request may have been blocked")
    candidate = candidates[0]
    if candidate.get("finishReason") != "STOP":
        raise ValueError("Gemma stopped before completing the translation; no result accepted")
    content = candidate.get("content", {})
    parts = content.get("parts") if isinstance(content, dict) else None
    if not isinstance(parts, list) or not all(isinstance(part, dict) for part in parts):
        raise ValueError("Gemma returned malformed content")
    final_parts = [part for part in parts if part.get("thought") is not True]
    if not all(isinstance(part.get("text"), str) and "functionCall" not in part for part in final_parts):
        raise ValueError("Gemma returned a non-text translation")
    usage = response.get("usageMetadata")
    if not isinstance(usage, dict):
        raise ValueError("Gemma returned no token usage")
    model = response.get("modelVersion")
    result = {
        "translation": "".join(part["text"] for part in final_parts).strip(),
        "response_model": model.removeprefix("models/") if isinstance(model, str) else model,
        "response_id": response.get("responseId"), "finish_reason": "STOP",
        "prompt_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "total_tokens": usage.get("totalTokenCount"), "thought_tokens": usage.get("thoughtsTokenCount", 0),
    }
    validate_result(result)
    return result


def call_api(session, key, body):
    try:
        response = session.post(URL, headers={"x-goog-api-key": key}, json=body, timeout=(10, 180))
    except requests.Timeout:
        raise RuntimeError("Gemma request timed out; it may have been processed. No automatic retry.") from None
    except requests.RequestException as error:
        raise RuntimeError(f"Gemma transport failed ({type(error).__name__}); no automatic retry.") from None
    if response.status_code != 200:
        advice = {400: "Check the model and generation settings.", 401: "Check the API key.",
                  403: "Check API-key permissions and model access.", 404: "Check model availability.",
                  429: "Quota or rate limit reached; resume later."}.get(response.status_code, "Resume after the provider recovers.")
        # Provider error bodies may echo credentials or input text; do not print them.
        raise RuntimeError(f"Gemma returned HTTP {response.status_code}. {advice} No automatic retry.")
    try:
        payload = response.json()
    except ValueError:
        raise RuntimeError("Gemma returned invalid JSON; no result accepted") from None
    return parse_response(payload)


def result_path(directory, source_id):
    return directory / (source_id.replace(":", "_") + ".json")


def load_saved(directory, sources, run_id, config, prompt):
    expected = {source["source_id"]: source for source in sources}
    records = {}
    for path in sorted(directory.glob("bar_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source = record.get("source") if isinstance(record, dict) else None
        source_id = source.get("source_id") if isinstance(source, dict) else None
        if source_id not in expected or source_id in records or source != expected[source_id]:
            raise ValueError(f"Unexpected, duplicate or changed saved source: {path}")
        if (path != result_path(directory, source_id) or record.get("run_id") != run_id
                or record.get("condition") != "blind_translation"
                or record.get("request") != body_for(source, config, prompt)):
            raise ValueError(f"Saved translation belongs to different inputs or settings: {path}")
        validate_result(record.get("result"))
        records[source_id] = record
    if len({record["result"]["response_model"] for record in records.values()}) > 1:
        raise ValueError("Saved translations contain different response model versions")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Send hosted Gemma requests")
    parser.add_argument("--max-new", type=int, default=53, help="Maximum new translations this invocation")
    parser.add_argument("--interval-seconds", type=float, default=15, help="Minimum interval between request starts")
    args = parser.parse_args()
    if args.max_new < 1 or not math.isfinite(args.interval_seconds) or not 0 <= args.interval_seconds <= 3600:
        parser.error("--max-new must be positive; --interval-seconds must be between 0 and 3600")
    config, prompt = load_experiment()
    sources = load_sources(config)
    directory = ROOT / "data/translations" / config["run_name"]
    manifest = {
        "condition": "blind_translation", "experiment": config, "experiment_sha256": sha256(EXPERIMENT),
        "prompt": prompt, "prompt_sha256": sha256(local_path(config["prompt_file"])),
        "runner_sha256": sha256(Path(__file__)), "python": platform.python_version(),
        "requests": version("requests"), "sources_sha256": digest_json(sources),
        "source_ids": [source["source_id"] for source in sources],
    }
    run_id = digest_json(manifest)
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Saved run manifest differs from the current code, sources or protocol; use a new run name")
    records = load_saved(directory, sources, run_id, config, prompt)
    print(f"{MODEL}: {len(records)}/{len(sources)} development translations saved; output {directory}", flush=True)
    if not args.execute or len(records) == len(sources):
        return
    key = load_api_key()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock, requests.Session() as session:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another translation process is already using this output directory") from None
        # Another process may have completed requests before this lock was acquired.
        save_json(manifest_path, manifest)
        records = load_saved(directory, sources, run_id, config, prompt)
        previous_start = None
        completed = 0
        for source in sources:
            source_id = source["source_id"]
            if source_id in records:
                continue
            if completed >= args.max_new:
                break
            if previous_start is not None:
                delay = args.interval_seconds - (time.monotonic() - previous_start)
                if delay > 0:
                    time.sleep(delay)
            body = body_for(source, config, prompt)
            print(f"Requesting {source_id}...", flush=True)
            previous_start = time.monotonic()
            try:
                result = call_api(session, key, body)
                previous_models = {r["result"]["response_model"] for r in records.values()}
                if previous_models and previous_models != {result["response_model"]}:
                    raise ValueError("Response model version changed; start a separate run to avoid mixing versions")
            except (RuntimeError, ValueError) as error:
                raise SystemExit(f"Stopped at {source_id}: {error}") from None
            record = {
                "condition": "blind_translation", "run_id": run_id, "source": source,
                "request": body, "result": result, "quality_status": "unassessed",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": round(time.monotonic() - previous_start, 3),
            }
            save_json(result_path(directory, source_id), record)
            records[source_id] = record
            completed += 1
            print(f"Saved {source_id} ({len(records)}/{len(sources)})", flush=True)
    print(f"Completed {completed} new translations; {len(records)}/{len(sources)} saved", flush=True)


if __name__ == "__main__":
    main()
