"""Read both pages -> commit evidence -> translate, without a blind draft.

Variant A's unchanged pure helpers enforce the same span and output rules.
Only source/page inputs, never its translations or selections, enter B.
"""

import csv
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import variant_a as shared
from variant_a import (MODEL, REVISION, atomic_json, digest, evidence_for,
                       numbered_page, output_group, record_path,
                       selection_grammar, strict_json, validate_selection)


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments/grounded_b_gemma4_modal_development_v1.json"


def selection_messages(manifest, item, page):
    payload = {"focus_source": item["source"], "bavarian_page": page["bar"], "english_page": page["en"]}
    return [{"role": "system", "content": manifest["prompts"]["selection"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def translation_messages(manifest, item, page, selection):
    validate_selection(selection, item, page)
    payload = {"focus_source": item["source"], "bavarian_page": page["bar"], "selection": selection,
               "selected_english_evidence": evidence_for(selection, page)}
    return [{"role": "system", "content": manifest["prompts"]["translation"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def stage_messages(stage, manifest, item, page, selection=None):
    if stage == "selection":
        return selection_messages(manifest, item, page)
    if stage == "translation":
        return translation_messages(manifest, item, page, selection)
    raise ValueError("Unknown Variant B stage")


def schema_for(stage, item, page):
    if stage not in {"selection", "translation"}:
        raise ValueError("Unknown Variant B stage")
    return shared.schema_for("selection" if stage == "selection" else "refinement", item, page)


def validate_inputs(manifest, items, pages):
    config = manifest["experiment"]
    expected = {"condition": "grounded_b", "protocol_version": 1, "split": "development",
                "source_language": "bar", "target_language": "en", "source_unit": "single_sentence",
                "provider": "modal_self_hosted", "model": MODEL, "model_revision": REVISION,
                "candidate_policy": "all_frozen_english_1_or_2_sentence_spans_in_linked_article",
                "preprocessing_freeze_sha256": "5cf98d0c8e9072cf4915f6679f5beb34d16396bd0ce3cd9022206078b590263f"}
    if (any(config.get(k) != v for k, v in expected.items())
            or config.get("generation") != {"temperature": 0.0, "seed": 42, "max_tokens": 1024,
                "enable_thinking": False, "structured_output": "selection_valid_span_grammar_translation_json_schema"}
            or config.get("inference") != {"vllm": "0.21.0", "transformers": "5.5.4", "gpu": "H200",
                "dtype": "bfloat16", "max_model_len": 65536, "gpu_memory_utilization": 0.9}
            or not re.fullmatch(r"[a-z0-9_]+", config.get("run_name", ""))
            or len(items) != 53 or set(pages) != {"2544", "59"}
            or manifest["inputs_sha256"] != digest({"items": items, "pages": pages})):
        raise ValueError("Variant B protocol/input fingerprint differs")
    ids = []
    for item in items:
        if (set(item) != {"source"} or set(item["source"]) != {
                "source_id", "bar_text", "bar_title", "bar_page_id", "bar_revision_id"}):
            raise ValueError("Variant B inputs must contain only frozen source fields, without a draft")
        source = item["source"]
        record_path(Path("."), source["source_id"])
        if not source["source_id"].startswith(f"bar_{source['bar_page_id']}:"):
            raise ValueError("Source ID and article differ")
        ids.append(source["source_id"])
    if len(set(ids)) != 53 or Counter(i["source"]["bar_page_id"] for i in items) != {2544: 39, 59: 14}:
        raise ValueError("Variant B requires the 53 unique frozen development sources")


def validate_stage(record, stage, manifest, item, page, selection=None):
    if (record.get("stage") != stage or record.get("source_id") != item["source"]["source_id"]
            or record.get("run_id") != digest(manifest)
            or record.get("messages") != stage_messages(stage, manifest, item, page, selection)
            or record.get("schema") != schema_for(stage, item, page)
            or record.get("grammar") != (selection_grammar(item, page) if stage == "selection" else None)
            or record.get("finish_reason") != "stop"
            or any(type(record.get(k)) is not int or record[k] < 0 for k in ("prompt_tokens", "output_tokens"))
            or record["prompt_tokens"] + manifest["experiment"]["generation"]["max_tokens"] > 65536
            or not isinstance(record.get("raw_response"), str)
            or strict_json(record["raw_response"]) != record.get("result")):
        raise ValueError(f"Saved {stage} record differs from the Variant B protocol")
    runtime = record.get("runtime", {})
    if any(runtime.get(k) != manifest["experiment"]["inference"][k] for k in ("vllm", "transformers")):
        raise ValueError("Variant B runtime differs from the pinned versions")
    timestamp = datetime.fromisoformat(record.get("recorded_at", ""))
    if timestamp.tzinfo is None:
        raise ValueError("Stage timestamp requires a timezone")
    if stage == "selection":
        validate_selection(record["result"], item, page)
    else:
        shared.validate_refinement(record["result"], item)


def read_saved(directory, manifest, items, pages):
    validate_inputs(manifest, items, pages)
    path = directory / "run_manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Saved Variant B run differs; use a new run name")
    expected = {i["source"]["source_id"]: i for i in items}
    selections, results = {}, {}
    for path in sorted((directory / "selections").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source_id = record.get("source_id")
        if source_id not in expected or source_id in selections or path != record_path(directory / "selections", source_id):
            raise ValueError("Unexpected or duplicate Variant B selection")
        item = expected[source_id]
        validate_stage(record, "selection", manifest, item, pages[str(item["source"]["bar_page_id"])])
        selections[source_id] = record
    for path in sorted((directory / "results").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        source_id = record.get("source", {}).get("source_id")
        if source_id not in selections or source_id in results or path != record_path(directory / "results", source_id):
            raise ValueError("Variant B result has no valid saved selection")
        item, selection = expected[source_id], selections[source_id]
        page = pages[str(item["source"]["bar_page_id"])]
        if (set(record) != {"condition", "run_id", "source", "selection", "selection_sha256", "evidence",
                            "en_page_id", "en_revision_id", "translation", "output_group", "quality_status"}
                or record["condition"] != "grounded_b" or record["run_id"] != digest(manifest)
                or record["source"] != item["source"] or record["selection"] != selection["result"]
                or record["selection_sha256"] != digest(selection)
                or record["evidence"] != evidence_for(selection["result"], page)
                or record["en_page_id"] != page["en_page_id"] or record["en_revision_id"] != page["en_revision_id"]
                or record["output_group"] != output_group(selection["result"])
                or record["quality_status"] != "unassessed"):
            raise ValueError("Saved Variant B result differs from its inputs/selection")
        validate_stage(record["translation"], "translation", manifest, item, page, selection["result"])
        if datetime.fromisoformat(selection["recorded_at"]) > datetime.fromisoformat(record["translation"]["recorded_at"]):
            raise ValueError("Translation predates evidence selection")
        results[source_id] = record
    return {"selections": selections, "results": results}


def local_run():
    from download_articles import sha256
    from translate_blind import load_sources, local_path
    from blind_records import optional_version, recorded_manifest
    import platform

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = load_sources(config)
    items, pages = [{"source": s} for s in sources], {}
    for source in sources:
        page_id = str(source["bar_page_id"])
        if page_id not in pages:
            prepared = json.loads((ROOT / f"data/prepared/bar_{page_id}.json").read_text(encoding="utf-8"))
            pages[page_id] = {"bar": numbered_page(prepared["bar"]), "en": numbered_page(prepared["en"]),
                "en_page_id": prepared["en"]["page_id"], "en_revision_id": prepared["en"]["revision_id"],
                "candidates": prepared["en"]["candidate_spans"]}
    # Check parity against A's fingerprinted input snapshot; none of its drafts/results are passed to B.
    a_directory = local_path(config["comparison_a_directory"])
    a_manifest = json.loads((a_directory / "run_manifest.json").read_text(encoding="utf-8"))
    a_inputs = json.loads((a_directory / "inputs.json").read_text(encoding="utf-8"))
    if (digest(a_manifest) != config["comparison_a_run_id"]
            or digest(a_inputs) != a_manifest["inputs_sha256"]
            or [{"source": i["source"]} for i in a_inputs["items"]] != items or a_inputs["pages"] != pages):
        raise ValueError("Variant B source/page/candidate pool differs from completed Variant A")
    prompts = {stage: local_path(config[stage + "_prompt"]).read_text(encoding="utf-8").strip()
               for stage in ("selection", "translation")}
    if not all(prompts.values()):
        raise ValueError("Variant B prompt is empty")
    inputs = {"experiment": config, "experiment_sha256": sha256(CONFIG), "prompts": prompts,
        "prompt_sha256": {s: sha256(local_path(config[s + "_prompt"])) for s in prompts},
        "source_loader_sha256": sha256(ROOT / "translate_blind.py"),
        "inputs_sha256": digest({"items": items, "pages": pages})}
    manifest = recorded_manifest(ROOT / "data/translations" / config["run_name"], inputs) or {
        **inputs,
        "protocol_sha256": sha256(Path(__file__)), "runner_sha256": sha256(ROOT / "runners/modal/variant_b.py"),
        "shared_protocol_sha256": sha256(ROOT / "variant_a.py"),
        "image_definition_sha256": sha256(ROOT / "runners/modal/blind.py"),
        "modal_sdk": optional_version("modal"), "python": platform.python_version()}
    validate_inputs(manifest, items, pages)
    return manifest, items, pages


def write_local(saved, manifest, items, pages):
    import tempfile
    directory = ROOT / "data/translations" / manifest["experiment"]["run_name"]
    read_saved(directory, manifest, items, pages)
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for kind in ("selections", "results"):
            for source_id, record in saved[kind].items():
                atomic_json(record_path(staging / kind, source_id), record)
        verified = read_saved(staging, manifest, items, pages)
        if verified != saved:
            raise ValueError("Remote Variant B bundle has inconsistent keys")
    atomic_json(directory / "run_manifest.json", manifest)
    atomic_json(directory / "inputs.json", {"items": items, "pages": pages})
    for kind in ("selections", "results"):
        for source_id, record in saved[kind].items():
            atomic_json(record_path(directory / kind, source_id), record)
    return directory, read_saved(directory, manifest, items, pages)


def export_results(directory, saved, manifest):
    records = [saved["results"][key] for key in sorted(saved["results"])]
    fields = ["source_id", "bar_title", "bar_text", "translation", "evidence_ids", "evidence_text",
              "match_status", "contradiction", "output_group", "selection_prompt_tokens",
              "selection_output_tokens", "translation_prompt_tokens", "translation_output_tokens"]
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            selection = saved["selections"][record["source"]["source_id"]]
            translated = record["translation"]
            writer.writerow({"source_id": record["source"]["source_id"], "bar_title": record["source"]["bar_title"],
                "bar_text": record["source"]["bar_text"], "translation": translated["result"]["translation"],
                "evidence_ids": ";".join(record["selection"]["evidence_ids"]),
                "evidence_text": "\n".join(e["text"] for e in record["evidence"]),
                "match_status": record["selection"]["match_status"], "contradiction": record["selection"]["contradiction"],
                "output_group": record["output_group"],
                "selection_prompt_tokens": selection["prompt_tokens"], "selection_output_tokens": selection["output_tokens"],
                "translation_prompt_tokens": translated["prompt_tokens"], "translation_output_tokens": translated["output_tokens"]})
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(directory / "comparison.csv")
    for group in ("grounded_candidates", "unmatched", "contradictory"):
        temporary = directory / (group + ".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as output:
            for record in records:
                if record["output_group"] == group:
                    output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(directory / (group + ".jsonl"))
    summary = {"run_id": digest(manifest), "completed": len(records), "expected": 53,
        "saved_selections": len(saved["selections"]),
        "match_status_counts": dict(Counter(r["selection"]["match_status"] for r in records)),
        "output_group_counts": dict(Counter(r["output_group"] for r in records)),
        "new_prompt_tokens": sum(s["prompt_tokens"] for s in saved["selections"].values()) + sum(r["translation"]["prompt_tokens"] for r in records),
        "new_output_tokens": sum(s["output_tokens"] for s in saved["selections"].values()) + sum(r["translation"]["output_tokens"] for r in records),
        "interpretation": "Model-selected evidence and translations, not gold judgments or accepted parallel pairs. Token counts are not Modal GPU billing."}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "summary.json")
    return summary
