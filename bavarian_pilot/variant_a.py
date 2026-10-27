"""Protocol, validation and exports for blind -> evidence -> refinement.

Pure-Python validation helpers also run inside the Modal container. Local input
loading verifies the existing blind run and frozen corpus before any GPU call.
"""

import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments/grounded_a_gemma4_modal_development_v2.json"
MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Saved Variant A artifact differs: {path.name}; use a new run name")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def record_path(directory, source_id):
    if not re.fullmatch(r"bar_(?:59|2544):v1\.0\.1:S\d+", source_id):
        raise ValueError("Unexpected development source ID")
    return directory / (source_id.replace(":", "_") + ".json")


def numbered_page(article):
    lines = [f"{article['title']} ({article['language']}), revision {article['revision_id']}"]
    previous = None
    for sentence in article["sentences"]:
        if not sentence["eligible_for_matching"]:
            continue
        paragraph = sentence["paragraph_id"]
        if paragraph != previous:
            lines.append(f"Paragraph {paragraph}; section: {' / '.join(sentence['heading_path']) or 'lead'}")
            previous = paragraph
        lines.append(f"[{sentence['sentence_id']}] {sentence['text']}")
    return "\n".join(lines)


def selection_messages(manifest, item, pages):
    payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
               "bavarian_page": pages["bar"], "english_page": pages["en"]}
    return [{"role": "system", "content": manifest["prompts"]["selection"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def evidence_for(selection, pages):
    ids = selection["evidence_ids"]
    return [] if not ids else [next(span for span in pages["candidates"] if span["sentence_ids"] == ids)]


def refinement_messages(manifest, item, pages, selection):
    payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
               "bavarian_page": pages["bar"], "selection": selection,
               "selected_english_evidence": evidence_for(selection, pages)}
    return [{"role": "system", "content": manifest["prompts"]["refinement"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def schema_for(stage, item, pages):
    properties = {"source_id": {"type": "string", "const": item["source"]["source_id"]}}
    if stage == "selection":
        ids = [span["sentence_ids"][0] for span in pages["candidates"] if len(span["sentence_ids"]) == 1]
        properties.update(evidence_ids={"type": "array", "minItems": 0, "maxItems": 2,
                                        "items": {"type": "string", "enum": ids},
                                        "enum": [[], *[span["sentence_ids"] for span in pages["candidates"]]]},
                          match_status={"type": "string", "enum": ["exact", "partial", "none"]},
                          contradiction={"type": "boolean"})
    elif stage == "refinement":
        properties["translation"] = {"type": "string", "minLength": 1}
    else:
        raise ValueError("Unknown Variant A stage")
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def selection_grammar(item, pages):
    """Constrain the whole evidence array, not just each individual ID.

    Each literal is JSON-escaped for the EBNF grammar. This enforces adjacency
    and status/contradiction consistency without silently editing model output.
    """
    prefix = '{"source_id": ' + json.dumps(item["source"]["source_id"]) + ', "evidence_ids": '
    spans = [json.dumps(span["sentence_ids"]) for span in pages["candidates"]]
    tails = [', "match_status": ' + json.dumps(status) + ', "contradiction": ' + json.dumps(contradiction)
             for status, contradiction in (("exact", False), ("partial", False), ("partial", True), ("none", True))]
    empty = '[], "match_status": "none", "contradiction": false'
    return '\n'.join([
        'root ::= ' + json.dumps(prefix) + ' (empty-selection | evidence selection-tail) ' + json.dumps('}'),
        'empty-selection ::= ' + json.dumps(empty),
        'evidence ::= ' + ' | '.join(json.dumps(span) for span in spans),
        'selection-tail ::= ' + ' | '.join(json.dumps(tail) for tail in tails),
    ]) + '\n'


def strict_json(text):
    def unique(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("Duplicate key in model JSON")
            values[key] = value
        return values
    def invalid(value):
        raise ValueError(f"Non-JSON constant: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def validate_selection(value, item, pages):
    if (not isinstance(value, dict) or set(value) != {"source_id", "evidence_ids", "match_status", "contradiction"}
            or value["source_id"] != item["source"]["source_id"]
            or value["match_status"] not in {"exact", "partial", "none"}
            or type(value["contradiction"]) is not bool):
        raise ValueError("Malformed evidence selection")
    ids = value["evidence_ids"]
    valid = {tuple(span["sentence_ids"]) for span in pages["candidates"]}
    if (not isinstance(ids, list) or any(not isinstance(v, str) for v in ids)
            or ids and tuple(ids) not in valid):
        raise ValueError("Evidence must be one frozen English span (one sentence or two consecutive sentences in a paragraph)")
    status, contradiction = value["match_status"], value["contradiction"]
    if (not ids and (status != "none" or contradiction)
            or ids and status == "none" and not contradiction
            or status == "exact" and contradiction):
        raise ValueError("Evidence IDs, match status and contradiction are inconsistent")
    return value


def validate_refinement(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "translation"}
            or value["source_id"] != item["source"]["source_id"]
            or not isinstance(value["translation"], str) or not value["translation"].strip()):
        raise ValueError("Malformed refinement")
    return value


def output_group(selection):
    if selection["contradiction"]:
        return "contradictory"
    return "unmatched" if selection["match_status"] == "none" else "grounded_candidates"


def validate_inputs(manifest, items, pages):
    config = manifest["experiment"]
    if (config["condition"] != "grounded_a" or config["split"] != "development"
            or config["model"] != MODEL or config["model_revision"] != REVISION
            or config["source_unit"] != "single_sentence" or len(items) != 53
            or config["provider"] != "modal_self_hosted"
            or config["preprocessing_freeze_sha256"] != "5cf98d0c8e9072cf4915f6679f5beb34d16396bd0ce3cd9022206078b590263f"
            or config["protocol_version"] != 2
            or config["generation"] != {"temperature": 0.0, "seed": 42, "max_tokens": 1024,
                                         "enable_thinking": False, "structured_output": "selection_valid_span_grammar_refinement_json_schema"}
            or config["inference"] != {"vllm": "0.21.0", "transformers": "5.5.4", "gpu": "H200",
                                        "dtype": "bfloat16", "max_model_len": 65536, "gpu_memory_utilization": 0.9}
            or not re.fullmatch(r"[a-z0-9_]+", config["run_name"])
            or set(pages) != {"2544", "59"}
            or manifest["inputs_sha256"] != digest({"items": items, "pages": pages})):
        raise ValueError("Variant A protocol/input fingerprint differs")
    ids = [item["source"]["source_id"] for item in items]
    if len(set(ids)) != 53 or {item["source"]["bar_page_id"] for item in items} != {2544, 59}:
        raise ValueError("Variant A requires the 53 unique frozen development sources")
    for item in items:
        if (item["blind"]["run_id"] != config["blind_run_id"]
                or not isinstance(item["blind"]["translation"], str) or not item["blind"]["translation"].strip()):
            raise ValueError("Missing/changed blind translation")


def stage_messages(stage, manifest, item, pages, selection=None):
    return (selection_messages(manifest, item, pages) if stage == "selection"
            else refinement_messages(manifest, item, pages, selection))


def validate_stage(record, stage, manifest, item, pages, selection=None):
    messages = stage_messages(stage, manifest, item, pages, selection)
    if (record.get("stage") != stage or record.get("source_id") != item["source"]["source_id"]
            or record.get("run_id") != digest(manifest) or record.get("messages") != messages
            or record.get("schema") != schema_for(stage, item, pages)
            or record.get("grammar") != (selection_grammar(item, pages) if stage == "selection" else None)
            or record.get("finish_reason") != "stop"
            or any(type(record.get(key)) is not int or record[key] < 0 for key in ("prompt_tokens", "output_tokens"))
            or not isinstance(record.get("raw_response"), str)
            or strict_json(record["raw_response"]) != record.get("result")):
        raise ValueError(f"Saved {stage} record differs from the Variant A protocol")
    if stage == "selection":
        validate_selection(record["result"], item, pages)
    else:
        validate_refinement(record["result"], item)


def read_saved(directory, manifest, items, pages):
    validate_inputs(manifest, items, pages)
    file = directory / "run_manifest.json"
    if file.exists() and json.loads(file.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Saved Variant A run uses different code/inputs/settings; use a new run name")
    expected = {item["source"]["source_id"]: item for item in items}
    selections, results = {}, {}
    for file in sorted((directory / "selections").glob("bar_*.json")):
        value = json.loads(file.read_text(encoding="utf-8"))
        source_id = value.get("source_id")
        if source_id not in expected or source_id in selections or file != record_path(directory / "selections", source_id):
            raise ValueError("Unexpected or duplicate Variant A selection")
        item = expected[source_id]
        validate_stage(value, "selection", manifest, item, pages[str(item["source"]["bar_page_id"])])
        selections[source_id] = value
    for file in sorted((directory / "results").glob("bar_*.json")):
        value = json.loads(file.read_text(encoding="utf-8"))
        source_id = value.get("source", {}).get("source_id")
        if source_id not in selections or file != record_path(directory / "results", source_id):
            raise ValueError("Variant A result has no valid saved selection")
        item = expected[source_id]
        page = pages[str(item["source"]["bar_page_id"])]
        selection = selections[source_id]
        if (value.get("condition") != "grounded_a" or value.get("run_id") != digest(manifest)
                or value.get("source") != item["source"] or value.get("blind") != item["blind"]
                or value.get("selection_sha256") != digest(selection)
                or value.get("selection") != selection["result"]
                or value.get("evidence") != evidence_for(selection["result"], page)
                or value.get("en_page_id") != page["en_page_id"]
                or value.get("en_revision_id") != page["en_revision_id"]
                or value.get("output_group") != output_group(selection["result"])
                or value.get("quality_status") != "unassessed"):
            raise ValueError("Saved Variant A result differs from its inputs/selection")
        validate_stage(value["refinement"], "refinement", manifest, item, page, selection["result"])
        results[source_id] = value
    return {"selections": selections, "results": results}


def local_run():
    from download_articles import sha256
    from translate_blind import local_path
    import blind_records as blind

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    blind_manifest, sources = blind.local_run()
    if digest(blind_manifest) != config["blind_run_id"]:
        raise ValueError("The saved blind-run protocol differs from the selected baseline")
    directory = local_path(config["blind_directory"])
    translations = blind.read_saved(directory, blind_manifest, sources)
    if set(translations) != {source["source_id"] for source in sources}:
        raise ValueError("Complete the 53 blind development translations first")
    items, pages = [], {}
    for source in sources:
        source_id = source["source_id"]
        record = translations[source_id]
        path = blind.record_path(directory, source_id)
        items.append({"source": source, "blind": {"translation": record["translation"],
                      "run_id": record["run_id"], "file": str(path.relative_to(ROOT)),
                      "file_sha256": sha256(path), "prompt_tokens": record["prompt_tokens"],
                      "output_tokens": record["output_tokens"], "duration_seconds": record["duration_seconds"]}})
        page_id = str(source["bar_page_id"])
        if page_id not in pages:
            prepared = json.loads((ROOT / f"data/prepared/bar_{page_id}.json").read_text(encoding="utf-8"))
            pages[page_id] = {"bar": numbered_page(prepared["bar"]), "en": numbered_page(prepared["en"]),
                              "en_page_id": prepared["en"]["page_id"], "en_revision_id": prepared["en"]["revision_id"],
                              "candidates": prepared["en"]["candidate_spans"]}
    from blind_records import optional_version, recorded_manifest
    import platform
    prompts = {stage: local_path(config[stage + "_prompt"]).read_text(encoding="utf-8").strip()
               for stage in ("selection", "refinement")}
    inputs = {"experiment": config, "experiment_sha256": sha256(CONFIG), "prompts": prompts,
                "prompt_sha256": {stage: sha256(local_path(config[stage + "_prompt"])) for stage in prompts},
                "blind_manifest_sha256": sha256(directory / "run_manifest.json"),
                "source_loader_sha256": sha256(ROOT / "translate_blind.py"),
                "inputs_sha256": digest({"items": items, "pages": pages})}
    manifest = recorded_manifest(ROOT / "data/translations" / config["run_name"], inputs) or {
                **inputs,
                "protocol_sha256": sha256(Path(__file__)), "runner_sha256": sha256(ROOT / "runners/modal/variant_a.py"),
                "blind_runner_sha256": sha256(ROOT / "runners/modal/blind.py"),
                "modal_sdk": optional_version("modal"), "python": platform.python_version()}
    validate_inputs(manifest, items, pages)
    return manifest, items, pages


def write_local(saved, manifest, items, pages):
    directory = ROOT / "data/translations" / manifest["experiment"]["run_name"]
    read_saved(directory, manifest, items, pages)
    # Validate the complete remote bundle in a temporary directory before local publication.
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for kind in ("selections", "results"):
            for source_id, value in saved[kind].items():
                atomic_json(record_path(staging / kind, source_id), value)
        read_saved(staging, manifest, items, pages)
    atomic_json(directory / "run_manifest.json", manifest)
    atomic_json(directory / "inputs.json", {"items": items, "pages": pages})
    for kind in ("selections", "results"):
        for source_id, value in saved[kind].items():
            atomic_json(record_path(directory / kind, source_id), value)
    verified = read_saved(directory, manifest, items, pages)
    print(f"Saved locally: {len(verified['results'])}/53 Variant A results; {len(verified['selections'])}/53 selections", flush=True)
    return directory, verified


def export_results(directory, saved, manifest):
    records = [saved["results"][key] for key in sorted(saved["results"])]
    fields = ["source_id", "bar_title", "bar_text", "blind_translation", "refined_translation",
              "evidence_ids", "evidence_text", "match_status", "contradiction", "output_group", "translation_changed",
              "selection_prompt_tokens", "selection_output_tokens", "refinement_prompt_tokens", "refinement_output_tokens",
              "blind_record_file", "blind_record_sha256"]
    temporary = directory / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for record in records:
            source_id = record["source"]["source_id"]
            selection = saved["selections"][source_id]
            refined = record["refinement"]
            writer.writerow({"source_id": source_id, "bar_title": record["source"]["bar_title"],
                "bar_text": record["source"]["bar_text"], "blind_translation": record["blind"]["translation"],
                "refined_translation": refined["result"]["translation"],
                "evidence_ids": ";".join(record["selection"]["evidence_ids"]),
                "evidence_text": "\n".join(e["text"] for e in record["evidence"]),
                "match_status": record["selection"]["match_status"], "contradiction": record["selection"]["contradiction"],
                "output_group": record["output_group"],
                "translation_changed": record["blind"]["translation"] != refined["result"]["translation"],
                "selection_prompt_tokens": selection["prompt_tokens"], "selection_output_tokens": selection["output_tokens"],
                "refinement_prompt_tokens": refined["prompt_tokens"], "refinement_output_tokens": refined["output_tokens"],
                "blind_record_file": record["blind"]["file"], "blind_record_sha256": record["blind"]["file_sha256"]})
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
               "translations_changed": sum(r["blind"]["translation"] != r["refinement"]["result"]["translation"] for r in records),
               "new_prompt_tokens": sum(s["prompt_tokens"] for s in saved["selections"].values()) + sum(r["refinement"]["prompt_tokens"] for r in records),
               "new_output_tokens": sum(s["output_tokens"] for s in saved["selections"].values()) + sum(r["refinement"]["output_tokens"] for r in records),
               "reused_blind_prompt_tokens": sum(r["blind"]["prompt_tokens"] for r in records),
               "reused_blind_output_tokens": sum(r["blind"]["output_tokens"] for r in records),
               "interpretation": "Model-selected evidence and model refinements, not gold judgments or accepted parallel pairs. Token counts are not Modal GPU billing."}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(directory / "summary.json")
    return summary
