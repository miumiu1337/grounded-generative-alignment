"""Export the existing draft-free B selections as a no-rewriting ablation.

No inference or translation stage is called. English text is resolved directly
from the frozen candidate spans; original selection requests remain auditable.
"""

import csv
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import variant_b as selection_protocol
from variant_a import digest, evidence_for, record_path, strict_json


ROOT = Path(__file__).resolve().parent
SOURCE_DIRECTORY = Path("data/translations/grounded_b_gemma4_modal_development_v1")
SOURCE_RUN_ID = "443db3a01d50baa81e8e55ed5069096c207e2eb64ad50cf7bcca890a0b3f9b64"
OUTPUT_DIRECTORY = Path("data/reranking/rerank_only_gemma4_development_v1")
GROUPS = ("exact_candidates", "partial_candidates", "unmatched", "contradictory")


def validate_selections(manifest, items, pages, selections):
    selection_protocol.validate_inputs(manifest, items, pages)
    expected = {item["source"]["source_id"] for item in items}
    if set(selections) != expected:
        raise ValueError("Reranking requires all 53 saved selections, with no unexpected IDs")
    for item in items:
        record = selections[item["source"]["source_id"]]
        page = pages[str(item["source"]["bar_page_id"])]
        selection_protocol.validate_stage(record, "selection", manifest, item, page)


def load_selections():
    manifest, items, pages = selection_protocol.local_run()
    directory = ROOT / SOURCE_DIRECTORY
    if (digest(manifest) != SOURCE_RUN_ID
            or strict_json((directory / "run_manifest.json").read_text()) != manifest
            or strict_json((directory / "inputs.json").read_text()) != {"items": items, "pages": pages}):
        raise ValueError("Reranking source run or frozen inputs differ from completed B")
    selections = {}
    for path in sorted((directory / "selections").glob("*.json")):
        record = strict_json(path.read_text())
        source_id = record.get("source_id")
        if source_id in selections or path != record_path(directory / "selections", source_id):
            raise ValueError("Unexpected or duplicate saved selection file")
        selections[source_id] = record
    validate_selections(manifest, items, pages, selections)
    return manifest, items, pages, selections


def output_group(selection):
    if selection["contradiction"]:
        return "contradictory"
    return {"exact": "exact_candidates", "partial": "partial_candidates", "none": "unmatched"}[selection["match_status"]]


def export_run(directory, source_manifest, items, pages, selections):
    validate_selections(source_manifest, items, pages, selections)
    manifest = {
        "condition": "rerank_only", "protocol_version": 1, "split": "development",
        "source_language": "bar", "target_language": "en", "generated_english": False,
        "source_directory": str(SOURCE_DIRECTORY), "source_run_id": digest(source_manifest),
        "reuse_policy": "verbatim_B_selection_stage_before_translation",
        "source_selection_sha256": {sid: digest(selections[sid]) for sid in sorted(selections)},
        "inputs_sha256": digest({"items": items, "pages": pages}),
        "model": source_manifest["experiment"]["model"],
        "model_revision": source_manifest["experiment"]["model_revision"],
        "inference": source_manifest["experiment"]["inference"],
        "generation": {key: source_manifest["experiment"]["generation"][key]
                       for key in ("temperature", "seed", "max_tokens", "enable_thinking")},
        "structured_output": "selection_valid_span_grammar",
        "selection_prompt": source_manifest["prompts"]["selection"],
        "protocol_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "interpretation": "Paired ablation sharing B's selection calls, not an independent inference replicate. No translation stage is used.",
    }
    run_id = digest(manifest)
    records = []
    for item in items:
        sid = item["source"]["source_id"]
        page = pages[str(item["source"]["bar_page_id"])]
        selection = selections[sid]["result"]
        evidence = evidence_for(selection, page)
        records.append({
            "condition": "rerank_only", "run_id": run_id, "source": item["source"],
            "selection": selection, "selection_sha256": digest(selections[sid]),
            "evidence": evidence, "english_text": evidence[0]["text"] if evidence else None,
            "en_page_id": page["en_page_id"], "en_revision_id": page["en_revision_id"],
            "output_group": output_group(selection), "quality_status": "unassessed",
        })
    records.sort(key=lambda record: record["source"]["source_id"])
    summary = {
        "run_id": run_id, "source_run_id": digest(source_manifest), "completed": len(records), "expected": 53,
        "match_status_counts": dict(Counter(r["selection"]["match_status"] for r in records)),
        "output_group_counts": {group: sum(r["output_group"] == group for r in records) for group in GROUPS},
        "per_article": {page: dict(Counter(r["selection"]["match_status"] for r in records
                            if str(r["source"]["bar_page_id"]) == page)) for page in sorted(pages)},
        "reused_selection_prompt_tokens": sum(r["prompt_tokens"] for r in selections.values()),
        "reused_selection_output_tokens": sum(r["output_tokens"] for r in selections.values()),
        "new_inference_calls": 0, "new_prompt_tokens": 0, "new_output_tokens": 0,
        "generated_translations": 0,
        "interpretation": "Model-selected unchanged English spans, not gold alignments or accepted parallel pairs. Partial evidence may contain extra or missing claims. Shared B selection compute is counted once in total study costs; a standalone reranker would incur its selection-stage cost. Tokens are not GPU billing.",
    }

    def json_text(value):
        return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"

    artifacts = {"run_manifest.json": json_text(manifest), "source_run_manifest.json": json_text(source_manifest),
                 "inputs.json": json_text({"items": items, "pages": pages}), "summary.json": json_text(summary)}
    for record in records:
        sid = record["source"]["source_id"]
        artifacts[str(record_path(Path("results"), sid))] = json_text(record)
        artifacts[str(record_path(Path("selections"), sid))] = json_text(selections[sid])
    for group in GROUPS:
        artifacts[group + ".jsonl"] = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n"
                                              for r in records if r["output_group"] == group)
    output = io.StringIO(newline="")
    fields = ["source_id", "bar_title", "bar_text", "english_text", "evidence_ids", "match_status",
              "contradiction", "output_group", "en_page_id", "en_revision_id", "quality_status",
              "reused_selection_prompt_tokens", "reused_selection_output_tokens"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for record in records:
        source, selection = record["source"], record["selection"]
        stage = selections[source["source_id"]]
        writer.writerow({"source_id": source["source_id"], "bar_title": source["bar_title"], "bar_text": source["bar_text"],
            **{key: record[key] for key in ("english_text", "output_group", "en_page_id", "en_revision_id", "quality_status")},
            "evidence_ids": ";".join(selection["evidence_ids"]), "match_status": selection["match_status"],
            "contradiction": selection["contradiction"], "reused_selection_prompt_tokens": stage["prompt_tokens"],
            "reused_selection_output_tokens": stage["output_tokens"]})
    artifacts["comparison.csv"] = output.getvalue()

    # Validate every existing artifact before writing, so changed/tampered runs
    # cannot be silently overwritten or partially refreshed.
    existing = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if existing - set(artifacts) - {name + ".tmp" for name in artifacts}:
        raise ValueError("Unexpected files in reranking output; use a new versioned directory")
    for name, text in artifacts.items():
        path = directory / name
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            raise ValueError(f"Saved reranking artifact differs: {name}; use a new versioned directory")
    for name, text in artifacts.items():
        path = directory / name
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        temporary.replace(path)
    return summary


if __name__ == "__main__":
    manifest, items, pages, selections = load_selections()
    print(json.dumps(export_run(ROOT / OUTPUT_DIRECTORY, manifest, items, pages, selections), indent=2), flush=True)
