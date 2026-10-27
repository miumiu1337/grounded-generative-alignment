"""Offline technical filtering of the completed development outputs.

This does not assess meaning, calibrate mining thresholds or run a judge.
All original outputs are retained and every method uses the same checks.
"""

import csv
import hashlib
import io
import json
import math
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import blind_records as blind
import variant_a as a
import variant_b as b
import variant_c as c


ROOT = Path(__file__).resolve().parent
OUTPUT = Path("data/quality/technical_development_v1")
RUNS = {
    "blind_translation": ("data/translations/blind_gemma4_modal_development_v3", "5a6e2f8b0916e5a31e3fac50b08e26a716e287c3db980326286b9b16c82c8a9d"),
    "grounded_a": ("data/translations/grounded_a_gemma4_modal_development_v2", "8f8b50509e883ed6cedc0a98496c14ae8a3b66d2b843c6426c1f361c4d6d4d09"),
    "grounded_b": ("data/translations/grounded_b_gemma4_modal_development_v1", "443db3a01d50baa81e8e55ed5069096c207e2eb64ad50cf7bcca890a0b3f9b64"),
    "grounded_c": ("data/translations/grounded_c_gemma4_modal_development_v1", "0b4b35285423b98cb54e0f4c8274a10da0e5b135c24b8b80bc5089f3d3bdd742"),
    "rerank_only": ("data/reranking/rerank_only_gemma4_development_v1", "77aa7864c232a8158e816ca2fe99f8c067882aa2e9904ef2de9af7f3f2d2b296"),
    "mining": ("data/mining/laser_bge_development_v1", "ce3019d872a7656223716872e408762e48ba36b6a04980991ebad4b12f83818a"),
}
POLICY = {
    "version": 1, "split": "development", "mining_candidates": "all_saved_top_5",
    "reject": ["empty_target", "punctuation_only_target", "invalid_control_character", "incomplete_generation"],
    "warn_only": ["replacement_character", "source_copy", "duplicate_pair", "reused_target"],
    "duplicate_normalization": "Unicode_NFC_and_collapsed_whitespace_case_sensitive",
    "duplicate_scope": "within_method_only", "automatic_deduplication": False,
    "mining_acceptance_threshold": None, "semantic_evaluation": "not_run",
    "model_labels_and_selfchecks_affect_technical_decisions": False,
}


def read_json(path):
    return a.strict_json(path.read_text(encoding="utf-8"))


def validate_mining_record(record, page):
    candidates = record["candidates"]
    if (record.get("acceptance_status") != "unthresholded"
            or [candidate["rank"] for candidate in candidates] != [1, 2, 3, 4, 5]
            or len({tuple(candidate["sentence_ids"]) for candidate in candidates}) != 5
            or record["candidate_count"] != len(page["candidates"])
            or record["en_page_id"] != page["en_page_id"] or record["en_revision_id"] != page["en_revision_id"]):
        raise ValueError("Invalid mining candidate ranks/page or unexpectedly thresholded results")
    for candidate in candidates:
        span = next((s for s in page["candidates"] if s["sentence_ids"] == candidate["sentence_ids"]), None)
        if (span is None or any(candidate.get(key) != value for key, value in span.items())
                or set(candidate) != set(span) | {"rank", "margin_ratio", "cosine"}
                or any(type(candidate[key]) not in (float, int) or not math.isfinite(candidate[key])
                       for key in ("margin_ratio", "cosine"))):
            raise ValueError("Changed/non-finite mining candidate")
    if [c["margin_ratio"] for c in candidates] != sorted((c["margin_ratio"] for c in candidates), reverse=True):
        raise ValueError("Mining candidate margins are not in the saved rank order")


def load_grounded_exports(method, directory, manifest, items, pages):
    """Validate A/B target responses from their readable aggregate exports.

    Older per-source selection-file reads can stall on this filesystem. These
    exports contain complete refinement responses and original selection IDs,
    but not raw selection-stage responses; that scope is recorded in provenance.
    """
    protocol = {"grounded_a": a, "grounded_b": b}[method]
    protocol.validate_inputs(manifest, items, pages)
    expected = {item["source"]["source_id"]: item for item in items}
    records, paths = {}, {}
    for group in ("grounded_candidates", "unmatched", "contradictory"):
        path = directory / (group + ".jsonl")
        for line in path.read_text(encoding="utf-8").splitlines():
            record = a.strict_json(line)
            sid = record["source"]["source_id"]
            if sid not in expected or sid in records:
                raise ValueError(f"Unexpected/duplicate source in {method} aggregate exports")
            item = expected[sid]
            page = pages[str(item["source"]["bar_page_id"])]
            a.validate_selection(record["selection"], item, page)
            if (record["condition"] != method or record["run_id"] != a.digest(manifest)
                    or record["source"] != item["source"]
                    or method == "grounded_a" and record["blind"] != item["blind"]
                    or record["evidence"] != a.evidence_for(record["selection"], page)
                    or record["en_page_id"] != page["en_page_id"] or record["en_revision_id"] != page["en_revision_id"]
                    or record["output_group"] != group or group != a.output_group(record["selection"])
                    or record["quality_status"] != "unassessed"):
                raise ValueError(f"{method} aggregate output differs from its frozen source/evidence")
            stage = "refinement" if method == "grounded_a" else "translation"
            protocol.validate_stage(record[stage], stage, manifest, item, page, record["selection"])
            records[sid], paths[sid] = record, path
    if set(records) != set(expected):
        raise ValueError(f"Incomplete {method} aggregate exports")
    return {"results": records}, paths


def load_outputs():
    manifests = {}
    for method, (relative, expected) in RUNS.items():
        manifest = read_json(ROOT / relative / "run_manifest.json")
        if a.digest(manifest) != expected:
            raise ValueError(f"Changed {method} run manifest; use a new versioned quality protocol")
        manifests[method] = manifest
    current_b, items, pages = b.local_run()  # Includes frozen preprocessing/split checks.
    if current_b != manifests["grounded_b"]:
        raise ValueError("Current source/page protocol differs from completed B")
    sources = {item["source"]["source_id"]: item["source"] for item in items}
    rows, bundles = [], {}

    def add(method, record, path, target, finishes, *, rank=None, evidence_ids=None,
            match_status=None, contradiction=None, selfcheck=None, margin=None, cosine=None):
        source = record["source"]
        sid = source["source_id"]
        if source != sources.get(sid):
            raise ValueError(f"Changed/non-development source in {method}: {sid}")
        if target is not None and not isinstance(target, str):
            raise ValueError(f"Non-string target in {method}: {sid}")
        if target is None and not (method == "rerank_only" and match_status == "none" and not contradiction):
            raise ValueError(f"Unexpected missing target in {method}: {sid}")
        rows.append({
            "output_id": f"{method}|{sid}" + (f"|{rank}" if rank is not None else ""),
            "method": method, "source": source, "target_text": target,
            "target_kind": "verbatim_evidence" if method in {"laser_1", "bge_m3", "rerank_only"} else "generated_translation",
            "candidate_rank": rank, "evidence_ids": evidence_ids or [],
            "original_match_status": match_status, "original_contradiction": contradiction,
            "original_selfcheck_verdict": selfcheck, "margin_ratio": margin, "cosine": cosine,
            "finish_reasons": finishes, "artifact_path": str(path.relative_to(ROOT)),
            "artifact_canonical_sha256": a.digest(record),
        })

    directory = ROOT / RUNS["blind_translation"][0]
    blind_records = blind.read_saved(directory, manifests["blind_translation"], list(sources.values()))
    if set(blind_records) != set(sources):
        raise ValueError("Incomplete blind development outputs")
    bundles["blind_translation"] = a.digest(blind_records)
    for sid, record in blind_records.items():
        add("blind_translation", record, blind.record_path(directory, sid), record["translation"], [record["finish_reason"]])

    saved_by_method = {}
    for method, protocol in (("grounded_a", a), ("grounded_b", b), ("grounded_c", c)):
        print(f"Validating {method} saved stages", flush=True)
        directory = ROOT / RUNS[method][0]
        inputs = read_json(directory / "inputs.json")
        if (a.digest(inputs) != manifests[method]["inputs_sha256"] or inputs["pages"] != pages
                or {i["source"]["source_id"]: i["source"] for i in inputs["items"]} != sources):
            raise ValueError(f"Changed shared source/page inputs for {method}")
        aggregate_paths = {}
        if method in {"grounded_a", "grounded_b"}:
            saved, aggregate_paths = load_grounded_exports(method, directory, manifests[method], inputs["items"], pages)
        else:
            saved = protocol.read_saved(directory, manifests[method], inputs["items"], pages)
        if (set(saved["results"]) != set(sources)
                or method == "grounded_c" and set(saved["refinements"]) != set(sources)):
            raise ValueError(f"Incomplete {method} development outputs")
        saved_by_method[method] = saved
        bundles[method] = a.digest(saved)
        for sid, record in saved["results"].items():
            if method == "grounded_c":
                finishes = [saved["refinements"][sid]["finish_reason"], record["verification"]["finish_reason"]]
                add(method, record, a.record_path(directory / "results", sid), record["refinement"]["translation"],
                    finishes, selfcheck=record["verification"]["result"]["verdict"])
            else:
                stage = record["refinement" if method == "grounded_a" else "translation"]
                selection = record["selection"]
                finishes = [stage["finish_reason"]]
                add(method, record, aggregate_paths[sid], stage["result"]["translation"],
                    finishes,
                    evidence_ids=selection["evidence_ids"], match_status=selection["match_status"],
                    contradiction=selection["contradiction"])

    directory = ROOT / RUNS["rerank_only"][0]
    reranked = {}
    rerank_exports = []
    for group in ("exact_candidates", "partial_candidates", "unmatched", "contradictory"):
        path = directory / (group + ".jsonl")
        rerank_exports.extend((a.strict_json(line), path, group) for line in path.read_text(encoding="utf-8").splitlines())
    for record, path, group in rerank_exports:
        sid = record["source"]["source_id"]
        if sid not in sources or sid in reranked or record["output_group"] != group:
            raise ValueError("Unexpected/duplicate reranking record")
        original_b = saved_by_method["grounded_b"]["results"][sid]
        selection = original_b["selection"]
        page = pages[str(record["source"]["bar_page_id"])]
        evidence = a.evidence_for(selection, page)
        if (record["condition"] != "rerank_only" or record["run_id"] != RUNS["rerank_only"][1]
                or record["selection"] != selection or record["selection_sha256"] != original_b["selection_sha256"]
                or record["evidence"] != evidence
                or record["english_text"] != (evidence[0]["text"] if evidence else None)
                or record["en_page_id"] != page["en_page_id"] or record["en_revision_id"] != page["en_revision_id"]):
            raise ValueError(f"Reranking output differs from its frozen selection: {sid}")
        reranked[sid] = record
        add("rerank_only", record, path, record["english_text"], [],
            evidence_ids=selection["evidence_ids"], match_status=selection["match_status"], contradiction=selection["contradiction"])
    if set(reranked) != set(sources):
        raise ValueError("Incomplete reranking development outputs")
    bundles["rerank_only"] = a.digest(reranked)

    directory = ROOT / RUNS["mining"][0]
    groups = read_json(directory / "inputs.json")
    if a.digest(groups) != manifests["mining"]["inputs_sha256"]:
        raise ValueError("Changed mining inputs")
    if {s["source_id"]: s for group in groups for s in group["sources"]} != sources:
        raise ValueError("Mining and generation source pools differ")
    for group in groups:
        if group["targets"] != pages[str(group["bar_page_id"])]["candidates"]:
            raise ValueError("Mining candidate pool differs from the frozen English spans")
    for method in ("laser_1", "bge_m3"):
        path = directory / method / "rankings.json"
        ranked = read_json(path)
        if ranked["run_id"] != RUNS["mining"][1] or len(ranked["records"]) != 53:
            raise ValueError(f"Incomplete/changed {method} rankings")
        seen = set()
        for record in ranked["records"]:
            sid = record["source"]["source_id"]
            if sid in seen or record.get("acceptance_status") != "unthresholded":
                raise ValueError(f"Duplicate or unexpectedly thresholded {method} source: {sid}")
            seen.add(sid)
            page = pages[str(record["source"]["bar_page_id"])]
            validate_mining_record(record, page)
            for candidate in record["candidates"]:
                add(method, record, path, candidate["text"], [], rank=candidate["rank"],
                    evidence_ids=candidate["sentence_ids"], margin=candidate["margin_ratio"], cosine=candidate["cosine"])
        if seen != set(sources):
            raise ValueError(f"{method} does not cover the same 53 sources")
        bundles[method] = a.digest(ranked)
    rows.sort(key=lambda row: (row["method"], row["source"]["source_id"], row["candidate_rank"] or 0))
    return rows, {"upstream_manifests": manifests, "upstream_records_sha256": bundles,
                  "aggregate_validation_scope": "A/B aggregate exports: all sources, selected spans and complete English target-stage responses validated. Reranking aggregate evidence and selection hashes matched to B. Raw A/B/reranker selector-stage responses not reread due slow/stalled individual file access. No English is generated by reranking. C refinement and verification responses fully validated from saved stages.",
                  "source_pool_sha256": a.digest(sources), "normalized_outputs_sha256": a.digest(rows)}


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def annotate(rows):
    if len({row["output_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate technical-audit output IDs")
    audited, pairs, targets = [], defaultdict(list), defaultdict(list)
    for row in rows:
        text, reasons, warnings = row["target_text"], [], []
        missing = text is None
        if missing:
            if not (row["method"] == "rerank_only" and row["original_match_status"] == "none"
                    and row["original_contradiction"] is False and row["evidence_ids"] == []):
                raise ValueError("A missing target is valid only for an unmatched reranking source")
        else:
            if not isinstance(text, str):
                raise ValueError("Target must be text or an explicit unmatched reranking value")
            if not text.strip():
                reasons.append("empty_target")
            elif not any(character.isalnum() for character in text):
                reasons.append("punctuation_only_target")
            if any(unicodedata.category(ch) == "Cc" and ch not in "\t\r\n" for ch in text):
                reasons.append("invalid_control_character")
            if "\ufffd" in text:
                warnings.append("replacement_character")
            if normalized(text) == normalized(row["source"]["bar_text"]):
                warnings.append("source_copy")
        if row["target_kind"] == "generated_translation" and (not row["finish_reasons"]
                or any(reason != "stop" for reason in row["finish_reasons"])):
            reasons.append("incomplete_generation")
        status = "reject" if reasons else "no_target" if missing else "pass"
        result = {**row, "technical_status": status, "technical_reasons": reasons, "warnings": warnings,
                  "semantic_status": "unassessed", "needs_semantic_assessment": status == "pass"}
        audited.append(result)
        if text and text.strip():
            pairs[(row["method"], normalized(row["source"]["bar_text"]), normalized(text))].append(result)
            targets[(row["method"], normalized(text))].append(result)
    duplicate_groups = []
    for code, groups in (("duplicate_pair", pairs), ("reused_target", targets)):
        for group in groups.values():
            if len(group) < 2 or code == "reused_target" and len({r["source"]["source_id"] for r in group}) < 2:
                continue
            duplicate_groups.append({"kind": code, "method": group[0]["method"],
                                     "output_ids": sorted(r["output_id"] for r in group)})
            for row in group:
                row["warnings"].append(code)
    return audited, duplicate_groups


def export_report(directory, rows, provenance):
    audited, duplicates = annotate(rows)
    manifest = {"policy": POLICY, "provenance": provenance, "outputs_sha256": a.digest(rows),
                "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    run_id = a.digest(manifest)
    methods = {}
    for method in sorted({row["method"] for row in audited}):
        group = [row for row in audited if row["method"] == method]
        methods[method] = {
            "rows": len(group), "sources": len({row["source"]["source_id"] for row in group}),
            "technical_status_counts": dict(Counter(row["technical_status"] for row in group)),
            "reason_counts": dict(Counter(reason for row in group for reason in row["technical_reasons"])),
            "warning_counts": dict(Counter(warning for row in group for warning in row["warnings"])),
        }
    summary = {"run_id": run_id, "split": "development", "rows": len(audited), "methods": methods,
        "semantic_evaluation": "not_run", "accepted_pairs": None, "new_inference_calls": 0,
        "mining_acceptance_threshold": None,
        "interpretation": "Technical passes are readable outputs, not faithful translations or accepted alignments. Mining has five candidates per source; row counts are not comparable yields. Duplicates are flagged and retained. All semantic judgments remain unassessed."}
    a.atomic_json(directory / "run_manifest.json", manifest)
    a.atomic_json(directory / "report.json", {"run_id": run_id, "outputs": audited})
    a.atomic_json(directory / "duplicate_groups.json", duplicates)
    a.atomic_json(directory / "summary.json", summary)
    artifacts = {f"technical_{status}.jsonl": "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                                                   for row in audited if row["technical_status"] == status)
                 for status in ("pass", "reject", "no_target")}
    output = io.StringIO(newline="")
    fields = ["output_id", "method", "source_id", "bar_title", "bar_text", "target_text", "target_kind", "candidate_rank",
              "technical_status", "technical_reasons", "warnings", "semantic_status", "evidence_ids",
              "original_match_status", "original_contradiction", "original_selfcheck_verdict", "margin_ratio",
              "cosine", "artifact_path", "artifact_canonical_sha256"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for row in audited:
        flat = {key: row.get(key) for key in fields}
        flat.update(source_id=row["source"]["source_id"], bar_title=row["source"]["bar_title"], bar_text=row["source"]["bar_text"])
        for key in ("technical_reasons", "warnings", "evidence_ids"):
            flat[key] = json.dumps(row[key], ensure_ascii=False)
        writer.writerow(flat)
    artifacts["comparison.csv"] = output.getvalue()
    for name, content in artifacts.items():
        path = directory / name
        if path.exists() and path.read_bytes() != content.encode("utf-8"):
            raise ValueError(f"Changed technical-quality export: {name}; use a new run directory")
    for name, content in artifacts.items():
        path = directory / name
        if path.exists():
            continue
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8", newline="") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    return summary


if __name__ == "__main__":
    rows, provenance = load_outputs()
    print(json.dumps(export_report(ROOT / OUTPUT, rows, provenance), indent=2), flush=True)
