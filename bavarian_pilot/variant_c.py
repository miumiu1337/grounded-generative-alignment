"""Search-cache preparation, protocol and exports for search-assisted refinement.

External search is collected with the connected web tool, not by Gemma. Model
replay is offline with respect to web search and uses an immutable full cache.
"""

import csv
import difflib
import ipaddress
import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import variant_a as shared
from variant_a import MODEL, REVISION, atomic_json, digest, record_path, strict_json

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments/grounded_c_gemma4_modal_development_v1.json"


def query_for(item, page):
    topic = page["en"].split(" (en), revision ", 1)[0]
    words = re.findall(r"[\w’'-]+", item["blind"]["translation"], re.UNICODE)
    return f'{topic} ' + ' '.join(words[:26]) + ' -site:wikipedia.org -site:wikimedia.org'


def search_plan(config, items, pages):
    return {"search_policy": config["search_policy"],
            "base_inputs_sha256": digest({"items": items, "pages": pages}),
            "sources": [{"source_id": i["source"]["source_id"],
                         "query": query_for(i, pages[str(i["source"]["bar_page_id"])])} for i in items]}


def external_url(url):
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Invalid external URL")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or "." not in host or host.endswith((".local", ".internal"))):
        raise ValueError("External evidence requires a public HTTPS URL")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP-address evidence URLs are not allowed")
    domains = ("wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org", "wikiquote.org", "wikisource.org")
    if any(host == d or host.endswith("." + d) for d in domains):
        raise ValueError("Variant C external evidence must be beyond Wikimedia")
    # Keep the original URL in the log; this key prevents tracking variants
    # of one page from masquerading as two independent references.
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"trk", "fbclid", "gclid"}]
    return urlunsplit(("https", host, parsed.path or "/", urlencode(query), ""))


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("A timezone is required")
    return parsed


def validate_search(value, item, page, policy):
    expected = {"source_id", "query", "provider", "searched_at", "status", "sources", "failures", "search_ref"}
    if (not isinstance(value, dict) or set(value) != expected
            or value["source_id"] != item["source"]["source_id"] or value["query"] != query_for(item, page)
            or value["provider"] != policy["provider"] or value["status"] not in {"ok", "no_results", "unavailable"}
            or not isinstance(value["search_ref"], str) or not value["search_ref"].strip()
            or not isinstance(value["sources"], list) or not isinstance(value["failures"], list)
            or len({s["url"] for s in value["sources"] + value["failures"]}) > policy["max_pages_per_source"]):
        raise ValueError("Search cache differs from the fixed source/query/budget")
    searched = timestamp(value["searched_at"])
    urls = set()
    for n, source in enumerate(value["sources"], 1):
        if (set(source) != {"reference_id", "url", "title", "text", "retrieved_at", "provider_ref", "evidence_kind"}
                or source["reference_id"] != f"WEB{n}"
                or source["evidence_kind"] not in {"page_excerpt", "search_snippet"}
                or any(not isinstance(source[k], str) or not source[k].strip() for k in ("title", "text", "provider_ref"))
                or len(source["title"]) > 500 or len(source["text"]) > 3000
                or len(source["text"].split()) > policy["max_excerpt_words_per_page"]
                or timestamp(source["retrieved_at"]) < searched):
            raise ValueError("Malformed or over-budget cached external source")
        url = external_url(source["url"])
        if url in urls:
            raise ValueError("Repeated external URL in one search cache")
        urls.add(url)
    for failure in value["failures"]:
        if set(failure) != {"url", "reason"} or not isinstance(failure["reason"], str) or not failure["reason"].strip():
            raise ValueError("Malformed search access failure")
        external_url(failure["url"])
    if ((value["status"] == "ok") != bool(value["sources"])
            or value["status"] == "no_results" and value["failures"]
            or value["status"] == "unavailable" and not value["failures"]):
        raise ValueError("Search status disagrees with evidence/access failures")
    return value


def stage_messages(stage, manifest, item, page, refinement=None):
    if stage == "refinement":
        payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
                   "bavarian_page": page["bar"], "english_wikipedia_page": page["en"],
                   "external_search": item["search"]}
    elif stage == "verification":
        validate_refinement(refinement, item)
        payload = {"focus_source": item["source"], "bavarian_page": page["bar"],
                   "translation": refinement["translation"], "used_reference_ids": refinement["used_reference_ids"],
                   "external_sources": item["search"]["sources"]}
    else:
        raise ValueError("Unknown Variant C stage")
    return [{"role": "system", "content": manifest["prompts"][stage]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def schema_for(stage, item, page=None):
    properties = {"source_id": {"type": "string", "const": item["source"]["source_id"]}}
    if stage == "refinement":
        ids = [s["reference_id"] for s in item["search"]["sources"]]
        properties.update(translation={"type": "string", "minLength": 1},
            used_reference_ids={"type": "array", "maxItems": len(ids),
                                "items": {"type": "string", **({"enum": ids} if ids else {})}})
    elif stage == "verification":
        properties.update(verdict={"type": "string", "enum": ["pass", "flag", "uncertain"]},
            issues={"type": "array", "maxItems": 5, "items": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["addition", "omission", "mistranslation", "contradiction", "citation", "uncertainty"]},
                "description": {"type": "string", "minLength": 1}}, "required": ["kind", "description"], "additionalProperties": False}})
    else:
        raise ValueError("Unknown Variant C stage")
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def validate_refinement(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "translation", "used_reference_ids"}
            or value["source_id"] != item["source"]["source_id"]
            or not isinstance(value["translation"], str) or not value["translation"].strip()
            or not isinstance(value["used_reference_ids"], list)
            or any(not isinstance(i, str) for i in value["used_reference_ids"])
            or len(set(value["used_reference_ids"])) != len(value["used_reference_ids"])
            or not set(value["used_reference_ids"]) <= {s["reference_id"] for s in item["search"]["sources"]}):
        raise ValueError("Malformed refinement or unknown/duplicate external citation")


def validate_verification(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "verdict", "issues"}
            or value["source_id"] != item["source"]["source_id"] or value["verdict"] not in {"pass", "flag", "uncertain"}
            or not isinstance(value["issues"], list) or len(value["issues"]) > 5
            or (value["verdict"] == "pass") != (not value["issues"])):
        raise ValueError("Malformed/inconsistent self-verification")
    for issue in value["issues"]:
        if (not isinstance(issue, dict) or set(issue) != {"kind", "description"}
                or issue["kind"] not in {"addition", "omission", "mistranslation", "contradiction", "citation", "uncertainty"}
                or not isinstance(issue["description"], str) or not issue["description"].strip()):
            raise ValueError("Malformed verification issue")


def edits_for(draft, translation):
    return [{"operation": op, "draft_start": a, "draft_end": b, "old": draft[a:b], "new": translation[c:d]}
            for op, a, b, c, d in difflib.SequenceMatcher(None, draft, translation, autojunk=False).get_opcodes() if op != "equal"]


def validate_inputs(manifest, items, pages):
    config = manifest["experiment"]
    if (config["condition"] != "grounded_c" or config["protocol_version"] != 1 or config["split"] != "development"
            or config["model"] != MODEL or config["model_revision"] != REVISION
            or config["provider"] != "modal_self_hosted" or config["source_unit"] != "single_sentence"
            or config["preprocessing_freeze_sha256"] != "5cf98d0c8e9072cf4915f6679f5beb34d16396bd0ce3cd9022206078b590263f"
            or config["generation"] != {"temperature": 0.0, "seed": 42, "max_tokens": 1024, "enable_thinking": False,
                                        "structured_output": "json_schema_refinement_and_verification"}
            or config["inference"] != {"vllm": "0.21.0", "transformers": "5.5.4", "gpu": "H200", "dtype": "bfloat16",
                                       "max_model_len": 65536, "gpu_memory_utilization": 0.9}
            or not re.fullmatch(r"[a-z0-9_]+", config["run_name"]) or set(pages) != {"2544", "59"}
            or len(items) != 53 or manifest["inputs_sha256"] != digest({"items": items, "pages": pages})
            or Counter(i["source"]["bar_page_id"] for i in items) != {2544: 39, 59: 14}
            or len({i["source"]["source_id"] for i in items}) != 53):
        raise ValueError("Variant C protocol/input fingerprint differs")
    for item in items:
        if set(item) != {"source", "blind", "search"} or item["blind"]["run_id"] != config["blind_run_id"]:
            raise ValueError("Missing/changed frozen source or blind draft")
        record_path(Path("."), item["source"]["source_id"])
        validate_search(item["search"], item, pages[str(item["source"]["bar_page_id"])], config["search_policy"])


def validate_stage(record, stage, manifest, item, page, refinement=None):
    if (record.get("stage") != stage or record.get("source_id") != item["source"]["source_id"]
            or record.get("run_id") != digest(manifest)
            or record.get("messages") != stage_messages(stage, manifest, item, page, refinement)
            or record.get("schema") != schema_for(stage, item, page) or record.get("grammar") is not None
            or record.get("finish_reason") != "stop"
            or any(type(record.get(k)) is not int or record[k] < 0 for k in ("prompt_tokens", "output_tokens"))
            or record["prompt_tokens"] + 1024 > 65536 or record["output_tokens"] > 1024
            or strict_json(record["raw_response"]) != record.get("result")
            or any(record.get("runtime", {}).get(k) != manifest["experiment"]["inference"][k] for k in ("vllm", "transformers"))):
        raise ValueError(f"Saved {stage} record differs from the Variant C protocol")
    timestamp(record["recorded_at"])
    (validate_refinement if stage == "refinement" else validate_verification)(record["result"], item)


def read_saved(directory, manifest, items, pages):
    validate_inputs(manifest, items, pages)
    file = directory / "run_manifest.json"
    if file.exists() and strict_json(file.read_text()) != manifest:
        raise ValueError("Saved Variant C protocol differs; use a new run name")
    expected = {i["source"]["source_id"]: i for i in items}
    refinements, results = {}, {}
    for file in sorted((directory / "refinements").glob("*.json")):
        record = strict_json(file.read_text())
        sid = record.get("source_id")
        if sid not in expected or sid in refinements or file != record_path(directory / "refinements", sid):
            raise ValueError("Unexpected/duplicate Variant C refinement")
        item = expected[sid]
        validate_stage(record, "refinement", manifest, item, pages[str(item["source"]["bar_page_id"])])
        refinements[sid] = record
    for file in sorted((directory / "results").glob("*.json")):
        record = strict_json(file.read_text())
        sid = record.get("source", {}).get("source_id")
        if sid not in refinements or sid in results or file != record_path(directory / "results", sid):
            raise ValueError("Variant C result has no valid committed refinement")
        item, refined = expected[sid], refinements[sid]
        if (record.get("condition") != "grounded_c" or record.get("run_id") != digest(manifest)
                or any(record.get(k) != item[k] for k in ("source", "blind", "search"))
                or record.get("refinement") != refined["result"] or record.get("refinement_sha256") != digest(refined)
                or record.get("edits") != edits_for(item["blind"]["translation"], refined["result"]["translation"])
                or record.get("quality_status") != "ai_self_checked_not_gold"):
            raise ValueError("Variant C result differs from frozen inputs/refinement")
        validate_stage(record["verification"], "verification", manifest, item,
                       pages[str(item["source"]["bar_page_id"])], refined["result"])
        if timestamp(refined["recorded_at"]) > timestamp(record["verification"]["recorded_at"]):
            raise ValueError("Verification predates refinement")
        results[sid] = record
    return {"refinements": refinements, "results": results}


def base_run():
    config = strict_json(CONFIG.read_text())
    base_manifest, items, pages = shared.local_run()
    if base_manifest["experiment"]["blind_run_id"] != config["blind_run_id"]:
        raise ValueError("C requires the existing Gemma 4 blind run")
    return config, base_manifest, items, pages


def prepare_search():
    from translate_blind import local_path
    config, _, items, pages = base_run()
    directory = local_path(config["search_directory"])
    plan = search_plan(config, items, pages)
    atomic_json(directory / "plan.json", plan)
    return plan


def local_run():
    from download_articles import sha256
    from translate_blind import local_path
    from blind_records import optional_version, recorded_manifest
    import platform
    config, base_manifest, base_items, pages = base_run()
    directory = local_path(config["search_directory"])
    plan = search_plan(config, base_items, pages)
    if strict_json((directory / "plan.json").read_text()) != plan:
        raise ValueError("Cached search plan differs from source/draft inputs")
    items, missing = [], []
    for base in base_items:
        path = record_path(directory / "sources", base["source"]["source_id"])
        if not path.exists():
            missing.append(base["source"]["source_id"])
            continue
        items.append({**base, "search": strict_json(path.read_text())})
    if missing:
        raise ValueError(f"Search cache incomplete: {len(missing)}/53 sources missing; first: {missing[0]}. Collect evidence before model execution.")
    prompts = {s: local_path(config[s + "_prompt"]).read_text().strip() for s in ("refinement", "verification")}
    inputs = {"experiment": config, "experiment_sha256": sha256(CONFIG), "prompts": prompts,
        "prompt_sha256": {s: sha256(local_path(config[s + "_prompt"])) for s in prompts},
        "base_manifest_sha256": digest(base_manifest),
        "search_audit_sha256": {i["source"]["source_id"]: sha256(record_path(directory / "audit", i["source"]["source_id"]))
                                for i in items},
        "search_plan_sha256": digest(plan), "inputs_sha256": digest({"items": items, "pages": pages})}
    manifest = recorded_manifest(ROOT / "data/translations" / config["run_name"], inputs) or {
        **inputs,
        "protocol_sha256": sha256(Path(__file__)), "runner_sha256": sha256(ROOT / "runners/modal/variant_c.py"),
        "shared_protocol_sha256": sha256(ROOT / "variant_a.py"),
        "image_definition_sha256": sha256(ROOT / "runners/modal/blind.py"),
        "search_collection_protocol_sha256": sha256(directory / "collection_protocol.json"),
        "modal_sdk": optional_version("modal"), "python": platform.python_version()}
    validate_inputs(manifest, items, pages)
    return manifest, items, pages


def write_local(saved, manifest, items, pages):
    import tempfile
    directory = ROOT / "data/translations" / manifest["experiment"]["run_name"]
    read_saved(directory, manifest, items, pages)
    with tempfile.TemporaryDirectory() as temporary:
        staging = Path(temporary)
        for kind in ("refinements", "results"):
            for sid, record in saved[kind].items():
                atomic_json(record_path(staging / kind, sid), record)
        if read_saved(staging, manifest, items, pages) != saved:
            raise ValueError("Remote C bundle keys differ from record IDs")
    atomic_json(directory / "run_manifest.json", manifest)
    atomic_json(directory / "inputs.json", {"items": items, "pages": pages})
    for kind in ("refinements", "results"):
        for sid, record in saved[kind].items():
            atomic_json(record_path(directory / kind, sid), record)
    return directory, read_saved(directory, manifest, items, pages)


def export_results(directory, saved, manifest):
    records = [saved["results"][sid] for sid in sorted(saved["results"])]
    fields = ["source_id", "bar_title", "bar_text", "blind_translation", "refined_translation", "verification_verdict",
              "verification_issues", "search_query", "search_status", "used_reference_ids", "reference_urls", "edits",
              "new_prompt_tokens", "new_output_tokens", "quality_status"]
    temporary = directory / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for r in records:
            sid = r["source"]["source_id"]
            stages = (saved["refinements"][sid], r["verification"])
            writer.writerow({"source_id": sid, "bar_title": r["source"]["bar_title"], "bar_text": r["source"]["bar_text"],
                "blind_translation": r["blind"]["translation"], "refined_translation": r["refinement"]["translation"],
                "verification_verdict": r["verification"]["result"]["verdict"],
                "verification_issues": json.dumps(r["verification"]["result"]["issues"], ensure_ascii=False),
                "search_query": r["search"]["query"], "search_status": r["search"]["status"],
                "used_reference_ids": ";".join(r["refinement"]["used_reference_ids"]),
                "reference_urls": ";".join(s["url"] for s in r["search"]["sources"]),
                "edits": json.dumps(r["edits"], ensure_ascii=False), "quality_status": r["quality_status"],
                **{f"new_{k}": sum(s[k] for s in stages) for k in ("prompt_tokens", "output_tokens")}})
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(directory / "comparison.csv")
    for verdict in ("pass", "flag", "uncertain"):
        temporary = directory / f"selfcheck_{verdict}.jsonl.tmp"
        with temporary.open("w", encoding="utf-8") as output:
            for r in records:
                if r["verification"]["result"]["verdict"] == verdict:
                    output.write(json.dumps(r, ensure_ascii=False) + "\n")
        temporary.replace(directory / f"selfcheck_{verdict}.jsonl")
    summary = {"run_id": digest(manifest), "completed": len(records), "expected": 53,
        "saved_refinements": len(saved["refinements"]),
        "selfcheck_verdicts": dict(Counter(r["verification"]["result"]["verdict"] for r in records)),
        "search_status_counts": dict(Counter(r["search"]["status"] for r in records)),
        "externally_cited_outputs": sum(bool(r["refinement"]["used_reference_ids"]) for r in records),
        "translations_changed": sum(bool(r["edits"]) for r in records),
        "new_prompt_tokens": sum(r["prompt_tokens"] for r in saved["refinements"].values()) + sum(r["verification"]["prompt_tokens"] for r in records),
        "new_output_tokens": sum(r["output_tokens"] for r in saved["refinements"].values()) + sum(r["verification"]["output_tokens"] for r in records),
        "reused_blind_prompt_tokens": sum(r["blind"]["prompt_tokens"] for r in records),
        "reused_blind_output_tokens": sum(r["blind"]["output_tokens"] for r in records),
        "interpretation": "Same-model AI self-checks, not independent evaluation, accepted pairs or gold labels. Tokens are not GPU billing."}
    temporary = directory / "summary.json.tmp"
    temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(directory / "summary.json")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-search", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(prepare_search(), ensure_ascii=False, indent=2), flush=True)
