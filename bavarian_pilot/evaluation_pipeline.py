"""Frozen eight-article evaluation, separate from the completed development runs.

python evaluation_pipeline.py --freeze          Commit settings before generation
python evaluation_pipeline.py                   Validate and print the offline plan
python evaluation_pipeline.py --mine            Run both existing local encoders
python evaluation_pipeline.py --prepare-search  Plan C after blind translation
python evaluation_pipeline.py --prepare-judge   Audit all methods and hide their labels
python evaluation_pipeline.py --score           Score complete recorded AI judgments
"""

import argparse
import csv
import fcntl
import hashlib
import io
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import variant_a as a
import variant_b as b
import variant_c as c
from semantic_evaluation import RUBRIC, prepare_items, summarize, validate_judgments

ROOT = Path(__file__).resolve().parent
FREEZE = ROOT / "experiments/evaluation_eight_v2.json"
OUTPUT = ROOT / "data/evaluation_runs/frozen_eight_v2"
CONFIGS = {
    "blind": "experiments/blind_gemma4_modal_development_v3.json",
    "mining": "experiments/mining_development_v1.json",
    "a": "experiments/grounded_a_gemma4_modal_development_v2.json",
    "b": "experiments/grounded_b_gemma4_modal_development_v1.json",
    "c": "experiments/grounded_c_gemma4_modal_development_v1.json",
}
STAGES = {"blind": ("blind",), "a": ("selection", "refinement"),
          "b": ("selection", "translation"), "c": ("refinement", "verification")}
METHODS = {"blind": "blind_translation", "a": "grounded_a", "b": "grounded_b", "c": "grounded_c"}
EVALUATION_PAGES = {56, 57, 58, 60, 62, 63, 65, 2241}


def read(path):
    return a.strict_json(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def corpus():
    """Verify the original freeze and load only its assigned evaluation articles."""
    from translate_blind import load_sources

    preprocessing = read(ROOT / "data/preprocessing_freeze.json")
    # This existing loader validates all prepared article hashes, preprocessing
    # code, split, provenance and dependencies without changing any input.
    load_sources(read(ROOT / CONFIGS["blind"]))
    split = read(ROOT / "data/article_split.json")
    selected = {p["bar_page_id"]: p for p in split["evaluation"]}
    sources, pages, groups = [], {}, []
    for entry in read(ROOT / preprocessing["manifest_file"])["articles"]:
        article = read(ROOT / preprocessing["prepared_directory"] / entry["file"])
        if article["split"] != "evaluation":
            continue
        bar, en = article["bar"], article["en"]
        pid = bar["page_id"]
        if (pid not in selected or str(pid) in pages or bar["title"] != selected[pid]["bar_title"]
                or en["page_id"] != selected[pid]["en_page_id"]):
            raise ValueError("Evaluation pages differ from the frozen assignment")
        singles = {s["sentence_ids"][0]: s["text"] for s in bar["candidate_spans"] if len(s["sentence_ids"]) == 1}
        eligible = [s for s in bar["sentences"] if s["eligible_for_matching"]]
        if singles != {s["sentence_id"]: s["text"] for s in eligible}:
            raise ValueError(f"Eligibility/candidate mismatch: {pid}")
        group_sources = [{"source_id": s["sentence_id"], "bar_text": s["text"], "bar_title": bar["title"],
                          "bar_page_id": pid, "bar_revision_id": bar["revision_id"]} for s in eligible]
        sources.extend(group_sources)
        pages[str(pid)] = {"bar": a.numbered_page(bar), "en": a.numbered_page(en),
                          "en_page_id": en["page_id"], "en_revision_id": en["revision_id"],
                          "candidates": en["candidate_spans"]}
        groups.append({"bar_page_id": pid, "bar_title": bar["title"], "en_page_id": en["page_id"],
                       "en_title": en["title"], "en_revision_id": en["revision_id"],
                       "sources": group_sources, "targets": en["candidate_spans"]})
    if (set(pages) != {str(p) for p in selected} or len(pages) != 8 or len(sources) != 524
            or len({s["source_id"] for s in sources}) != 524):
        raise ValueError("Expected the eight fixed evaluation pages and 524 unique sources")
    return sources, pages, groups


def frozen_protocol():
    sources, pages, groups = corpus()
    paths = set(CONFIGS.values()) | {
        "data/preprocessing_freeze.json", "data/article_split.json", "data/prepared/manifest.json",
        "evaluation_pipeline.py", "modal_evaluation.py", "tests/test_evaluation_pipeline.py",
        "translate_blind.py", "modal_translate_blind.py", "mine_baselines.py", "variant_a.py", "variant_b.py",
        "variant_c.py", "output_quality.py", "semantic_evaluation.py", "requirements-mining.txt", "requirements-modal.txt",
        "data/search/grounded_c_web_development_v1/collection_protocol.json",
        "data/quality/technical_development_v1/run_manifest.json",
    }
    protocols = {}
    for method, relative in CONFIGS.items():
        config = read(ROOT / relative)
        paths.update(v for k, v in config.items() if k.endswith("_prompt") or k == "prompt_file")
        # These are development artifact dependencies, not inference settings.
        config = {k: v for k, v in config.items() if k not in {
            "blind_directory", "blind_run_id", "comparison_a_directory", "comparison_a_run_id", "search_directory"}}
        config.update(split="evaluation", run_name=f"frozen_eight_v2_{method}")
        protocols[method] = config
    policy = read(ROOT / "data/quality/technical_development_v1/run_manifest.json")["policy"]
    return {"protocol_version": 2, "created_on": "2026-10-26", "split": "evaluation", "sources": 524,
        "previous_attempt": {"freeze": "experiments/evaluation_eight_v1.json",
            "freeze_sha256": sha(ROOT / "experiments/evaluation_eight_v1.json"),
            "reason": "CPU worker import failed before inference: missing uploaded modal_translate_blind helper. Packaging fixed; scientific settings unchanged."},
        "articles": [{"bar_page_id": g["bar_page_id"], "bar_title": g["bar_title"],
                      "source_count": len(g["sources"]), "english_spans": len(g["targets"])} for g in groups],
        "sources_sha256": a.digest(sources), "pages_sha256": a.digest(pages), "groups_sha256": a.digest(groups),
        "methods": protocols, "files_sha256": {p: sha(ROOT / p) for p in sorted(paths)},
        "technical_policy": {**policy, "split": "evaluation"},
        "assessment": {"rubric": RUBRIC, "seed": 20261026, "primary_candidates": "rank1_mining_final_output_others",
            "strict_retention": "score_4_and_technical_pass", "missing_outputs": "remain_in_yield_denominator",
            "mean_fidelity": "conditional_on_nonempty_confidently_scored_outputs",
            "deduplication": "one_judgment_per_identical_source_and_english_pair",
            "report": ["mean_fidelity", "strict_yield", "mostly_or_fully_faithful", "issue_counts", "per_article", "paired_vs_blind"],
            "assessor": "Codex conversational AI; exact model/settings unavailable; no validated Bavarian expertise",
            "blinding": "method_names_hidden_in_new_inputs; prior_project_context_prevents_perfect_blinding",
            "gold_alignment_precision_recall": None, "cost": "actual_billing_required; tokens_are_not_billing"},
        "reranking": "reuse_B_selection_verbatim_without_new_inference",
        "search": "same_cached_web_collection_protocol_as_development; freeze_all_524_records_before_C_inference",
        "evaluation_articles_previously_used": True,
        "change_policy": "Do not tune on evaluation outputs. Amendments require a new named protocol and explicit reporting."}


def load_freeze():
    frozen = read(FREEZE)
    if frozen != frozen_protocol():
        raise ValueError("Frozen evaluation settings/code/inputs changed; do not run or overwrite this experiment")
    return frozen


def path_for(directory, sid):
    if not re.fullmatch(r"bar_\d+:v1\.0\.1:S\d+", sid):
        raise ValueError("Invalid evaluation source ID")
    return directory / (sid.replace(":", "_") + ".json")


def messages(method, stage, manifest, item, page, previous=None):
    if method == "blind":
        return [{"role": "system", "content": manifest["prompts"]["blind"]},
                {"role": "user", "content": f"Bavarian text:\n{item['source']['bar_text']}"}]
    return {"a": a, "b": b, "c": c}[method].stage_messages(stage, manifest, item, page, previous)


def constraints(method, stage, item, page):
    if method == "blind":
        return None, None
    protocol = {"a": a, "b": b, "c": c}[method]
    return protocol.schema_for(stage, item, page), a.selection_grammar(item, page) if stage == "selection" else None


def validate_bundle(manifest, items, pages):
    if (manifest["experiment"]["split"] != "evaluation"
            or manifest["method"] not in STAGES or len(items) != 524
            or manifest["inputs_sha256"] != a.digest({"items": items, "pages": pages})
            or len({i["source"]["source_id"] for i in items}) != 524
            or {i["source"]["bar_page_id"] for i in items} != EVALUATION_PAGES):
        raise ValueError("Changed or incomplete evaluation inputs")
    method = manifest["method"]
    if method == "blind" and pages:
        raise ValueError("Blind payload must not include English pages")
    if method != "blind" and set(pages) != {str(p) for p in EVALUATION_PAGES}:
        raise ValueError("Unexpected evaluation page set")
    for item in items:
        sid = item["source"]["source_id"]
        path_for(Path("."), sid)
        expected = {"source"} | ({"blind"} if method in {"a", "c"} else set()) | ({"search"} if method == "c" else set())
        if set(item) != expected or set(item["source"]) != {"source_id", "bar_text", "bar_title", "bar_page_id", "bar_revision_id"}:
            raise ValueError("Unexpected fields/draft contamination in evaluation inputs")
        if not sid.startswith(f"bar_{item['source']['bar_page_id']}:"):
            raise ValueError("Source/page mismatch")
        if method != "blind" and str(item["source"]["bar_page_id"]) not in pages:
            raise ValueError("Missing frozen context")
        if method == "c":
            c.validate_search(item["search"], item, pages[str(item["source"]["bar_page_id"])], manifest["experiment"]["search_policy"])


def validate_stage(record, method, stage, manifest, item, page, previous=None):
    schema, grammar = constraints(method, stage, item, page)
    config = manifest["experiment"]
    if (record.get("run_id") != a.digest(manifest) or record.get("source_id") != item["source"]["source_id"]
            or record.get("stage") != stage or record.get("messages") != messages(method, stage, manifest, item, page, previous)
            or record.get("schema") != schema or record.get("grammar") != grammar
            or record.get("finish_reason") != "stop"
            or any(type(record.get(k)) is not int or record[k] < 0 for k in ("prompt_tokens", "output_tokens"))
            or record["prompt_tokens"] + config["generation"]["max_tokens"] > config["inference"]["max_model_len"]
            or record["output_tokens"] > config["generation"]["max_tokens"]
            or any(record.get("runtime", {}).get(k) != config["inference"][k] for k in ("vllm", "transformers"))):
        raise ValueError(f"Invalid or changed evaluation {method}/{stage} response")
    c.timestamp(record["recorded_at"])
    if method == "blind":
        if (not isinstance(record.get("raw_response"), str) or not record["raw_response"].strip()
                or record.get("result") != {"translation": record["raw_response"].strip()}):
            raise ValueError("Empty or changed blind response")
    else:
        if a.strict_json(record["raw_response"]) != record.get("result"):
            raise ValueError("Raw response differs from parsed output")
        if stage == "selection":
            a.validate_selection(record["result"], item, page)
        elif method == "c":
            (c.validate_refinement if stage == "refinement" else c.validate_verification)(record["result"], item)
        else:
            a.validate_refinement(record["result"], item)


def read_saved(directory, manifest, items, pages):
    validate_bundle(manifest, items, pages)
    if (directory / "run_manifest.json").exists() and read(directory / "run_manifest.json") != manifest:
        raise ValueError("Saved evaluation run differs; refuse to reuse/overwrite")
    expected = {i["source"]["source_id"]: i for i in items}
    saved = {}
    for stage in STAGES[manifest["method"]]:
        saved[stage] = {}
        for path in sorted((directory / stage).glob("*.json")):
            record = read(path)
            sid = record.get("source_id")
            if sid not in expected or path != path_for(directory / stage, sid):
                raise ValueError("Unknown evaluation result or incorrect filename")
            item = expected[sid]
            previous_stage = STAGES[manifest["method"]][0]
            previous = saved.get(previous_stage, {}).get(sid)
            if stage != previous_stage and previous is None:
                raise ValueError("Second-stage result has no committed first stage")
            validate_stage(record, manifest["method"], stage, manifest, item,
                           pages.get(str(item["source"]["bar_page_id"])), previous["result"] if previous else None)
            if previous and c.timestamp(record["recorded_at"]) < c.timestamp(previous["recorded_at"]):
                raise ValueError("Second stage predates the first")
            saved[stage][sid] = record
    return saved


def build_run(method, frozen=None):
    frozen = load_freeze() if frozen is None else frozen
    sources, pages, _ = corpus()
    config = frozen["methods"][method]
    items = [{"source": s} for s in sources]
    if method in {"a", "c"}:
        baseline, base_items, _ = build_run("blind", frozen)
        saved = read_saved(OUTPUT / "blind", baseline, base_items, {})["blind"]
        if len(saved) != len(sources):
            raise ValueError(f"Complete blind translations first: {len(saved)}/{len(sources)}")
        for item in items:
            sid = item["source"]["source_id"]
            item["blind"] = {"translation": saved[sid]["result"]["translation"],
                             "run_id": a.digest(baseline), "record_sha256": a.digest(saved[sid])}
    if method == "c":
        plan = c.search_plan(config, items, pages)
        if read(OUTPUT / "search/plan.json") != plan:
            raise ValueError("External search plan changed")
        for item in items:
            item["search"] = read(path_for(OUTPUT / "search/sources", item["source"]["source_id"]))
            audit = read(path_for(OUTPUT / "search/audit", item["source"]["source_id"]))
            if audit["source_id"] != item["source"]["source_id"] or audit["query"] != item["search"]["query"]:
                raise ValueError("External search audit differs from its query/source")
        cache = {"plan_sha256": a.digest(plan), "sources_sha256": a.digest([i["search"] for i in items]),
                 "audit_sha256": {i["source"]["source_id"]: sha(path_for(OUTPUT / "search/audit", i["source"]["source_id"])) for i in items}}
    if method == "blind":
        pages = {}
    prompts = ({"blind": read_prompt(config["prompt_file"])} if method == "blind" else
               {stage: read_prompt(config[stage + "_prompt"]) for stage in STAGES[method]})
    manifest = {"method": method, "experiment": config, "evaluation_freeze_sha256": sha(FREEZE),
                "prompts": prompts, "inputs_sha256": a.digest({"items": items, "pages": pages})}
    if method == "c":
        manifest["search_cache_sha256"] = a.digest(cache)
    validate_bundle(manifest, items, pages)
    if method == "c":
        a.atomic_json(OUTPUT / "search/cache_freeze.json", cache)
    return manifest, items, pages


def read_prompt(relative):
    return (ROOT / relative).read_text(encoding="utf-8").strip()


def write_local(directory, saved, manifest, items, pages):
    # Validate every received record before publishing any of the response bundle.
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        for stage, records in saved.items():
            if stage not in STAGES[manifest["method"]]:
                raise ValueError("Unknown stage in remote response")
            for sid, record in records.items():
                a.atomic_json(path_for(Path(tmp) / stage, sid), record)
        verified = read_saved(Path(tmp), manifest, items, pages)
        if verified != saved:
            raise ValueError("Incomplete remote stage bundle")
    read_saved(directory, manifest, items, pages)
    a.atomic_json(directory / "run_manifest.json", manifest)
    a.atomic_json(directory / "inputs.json", {"items": items, "pages": pages})
    for stage, records in saved.items():
        for sid, record in records.items():
            a.atomic_json(path_for(directory / stage, sid), record)
    if manifest["method"] == "c":
        for item in items:
            sid = item["source"]["source_id"]
            if sid in saved["refinement"]:
                refined = saved["refinement"][sid]["result"]
                a.atomic_json(path_for(directory / "edits", sid), {
                    "source_id": sid, "blind_translation": item["blind"]["translation"],
                    "refined_translation": refined["translation"], "used_reference_ids": refined["used_reference_ids"],
                    "edits": c.edits_for(item["blind"]["translation"], refined["translation"])})


def prepare_search():
    manifest, items, pages = build_run("a")
    policy = load_freeze()["methods"]["c"]
    plan = c.search_plan(policy, items, pages)
    a.atomic_json(OUTPUT / "search/plan.json", plan)
    print(f"Prepared {len(plan['sources'])} fixed web queries; no search has been run")


def mine():
    import mine_baselines as mining
    frozen = load_freeze()
    config = {**frozen["methods"]["mining"], "evaluation_freeze_sha256": sha(FREEZE)}
    _, _, groups = corpus()
    directory = OUTPUT / "mining"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # execute() reuses the same model checks, embedding caches and scoring;
        # the development-only source selector is deliberately not invoked.
        mining.EXPERIMENT = FREEZE
        mining.execute(config, groups, directory)
        comparison = directory / "comparison.md"
        text = comparison.read_text(encoding="utf-8")
        comparison.write_text(text.replace("# LASER-1 and BGE-M3 development comparison",
                                           "# LASER-1 and BGE-M3 fixed evaluation comparison", 1), encoding="utf-8")


def collect():
    from output_quality import annotate, validate_mining_record
    frozen = load_freeze()
    rows, runs = [], {}
    sources, pages, groups = corpus()
    expected = {s["source_id"]: s for s in sources}

    def add(method, source, text, kind, record, rank=None, selection=None):
        rows.append({"output_id": f"{method}:{source['source_id']}:{rank or 0}", "method": method,
            "source": source, "target_text": text, "target_kind": kind, "candidate_rank": rank,
            "finish_reasons": ["stop"] if kind == "generated_translation" else [],
            "evidence_ids": selection["evidence_ids"] if selection else [],
            "original_match_status": selection["match_status"] if selection else None,
            "original_contradiction": selection["contradiction"] if selection else None,
            "artifact_canonical_sha256": a.digest(record)})

    for method in STAGES:
        manifest, items, method_pages = build_run(method, frozen)
        saved = read_saved(OUTPUT / method, manifest, items, method_pages)
        final = STAGES[method][-1]
        if set(saved[final]) != set(expected):
            raise ValueError(f"Incomplete {method}: {len(saved[final])}/524; do not prepare assessment yet")
        runs[method] = a.digest(manifest)
        for item in items:
            sid = item["source"]["source_id"]
            record = saved[final][sid]
            translation = saved["refinement"][sid] if method == "c" else record
            add(METHODS[method], item["source"], translation["result"]["translation"], "generated_translation", record)
            if method == "b":
                selected = saved["selection"][sid]
                selection = selected["result"]
                evidence = a.evidence_for(selection, method_pages[str(item["source"]["bar_page_id"])])
                # Contradictory selections retain their original evidence and flags,
                # as in development; only an empty selection is an abstention.
                add("rerank_only", item["source"], evidence[0]["text"] if evidence else None,
                    "existing_english_evidence", selected, selection=selection)
    mining_manifest = read(OUTPUT / "mining/run_manifest.json")
    config = {**frozen["methods"]["mining"], "evaluation_freeze_sha256": sha(FREEZE)}
    if mining_manifest["experiment"] != config or mining_manifest["inputs_sha256"] != a.digest(groups):
        raise ValueError("Mining run differs from frozen settings/inputs")
    for name in ("laser_1", "bge_m3"):
        data = read(OUTPUT / f"mining/{name}/rankings.json")
        records = data["records"]
        if (data["run_id"] != a.digest(mining_manifest) or len(records) != len(sources)
                or {r["source"]["source_id"] for r in records} != set(expected)):
            raise ValueError("Incomplete or changed mining rankings")
        for record in records:
            source = record["source"]
            if source != expected[source["source_id"]]:
                raise ValueError("Mining source changed")
            validate_mining_record(record, pages[str(source["bar_page_id"])])
            for candidate in record["candidates"]:
                add(name, source, candidate["text"], "existing_english_evidence", record, candidate["rank"])
    audited, duplicates = annotate(rows)
    a.atomic_json(OUTPUT / "quality/report.json", {"evaluation_freeze_sha256": sha(FREEZE),
                  "runs": runs, "outputs": audited, "duplicate_groups": duplicates})
    return audited, pages


def prepare_judge():
    frozen = load_freeze()
    rows, pages = collect()
    blinded, mapping = prepare_items(rows, frozen["assessment"]["seed"])
    directory = OUTPUT / "assessment"
    contexts = {pid: page["bar"] for pid, page in pages.items()}
    manifest = {"evaluation_freeze_sha256": sha(FREEZE), "assessment": frozen["assessment"],
                "blinded_items_sha256": a.digest(blinded), "contexts_sha256": a.digest(contexts),
                "method_key_sha256": a.digest(mapping), "technical_outputs_sha256": a.digest(rows)}
    a.atomic_json(directory / "run_manifest.json", manifest)
    a.atomic_json(directory / "blinded_items.json", blinded)
    a.atomic_json(directory / "bavarian_contexts.json", contexts)
    a.atomic_json(directory / "private/method_key.json", mapping)
    print(json.dumps({"sources": len(blinded), "unique_judgments": sum(len(g["candidates"]) for g in blinded), "rows": len(mapping)}))


def score():
    load_freeze()
    directory = OUTPUT / "assessment"
    manifest = read(directory / "run_manifest.json")
    blinded, contexts = read(directory / "blinded_items.json"), read(directory / "bavarian_contexts.json")
    if (manifest["evaluation_freeze_sha256"] != sha(FREEZE) or a.digest(blinded) != manifest["blinded_items_sha256"]
            or a.digest(contexts) != manifest["contexts_sha256"]):
        raise ValueError("Assessment inputs changed")
    recorded = read(directory / "judgments.json")
    judgments = validate_judgments(recorded, {i["case_id"] for group in blinded for i in group["candidates"]})
    mapping = read(directory / "private/method_key.json")  # Unmask only after complete judgments.
    if a.digest(mapping) != manifest["method_key_sha256"]:
        raise ValueError("Method mapping changed")
    methods = summarize(mapping, judgments)
    per_article = {article: summarize([r for r in mapping if r["article"] == article], judgments)
                   for article in sorted({r["article"] for r in mapping})}
    baseline = {r["source_id"]: judgments[r["case_id"]]["score"] for r in mapping if r["method"] == "blind_translation"}
    paired = {}
    for method in methods:
        counts = Counter()
        for row in (r for r in mapping if r["method"] == method):
            value = judgments[row["case_id"]]["score"] if row["case_id"] else None
            base = baseline[row["source_id"]]
            counts["unscored_or_missing" if value is None or base is None else
                   "higher" if value > base else "lower" if value < base else "tied"] += 1
        paired[method] = dict(counts)
    summary = {"scope": "AI_assessed_fixed_evaluation_split_previously_exposed_in_older_pilot",
               "evaluation_freeze_sha256": sha(FREEZE), "judgments_sha256": a.digest(recorded),
               "methods": methods, "per_article": per_article, "paired_vs_blind": paired}
    a.atomic_json(directory / "summary.json", summary)
    a.atomic_json(directory / "per_output.json", [{**r, "judgment": judgments.get(r["case_id"])} for r in mapping])
    rows = read(OUTPUT / "quality/report.json")["outputs"]
    if a.digest(rows) != manifest["technical_outputs_sha256"]:
        raise ValueError("Technical output report changed after assessment preparation")
    outputs = {r["output_id"]: r for r in rows}
    details = io.StringIO(newline="")
    writer = csv.DictWriter(details, fieldnames=["method", "source_id", "article", "bar_text", "english", "technical_status", "score", "issues", "note"])
    writer.writeheader()
    for row in mapping:
        original, judgment = outputs[row["output_id"]], judgments.get(row["case_id"])
        writer.writerow({"method": row["method"], "source_id": row["source_id"], "article": row["article"],
            "bar_text": original["source"]["bar_text"], "english": original["target_text"],
            "technical_status": row["technical_status"], "score": judgment["score"] if judgment else None,
            "issues": ";".join(judgment["issues"]) if judgment else "", "note": judgment["note"] if judgment else "No English output"})
    detail_path = directory / "comparison.csv"
    if detail_path.exists() and detail_path.read_bytes() != details.getvalue().encode():
        raise ValueError("Existing pair-level results differ")
    if not detail_path.exists():
        detail_path.write_bytes(details.getvalue().encode())
    output = io.StringIO(newline="")
    fields = ["method", "sources", "produced", "missing", "uncertain", "mean_fidelity_0_4", "ai_retained_strict", "strict_yield", "mostly_or_fully_faithful"]
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    for method, stats in methods.items():
        writer.writerow({"method": method, **{k: stats[k] for k in fields if k != "method"}})
    path = directory / "results.csv"
    if path.exists() and path.read_bytes() != output.getvalue().encode():
        raise ValueError("Existing results differ; refuse to overwrite")
    if not path.exists():
        path.write_bytes(output.getvalue().encode())
    print(json.dumps(methods, indent=2))


def plan():
    frozen = load_freeze()
    for article in frozen["articles"]:
        print(f"{article['bar_title']}: {article['source_count']} Bavarian sources; {article['english_spans']} English spans")
    for method in ("blind", "b", "a", "c"):
        try:
            manifest, items, pages = build_run(method, frozen)
            saved = read_saved(OUTPUT / method, manifest, items, pages)
            print(f"{method}: {len(saved[STAGES[method][-1]])}/524 complete; run {a.digest(manifest)}")
        except FileNotFoundError as error:
            print(f"{method}: prerequisite missing: {error.filename}")
        except ValueError as error:
            if str(error).startswith("Complete blind translations first:"):
                print(f"{method}: waiting for blind translations")
            else:
                raise
    print(f"Freeze: {sha(FREEZE)}; offline plan only; no cloud/model calls")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    for name in ("freeze", "mine", "prepare-search", "prepare-judge", "score"):
        action.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    if args.freeze:
        a.atomic_json(FREEZE, frozen_protocol())
        plan()
    elif args.mine:
        mine()
    elif args.prepare_search:
        prepare_search()
    elif args.prepare_judge:
        prepare_judge()
    elif args.score:
        score()
    else:
        plan()
