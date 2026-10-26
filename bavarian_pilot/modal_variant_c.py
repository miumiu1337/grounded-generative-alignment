"""Gemma 4 Variant C on Modal: cached web evidence -> refine -> verify.

python modal_variant_c.py                         Offline plan
python modal_variant_c.py --execute               Same Modal job via Python SDK
modal run modal_variant_c.py --execute            Two representative sources
modal run modal_variant_c.py --execute --max-new 53
modal run modal_variant_c.py --sync               Recover saved stages; no GPU
"""

import fcntl
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal

import variant_c as protocol
from modal_translate_blind import cache_volume, gpu_image, results_volume


app = modal.App("gga-clean-gemma4-grounded-c-dev")
image = gpu_image.add_local_python_source("variant_c", "variant_a", "modal_translate_blind")
REMOTE = Path("/results")


@app.function(image=image, volumes={"/results": results_volume}, timeout=180, retries=0)
def saved_results(manifest, items, pages):
    results_volume.reload()
    return protocol.read_saved(REMOTE / manifest["experiment"]["run_name"], manifest, items, pages)


@app.function(image=image, volumes={"/cache": cache_volume}, timeout=240, retries=0)
def preflight(manifest, items, pages):
    from transformers import AutoConfig, AutoTokenizer
    import xgrammar

    protocol.validate_inputs(manifest, items, pages)
    if version("transformers") != manifest["experiment"]["inference"]["transformers"]:
        raise RuntimeError("The container Transformers version differs from the protocol")
    config = AutoConfig.from_pretrained(protocol.MODEL, revision=protocol.REVISION)
    if config.model_type != "gemma4":
        raise RuntimeError("The checkpoint is not Gemma 4")
    tokenizer = AutoTokenizer.from_pretrained(protocol.MODEL, revision=protocol.REVISION)
    maximum = manifest["experiment"]["inference"]["max_model_len"]
    reserve = manifest["experiment"]["generation"]["max_tokens"]
    counts = {}
    for item in items:
        source_id = item["source"]["source_id"]
        page = pages[str(item["source"]["bar_page_id"])]
        for stage in ("refinement", "verification"):
            xgrammar.Grammar.from_json_schema(protocol.schema_for(stage, item, page))
        text = tokenizer.apply_chat_template(protocol.stage_messages("refinement", manifest, item, page),
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        count = len(tokenizer.encode(text, add_special_tokens=False))
        if count + reserve > maximum:
            raise RuntimeError(f"Full-page refinement exceeds context for {source_id}: {count}+{reserve}>{maximum}; no truncation allowed")
        counts[source_id] = count
    cache_volume.commit()
    return {"sources_checked": len(counts), "min_refinement_prompt_tokens": min(counts.values()),
            "max_refinement_prompt_tokens": max(counts.values()), "max_model_len": maximum,
            "schemas_checked": len(counts) * 2}


def work_order(items):
    # Smoke-test both page lengths before continuing with the remaining sources.
    first, remaining, seen = [], [], set()
    for item in items:
        page_id = item["source"]["bar_page_id"]
        (remaining if page_id in seen else first).append(item)
        seen.add(page_id)
    return first + remaining


def generate_stage(llm, tokenizer, manifest, item, page, stage, runtime, refinement=None):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    config = manifest["experiment"]
    messages = protocol.stage_messages(stage, manifest, item, page, refinement)
    schema = protocol.schema_for(stage, item, page)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    if prompt_tokens + config["generation"]["max_tokens"] > config["inference"]["max_model_len"]:
        raise RuntimeError(f"{stage} context limit exceeded; no input will be truncated")
    grammar = None
    structured = StructuredOutputsParams(json=schema)
    sampling = SamplingParams(temperature=config["generation"]["temperature"], seed=config["generation"]["seed"],
                              max_tokens=config["generation"]["max_tokens"], structured_outputs=structured)
    started = time.monotonic()
    outputs = llm.generate([text], sampling, use_tqdm=False)
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise RuntimeError(f"No single response for {item['source']['source_id']} {stage}")
    response, completion = outputs[0], outputs[0].outputs[0]
    raw = completion.text.strip()
    record = {"run_id": protocol.digest(manifest), "source_id": item["source"]["source_id"], "stage": stage,
              "messages": messages, "schema": schema, "grammar": grammar, "raw_response": raw,
              "finish_reason": completion.finish_reason, "prompt_tokens": len(response.prompt_token_ids),
              "output_tokens": len(completion.token_ids), "runtime": runtime,
              "duration_seconds": round(time.monotonic() - started, 3),
              "recorded_at": datetime.now(timezone.utc).isoformat()}
    return record


@app.function(image=image, gpu="H200", timeout=3600, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/results": results_volume, "/cache": cache_volume})
def run_variant(manifest, items, pages, max_new):
    protocol.validate_inputs(manifest, items, pages)
    if type(max_new) is not int or not 1 <= max_new <= 53:
        raise ValueError("--max-new must be between 1 and 53")
    results_volume.reload()
    directory = REMOTE / manifest["experiment"]["run_name"]
    saved = protocol.read_saved(directory, manifest, items, pages)
    pending = [item for item in work_order(items) if item["source"]["source_id"] not in saved["results"]][:max_new]
    if not pending:
        return saved
    protocol.atomic_json(directory / "run_manifest.json", manifest)
    protocol.atomic_json(directory / "inputs.json", {"items": items, "pages": pages})
    results_volume.commit()
    from vllm import LLM

    config = manifest["experiment"]
    llm = LLM(model=protocol.MODEL, revision=protocol.REVISION, dtype=config["inference"]["dtype"],
              max_model_len=config["inference"]["max_model_len"],
              gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
              enforce_eager=True, limit_mm_per_prompt={"image": 0})
    cache_volume.commit()
    tokenizer = llm.get_tokenizer()
    runtime = {name: version(name) for name in ("vllm", "torch", "transformers")}
    runtime["python"] = platform.python_version()
    for item in pending:
        source_id = item["source"]["source_id"]
        page = pages[str(item["source"]["bar_page_id"])]
        for stage in ("refinement", "verification"):
            if stage == "refinement" and source_id in saved["refinements"]:
                continue
            refinement = saved["refinements"].get(source_id, {}).get("result")
            record = generate_stage(llm, tokenizer, manifest, item, page, stage, runtime, refinement)
            try:
                record["result"] = protocol.strict_json(record["raw_response"])
                protocol.validate_stage(record, stage, manifest, item, page, refinement)
            except (ValueError, TypeError) as error:
                protocol.atomic_json(protocol.record_path(directory / "failures" / stage, source_id),
                                     {"response": record, "validation_error": str(error)})
                results_volume.commit()
                raise RuntimeError(f"Invalid {stage} for {source_id}; raw response and completed stages are committed: {error}") from error
            if stage == "refinement":
                protocol.atomic_json(protocol.record_path(directory / "refinements", source_id), record)
                saved["refinements"][source_id] = record
            else:
                refined = saved["refinements"][source_id]
                result = {"condition": "grounded_c", "run_id": protocol.digest(manifest),
                          "source": item["source"], "blind": item["blind"], "search": item["search"],
                          "refinement": refined["result"], "refinement_sha256": protocol.digest(refined),
                          "edits": protocol.edits_for(item["blind"]["translation"], refined["result"]["translation"]),
                          "verification": record, "quality_status": "ai_self_checked_not_gold"}
                protocol.atomic_json(protocol.record_path(directory / "results", source_id), result)
                saved["results"][source_id] = result
            results_volume.commit()
            print(f"Committed {source_id} {stage}; completed {len(saved['results'])}/53 sources", flush=True)
    return saved


@app.local_entrypoint()
def main(execute: bool = False, sync: bool = False, max_new: int = 2):
    if execute and sync or not 1 <= max_new <= 53:
        raise ValueError("Choose --execute or --sync; --max-new must be between 1 and 53")
    manifest, items, pages = protocol.local_run()
    directory = protocol.ROOT / "data/translations" / manifest["experiment"]["run_name"]
    local = protocol.read_saved(directory, manifest, items, pages)
    print(f"Variant C: {len(local['results'])}/53 complete; {len(local['refinements'])}/53 refinements; max new sources: {max_new}")
    print(f"Run fingerprint: {protocol.digest(manifest)}", flush=True)
    if not execute and not sync:
        print("Offline plan only. Search evidence and blind drafts are cached; no inference call.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local Variant C process is running") from None
        saved = saved_results.remote(manifest, items, pages)
        protocol.write_local(saved, manifest, items, pages)
        if execute and len(saved["results"]) < len(items):
            print(f"CPU context preflight: {preflight.remote(manifest, items, pages)}", flush=True)
            try:
                saved = run_variant.remote(manifest, items, pages, max_new)
            except Exception:
                # Recover validated completed stages; the original generation failure still propagates.
                recovered = saved_results.remote(manifest, items, pages)
                protocol.write_local(recovered, manifest, items, pages)
                protocol.export_results(directory, recovered, manifest)
                raise
            protocol.write_local(saved, manifest, items, pages)
        print(protocol.export_results(directory, saved, manifest), flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--sync", action="store_true")
    parser.add_argument("--max-new", type=int, default=2)
    parser.add_argument("--detach", action="store_true", help="Continue the remote worker if this client disconnects")
    args = parser.parse_args()
    if not 1 <= args.max_new <= 53:
        parser.error("--max-new must be between 1 and 53")
    if args.detach and not args.execute:
        parser.error("--detach requires --execute")
    if args.execute or args.sync:
        with modal.enable_output(), app.run(detach=args.detach):
            main(execute=args.execute, sync=args.sync, max_new=args.max_new)
    else:
        main(max_new=args.max_new)
