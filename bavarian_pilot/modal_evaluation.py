"""Run the frozen eight-article Gemma experiment; default is an offline plan.

python modal_evaluation.py --method blind
python modal_evaluation.py --method blind --execute --max-new 2 --detach
python modal_evaluation.py --method blind --sync

Use --method a, b or c for the other conditions. A/C require all blind drafts;
C also requires the completed and frozen web-search cache. Reranking reuses B.
"""

import argparse
import fcntl
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import modal

import evaluation_pipeline as protocol
from modal_translate_blind import cache_volume, gpu_image, results_volume

app = modal.App("gga-clean-gemma4-frozen-eight-v2")
LOCAL_MODULES = ("evaluation_pipeline", "variant_a", "variant_b", "variant_c", "semantic_evaluation", "modal_translate_blind")
image = gpu_image.add_local_python_source(*LOCAL_MODULES)
REMOTE = Path("/results")


def work_order(items, pages):
    # Exercise the largest page first; then one source from every other page.
    first, remaining, seen = [], [], set()
    for item in items:
        pid = str(item["source"]["bar_page_id"])
        (remaining if pid in seen else first).append(item)
        seen.add(pid)
    first.sort(key=lambda i: -len(pages.get(str(i["source"]["bar_page_id"]), {}).get("en", "")))
    return first + remaining


def stage_prompt(tokenizer, method, stage, manifest, item, page, previous=None):
    messages = protocol.messages(method, stage, manifest, item, page, previous)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    count = len(tokenizer.encode(text, add_special_tokens=False))
    config = manifest["experiment"]
    if count + config["generation"]["max_tokens"] > config["inference"]["max_model_len"]:
        raise RuntimeError(f"{item['source']['source_id']} {stage}: {count} prompt tokens exceed the frozen context budget; no truncation allowed")
    return messages, text, count


@app.function(image=image, volumes={"/results": results_volume}, timeout=300, retries=0)
def saved_results(manifest, items, pages):
    results_volume.reload()
    return protocol.read_saved(REMOTE / manifest["experiment"]["run_name"], manifest, items, pages)


@app.function(image=image, volumes={"/cache": cache_volume}, timeout=900, retries=0)
def preflight(manifest, items, pages):
    from transformers import AutoConfig, AutoTokenizer
    import xgrammar

    protocol.validate_bundle(manifest, items, pages)
    config = manifest["experiment"]
    if version("transformers") != config["inference"]["transformers"]:
        raise RuntimeError("Transformers differs from the frozen protocol")
    if AutoConfig.from_pretrained(config["model"], revision=config["model_revision"]).model_type != "gemma4":
        raise RuntimeError("The pinned checkpoint is not Gemma 4")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["model_revision"])
    method = manifest["method"]
    first_stage = protocol.STAGES[method][0]
    counts, grammars = [], set()
    for item in items:
        page_id = str(item["source"]["bar_page_id"])
        page = pages.get(page_id)
        _, _, count = stage_prompt(tokenizer, method, first_stage, manifest, item, page)
        counts.append(count)
        if first_stage == "selection" and page_id not in grammars:
            xgrammar.Grammar.from_ebnf(protocol.a.selection_grammar(item, page))
            grammars.add(page_id)
    cache_volume.commit()
    return {"sources_checked": len(counts), "minimum_prompt_tokens": min(counts), "maximum_prompt_tokens": max(counts),
            "context_limit": config["inference"]["max_model_len"], "span_grammars_checked": len(grammars)}


def generate_stage(llm, tokenizer, manifest, item, page, stage, runtime, previous=None):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    method, config = manifest["method"], manifest["experiment"]
    messages, text, _ = stage_prompt(tokenizer, method, stage, manifest, item, page, previous)
    schema, grammar = protocol.constraints(method, stage, item, page)
    settings = {"temperature": config["generation"]["temperature"], "seed": config["generation"]["seed"],
                "max_tokens": config["generation"]["max_tokens"]}
    if grammar is not None:
        settings["structured_outputs"] = StructuredOutputsParams(grammar=grammar)
    elif schema is not None:
        settings["structured_outputs"] = StructuredOutputsParams(json=schema)
    started = time.monotonic()
    response = llm.generate([text], SamplingParams(**settings), use_tqdm=False)
    if len(response) != 1 or len(response[0].outputs) != 1:
        raise RuntimeError(f"No single response for {item['source']['source_id']} {stage}")
    result, completion = response[0], response[0].outputs[0]
    return {"run_id": protocol.a.digest(manifest), "source_id": item["source"]["source_id"], "stage": stage,
        "messages": messages, "schema": schema, "grammar": grammar, "raw_response": completion.text.strip(),
        "finish_reason": completion.finish_reason, "prompt_tokens": len(result.prompt_token_ids),
        "output_tokens": len(completion.token_ids), "runtime": runtime,
        "duration_seconds": round(time.monotonic() - started, 3), "recorded_at": datetime.now(timezone.utc).isoformat()}


@app.function(image=image, gpu="H200", timeout=3600, retries=0,
              max_containers=1, single_use_containers=True,
              volumes={"/results": results_volume, "/cache": cache_volume})
def run(manifest, items, pages, max_new):
    protocol.validate_bundle(manifest, items, pages)
    if type(max_new) is not int or not 1 <= max_new <= len(items):
        raise ValueError("--max-new must be between 1 and 524")
    results_volume.reload()
    method, config = manifest["method"], manifest["experiment"]
    directory = REMOTE / config["run_name"]
    saved = protocol.read_saved(directory, manifest, items, pages)
    stages = protocol.STAGES[method]
    pending = [i for i in work_order(items, pages) if i["source"]["source_id"] not in saved[stages[-1]]][:max_new]
    if not pending:
        return saved
    protocol.a.atomic_json(directory / "run_manifest.json", manifest)
    results_volume.commit()
    from vllm import LLM

    llm = LLM(model=config["model"], revision=config["model_revision"], dtype=config["inference"]["dtype"],
              max_model_len=config["inference"]["max_model_len"], gpu_memory_utilization=config["inference"]["gpu_memory_utilization"],
              enforce_eager=True, limit_mm_per_prompt={"image": 0})
    cache_volume.commit()
    tokenizer = llm.get_tokenizer()
    runtime = {name: version(name) for name in ("vllm", "torch", "transformers")}
    runtime["python"] = platform.python_version()
    for item in pending:
        sid = item["source"]["source_id"]
        page = pages.get(str(item["source"]["bar_page_id"]))
        for stage in stages:
            if sid in saved[stage]:
                continue
            previous = saved[stages[0]].get(sid)
            record = generate_stage(llm, tokenizer, manifest, item, page, stage, runtime,
                                    previous["result"] if previous else None)
            try:
                record["result"] = ({"translation": record["raw_response"]} if method == "blind" else
                                    protocol.a.strict_json(record["raw_response"]))
                protocol.validate_stage(record, method, stage, manifest, item, page,
                                        previous["result"] if previous else None)
                if previous and protocol.c.timestamp(record["recorded_at"]) < protocol.c.timestamp(previous["recorded_at"]):
                    raise ValueError("Second stage predates the first")
            except (ValueError, TypeError, KeyError) as error:
                failure = directory / "failures" / stage / f"{sid.replace(':', '_')}_{time.time_ns()}.json"
                protocol.a.atomic_json(failure, {"response": record, "validation_error": str(error)})
                results_volume.commit()
                raise RuntimeError(f"Invalid {method}/{stage} for {sid}; failure and prior stages committed: {error}") from error
            protocol.a.atomic_json(protocol.path_for(directory / stage, sid), record)
            results_volume.commit()
            saved[stage][sid] = record
            print(f"Committed {method}/{stage} {sid}; {len(saved[stages[-1]])}/524 complete", flush=True)
    return saved


def main(method, execute=False, sync=False, max_new=2):
    if execute and sync or not 1 <= max_new <= 524:
        raise ValueError("Choose execute or sync; max-new must be between 1 and 524")
    manifest, items, pages = protocol.build_run(method)
    directory = protocol.OUTPUT / method
    saved = protocol.read_saved(directory, manifest, items, pages)
    print(f"{method}: {len(saved[protocol.STAGES[method][-1]])}/524 complete; run {protocol.a.digest(manifest)}", flush=True)
    if not execute and not sync:
        print("Offline plan; no cloud calls")
        return
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = saved_results.remote(manifest, items, pages)
        protocol.write_local(directory, saved, manifest, items, pages)
        if execute and len(saved[protocol.STAGES[method][-1]]) < len(items):
            checked = preflight.remote(manifest, items, pages)
            protocol.a.atomic_json(directory / "context_preflight.json", checked)
            print(f"CPU preflight: {checked}", flush=True)
            try:
                saved = run.remote(manifest, items, pages, max_new)
            except Exception as error:
                # Recovery cannot replace or hide the original generation error.
                try:
                    recovered = saved_results.remote(manifest, items, pages)
                    protocol.write_local(directory, recovered, manifest, items, pages)
                except Exception as recovery_error:
                    error.add_note(f"Recovery also failed: {recovery_error}; use --sync")
                raise
            protocol.write_local(directory, saved, manifest, items, pages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=protocol.STAGES, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--sync", action="store_true")
    parser.add_argument("--max-new", type=int, default=2)
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach and not args.execute:
        parser.error("--detach requires --execute")
    if not 1 <= args.max_new <= 524:
        parser.error("--max-new must be between 1 and 524")
    if args.execute or args.sync:
        with modal.enable_output(), app.run(detach=args.detach):
            main(args.method, args.execute, args.sync, args.max_new)
    else:
        main(args.method, max_new=args.max_new)
