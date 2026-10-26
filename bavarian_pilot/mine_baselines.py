"""LASER-1/BGE-M3 development mining; scoring/cache pattern adapted from mine_laser.py.

Default is an offline plan. --execute runs locally; --download-models fetches
only the pinned LASER assets. Neither option selects evaluation articles.
"""

import argparse
import csv
import fcntl
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import requests

from download_articles import sha256
from translate_blind import digest_json, load_sources, local_path


ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "experiments/mining_development_v1.json"
MODEL_FILES = ("93langs.fcodes", "93langs.fvocab", "bilstm.93langs.2018-12-26.pt")
LASER_URL = "https://dl.fbaipublicfiles.com/laser/models/"


def atomic_json(path, value):
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise ValueError(f"Saved artifact differs: {path}; choose a new run name")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def load_inputs(config):
    if (config["split"] != "development" or config["source_unit"] != "single_sentence"
            or config["neighbors"] != 4 or config["margin_method"] != "ratio"
            or config["acceptance_threshold"] is not None):
        raise ValueError("This runner supports only the unthresholded development protocol")
    sources = load_sources(config)  # Validates the entire frozen preprocessing manifest.
    groups = []
    for page_id in dict.fromkeys(source["bar_page_id"] for source in sources):
        article = json.loads((ROOT / f"data/prepared/bar_{page_id}.json").read_text(encoding="utf-8"))
        if article["split"] != "development":
            raise ValueError("Evaluation article reached the mining runner")
        group_sources = [source for source in sources if source["bar_page_id"] == page_id]
        targets = article["en"]["candidate_spans"]
        ids = [tuple(target["sentence_ids"]) for target in targets]
        if len(set(ids)) != len(ids) or any(not 1 <= len(ids_) <= 2 for ids_ in ids):
            raise ValueError("Invalid or repeated English candidate span IDs")
        groups.append({"bar_page_id": page_id, "bar_title": article["bar"]["title"],
                       "en_page_id": article["en"]["page_id"], "en_title": article["en"]["title"],
                       "en_revision_id": article["en"]["revision_id"],
                       "sources": group_sources, "targets": targets})
    if len(groups) != 2 or sum(len(g["sources"]) for g in groups) != 53:
        raise ValueError("Expected exactly the two development articles and 53 source sentences")
    return groups


def laser_assets(config):
    directory = local_path(config["laser"]["model_directory"])
    hashes = {name: sha256(directory / name) for name in MODEL_FILES}
    identity = hashlib.sha256((config["laser"]["package_version"] + "|" + "|".join(
        hashes[name] for name in MODEL_FILES) + "|bar=de|en=en").encode()).hexdigest()
    if identity != config["laser"]["model_hash"]:
        raise ValueError("LASER assets differ from the original pinned checkpoint/tokenizers")
    return {name: {"sha256": value, "url": LASER_URL + name} for name, value in hashes.items()}


def download_models(config):
    directory = local_path(config["laser"]["model_directory"])
    directory.mkdir(parents=True, exist_ok=True)
    for name in MODEL_FILES:
        path = directory / name
        if path.exists():
            continue
        temporary = path.with_suffix(path.suffix + ".part")
        started = time.monotonic()
        try:
            with requests.get(LASER_URL + name, stream=True, timeout=(10, 60)) as response:
                response.raise_for_status()
                size = 0
                with temporary.open("wb") as output:
                    for block in response.iter_content(1024 * 1024):
                        size += len(block)
                        if size > 300 * 1024 * 1024 or time.monotonic() - started > 300:
                            raise RuntimeError(f"LASER asset download exceeded size/time limit: {name}")
                        output.write(block)
                    output.flush()
                    os.fsync(output.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Downloaded {name}", flush=True)
    print(json.dumps(laser_assets(config), indent=2))


def ollama_json(config, route, body=None):
    url = config["bge"]["url"] + route
    try:
        response = requests.get(url, timeout=(5, 10)) if body is None else requests.post(
            url, json=body, timeout=(5, 120))
        response.raise_for_status()
        return response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"Local Ollama {route} failed ({type(error).__name__}); completed batches are saved") from error


def check_bge(config):
    matches = [model for model in ollama_json(config, "/api/tags")["models"]
               if model["name"] == config["bge"]["model"]]
    if (len(matches) != 1 or matches[0]["digest"] != config["bge"]["model_digest"]
            or matches[0]["details"]["quantization_level"] != config["bge"]["precision"]):
        raise ValueError("The local BGE-M3 checkpoint or precision differs from the protocol")


def normalize(vectors, rows=None):
    vectors = np.asarray(vectors, dtype=np.float32)
    if (vectors.ndim != 2 or vectors.shape[1] != 1024 or rows is not None and len(vectors) != rows
            or not np.isfinite(vectors).all()):
        raise ValueError("Malformed/non-finite embedding matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ValueError("Zero embedding norm")
    return vectors / norms


def rank(source, target, neighbors=4):
    if min(len(source), len(target)) < neighbors:
        raise ValueError("Too few items for the margin neighborhood")
    cosine = normalize(source) @ normalize(target).T
    source_near = np.partition(cosine, -neighbors, axis=1)[:, -neighbors:].mean(axis=1)
    target_near = np.partition(cosine, -neighbors, axis=0)[-neighbors:, :].mean(axis=0)
    neighborhood = (source_near[:, None] + target_near[None, :]) / 2
    if np.any(neighborhood <= 0):
        raise ValueError("Non-positive margin denominator")
    return cosine / neighborhood, cosine


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as output:
        np.savez_compressed(output, **arrays)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def cached_embeddings(path, texts, run_id, batch_size, embed):
    input_hash = digest_json(texts)
    vectors = np.empty((0, 1024), dtype=np.float32)
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["run_id"]) != run_id or str(cached["input_hash"]) != input_hash:
                raise ValueError(f"Changed embedding cache inputs/protocol: {path}")
            vectors = cached["vectors"]
        normalize(vectors)
        if len(vectors) > len(texts) or len(vectors) < len(texts) and len(vectors) % batch_size:
            raise ValueError("Incomplete/misaligned cached batch")
    for start in range(len(vectors), len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = normalize(embed(batch), len(batch))
        vectors = np.concatenate((vectors, encoded))
        save_npz(path, vectors=vectors, run_id=run_id, input_hash=input_hash)
        print(f"Saved {path.parent.name}/{path.stem}: {len(vectors)}/{len(texts)}", flush=True)
    return vectors


def ranked_records(group, source, target, config):
    margin, cosine = rank(source, target, config["neighbors"])
    records = []
    for i, sentence in enumerate(group["sources"]):
        indices = np.argsort(-margin[i], kind="stable")[:config["top_k"]]
        best = int(indices[0])
        candidates = [{"rank": position, **group["targets"][int(j)],
                       "margin_ratio": float(margin[i, j]), "cosine": float(cosine[i, j])}
                      for position, j in enumerate(indices, 1)]
        records.append({"source": sentence, "en_page_id": group["en_page_id"],
                        "en_revision_id": group["en_revision_id"], "candidate_count": len(target),
                        "candidates": candidates,
                        "best_cosine_sentence_ids": group["targets"][int(np.argmax(cosine[i]))]["sentence_ids"],
                        "reciprocal_top1": int(np.argmax(margin[:, best])) == i,
                        "quality_status": "unassessed", "acceptance_status": "unthresholded"})
    return records, margin, cosine


def compare(results, config):
    first, second = results["laser_1"], results["bge_m3"]
    if [r["source"] for r in first] != [r["source"] for r in second]:
        raise ValueError("Model outputs do not share the same ordered source units")
    comparisons = []
    for a, b in zip(first, second):
        aset = {tuple(c["sentence_ids"]) for c in a["candidates"]}
        bset = {tuple(c["sentence_ids"]) for c in b["candidates"]}
        at, bt = a["candidates"][0], b["candidates"][0]
        comparisons.append({"source_id": a["source"]["source_id"], "bar_text": a["source"]["bar_text"],
                            "bar_title": a["source"]["bar_title"],
                            "exact_top1_agreement": at["sentence_ids"] == bt["sentence_ids"],
                            "top1_sentence_overlap": bool(set(at["sentence_ids"]) & set(bt["sentence_ids"])),
                            "top5_common_spans": len(aset & bset),
                            "top5_jaccard": len(aset & bset) / len(aset | bset),
                            "laser_1_top1": at, "bge_m3_top1": bt})
    statistics = {}
    for name, records in results.items():
        margins = [r["candidates"][0]["margin_ratio"] for r in records]
        statistics[name] = {
            "sources": len(records), "reciprocal_top1": sum(r["reciprocal_top1"] for r in records),
            "two_sentence_top1": sum(len(r["candidates"][0]["sentence_ids"]) == 2 for r in records),
            "top1_margin_quantiles": dict(zip(("min", "q25", "median", "q75", "max"),
                map(float, np.quantile(margins, [0, .25, .5, .75, 1])))),
            "threshold_sensitivity_top1_counts": {str(t): sum(m >= t for m in margins)
                                                   for t in config["sensitivity_thresholds"]}}
    summary = {"source_count": len(comparisons),
               "exact_top1_agreement": sum(r["exact_top1_agreement"] for r in comparisons),
               "top1_sentence_overlap": sum(r["top1_sentence_overlap"] for r in comparisons),
               "mean_top5_jaccard": float(np.mean([r["top5_jaccard"] for r in comparisons])),
               "models": statistics,
               "interpretation": "Agreement and score diagnostics only; no gold alignments, precision, recall or quality winner. Threshold counts are sensitivity analysis, not accepted pairs."}
    return {"summary": summary, "per_source": comparisons}


def export_comparison(directory, comparison):
    atomic_json(directory / "comparison.json", comparison)
    fields = ["source_id", "bar_title", "bar_text", "exact_top1_agreement", "top1_sentence_overlap"]
    for model in ("laser_1", "bge_m3"):
        fields += [model + suffix for suffix in ("_sentence_ids", "_text", "_margin", "_cosine")]
    temporary = directory / "comparison.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row in comparison["per_source"]:
            record = {key: row[key] for key in fields if key in row}
            for model in ("laser_1", "bge_m3"):
                candidate = row[model + "_top1"]
                record.update({model + "_sentence_ids": ";".join(candidate["sentence_ids"]),
                               model + "_text": candidate["text"], model + "_margin": candidate["margin_ratio"],
                               model + "_cosine": candidate["cosine"]})
            writer.writerow(record)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(directory / "comparison.csv")
    s = comparison["summary"]
    lines = ["# LASER-1 and BGE-M3 development comparison", "",
             f"Sources: {s['source_count']}. Exact top-1 agreement: {s['exact_top1_agreement']}/{s['source_count']}.",
             f"Top-1 spans sharing an English sentence: {s['top1_sentence_overlap']}/{s['source_count']}.",
             f"Mean top-5 span Jaccard: {s['mean_top5_jaccard']:.3f}.", "",
             "| Model | Sources | Reciprocal top-1 | Two-sentence top-1 | Median ratio margin |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, m in s["models"].items():
        lines.append(f"| {name} | {m['sources']} | {m['reciprocal_top1']} | {m['two_sentence_top1']} | {m['top1_margin_quantiles']['median']:.4f} |")
    lines += ["", s["interpretation"], "",
              "Higher scores across different encoders do not establish better alignment quality.",
              "Overlapping one/two-sentence candidates can fill multiple neighborhood positions.",
              "No deduplication or one-to-one assignment is applied to either model.", "",
              "See comparison.csv for all source texts and both top choices; per-model rankings.json retain five candidates per source.", ""]
    (directory / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def execute(config, groups, directory):
    assets = laser_assets(config)
    if version("laserembeddings") != config["laser"]["package_version"]:
        raise ValueError("Unexpected LASER package version")
    check_bge(config)
    runtime = {name: version(name) for name in ("laserembeddings", "torch", "numpy", "sacremoses", "subword-nmt", "requests")}
    runtime.update(python=platform.python_version(), ollama=ollama_json(config, "/api/version")["version"])
    manifest = {"experiment": config, "experiment_sha256": sha256(EXPERIMENT),
                "runner_sha256": sha256(Path(__file__)), "source_loader_sha256": sha256(ROOT / "translate_blind.py"),
                "requirements_sha256": sha256(ROOT / "requirements-mining.txt"),
                "inputs_sha256": digest_json(groups), "laser_assets": assets, "runtime": runtime,
                "adapted_from": "../grounded-generative-alignment/mine_laser.py; batch cache and ratio-margin scoring"}
    run_id = digest_json(manifest)
    atomic_json(directory / "run_manifest.json", manifest)
    atomic_json(directory / "inputs.json", groups)
    from laserembeddings import Laser
    import torch

    torch.set_num_threads(config["laser"]["cpu_threads"])
    torch.manual_seed(42)
    paths = [str(local_path(config["laser"]["model_directory"]) / name) for name in MODEL_FILES]
    laser = Laser(*paths, embedding_options={"cpu": True, "max_sentences": config["batch_size"]})
    results = {}
    for name in ("laser_1", "bge_m3"):
        records = []
        started = time.monotonic()
        for group in groups:
            vectors = []
            for language in ("bar", "en"):
                texts = ([source["bar_text"] for source in group["sources"]] if language == "bar"
                         else [target["text"] for target in group["targets"]])
                def embed(batch):
                    if name == "laser_1":
                        return laser.embed_sentences(batch, lang="de" if language == "bar" else "en")
                    check_bge(config)
                    response = ollama_json(config, "/api/embed", {
                        "model": config["bge"]["model"], "input": batch,
                        "truncate": False, "options": {"num_ctx": config["bge"]["num_ctx"]}, "keep_alive": "5m"})
                    if response.get("model") != config["bge"]["model"]:
                        raise ValueError("Unexpected model in Ollama embedding response")
                    check_bge(config)
                    return response["embeddings"]
                path = directory / name / f"bar_{group['bar_page_id']}_{language}.npz"
                vectors.append(cached_embeddings(path, texts, run_id, config["batch_size"], embed))
            rows, margin, cosine = ranked_records(group, *vectors, config)
            records.extend(rows)
            save_npz(directory / name / f"bar_{group['bar_page_id']}_scores.npz", margin=margin, cosine=cosine, run_id=run_id)
        atomic_json(directory / name / "rankings.json", {"run_id": run_id, "records": records})
        log = {"at": datetime.now(timezone.utc).isoformat(), "model": name,
               "elapsed_seconds_including_cache_reads": round(time.monotonic() - started, 3), "records": len(records)}
        with (directory / "run_log.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(log) + "\n")
        print(json.dumps(log), flush=True)
        results[name] = records
    comparison = compare(results, config)
    export_comparison(directory, comparison)
    print(json.dumps(comparison["summary"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--download-models", action="store_true")
    args = parser.parse_args()
    config = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    groups = load_inputs(config)
    for group in groups:
        print(f"{group['bar_title']}: {len(group['sources'])} original Bavarian sentences; {len(group['targets'])} English spans", flush=True)
    if not args.execute and not args.download_models:
        print("Offline plan only; no model inference or network requests.")
        return
    directory = local_path("data/mining/" + config["run_name"])
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another local mining run is active") from None
        if args.download_models:
            download_models(config)
        else:
            execute(config, groups, directory)


if __name__ == "__main__":
    main()
