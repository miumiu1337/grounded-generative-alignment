"""Training rows and complete-cohort coverage for the seven pilot conditions."""

import preprocessing.finewiki_pilot as data


def training_rows(bundle, condition, outputs):
    sources = {s["source_id"]: s for s in bundle["sources"]}
    if condition not in data.CONDITIONS or set(outputs) - set(sources):
        raise ValueError("Unknown export condition or source outside the common cohort")
    rows, coverage = [], []
    for sid, source in sources.items():
        result = outputs.get(sid)
        if result is None:
            coverage.append({"source_id": sid, "status": "missing"})
            continue
        target = result["target_text"]
        if target is None:
            unmatched = result.get("evidence_ids") == []
            rejected = result.get("match_status") == "none" or result.get("contradiction") is True
            if condition != "rerank_only" or not (unmatched or rejected):
                raise ValueError("Only an explicitly unmatched or rejected reranker may export no target")
            coverage.append({"source_id": sid, "status": "unmatched" if unmatched else "rejected",
                             "match_status": result.get("match_status"),
                             "contradiction": result.get("contradiction"),
                             "evidence_ids": result.get("evidence_ids", [])})
            continue
        if condition == "rerank_only" and (result.get("match_status") == "none" or result.get("contradiction") is True):
            raise ValueError("Rejected reranking evidence must not be exported as a training pair")
        provenance = bundle["provenance"][sid]
        row = data.training_row(source["bar_text"], target, condition, provenance)
        row.pop("provenance")
        row.update(provenance)
        row.update(source_id=sid, method=condition, quality_status="unassessed", cohort_sha256=bundle["sources_sha256"],
                   in_common_cohort=result.get("in_common_cohort"),
                   source_revision_url=f"https://bar.wikipedia.org/w/index.php?oldid={provenance['source_revision']}",
                   target_revision_url=f"https://en.wikipedia.org/w/index.php?oldid={provenance['target_revision']}",
                   evidence_ids=result.get("evidence_ids", []), match_status=result.get("match_status"),
                   contradiction=result.get("contradiction"), model=result["model"],
                   model_revision=result.get("model_revision"),
                   output_sha256=result["output_sha256"], mining_margin_ratio=result.get("mining_margin_ratio"),
                   mining_cosine=result.get("mining_cosine"), external_reference_urls=result.get("external_reference_urls", []),
                   available_external_reference_urls=result.get("available_external_reference_urls", []),
                   external_search_query=result.get("external_search_query"), external_search_ref=result.get("external_search_ref"),
                   external_search_status=result.get("external_search_status"),
                   self_verification_sha256=result.get("self_verification_sha256"),
                   self_verification_verdict=result.get("self_verification_verdict"))
        if condition in {"laser", "bge_m3", "rerank_only"}:
            candidates = bundle["pages"][str(source["bar_page_id"])]["candidates"]
            if not any(c["sentence_ids"] == row["evidence_ids"] and c["text"] == target for c in candidates):
                raise ValueError("Extractive export differs from its original English candidate span")
        rows.append(row)
        coverage.append({"source_id": sid, "status": "complete"})
    return rows, {"condition": condition, "input_sources": len(sources), "training_rows": len(rows),
                  "cohort_sha256": bundle["sources_sha256"], "coverage": coverage,
                  "quality_status": "unassessed_candidates;not_gold_parallel_pairs"}
