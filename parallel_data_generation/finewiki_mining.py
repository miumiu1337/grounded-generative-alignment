"""The pilot's linked-article ratio-margin retrieval, shared by both encoders."""

import preprocessing.finewiki_pilot as data


def rank_group(sources, targets, source_vectors, target_vectors):
    import numpy as np

    def normalized(vectors, rows):
        values = np.asarray(vectors, dtype=np.float32)
        if values.shape != (rows, 1024) or not np.isfinite(values).all():
            raise ValueError("Mining embeddings must be finite 1024-dimensional vectors")
        norms = np.linalg.norm(values, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError("Zero mining embedding")
        return values / norms

    if min(len(sources), len(targets)) < 4:
        raise ValueError("This development protocol requires four neighbours on each side")
    similarity = normalized(source_vectors, len(sources)) @ normalized(target_vectors, len(targets)).T
    x_near = np.partition(similarity, -4, axis=1)[:, -4:].mean(axis=1)
    y_near = np.partition(similarity, -4, axis=0)[-4:, :].mean(axis=0)
    denominator = (x_near[:, None] + y_near[None, :]) / 2
    if (denominator <= 0).any():
        raise ValueError("Non-positive ratio-margin neighbourhood; preserve inputs for review")
    margins = similarity / denominator
    records = []
    for index, source in enumerate(sources):
        chosen = np.argsort(-margins[index], kind="stable")[:5]
        candidates = [{"rank": rank, **targets[int(j)], "margin_ratio": float(margins[index, j]),
                       "cosine": float(similarity[index, j])} for rank, j in enumerate(chosen, 1)]
        records.append({"source": source, "candidates": candidates, "candidate_count": len(targets),
                        "reciprocal_top1": int(np.argmax(margins[:, chosen[0]])) == index,
                        "quality_status": "unassessed", "acceptance_status": "unthresholded"})
    return records


def groups(bundle):
    if data.digest(bundle["sources"]) != bundle["sources_sha256"] or data.digest(bundle["pages"]) != bundle["pages_sha256"]:
        raise ValueError("Mining source cohort or candidate pool changed")
    return [(page_id, [s for s in bundle["sources"] if str(s["bar_page_id"]) == page_id], page["candidates"])
            for page_id, page in sorted(bundle["pages"].items())]
