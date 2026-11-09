"""Article-local ratio margins with bounded matrices and explicit short-page behavior."""


def rank_group(sources, targets, source_vectors, target_vectors, block_rows=128):
    import numpy as np

    if type(block_rows) is not int or block_rows < 1:
        raise ValueError('A positive matrix block size is required')

    def normalized(values, size):
        vectors = np.asarray(values, dtype=np.float32)
        if vectors.shape != (size, 1024) or not np.isfinite(vectors).all():
            raise ValueError('Mining requires finite 1024-dimensional vectors')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms <= 0).any():
            raise ValueError('Mining received zero embeddings')
        return vectors / norms

    x, y = normalized(source_vectors, len(sources)), normalized(target_vectors, len(targets))
    if not sources:
        return []
    if not targets:
        return [{'source': s, 'candidates': [], 'candidate_count': 0, 'reciprocal_top1': None,
                 'quality_status': 'unassessed', 'acceptance_status': 'no_eligible_target_span',
                 'neighbors': {'source': min(4, len(sources)), 'target': 0}} for s in sources]
    kx, ky = min(4, len(targets)), min(4, len(sources))
    x_near = np.empty(len(sources), dtype=np.float32)
    y_top = np.full((ky, len(targets)), -np.inf, dtype=np.float32)
    for start in range(0, len(sources), block_rows):
        similarity = x[start:start + block_rows] @ y.T
        x_near[start:start + len(similarity)] = np.partition(similarity, -kx, axis=1)[:, -kx:].mean(axis=1)
        y_top = np.partition(np.concatenate((y_top, similarity)), -ky, axis=0)[-ky:]
    y_near = y_top.mean(axis=0)
    y_best_margin = np.full(len(targets), -np.inf, dtype=np.float32)
    y_best_source = np.full(len(targets), -1, dtype=np.int64)
    records, chosen_targets = [], []
    for start in range(0, len(sources), block_rows):
        similarity = x[start:start + block_rows] @ y.T
        denominator = (x_near[start:start + len(similarity), None] + y_near[None, :]) / 2
        if (denominator <= 0).any():
            raise ValueError('Non-positive ratio-margin neighborhood; preserve inputs for review')
        margins = similarity / denominator
        column_best = margins.argmax(axis=0)
        values = margins[column_best, np.arange(len(targets))]
        improved = values > y_best_margin
        y_best_margin[improved] = values[improved]
        y_best_source[improved] = start + column_best[improved]
        for offset, source in enumerate(sources[start:start + len(similarity)]):
            chosen = np.argsort(-margins[offset], kind='stable')[:5]
            records.append({'source': source, 'candidates': [dict(rank=rank, **targets[int(j)],
                margin_ratio=float(margins[offset, j]), cosine=float(similarity[offset, j]))
                for rank, j in enumerate(chosen, 1)], 'candidate_count': len(targets),
                'quality_status': 'unassessed', 'acceptance_status': 'unthresholded',
                'neighbors': {'source': ky, 'target': kx}})
            chosen_targets.append(int(chosen[0]))
    for index, record in enumerate(records):
        record['reciprocal_top1'] = int(y_best_source[chosen_targets[index]]) == index
    return records
