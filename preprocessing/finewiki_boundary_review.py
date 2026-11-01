"""Apply recorded development boundary decisions without editing FineWiki text."""

import preprocessing.finewiki_pilot as data


def apply_review(rows, baseline, disagreements, plan):
    if (plan["version"] != 1 or plan["dataset_revision"] != data.REVISION
            or plan["scope"] != "development_only" or plan["base_splitter"] != "datatrove"):
        raise ValueError("Boundary review differs from the pinned development comparison")
    methods = {"pysbd", "srx", "datatrove"}
    expected = {(r["article_id"], r["offset"]) for r in disagreements
                if 0 < len(methods.intersection(r["methods_with_boundary"])) < 3}
    decisions = {(article_id, offset) for article_id, items in plan["decisions"].items() for offset, _, _ in items}
    if decisions != expected or set(plan["decisions"]) != {row["id"] for row in rows}:
        raise ValueError("Boundary review must cover every disputed development boundary exactly")
    prepared, audits = {}, []
    for row in rows:
        original = baseline[row["id"]]
        if (original["dataset_id"] != row["id"] or original["text"] != row["text"]
                or original["revision_id"] != row["version"] or original["sentence_splitter"] != "datatrove"):
            raise ValueError("Reviewed baseline differs from its FineWiki original")
        boundaries = {unit["end"] for unit in original["sentences"] + original["excluded_units"]}
        items = plan["decisions"][row["id"]]
        if len({item[0] for item in items}) != len(items):
            raise ValueError("Repeated boundary decision")
        changes = []
        for offset, accepted, reason in items:
            if type(offset) is not int or type(accepted) is not bool or not isinstance(reason, str) or not reason.strip():
                raise ValueError("Malformed boundary decision")
            previous = offset in boundaries
            if accepted:
                boundaries.add(offset)
            else:
                boundaries.discard(offset)
            if previous != accepted:
                changes.append({"offset": offset, "action": "add" if accepted else "remove", "reason": reason})
        article = data.prepare_article(row, data.REVIEWED_METHOD, boundaries=boundaries)
        prepared[row["id"]] = article
        audits.append({"article_id": row["id"], "baseline_units": len(original["sentences"]),
                       "reviewed_units": len(article["sentences"]), "reviewed_decisions": len(items),
                       "changes": changes, "text_sha256": article["text_sha256"], "artifact_sha256": data.digest(article)})
    return prepared, {"plan": plan, "plan_sha256": data.digest(plan), "articles": audits,
                      "quality_claim": "AI-assisted segmentation review, not translation evaluation or human gold",
                      "text_policy": "original FineWiki text and exact character offsets retained"}
