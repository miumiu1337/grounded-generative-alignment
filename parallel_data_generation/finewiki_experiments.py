"""Seven FineWiki pilot conditions, sharing one fingerprinted source cohort."""

import json
import hashlib

import preprocessing.finewiki_pilot as data

from parallel_data_generation import translate_blind as blind
from parallel_data_generation import variant_a as variant_a
from parallel_data_generation import variant_b as variant_b
from parallel_data_generation import variant_c as variant_c

MODEL, REVISION = blind.MODEL, blind.REVISION


def ordered_selection(selection):
    return {key: selection[key] for key in ("source_id", "evidence_ids", "match_status", "contradiction")}


def freeze_inputs(pairs, prepared, methods, split="development"):
    if split not in {"development", "training", "evaluation"}:
        raise ValueError("Unknown pilot split")
    if set(methods) != {"bar", "en"} or any(method not in (*data.METHODS, data.REVIEWED_METHOD) for method in methods.values()):
        raise ValueError("A compared splitter must be selected for each language")
    pages, sources, provenance = {}, [], {}
    for pair in pairs:
        if pair["split"] != split:
            continue
        bar, en = (prepared[(pair[language]["id"], methods[language])] for language in ("bar", "en"))
        for language, article in (("bar", bar), ("en", en)):
            original = pair[language]
            if (article["dataset_id"] != original["id"] or article["page_id"] != original["page_id"]
                    or article["revision_id"] != original["version"] or article["language"] != language
                    or article["sentence_splitter"] != methods[language] or article["text"] != original["text"]
                    or article["text_sha256"] != hashlib.sha256(original["text"].encode()).hexdigest()):
                raise ValueError("Prepared article differs from its pinned FineWiki original")
            if any(s["text"] != original["text"][s["start"]:s["end"]] for s in article["sentences"]):
                raise ValueError("Prepared sentence offsets differ from FineWiki text")
        if (bar["dataset_revision"] != data.REVISION or en["dataset_revision"] != data.REVISION
                or bar["wikidata_id"] != en["wikidata_id"] or bar["wikidata_id"] != pair["wikidata_id"]):
            raise ValueError("Prepared FineWiki snapshot or topic pair differs")
        page_id = str(bar["page_id"])
        pages[page_id] = {
            "bar": f"{bar['title']} (bar), revision {bar['revision_id']}\n" + data.numbered_page(bar),
            "en": f"{en['title']} (en), revision {en['revision_id']}\n" + data.numbered_page(en),
            "candidates": en["candidate_spans"], "en_page_id": en["page_id"], "en_revision_id": en["revision_id"],
            "wikidata_id": pair["wikidata_id"], "bar_article_id": bar["dataset_id"], "en_article_id": en["dataset_id"],
            "bar_url": bar["url"], "en_url": en["url"], "bar_text_sha256": bar["text_sha256"],
            "en_text_sha256": en["text_sha256"],
        }
        for sentence in bar["sentences"]:
            if not sentence["eligible_for_matching"]:
                continue
            source = {"source_id": sentence["sentence_id"], "bar_text": sentence["text"],
                      "bar_page_id": bar["page_id"], "bar_title": bar["title"], "bar_revision_id": bar["revision_id"]}
            sources.append(source)
            provenance[source["source_id"]] = {"dataset": data.DATASET, "dataset_revision": data.REVISION,
                "wikidata_id": pair["wikidata_id"], "source_article_id": bar["dataset_id"],
                "target_article_id": en["dataset_id"], "source_url": bar["url"], "target_url": en["url"],
                "source_revision": bar["revision_id"], "target_revision": en["revision_id"],
                "source_segmentation": methods["bar"], "target_segmentation": methods["en"],
                "source_start": sentence["start"], "source_end": sentence["end"], "split": split}
    sources.sort(key=lambda source: (source["bar_page_id"], source["source_id"]))
    if (len(pages) != {"development": 2, "training": 8, "evaluation": 8}[split] or not sources or len({s["source_id"] for s in sources}) != len(sources)
            or set(provenance) != {s["source_id"] for s in sources}):
        raise ValueError("FineWiki cohort is empty, repeated or incomplete")
    return {"sources": sources, "pages": pages, "provenance": provenance,
            "sources_sha256": data.digest(sources), "pages_sha256": data.digest(pages),
            "segmentation": methods, "dataset_revision": data.REVISION}


def request(condition, stage, source, page, prompts, *, draft=None, selection=None, search=None, refinement=None):
    if condition == "blind":
        if stage != "translation" or any(value is not None for value in (draft, selection, search, refinement)):
            raise ValueError("Blind translation must not receive English context or other outputs")
        return {"messages": blind.messages_for(source, prompts["blind"]), "schema": None, "grammar": None}
    if condition == "variant_a":
        if not isinstance(draft, str) or not draft.strip() or stage not in {"selection", "refinement"}:
            raise ValueError("Variant A needs its original blind translation")
        item = {"source": source, "blind": {"translation": draft}}
        settings = {"prompts": prompts["variant_a"]}
        selected = ordered_selection(selection) if stage == "refinement" else None
        messages = variant_a.stage_messages(stage, settings, item, page, selected)
        schema = variant_a.schema_for(stage, item, page)
        grammar = variant_a.selection_grammar(item, page) if stage == "selection" and page["candidates"] else None
    elif condition == "variant_b":
        if draft is not None or search is not None or refinement is not None or stage not in {"selection", "translation"}:
            raise ValueError("Variant B must not receive a blind draft or A/C outputs")
        item, settings = {"source": source}, {"prompts": prompts["variant_b"]}
        selected = ordered_selection(selection) if stage == "translation" else None
        messages = variant_b.stage_messages(stage, settings, item, page, selected)
        schema = variant_b.schema_for(stage, item, page)
        grammar = variant_a.selection_grammar(item, page) if stage == "selection" and page["candidates"] else None
    elif condition == "variant_c":
        if not isinstance(draft, str) or not draft.strip() or stage not in {"refinement", "verification"}:
            raise ValueError("Variant C needs the blind translation and recorded search")
        item = {"source": source, "blind": {"translation": draft}, "search": search}
        settings = {"prompts": prompts["variant_c"]}
        variant_c.validate_search(search, item, page, SEARCH_POLICY)
        messages = variant_c.stage_messages(stage, settings, item, page, refinement)
        schema, grammar = variant_c.schema_for(stage, item, page), None
    else:
        raise ValueError("This condition does not generate text")
    if stage == "selection" and not page["candidates"]:
        only = {"source_id": source["source_id"], "evidence_ids": [], "match_status": "none", "contradiction": False}
        grammar = "root ::= " + json.dumps(json.dumps(only)) + "\n"
    return {"messages": messages, "schema": schema, "grammar": grammar}


SEARCH_POLICY = {"provider": "codex_web_tool", "query_version": 1, "queries_per_source": 1,
                 "max_pages_per_source": 2, "max_excerpt_words_per_page": 25, "exclude_wikimedia": True,
                 "selection": "prefer_english_primary_sources_in_returned_order", "collector": "ai_assisted"}


def parse_result(condition, stage, raw, source, page, search=None):
    if condition == "blind":
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("Empty blind translation")
        return {"translation": raw.strip()}
    value = variant_a.strict_json(raw)
    item = {"source": source}
    if stage == "selection":
        variant_a.validate_selection(value, item, page)
    elif condition == "variant_c":
        if stage == "refinement":
            variant_c.validate_refinement(value, {**item, "search": search})
        else:
            variant_c.validate_verification(value, item)
    else:
        variant_a.validate_refinement(value, item)
    return value


def validate_record(record, source, manifest, expected_request, page, search=None):
    if (record["source"] != source or record["run_id"] != data.digest(manifest)
            or record["model"] != MODEL or record["model_revision"] != REVISION
            or record["condition"] != manifest["condition"] or record["stage"] != manifest["stage"]
            or record["request_sha256"] != data.digest(expected_request) or record["finish_reason"] != "stop"
            or record["quality_status"] != "unassessed"
            or any(type(record[key]) is not int or record[key] < 0 for key in ("prompt_tokens", "output_tokens"))):
        raise ValueError("FineWiki stage record differs from its frozen request")
    parsed = parse_result(manifest["condition"], manifest["stage"], record["raw_response"], source, page, search)
    if parsed != record["result"]:
        raise ValueError("Saved model result differs from its raw response")
