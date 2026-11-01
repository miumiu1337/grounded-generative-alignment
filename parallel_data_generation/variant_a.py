"""Evidence selection and refinement protocol shared by the FineWiki methods."""

import json
import re


def selection_messages(manifest, item, pages):
    payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
               "bavarian_page": pages["bar"], "english_page": pages["en"]}
    return [{"role": "system", "content": manifest["prompts"]["selection"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def evidence_for(selection, pages):
    ids = selection["evidence_ids"]
    return [] if not ids else [next(span for span in pages["candidates"] if span["sentence_ids"] == ids)]


def refinement_messages(manifest, item, pages, selection):
    payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
               "bavarian_page": pages["bar"], "selection": selection,
               "selected_english_evidence": evidence_for(selection, pages)}
    return [{"role": "system", "content": manifest["prompts"]["refinement"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def schema_for(stage, item, pages):
    properties = {"source_id": {"type": "string", "const": item["source"]["source_id"]}}
    if stage == "selection":
        ids = [span["sentence_ids"][0] for span in pages["candidates"] if len(span["sentence_ids"]) == 1]
        properties.update(evidence_ids={"type": "array", "minItems": 0, "maxItems": 2,
                                        "items": {"type": "string", "enum": ids},
                                        "enum": [[], *[span["sentence_ids"] for span in pages["candidates"]]]},
                          match_status={"type": "string", "enum": ["exact", "partial", "none"]},
                          contradiction={"type": "boolean"})
    elif stage == "refinement":
        properties["translation"] = {"type": "string", "minLength": 1}
    else:
        raise ValueError("Unknown Variant A stage")
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def selection_grammar(item, pages):
    """Constrain the whole evidence array, not just each individual ID.

    Each literal is JSON-escaped for the EBNF grammar. This enforces adjacency
    and status/contradiction consistency without silently editing model output.
    """
    prefix = '{"source_id": ' + json.dumps(item["source"]["source_id"]) + ', "evidence_ids": '
    spans = [json.dumps(span["sentence_ids"]) for span in pages["candidates"]]
    tails = [', "match_status": ' + json.dumps(status) + ', "contradiction": ' + json.dumps(contradiction)
             for status, contradiction in (("exact", False), ("partial", False), ("partial", True), ("none", True))]
    empty = '[], "match_status": "none", "contradiction": false'
    return '\n'.join([
        'root ::= ' + json.dumps(prefix) + ' (empty-selection | evidence selection-tail) ' + json.dumps('}'),
        'empty-selection ::= ' + json.dumps(empty),
        'evidence ::= ' + ' | '.join(json.dumps(span) for span in spans),
        'selection-tail ::= ' + ' | '.join(json.dumps(tail) for tail in tails),
    ]) + '\n'


def strict_json(text):
    def unique(pairs):
        values = {}
        for key, value in pairs:
            if key in values:
                raise ValueError("Duplicate key in model JSON")
            values[key] = value
        return values
    def invalid(value):
        raise ValueError(f"Non-JSON constant: {value}")
    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)


def validate_selection(value, item, pages):
    if (not isinstance(value, dict) or set(value) != {"source_id", "evidence_ids", "match_status", "contradiction"}
            or value["source_id"] != item["source"]["source_id"]
            or value["match_status"] not in {"exact", "partial", "none"}
            or type(value["contradiction"]) is not bool):
        raise ValueError("Malformed evidence selection")
    ids = value["evidence_ids"]
    valid = {tuple(span["sentence_ids"]) for span in pages["candidates"]}
    if (not isinstance(ids, list) or any(not isinstance(v, str) for v in ids)
            or ids and tuple(ids) not in valid):
        raise ValueError("Evidence must be one frozen English span (one sentence or two consecutive sentences in a paragraph)")
    status, contradiction = value["match_status"], value["contradiction"]
    if (not ids and (status != "none" or contradiction)
            or ids and status == "none" and not contradiction
            or status == "exact" and contradiction):
        raise ValueError("Evidence IDs, match status and contradiction are inconsistent")
    return value


def validate_refinement(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "translation"}
            or value["source_id"] != item["source"]["source_id"]
            or not isinstance(value["translation"], str) or not value["translation"].strip()):
        raise ValueError("Malformed refinement")
    return value


def stage_messages(stage, manifest, item, pages, selection=None):
    return (selection_messages(manifest, item, pages) if stage == "selection"
            else refinement_messages(manifest, item, pages, selection))
