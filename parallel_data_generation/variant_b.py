"""FineWiki protocol: select evidence before translating, without a blind draft."""

import json

from parallel_data_generation import variant_a as shared
from parallel_data_generation.variant_a import evidence_for, validate_selection


def selection_messages(manifest, item, page):
    payload = {"focus_source": item["source"], "bavarian_page": page["bar"], "english_page": page["en"]}
    return [{"role": "system", "content": manifest["prompts"]["selection"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def translation_messages(manifest, item, page, selection):
    validate_selection(selection, item, page)
    payload = {"focus_source": item["source"], "bavarian_page": page["bar"], "selection": selection,
               "selected_english_evidence": evidence_for(selection, page)}
    return [{"role": "system", "content": manifest["prompts"]["translation"]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def stage_messages(stage, manifest, item, page, selection=None):
    if stage == "selection":
        return selection_messages(manifest, item, page)
    if stage == "translation":
        return translation_messages(manifest, item, page, selection)
    raise ValueError("Unknown Variant B stage")


def schema_for(stage, item, page):
    if stage not in {"selection", "translation"}:
        raise ValueError("Unknown Variant B stage")
    return shared.schema_for("selection" if stage == "selection" else "refinement", item, page)
