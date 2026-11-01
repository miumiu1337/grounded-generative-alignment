"""FineWiki search-assisted refinement, citation validation and self-verification."""

import ipaddress
import json
import re
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def query_for(item, page):
    topic = page["en"].split(" (en), revision ", 1)[0]
    words = re.findall(r"[\w’'-]+", item["blind"]["translation"], re.UNICODE)
    return f'{topic} ' + ' '.join(words[:26]) + ' -site:wikipedia.org -site:wikimedia.org'


def external_url(url):
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Invalid external URL")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().rstrip(".")
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or "." not in host or host.endswith((".local", ".internal"))):
        raise ValueError("External evidence requires a public HTTPS URL")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError("IP-address evidence URLs are not allowed")
    domains = ("wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org", "wikiquote.org", "wikisource.org")
    if any(host == d or host.endswith("." + d) for d in domains):
        raise ValueError("Variant C external evidence must be beyond Wikimedia")
    # Keep the original URL in the log; this key prevents tracking variants
    # of one page from masquerading as two independent references.
    query = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in {"trk", "fbclid", "gclid"}]
    return urlunsplit(("https", host, parsed.path or "/", urlencode(query), ""))


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("A timezone is required")
    return parsed


def validate_search(value, item, page, policy):
    expected = {"source_id", "query", "provider", "searched_at", "status", "sources", "failures", "search_ref"}
    if (not isinstance(value, dict) or set(value) != expected
            or value["source_id"] != item["source"]["source_id"] or value["query"] != query_for(item, page)
            or value["provider"] != policy["provider"] or value["status"] not in {"ok", "no_results", "unavailable"}
            or not isinstance(value["search_ref"], str) or not value["search_ref"].strip()
            or not isinstance(value["sources"], list) or not isinstance(value["failures"], list)
            or len({s["url"] for s in value["sources"] + value["failures"]}) > policy["max_pages_per_source"]):
        raise ValueError("Search cache differs from the fixed source/query/budget")
    searched = timestamp(value["searched_at"])
    urls = set()
    for n, source in enumerate(value["sources"], 1):
        if (set(source) != {"reference_id", "url", "title", "text", "retrieved_at", "provider_ref", "evidence_kind"}
                or source["reference_id"] != f"WEB{n}"
                or source["evidence_kind"] not in {"page_excerpt", "search_snippet"}
                or any(not isinstance(source[k], str) or not source[k].strip() for k in ("title", "text", "provider_ref"))
                or len(source["title"]) > 500 or len(source["text"]) > 3000
                or len(source["text"].split()) > policy["max_excerpt_words_per_page"]
                or timestamp(source["retrieved_at"]) < searched):
            raise ValueError("Malformed or over-budget cached external source")
        url = external_url(source["url"])
        if url in urls:
            raise ValueError("Repeated external URL in one search cache")
        urls.add(url)
    for failure in value["failures"]:
        if set(failure) != {"url", "reason"} or not isinstance(failure["reason"], str) or not failure["reason"].strip():
            raise ValueError("Malformed search access failure")
        external_url(failure["url"])
    if ((value["status"] == "ok") != bool(value["sources"])
            or value["status"] == "no_results" and value["failures"]
            or value["status"] == "unavailable" and not value["failures"]):
        raise ValueError("Search status disagrees with evidence/access failures")
    return value


def stage_messages(stage, manifest, item, page, refinement=None):
    if stage == "refinement":
        payload = {"focus_source": item["source"], "blind_translation": item["blind"]["translation"],
                   "bavarian_page": page["bar"], "english_wikipedia_page": page["en"],
                   "external_search": item["search"]}
    elif stage == "verification":
        validate_refinement(refinement, item)
        payload = {"focus_source": item["source"], "bavarian_page": page["bar"],
                   "translation": refinement["translation"], "used_reference_ids": refinement["used_reference_ids"],
                   "external_sources": item["search"]["sources"]}
    else:
        raise ValueError("Unknown Variant C stage")
    return [{"role": "system", "content": manifest["prompts"][stage]},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]


def schema_for(stage, item, page=None):
    properties = {"source_id": {"type": "string", "const": item["source"]["source_id"]}}
    if stage == "refinement":
        ids = [s["reference_id"] for s in item["search"]["sources"]]
        properties.update(translation={"type": "string", "minLength": 1},
            used_reference_ids={"type": "array", "maxItems": len(ids),
                                "items": {"type": "string", **({"enum": ids} if ids else {})}})
    elif stage == "verification":
        properties.update(verdict={"type": "string", "enum": ["pass", "flag", "uncertain"]},
            issues={"type": "array", "maxItems": 5, "items": {"type": "object", "properties": {
                "kind": {"type": "string", "enum": ["addition", "omission", "mistranslation", "contradiction", "citation", "uncertainty"]},
                "description": {"type": "string", "minLength": 1}}, "required": ["kind", "description"], "additionalProperties": False}})
    else:
        raise ValueError("Unknown Variant C stage")
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def validate_refinement(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "translation", "used_reference_ids"}
            or value["source_id"] != item["source"]["source_id"]
            or not isinstance(value["translation"], str) or not value["translation"].strip()
            or not isinstance(value["used_reference_ids"], list)
            or any(not isinstance(i, str) for i in value["used_reference_ids"])
            or len(set(value["used_reference_ids"])) != len(value["used_reference_ids"])
            or not set(value["used_reference_ids"]) <= {s["reference_id"] for s in item["search"]["sources"]}):
        raise ValueError("Malformed refinement or unknown/duplicate external citation")


def validate_verification(value, item):
    if (not isinstance(value, dict) or set(value) != {"source_id", "verdict", "issues"}
            or value["source_id"] != item["source"]["source_id"] or value["verdict"] not in {"pass", "flag", "uncertain"}
            or not isinstance(value["issues"], list) or len(value["issues"]) > 5
            or (value["verdict"] == "pass") != (not value["issues"])):
        raise ValueError("Malformed/inconsistent self-verification")
    for issue in value["issues"]:
        if (not isinstance(issue, dict) or set(issue) != {"kind", "description"}
                or issue["kind"] not in {"addition", "omission", "mistranslation", "contradiction", "citation", "uncertainty"}
                or not isinstance(issue["description"], str) or not issue["description"].strip()):
            raise ValueError("Malformed verification issue")
