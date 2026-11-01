"""FineWiki pilot inputs, lossless sentence offsets and training exports."""

from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from functools import lru_cache

DATASET = "HuggingFaceFW/finewiki"
REVISION = "8bd13e72e6a002407649b3e898535f42ceb1aeb9"
METHODS = ("newline", "pysbd", "srx", "datatrove")
REVIEWED_METHOD = "reviewed_datatrove_v1"
CONDITIONS = ("blind", "laser", "bge_m3", "variant_a", "variant_b", "variant_c", "rerank_only")
PROXIES = {"bar": "de", "en": "en"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def read_json(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path, value, replace=False):
    """Use the existing project's immutable, atomic artifact convention."""
    path = Path(path)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    if path.exists() and not replace:
        if read_json(path) != json.loads(payload):
            raise FileExistsError("Immutable FineWiki artifact differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            if str(path).endswith(".gz"):
                with gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
                    compressed.write(payload)
            else:
                stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_article(row, language):
    required = {"id", "page_id", "title", "wikidata_id", "version", "text", "url", "in_language"}
    if (not isinstance(row, dict) or not required.issubset(row)
            or row["in_language"] != language or type(row["page_id"]) is not int or row["page_id"] < 1
            or row["id"] != f"{language}wiki/{row['page_id']}"
            or not isinstance(row["text"], str) or not row["text"].strip()
            or not isinstance(row["title"], str) or not row["title"].strip()
            or not isinstance(row["version"], (int, str)) or isinstance(row["version"], bool)
            or not isinstance(row["url"], str) or not row["url"].startswith(f"https://{language}.wikipedia.org/")):
        raise ValueError("Malformed or wrong-language FineWiki article")
    qid = row["wikidata_id"]
    if qid is not None and (not isinstance(qid, str) or not re.fullmatch(r"Q[1-9][0-9]*", qid)):
        raise ValueError("Malformed Wikidata ID")


def select_latest_versions(rows, language):
    """Choose the latest available MediaWiki revision per page, retaining an audit."""
    groups = {}
    for row in rows:
        validate_article(row, language)
        if not str(row["version"]).isdigit() or int(row["version"]) <= 0:
            raise ValueError("FineWiki revision must be a positive MediaWiki revision ID")
        groups.setdefault(row["page_id"], {}).setdefault(int(row["version"]), []).append(row)
    selected, audit = [], []
    for page_id, versions in sorted(groups.items()):
        for revision, copies in versions.items():
            if len({digest(row) for row in copies}) != 1:
                raise ValueError(f"Conflicting FineWiki copies of {language} page {page_id}, revision {revision}")
        latest = versions[max(versions)][0]
        selected.append(latest)
        audit.append({"article_id": latest["id"], "selected_version": latest["version"],
                      "available_versions": sorted(versions), "retrieved_rows": sum(map(len, versions.values())),
                      "selected_text_sha256": hashlib.sha256(latest["text"].encode()).hexdigest()})
    return selected, audit


def match_pairs(requested, source_rows, target_rows):
    by_page = {}
    for row in source_rows:
        validate_article(row, "bar")
        if row["page_id"] in by_page:
            raise ValueError("Repeated Bavarian article ID")
        by_page[row["page_id"]] = row
    by_qid = {}
    for row in target_rows:
        validate_article(row, "en")
        if row["wikidata_id"]:
            by_qid.setdefault(row["wikidata_id"], []).append(row)
    pairs = []
    seen_qids = set()
    for split, original in requested:
        source = by_page.get(original["bar_page_id"])
        if source is None or not source["wikidata_id"]:
            raise ValueError("Requested FineWiki topic is missing or has no Wikidata ID: " + original["bar_title"])
        qid = source["wikidata_id"]
        candidates = by_qid.get(qid, [])
        if len(candidates) != 1 or qid in seen_qids:
            raise ValueError("FineWiki English match is missing, repeated or ambiguous: " + qid)
        seen_qids.add(qid)
        pairs.append({"split": split, "wikidata_id": qid, "bar": source, "en": candidates[0]})
    return pairs


def blocks(text):
    """Keep every non-whitespace character and its original document offsets.

    Structural Markdown remains available as article context. Only prose and
    list bodies are proposed as sentence matching units; this is recorded in
    the audit and does not rewrite FineWiki's original text.
    """
    output, pending, kind, fence = [], [], None, None
    position = 0

    def flush():
        nonlocal pending, kind
        if pending:
            start, end = pending[0][0], pending[-1][1]
            output.append({"start": start, "end": end, "kind": kind, "text": text[start:end]})
        pending, kind = [], None

    for line in text.splitlines(keepends=True):
        start, end = position, position + len(line)
        position = end
        stripped = line.strip()
        marker = re.match(r"^(`{3,}|~{3,})", stripped)
        if fence:
            pending.append((start, end))
            if marker and marker.group(1)[0] == fence:
                flush()
                fence = None
            continue
        if marker:
            flush()
            kind, fence = "code", marker.group(1)[0]
            pending.append((start, end))
            continue
        if not stripped:
            flush()
            continue
        current = ("heading" if re.match(r"^#{1,6}\s", stripped) else
                   "table" if stripped.startswith("|") else
                   "list" if re.match(r"^(?:[-*+]\s|\d+[.)]\s)", stripped) else
                   "markup" if not any(character.isalnum() for character in stripped) else "prose")
        if current != kind or current in {"heading", "list", "markup"}:
            flush()
            kind = current
        pending.append((start, end))
        if current in {"heading", "list", "markup"}:
            flush()
    flush()
    previous = 0
    for block in output:
        if text[previous:block["start"]].strip():
            raise ValueError("FineWiki block extraction lost non-whitespace text")
        previous = block["end"]
    if text[previous:].strip():
        raise ValueError("FineWiki block extraction lost its text tail")
    return output


@lru_cache(maxsize=8)
def splitter(method, language):
    if language not in PROXIES:
        raise ValueError("Configure and audit a sentence splitter for this language first")
    if method == "pysbd":
        import pysbd
        return pysbd.Segmenter(language=PROXIES[language], clean=False)
    if method == "srx":
        from sentence_splitter import SentenceSplitter
        return SentenceSplitter(language=PROXIES[language])
    if method == "datatrove":
        from datatrove.utils.word_tokenizers import load_word_tokenizer
        return load_word_tokenizer(language)
    raise ValueError("Unknown sentence splitter")


def sentence_values(text, method, language):
    if method == "newline":
        return [line for line in text.splitlines() if line.strip()]
    if method == "pysbd":
        return splitter(method, language).segment(text)
    if method == "srx":
        return splitter(method, language).split(text)
    if method == "datatrove":
        from datatrove.utils.text import split_into_sentences
        return split_into_sentences(text, language=language)
    raise ValueError("Unknown sentence splitter")


def align_segments(text, values):
    spans, previous = [], 0
    for value in values:
        if not isinstance(value, str):
            raise ValueError("Sentence splitter returned non-text output")
        value = value.strip()
        if not value:
            continue
        start = previous
        while start < len(text) and text[start].isspace():
            start += 1
        if text.startswith(value, start):
            end = start + len(value)
        else:
            pattern = "".join(r"\s+" if part.isspace() else re.escape(part)
                              for part in re.split(r"(\s+)", value) if part)
            match = re.compile(pattern).match(text, start)
            if match is None:
                raise ValueError("Sentence splitter changed or lost source characters")
            end = match.end()
        if start >= end:
            raise ValueError("Sentence splitter changed or lost source characters")
        spans.append((start, end))
        previous = end
    if text[previous:].strip():
        raise ValueError("Sentence splitter lost the final source characters")
    return spans


def prepare_article(row, method, boundaries=None):
    language = row["in_language"]
    validate_article(row, language)
    if boundaries is not None and (method != REVIEWED_METHOD or any(type(b) is not int or not 0 < b <= len(row["text"]) for b in boundaries)):
        raise ValueError("Reviewed boundaries require explicit valid offsets and a distinct method name")
    sentences, structural, excluded = [], Counter(), []
    used_boundaries = set()
    for index, block in enumerate(blocks(row["text"]), 1):
        structural[block["kind"]] += 1
        if block["kind"] not in {"prose", "list"}:
            continue
        if boundaries is None:
            values = sentence_values(block["text"], method, language)
        else:
            cuts = sorted({b for b in boundaries if block["start"] < b <= block["end"]})
            used_boundaries.update(cuts)
            cuts = [block["start"], *cuts, block["end"]]
            values = [row["text"][a:b] for a, b in zip(cuts, cuts[1:]) if a < b]
        for segment_index, (start, end) in enumerate(align_segments(block["text"], values)):
            start, end = start + block["start"], end + block["start"]
            value = row["text"][start:end]
            if not any(character.isalpha() for character in value):
                excluded.append({"start": start, "end": end, "reason": "nonalphabetic_unit", "text": value})
                continue
            tag = ":" + REVIEWED_METHOD if method == REVIEWED_METHOD else ""
            sid = f"{language}_{row['page_id']}:fw{REVISION[:8]}:r{row['version']}{tag}:S{len(sentences)+1:04d}"
            sentences.append({"sentence_id": sid, "text": value, "start": start, "end": end,
                              "paragraph_id": index, "segment_index": segment_index,
                              "heading_path": [], "eligible_for_matching": True})
    if boundaries is not None and used_boundaries != set(boundaries):
        raise ValueError("Reviewed boundary falls outside prose/list matching scope")
    candidates = []
    for index, sentence in enumerate(sentences):
        candidates.append({"sentence_ids": [sentence["sentence_id"]], "text": sentence["text"]})
        if (index + 1 < len(sentences) and sentence["paragraph_id"] == sentences[index+1]["paragraph_id"]
                and sentence["segment_index"] + 1 == sentences[index+1]["segment_index"]):
            following = sentences[index+1]
            candidates.append({"sentence_ids": [sentence["sentence_id"], following["sentence_id"]],
                               "text": row["text"][sentence["start"]:following["end"]]})
    return {"language": language, "title": row["title"], "page_id": row["page_id"],
            "revision_id": row["version"], "wikidata_id": row["wikidata_id"], "url": row["url"],
            "dataset_id": row["id"], "dataset_revision": REVISION, "text": row["text"],
            "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
            "sentence_splitter": method, "structural_block_counts": dict(structural), "excluded_units": excluded,
            "matching_scope": "prose_and_lists;other_blocks_retained_in_article_context",
            "sentences": sentences, "candidate_spans": candidates}


def numbered_page(article):
    text, position, parts = article["text"], 0, []
    for sentence in article["sentences"]:
        parts.extend((text[position:sentence["start"]], f"[{sentence['sentence_id']}] ", sentence["text"]))
        position = sentence["end"]
    parts.append(text[position:])
    return "".join(parts)


def training_row(source, target, condition, provenance):
    if condition not in CONDITIONS or not isinstance(target, str) or not target.strip():
        raise ValueError("Only completed nonempty translations belong in a training subset")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Empty training source")
    return {"source_lang": "bar", "target_lang": "en", "source_text": source,
            "target_text": target, "method": condition, "provenance": provenance}
