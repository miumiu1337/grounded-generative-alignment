"""Prepare a common corpus from pinned HTML snapshots; no bilingual alignment."""

import argparse
import hashlib
import json
import platform
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import pysbd
from bs4 import BeautifulSoup

from download_articles import ROOT, load_raw_pairs, load_selection, request_api, save_json, sha256
from source_quality import POLICY, exclusions


PREPROCESSING_VERSION = "1.0.1"
BOUNDARY_CORRECTIONS_PATH = ROOT / "data/sentence_boundary_corrections.json"
BACK_MATTER_HEADINGS = {
    "bar": {
        "Beleg", "Belege", "Biacha", "Literatua", "Literatur", "Im Netz", "Schau aa",
        "Quellen", "Einzelnachweise", "Anmerkungen", "Weblinks",
    },
    "en": {
        "Bibliography", "External links", "Footnotes", "Further reading", "Notes",
        "References", "See also", "Sources", "Notes and references", "References and notes",
        "Citations", "Works cited", "References and sources",
    },
}
NAVIGATION_PREFIXES = {"bar": ("Hauptartikl:", "Schaug:"), "en": ("Main article:",)}
REMOVED_SELECTORS = (
    "table, figure, script, style, link, meta, .thumb, .navbox, .hatnote, .metadata, "
    ".ambox, .reflist, .refbegin, .mw-references-wrap, .reference, .shortdescription, "
    ".noprint, .mw-editsection, .toc, .gallery, ul, ol, dl"
)


def normalized_heading(text):
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("&", " and ")
    return re.sub(r"[\W_]+", " ", text).strip()


def inline_text(element):
    # Joining with a space would insert spaces before punctuation around links.
    return re.sub(r"\s+", " ", element.get_text("", strip=False)).strip()


def html_snapshot(article, directory, fetch=False):
    language, revision_id = article["language"], article["revision_id"]
    path = directory / f"{language}_{article['page_id']}_{revision_id}.json"
    if not path.exists():
        if not fetch:
            raise FileNotFoundError(f"Missing HTML snapshot: {path}. Run with --fetch-html once.")
        data = request_api(language, {
            "action": "parse", "oldid": revision_id, "prop": "text|revid",
            "disableeditsection": 1,
        })
        parsed = data.get("parse", {})
        if parsed.get("revid") != revision_id or not isinstance(parsed.get("text"), str):
            raise ValueError(f"Invalid parsed revision for {language}:{revision_id}")
        save_json(path, {
            "language": language, "page_id": article["page_id"], "revision_id": revision_id,
            "rendered_at": datetime.now(timezone.utc).isoformat(),
            "html": parsed["text"],
        })
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    for key in ("language", "page_id", "revision_id"):
        if snapshot.get(key) != article[key]:
            raise ValueError(f"HTML snapshot identity mismatch ({key}): {path}")
    if not isinstance(snapshot.get("html"), str) or not snapshot["html"].strip():
        raise ValueError(f"HTML snapshot is empty: {path}")
    if not isinstance(snapshot.get("rendered_at"), str) or not snapshot["rendered_at"]:
        raise ValueError(f"HTML snapshot lacks render timestamp: {path}")
    return path, snapshot["html"]


def extract_paragraphs(html, language):
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".mw-parser-output")
    if body is None:
        raise ValueError("Article content container was not found")
    for line_break in body.find_all("br"):
        line_break.replace_with(" ")
    next_elements = {}
    for paragraph in body.find_all("p"):
        sibling = paragraph.find_next_sibling()
        while sibling is not None and sibling.name in {"link", "meta", "script", "style"}:
            sibling = sibling.find_next_sibling()
        next_elements[id(paragraph)] = sibling.name if sibling is not None else None
    for element in body.select(REMOVED_SELECTORS):
        if element.parent is not None:
            element.decompose()

    headings = {normalized_heading(text) for text in BACK_MATTER_HEADINGS[language]}
    sections = [{"section_id": "lead", "heading": None, "level": 1, "heading_path": []}]
    stack = []
    paragraphs = []
    removed = Counter()
    excluded_level = None
    navigation = tuple(prefix.casefold() for prefix in NAVIGATION_PREFIXES[language])
    for element in body.find_all(["h2", "h3", "h4", "h5", "h6", "p"]):
        if element.name != "p":
            level = int(element.name[1])
            heading = inline_text(element)
            if excluded_level is not None and level <= excluded_level:
                excluded_level = None
            if excluded_level is None and normalized_heading(heading) in headings:
                excluded_level = level
            if excluded_level is not None:
                continue
            while stack and stack[-1]["level"] >= level:
                stack.pop()
            section = {
                "section_id": f"section_{len(sections):03d}", "heading": heading, "level": level,
                "heading_path": [item["heading"] for item in stack] + [heading],
            }
            stack.append(section)
            sections.append(section)
            continue
        if excluded_level is not None:
            removed["back_matter_paragraphs"] += 1
            continue
        text = inline_text(element)
        if not text:
            continue
        coordinate = element.find("span", class_="geo-inline-hidden", recursive=False)
        if coordinate and text == inline_text(coordinate):
            removed["coordinate_only_paragraphs"] += 1
            continue
        bold = element.find(["b", "strong"])
        small = element.find("small")
        next_kind = next_elements[id(element)]
        reason = None
        if text.casefold().startswith(navigation) or (
            text.startswith("→") and len(text) < 100 and element.find("a")
        ):
            reason = "navigation_note"
        elif len(text) < 100 and not text.endswith((".", "!", "?", ":")) and (
            (bold is not None and inline_text(bold) == text)
            or (small is not None and inline_text(small) == text and next_kind in {"ul", "ol", "dl"})
        ):
            reason = "standalone_label"
        elif text.endswith(":") and next_kind in {"ul", "ol", "dl", "table"}:
            reason = "incomplete_list_intro"
        section = stack[-1] if stack else sections[0]
        paragraphs.append({
            "paragraph_id": len(paragraphs) + 1, "section_id": section["section_id"],
            "heading_path": section["heading_path"], "text": text,
            "matching_exclusion_reason": reason,
        })
    if not paragraphs:
        raise ValueError("No prose paragraphs extracted")
    return paragraphs, sections, dict(removed)


def apply_boundary_corrections(article, paragraphs, corrections):
    applied = []
    for correction in corrections:
        if any(correction[key] != article[key] for key in ("language", "page_id", "revision_id")):
            continue
        matches = [paragraph for paragraph in paragraphs
                   if hashlib.sha256(paragraph["text"].encode("utf-8")).hexdigest()
                   == correction["paragraph_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Boundary correction {correction['id']} must match exactly one paragraph")
        paragraph = matches[0]
        before = correction["split_after"]
        if not before or paragraph["text"].count(before) != 1:
            raise ValueError(f"Boundary correction {correction['id']} has an ambiguous split position")
        offset = paragraph["text"].index(before) + len(before)
        if offset == len(paragraph["text"]) or not paragraph["text"][offset].isspace():
            raise ValueError(f"Boundary correction {correction['id']} must split between sentences")
        boundaries = paragraph.setdefault("forced_sentence_boundaries", [])
        if offset in boundaries:
            raise ValueError(f"Repeated boundary correction: {correction['id']}")
        boundaries.append(offset)
        applied.append(correction["id"])
    return applied


def prepare_article(article, html, boundary_corrections=()):
    paragraphs, sections, removed = extract_paragraphs(html, article["language"])
    applied = apply_boundary_corrections(article, paragraphs, boundary_corrections)
    rule_language = "de" if article["language"] == "bar" else "en"
    splitter = pysbd.Segmenter(language=rule_language, clean=False)
    prefix = f"{article['language']}_{article['page_id']}:v{PREPROCESSING_VERSION}"
    sentences = []
    for paragraph in paragraphs:
        boundaries = [0, *sorted(paragraph.get("forced_sentence_boundaries", [])), len(paragraph["text"])]
        segments = [text.strip() for start, end in zip(boundaries, boundaries[1:])
                    for text in splitter.segment(paragraph["text"][start:end]) if text.strip()]
        if not segments:
            raise ValueError(f"Segmenter returned no text: {prefix}, paragraph {paragraph['paragraph_id']}")
        for index, text in enumerate(segments):
            reasons = []
            reason = paragraph["matching_exclusion_reason"]
            if reason and (reason != "incomplete_list_intro" or index == len(segments) - 1):
                reasons.append(reason)
            sentences.append({
                "sentence_id": f"{prefix}:S{len(sentences) + 1:04d}",
                "paragraph_id": paragraph["paragraph_id"], "section_id": paragraph["section_id"],
                "heading_path": paragraph["heading_path"], "text": text,
                "exclusion_reasons": reasons,
            })
    flagged = exclusions(sentences, article["language"])
    for sentence in sentences:
        sentence["exclusion_reasons"] = sorted(set(
            sentence["exclusion_reasons"] + flagged.get(sentence["sentence_id"], [])
        ))
        sentence["eligible_for_matching"] = not sentence["exclusion_reasons"]
    spans = []
    for start in range(len(sentences)):
        for length in (1, 2):
            group = sentences[start:start + length]
            if len(group) != length or group[0]["paragraph_id"] != group[-1]["paragraph_id"]:
                continue
            if any(not item["eligible_for_matching"] for item in group):
                continue
            spans.append({
                "sentence_ids": [item["sentence_id"] for item in group],
                "paragraph_id": group[0]["paragraph_id"], "section_id": group[0]["section_id"],
                "text": " ".join(item["text"] for item in group),
            })
    metadata = {key: value for key, value in article.items() if key != "wikitext"}
    return {
        **metadata, "sentence_rules": rule_language, "sentence_count": len(sentences),
        "clean_text": "\n\n".join(item["text"] for item in paragraphs),
        "sections": sections, "paragraphs": paragraphs, "sentences": sentences,
        "candidate_spans": spans, "removed": removed,
        "sentence_boundary_corrections_applied": applied,
    }


def validate_split(split, rows):
    selected = {(row["x_page_id"], row["en_page_id"]): row for row in rows}
    seen = set()
    assignments = {}
    for name, expected in (("development", 2), ("evaluation", 8)):
        pairs = split.get(name, [])
        if len(pairs) != expected:
            raise ValueError(f"Expected {expected} {name} pairs")
        for pair in pairs:
            key = (pair["bar_page_id"], pair["en_page_id"])
            if key not in selected or key in seen:
                raise ValueError(f"Unselected or repeated pair in split: {key}")
            row = selected[key]
            if pair["bar_title"] != row["x_title"] or pair["en_title"] != row["en_title"]:
                raise ValueError(f"Split title differs from selection: {key}")
            seen.add(key)
            assignments[key[0]] = name
    if seen != set(selected):
        raise ValueError("Split does not cover the selection")
    return assignments


def quality_report(prepared_pairs, assignments):
    duplicates = defaultdict(list)
    summaries = []
    for pair in prepared_pairs:
        split = assignments[pair["bar"]["page_id"]]
        for language in ("bar", "en"):
            article = pair[language]
            counts = Counter(reason for s in article["sentences"] for reason in s["exclusion_reasons"])
            summaries.append({
                "language": language, "title": article["title"], "split": split,
                "sentences": article["sentence_count"],
                "eligible_sentences": sum(s["eligible_for_matching"] for s in article["sentences"]),
                "candidate_spans": len(article["candidate_spans"]),
                "exclusion_counts": dict(sorted(counts.items())), "removed": article["removed"],
                "sentence_boundary_corrections_applied": article["sentence_boundary_corrections_applied"],
            })
            for sentence in article["sentences"]:
                key = (language, unicodedata.normalize("NFC", re.sub(r"\s+", " ", sentence["text"])).strip())
                duplicates[key].append({
                    "sentence_id": sentence["sentence_id"], "split": split,
                    "eligible_for_matching": sentence["eligible_for_matching"],
                })
    groups = [{"language": lang, "text": text, "cross_split": len({s["split"] for s in locations}) > 1,
               "occurrences": locations}
              for (lang, text), locations in sorted(duplicates.items()) if len(locations) > 1]
    return {
        "preprocessing_version": PREPROCESSING_VERSION, "sentence_quality_policy": POLICY,
        "articles": summaries, "exact_duplicate_groups": groups,
        "duplicate_policy": "Report Unicode-NFC/whitespace-normalized exact repeats without deleting occurrences. Near-duplicates are not detected.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch-html", action="store_true", help="Fetch missing pinned HTML snapshots; otherwise remain offline")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/prepared")
    args = parser.parse_args()
    rows = load_selection()
    pairs = load_raw_pairs(rows)
    if len(pairs) != len(rows):
        raise FileNotFoundError("Missing raw snapshots; run download_articles.py first")
    split_path = ROOT / "data/article_split.json"
    assignments = validate_split(json.loads(split_path.read_text(encoding="utf-8")), rows)
    provenance_path = ROOT / "data/provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    for entry in provenance["copied_raw_files"]:
        if sha256(ROOT / entry["file"]) != entry["sha256"]:
            raise ValueError(f"Imported raw snapshot changed: {entry['file']}")
    code_hashes = {name: sha256(ROOT / name) for name in (
        "download_articles.py", "prepare_articles.py", "source_quality.py",
    )}
    environment = {name: version(name) for name in (
        "beautifulsoup4", "pysbd", "requests", "urllib3", "certifi",
        "charset-normalizer", "idna", "soupsieve", "typing-extensions",
    )}
    environment["python"] = platform.python_version()
    corrections = json.loads(BOUNDARY_CORRECTIONS_PATH.read_text(encoding="utf-8"))
    correction_ids = [item["id"] for item in corrections]
    if len(correction_ids) != len(set(correction_ids)):
        raise ValueError("Duplicate sentence boundary correction IDs")
    prepared_pairs = []
    records = []
    for row in rows:
        raw_path, pair = pairs[row["x_page_id"]]
        prepared = {
            "preprocessing_version": PREPROCESSING_VERSION, "split": assignments[row["x_page_id"]],
            "inputs": {"raw_file": str(raw_path.relative_to(ROOT)), "raw_sha256": sha256(raw_path)},
            "preparation": {**environment, "code_sha256": code_hashes,
                            "maximum_span_sentences": 2, "sentence_quality_policy": POLICY,
                            "sentence_boundary_corrections_sha256": sha256(BOUNDARY_CORRECTIONS_PATH)},
        }
        for language in ("bar", "en"):
            html_path, html = html_snapshot(pair[language], ROOT / "data/html", args.fetch_html)
            prepared["inputs"][f"{language}_html_file"] = str(html_path.relative_to(ROOT))
            prepared["inputs"][f"{language}_html_sha256"] = sha256(html_path)
            prepared[language] = prepare_article(pair[language], html, corrections)
        output_path = args.output_dir / f"bar_{row['x_page_id']}.json"
        save_json(output_path, prepared)
        prepared_pairs.append(prepared)
        records.append({"file": output_path.name, "sha256": sha256(output_path), "inputs": prepared["inputs"]})
        print(f"{row['x_title']}: {prepared['bar']['sentence_count']} BAR / {prepared['en']['sentence_count']} EN sentences")
    applied = [correction_id for pair in prepared_pairs for language in ("bar", "en")
               for correction_id in pair[language]["sentence_boundary_corrections_applied"]]
    if sorted(applied) != sorted(correction_ids):
        raise ValueError("Each sentence boundary correction must be applied exactly once in the corpus")
    report = quality_report(prepared_pairs, assignments)
    save_json(args.output_dir / "quality_report.json", report)
    save_json(args.output_dir / "manifest.json", {
        "preprocessing_version": PREPROCESSING_VERSION, "preparation": prepared_pairs[0]["preparation"],
        "selection_sha256": sha256(ROOT / "data/articles.csv"), "split_sha256": sha256(split_path),
        "provenance_sha256": sha256(provenance_path),
        "requirements_lock_sha256": sha256(ROOT / "requirements.lock"),
        "quality_report_sha256": sha256(args.output_dir / "quality_report.json"),
        "articles": records,
    })
    print(f"Saved {len(records)} pairs, quality report and manifest to {args.output_dir}")


if __name__ == "__main__":
    main()
