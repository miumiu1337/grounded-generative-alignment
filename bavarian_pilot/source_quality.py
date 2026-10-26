"""Conservative sentence-fragment checks; source text is never rewritten here."""

import re
from collections import defaultdict


POLICY = "sentence_quality_v1"
ABBREVIATION_FRAGMENT = re.compile(
    r"(?:[/(\s]*(?:fem|mask|masc|neutr|sg|pl|akk|dat|nom|gen|engl|bzw)\.[)\s]*)+",
    re.IGNORECASE,
)
LABELS = {"bar": {"beispui", "beispiele", "schaug"}, "en": {"example", "examples"}}


def exclusions(sentences, language):
    reasons = defaultdict(list)
    paragraphs = defaultdict(list)
    seen = set()
    for sentence in sentences:
        sentence_id = sentence["sentence_id"]
        text = sentence["text"].strip()
        paragraph_id = sentence["paragraph_id"]
        if sentence_id in seen or not text or type(paragraph_id) is not int:
            raise ValueError(f"Malformed or duplicate sentence: {sentence_id}")
        seen.add(sentence_id)
        paragraphs[paragraph_id].append(sentence)
        if not any(character.isalnum() for character in text):
            reasons[sentence_id].append("punctuation_only")
        if text.startswith((",", ";")):
            reasons[sentence_id].append("leading_punctuation")
        if ABBREVIATION_FRAGMENT.fullmatch(text) or re.fullmatch(r"\([mfn]\)\.?", text):
            reasons[sentence_id].append("abbreviation_fragment")
        if text.rstrip(":").casefold() in LABELS[language]:
            reasons[sentence_id].append("standalone_label")

    for paragraph in paragraphs.values():
        for index, sentence in enumerate(paragraph[1:], start=1):
            if sentence["text"].strip().startswith((",", ";")):
                reasons[paragraph[index - 1]["sentence_id"]].append("continued_by_leading_punctuation")
        quote_start = None
        for index, sentence in enumerate(paragraph):
            if sentence["text"].count('"') % 2 == 0:
                continue
            if quote_start is None:
                quote_start = index
            else:
                for fragment in paragraph[quote_start:index + 1]:
                    reasons[fragment["sentence_id"]].append("split_quotation")
                quote_start = None
    return dict(reasons)
