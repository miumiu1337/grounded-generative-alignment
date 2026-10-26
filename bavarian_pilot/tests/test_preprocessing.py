import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from download_articles import ROOT, load_selection, request_api, save_json
from prepare_articles import (
    BOUNDARY_CORRECTIONS_PATH, PREPROCESSING_VERSION, apply_boundary_corrections,
    extract_paragraphs, html_snapshot, prepare_article, quality_report, validate_split,
)
from source_quality import exclusions


def article(language="bar", page_id=1):
    return {
        "language": language, "page_id": page_id, "revision_id": 10,
        "title": "Example", "wikitext": "Original source",
    }


def html(body):
    return '<div class="mw-parser-output">' + body + '</div>'


class ExtractionTests(unittest.TestCase):
    def test_preserve_inline_spacing_and_line_breaks(self):
        paragraphs, _, _ = extract_paragraphs(html(
            '<p>A <a>word</a>, <i>un</i>broken.<br>Next line.</p>'
        ), "en")
        self.assertEqual(paragraphs[0]["text"], "A word, unbroken. Next line.")

    def test_heading_normalization_nested_exclusion_and_resume(self):
        paragraphs, sections, removed = extract_paragraphs(html(
            '<p>Lead.</p><h2>History</h2><h3>Early years</h3><p>Body.</p>'
            '<h2> REFERENCES &amp; NOTES <span class="mw-editsection">[edit]</span></h2>'
            '<p>Exclude.</p><h3>Sources</h3><p>Exclude too.</p>'
            '<h2>Culture</h2><p>Keep.</p>'
        ), "en")
        self.assertEqual([p["text"] for p in paragraphs], ["Lead.", "Body.", "Keep."])
        self.assertEqual(paragraphs[1]["heading_path"], ["History", "Early years"])
        self.assertEqual(paragraphs[2]["heading_path"], ["Culture"])
        self.assertEqual(sections[0]["section_id"], "lead")
        self.assertEqual(removed["back_matter_paragraphs"], 2)

    def test_bavarian_literature_alias_and_no_substring_filtering(self):
        paragraphs, _, _ = extract_paragraphs(html(
            '<h2>References in popular culture</h2><p>Keep.</p>'
            '<h2>LITERATUA:</h2><p>Exclude.</p>'
        ), "bar")
        self.assertEqual([p["text"] for p in paragraphs], ["Keep."])

    def test_lists_tables_citations_and_list_introduction(self):
        paragraphs, _, _ = extract_paragraphs(html(
            '<p>Prose<sup class="reference">[1]</sup>.</p>'
            '<p>Examples:</p><ul><li><p>Nested list prose.</p></li></ul>'
            '<table><tr><td><p>Table prose.</p></td></tr></table>'
        ), "en")
        self.assertEqual([p["text"] for p in paragraphs], ["Prose.", "Examples:"])
        self.assertEqual(paragraphs[1]["matching_exclusion_reason"], "incomplete_list_intro")

    def test_missing_body_and_no_prose_fail(self):
        for document in ("<p>No container.</p>", html("<table><tr><td>Only table.</td></tr></table>")):
            with self.subTest(document=document), self.assertRaises(ValueError):
                extract_paragraphs(document, "en")


class SentenceTests(unittest.TestCase):
    def test_bavarian_vowel_label_ends_a_sentence_in_saved_article(self):
        selection = next(row for row in load_selection() if row["x_page_id"] == 58)
        raw_path = next((ROOT / "data/raw").glob("bar_58_*.json"))
        source = json.loads(raw_path.read_text(encoding="utf-8"))["bar"]
        snapshot = ROOT / "data/html" / f"bar_58_{selection['x_revision_id']}.json"
        document = json.loads(snapshot.read_text(encoding="utf-8"))["html"]
        corrections = json.loads(BOUNDARY_CORRECTIONS_PATH.read_text(encoding="utf-8"))
        prepared = prepare_article(source, document, corrections)
        first = "Es liegt fia deitsch Ohrn a näha am o."
        second = (
            "Das des zum Zammafoi mitn gschlossanan o fiaht, wo im Boarischn moastns nah am ou liegt, "
            "wead vo Kritikern gern ois Oloss gnumma, de Eifiahrung vo – vom Deitschn divergiaradn – "
            "Sondazoachn in da boarischn Schreibweis z begrindn."
        )
        sentences = prepared["sentences"]
        index = next(i for i, sentence in enumerate(sentences) if sentence["text"] == first)
        self.assertEqual(sentences[index + 1]["text"], second)
        self.assertEqual(sentences[index]["paragraph_id"], sentences[index + 1]["paragraph_id"])
        self.assertTrue(all(s["eligible_for_matching"] for s in sentences[index:index + 2]))
        span_texts = [span["text"] for span in prepared["candidate_spans"]]
        for text in (first, second, first + " " + second):
            self.assertIn(text, span_texts)
        paragraph = next(p for p in prepared["paragraphs"] if p["paragraph_id"] == sentences[index]["paragraph_id"])
        self.assertEqual(paragraph["text"], " ".join(
            s["text"] for s in sentences if s["paragraph_id"] == paragraph["paragraph_id"]
        ))
        self.assertEqual(prepared["sentence_boundary_corrections_applied"], [corrections[0]["id"]])

    def test_stale_boundary_correction_fails_and_other_revision_is_unchanged(self):
        correction = json.loads(BOUNDARY_CORRECTIONS_PATH.read_text(encoding="utf-8"))[0]
        source = {"language": "bar", "page_id": 58, "revision_id": correction["revision_id"]}
        with self.assertRaisesRegex(ValueError, "must match exactly one paragraph"):
            apply_boundary_corrections(source, [{"text": "Changed input."}], [correction])
        source["revision_id"] += 1
        paragraph = {"text": "Changed input."}
        self.assertEqual(apply_boundary_corrections(source, [paragraph], [correction]), [])
        self.assertNotIn("forced_sentence_boundaries", paragraph)

    def test_existing_abbreviation_and_initial_boundaries_remain_intact(self):
        prepared = prepare_article(article(), html("<p>Dr. Müller kummt. J. Bach spuit.</p>"))
        self.assertEqual([s["text"] for s in prepared["sentences"]],
                         ["Dr. Müller kummt.", "J. Bach spuit."])

    def test_versioned_ids_sections_and_paragraph_bounded_spans(self):
        prepared = prepare_article(article(), html(
            '<p>Minga liegt an da Isar. Des is schee.</p><h2>Gschicht</h2><p>Da Text bleibt.</p>'
        ))
        self.assertEqual(prepared["sentence_rules"], "de")
        self.assertEqual(len(prepared["sentences"]), 3)
        self.assertTrue(prepared["sentences"][0]["sentence_id"].startswith(f"bar_1:v{PREPROCESSING_VERSION}:"))
        self.assertEqual(prepared["sentences"][2]["heading_path"], ["Gschicht"])
        self.assertEqual(len(prepared["candidate_spans"]), 4)
        self.assertEqual([len(s["sentence_ids"]) for s in prepared["candidate_spans"]], [1, 2, 1, 1])

    def test_abbreviation_and_label_fragments_excluded_in_both_languages(self):
        for language in ("bar", "en"):
            with self.subTest(language=language):
                prepared = prepare_article(article(language), html('<p>Sg.</p><p>Go!</p><p>Examples:</p>'))
                self.assertFalse(prepared["sentences"][0]["eligible_for_matching"])
                self.assertIn("abbreviation_fragment", prepared["sentences"][0]["exclusion_reasons"])
                self.assertTrue(prepared["sentences"][1]["eligible_for_matching"])

    def test_no_span_crosses_excluded_sentence(self):
        prepared = prepare_article(article("en"), html('<p>Keep this. Sg. Keep that.</p>'))
        self.assertEqual(len(prepared["candidate_spans"]), 2)
        self.assertTrue(all(len(s["sentence_ids"]) == 1 for s in prepared["candidate_spans"]))

    def test_quotation_and_continuation_checks(self):
        sentences = [
            {"sentence_id": "s1", "paragraph_id": 1, "text": 'He said "One.'},
            {"sentence_id": "s2", "paragraph_id": 1, "text": 'Two."'},
            {"sentence_id": "s3", "paragraph_id": 1, "text": ', and continued.'},
        ]
        flagged = exclusions(sentences, "en")
        self.assertIn("split_quotation", flagged["s1"])
        self.assertIn("continued_by_leading_punctuation", flagged["s2"])
        self.assertIn("leading_punctuation", flagged["s3"])

    def test_duplicate_report_preserves_occurrences_and_identifies_cross_split(self):
        pairs = []
        for page_id in (1, 2):
            pairs.append({language: prepare_article(article(language, page_id), html('<p>Same text.</p>'))
                          for language in ("bar", "en")})
        report = quality_report(pairs, {1: "development", 2: "evaluation"})
        self.assertEqual(len(report["exact_duplicate_groups"]), 2)
        self.assertTrue(all(group["cross_split"] for group in report["exact_duplicate_groups"]))
        self.assertEqual(len(pairs[0]["bar"]["sentences"]), 1)


class SnapshotTests(unittest.TestCase):
    def test_missing_snapshot_is_offline_by_default(self):
        with tempfile.TemporaryDirectory() as directory, patch("prepare_articles.request_api") as request:
            with self.assertRaises(FileNotFoundError):
                html_snapshot(article(), Path(directory))
            request.assert_not_called()

    def test_fetch_validate_and_reuse_snapshot(self):
        response = {"parse": {"revid": 10, "text": html("<p>Text.</p>")}}
        with tempfile.TemporaryDirectory() as directory, patch("prepare_articles.request_api", return_value=response) as request:
            path, text = html_snapshot(article(), Path(directory), fetch=True)
            self.assertEqual(text, response["parse"]["text"])
            html_snapshot(article(), Path(directory), fetch=True)
            request.assert_called_once()
            saved = json.loads(path.read_text())
            saved["revision_id"] = 11
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                html_snapshot(article(), Path(directory))

    def test_wrong_revision_not_cached(self):
        with tempfile.TemporaryDirectory() as directory, patch("prepare_articles.request_api", return_value={"parse": {"revid": 11, "text": "wrong"}}):
            with self.assertRaises(ValueError):
                html_snapshot(article(), Path(directory), fetch=True)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_output_cannot_silently_change(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            save_json(path, {"revision": 1})
            save_json(path, {"revision": 1})
            with self.assertRaises(FileExistsError):
                save_json(path, {"revision": 2})
            self.assertEqual(json.loads(path.read_text()), {"revision": 1})
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_network_timeout_http_errors_and_api_errors_surface(self):
        response = Mock()
        response.json.return_value = {"error": {"code": "maxlag"}}
        with patch("download_articles.session.get", return_value=response) as get:
            with self.assertRaises(RuntimeError):
                request_api("bar", {"action": "parse"})
            self.assertEqual(get.call_args.kwargs["timeout"], (10, 60))
        with patch("download_articles.session.get", side_effect=requests.Timeout):
            with self.assertRaises(requests.Timeout):
                request_api("bar", {"action": "parse"})
        response.raise_for_status.side_effect = requests.HTTPError
        with patch("download_articles.session.get", return_value=response):
            with self.assertRaises(requests.HTTPError):
                request_api("bar", {"action": "parse"})

    def test_invalid_selection_and_split_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.csv"
            path.write_text("x_page_id,en_title\n1,Example\n")
            with self.assertRaises(ValueError):
                load_selection(path)
        with self.assertRaises(ValueError):
            validate_split({"development": [], "evaluation": []}, [])

    def test_split_must_cover_each_selected_pair_once(self):
        rows = [{"x_page_id": i, "en_page_id": 100 + i, "x_title": str(i), "en_title": str(i)}
                for i in range(10)]
        pairs = [{"bar_page_id": row["x_page_id"], "en_page_id": row["en_page_id"],
                  "bar_title": row["x_title"], "en_title": row["en_title"]} for row in rows]
        split = {"development": pairs[:2], "evaluation": pairs[2:]}
        self.assertEqual(len(validate_split(split, rows)), 10)
        split["evaluation"][0] = pairs[0]
        with self.assertRaises(ValueError):
            validate_split(split, rows)


if __name__ == "__main__":
    unittest.main()
