import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

import output_quality as quality


def row(output_id="one", method="blind_translation", source_id="source1", source="A Bavarian source.", target="An English sentence."):
    return {"output_id": output_id, "method": method,
            "source": {"source_id": source_id, "bar_text": source, "bar_title": "Title"},
            "target_text": target, "target_kind": "generated_translation", "candidate_rank": None,
            "evidence_ids": [], "original_match_status": None, "original_contradiction": None,
            "original_selfcheck_verdict": None, "finish_reasons": ["stop"],
            "artifact_path": "original.json", "artifact_canonical_sha256": "original-hash"}


class OutputQualityTests(unittest.TestCase):
    def test_shared_technical_rejections_and_normal_whitespace(self):
        for target, reason in (("  ", "empty_target"), ("...!?", "punctuation_only_target"),
                               ("An English\x00 sentence.", "invalid_control_character")):
            audited, _ = quality.annotate([row(target=target)])
            self.assertEqual(audited[0]["technical_status"], "reject")
            self.assertIn(reason, audited[0]["technical_reasons"])
        audited, _ = quality.annotate([row(target="An English\n\t sentence.")])
        self.assertEqual(audited[0]["technical_status"], "pass")
        self.assertEqual(audited[0]["semantic_status"], "unassessed")

    def test_truncated_and_missing_completion_metadata_do_not_pass(self):
        for reasons in (["length"], ["stop", "length"], []):
            sample = {**row(), "finish_reasons": reasons}
            audited, _ = quality.annotate([sample])
            self.assertEqual(audited[0]["technical_reasons"], ["incomplete_generation"])

    def test_verbatim_evidence_does_not_require_english_generation_completion_metadata(self):
        sample = {**row(method="rerank_only"), "target_kind": "verbatim_evidence", "finish_reasons": []}
        audited, _ = quality.annotate([sample])
        self.assertEqual(audited[0]["technical_status"], "pass")
        self.assertEqual(audited[0]["technical_reasons"], [])
        self.assertTrue(audited[0]["needs_semantic_assessment"])

    def test_only_explicit_unmatched_reranking_can_have_no_target(self):
        unmatched = {**row(method="rerank_only", target=None),
                     "original_match_status": "none", "original_contradiction": False}
        audited, _ = quality.annotate([unmatched])
        self.assertEqual(audited[0]["technical_status"], "no_target")
        self.assertFalse(audited[0]["needs_semantic_assessment"])
        for updates in ({"method": "grounded_b"}, {"original_match_status": "partial"},
                        {"evidence_ids": ["EnglishID"]}, {"original_contradiction": True}):
            with self.assertRaises(ValueError):
                quality.annotate([{**unmatched, **updates}])

    def test_selfcheck_and_match_labels_do_not_decide_technical_acceptance(self):
        for method in ("grounded_a", "grounded_b", "grounded_c", "rerank_only", "laser_1", "bge_m3"):
            sample = {**row(method=method), "original_match_status": "none", "original_contradiction": True,
                      "original_selfcheck_verdict": "flag"}
            audited, _ = quality.annotate([sample])
            self.assertEqual(audited[0]["technical_status"], "pass")
            self.assertEqual(audited[0]["semantic_status"], "unassessed")

    def test_normalized_duplicates_are_within_method_warnings_and_text_is_preserved(self):
        first = row(target="An English café.")
        second = row("two", source_id="source2", target="An  English cafe\u0301.")
        other_method = row("three", method="grounded_b", target=first["target_text"])
        original = copy.deepcopy([first, second, other_method])
        audited, groups = quality.annotate([first, second, other_method])
        self.assertEqual([first, second, other_method], original)
        self.assertEqual(audited[1]["target_text"], original[1]["target_text"])
        self.assertEqual({g["kind"] for g in groups}, {"duplicate_pair", "reused_target"})
        self.assertEqual(audited[2]["warnings"], [])
        self.assertTrue(all(r["technical_status"] == "pass" for r in audited))

    def test_different_sources_reusing_target_are_not_duplicate_pairs(self):
        samples = [row(), row("two", source_id="source2", source="A different Bavarian source.")]
        audited, groups = quality.annotate(samples)
        self.assertEqual([g["kind"] for g in groups], ["reused_target"])
        self.assertEqual(audited[0]["warnings"], ["reused_target"])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            quality.annotate([row(), row()])

    def test_mining_candidates_must_be_verbatim_unique_ranked_finite_spans(self):
        spans = [{"sentence_ids": [f"English{i}"], "text": f"Sentence {i}."} for i in range(5)]
        page = {"candidates": spans, "en_page_id": 1, "en_revision_id": 2}
        record = {"candidates": [{**span, "rank": i+1, "margin_ratio": 1.1-i/100, "cosine": .9-i/100}
                                 for i, span in enumerate(spans)],
                  "candidate_count": 5, "en_page_id": 1, "en_revision_id": 2, "acceptance_status": "unthresholded"}
        quality.validate_mining_record(record, page)
        for mutate in (lambda r: r["candidates"][0].update(text="rewritten"),
                       lambda r: r["candidates"][0].update(sentence_ids=["invented"]),
                       lambda r: r["candidates"][0].update(margin_ratio=float("nan")),
                       lambda r: r["candidates"][0].update(margin_ratio=.5),
                       lambda r: r["candidates"][0].update(rank=2),
                       lambda r: r.update(acceptance_status="accepted")):
            changed = copy.deepcopy(record)
            mutate(changed)
            with self.assertRaises(ValueError):
                quality.validate_mining_record(changed, page)

    def test_a_aggregate_exports_validate_complete_target_responses(self):
        a = quality.a
        config = json.loads(a.CONFIG.read_text())
        items = [{"source": {"source_id": f"bar_{page}:v1.0.1:S{n:04d}", "bar_page_id": page},
                  "blind": {"run_id": config["blind_run_id"], "translation": "Blind draft."}}
                 for page, count in ((2544, 39), (59, 14)) for n in range(1, count+1)]
        pages = {str(page): {"bar": "Bavarian page", "en": "English page", "candidates": [],
                            "en_page_id": page+1, "en_revision_id": 456} for page in (2544, 59)}
        manifest = {"experiment": config, "inputs_sha256": a.digest({"items": items, "pages": pages}),
                    "prompts": {"refinement": "Refine faithfully."}}
        records = []
        for item in items:
            page = pages[str(item["source"]["bar_page_id"])]
            sid = item["source"]["source_id"]
            selection = {"source_id": sid, "evidence_ids": [], "match_status": "none", "contradiction": False}
            result = {"source_id": sid, "translation": "A translation."}
            stage = {"source_id": sid, "run_id": a.digest(manifest), "stage": "refinement",
                     "messages": a.stage_messages("refinement", manifest, item, page, selection),
                     "schema": a.schema_for("refinement", item, page), "grammar": None,
                     "finish_reason": "stop", "prompt_tokens": 20, "output_tokens": 5,
                     "raw_response": json.dumps(result), "result": result}
            records.append({"condition": "grounded_a", "run_id": a.digest(manifest), **item,
                            "selection": selection, "evidence": [], "en_page_id": page["en_page_id"],
                            "en_revision_id": 456, "output_group": "unmatched", "quality_status": "unassessed",
                            "refinement": stage})
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ("grounded_candidates", "contradictory"):
                (directory/(name+".jsonl")).write_text("")
            file = directory/"unmatched.jsonl"
            file.write_text("\n".join(json.dumps(record) for record in records))
            saved, paths = quality.load_grounded_exports("grounded_a", directory, manifest, items, pages)
            self.assertEqual(len(saved["results"]), 53)
            self.assertEqual(set(paths.values()), {file})
            records[0]["refinement"]["finish_reason"] = "length"
            file.write_text("\n".join(json.dumps(record) for record in records))
            with self.assertRaisesRegex(ValueError, "refinement record"):
                quality.load_grounded_exports("grounded_a", directory, manifest, items, pages)

    def test_exports_keep_all_rows_partition_them_and_reject_overwrite(self):
        samples = [row(), row("two", target=" "),
                   {**row("three", method="rerank_only", target=None),
                    "original_match_status": "none", "original_contradiction": False}]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            summary = quality.export_report(directory, samples, {"test": "provenance"})
            self.assertEqual(summary["accepted_pairs"], None)
            self.assertEqual(summary["new_inference_calls"], 0)
            self.assertEqual(summary, quality.export_report(directory, samples, {"test": "provenance"}))
            ids = []
            for status in ("pass", "reject", "no_target"):
                records = [json.loads(line) for line in (directory/f"technical_{status}.jsonl").read_text().splitlines()]
                self.assertEqual(len(records), 1)
                ids.extend(r["output_id"] for r in records)
            self.assertEqual(set(ids), {r["output_id"] for r in samples})
            with (directory/"comparison.csv").open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(rows[1]["target_text"], " ")
            file = directory/"comparison.csv"
            file.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "Changed technical-quality"):
                quality.export_report(directory, samples, {"test": "provenance"})
            self.assertEqual(file.read_text(), "tampered")


if __name__ == "__main__":
    unittest.main()
