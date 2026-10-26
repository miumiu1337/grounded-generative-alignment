import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rerank_only as protocol
import variant_b as b


class RerankOnlyTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"source": {"source_id": f"bar_{page}:v1.0.1:S{n:04d}",
            "bar_text": "Unchanged source.", "bar_title": str(page), "bar_page_id": page,
            "bar_revision_id": 123}} for page, count in ((2544, 39), (59, 14)) for n in range(1, count + 1)]
        self.pages = {str(page): {"bar": "Numbered Bavarian page", "en": "Numbered English page",
            "en_page_id": page + 1, "en_revision_id": 456,
            "candidates": [{"sentence_ids": [f"en_{page}:S1"], "text": "Original English.  Keep spacing."},
                           {"sentence_ids": [f"en_{page}:S2"], "text": "Second sentence."},
                           {"sentence_ids": [f"en_{page}:S1", f"en_{page}:S2"],
                            "text": "Original English.  Keep spacing. Second sentence."}]}
            for page in (2544, 59)}
        self.manifest = {"experiment": json.loads(b.CONFIG.read_text()),
            "prompts": {"selection": "Select only IDs.", "translation": "NEVER USE THIS PROMPT"},
            "inputs_sha256": b.digest({"items": self.items, "pages": self.pages})}
        self.selections = {}
        for index, item in enumerate(self.items):
            page = self.pages[str(item["source"]["bar_page_id"])]
            ids, status, contradiction = [], "none", False
            if index in (0, 1, 2):
                ids = page["candidates"][2 if index == 1 else 0]["sentence_ids"]
                status, contradiction = ("exact", False) if index == 0 else ("partial", index == 2)
            result = {"source_id": item["source"]["source_id"], "evidence_ids": ids,
                      "match_status": status, "contradiction": contradiction}
            self.selections[result["source_id"]] = {
                "run_id": b.digest(self.manifest), "source_id": result["source_id"], "stage": "selection",
                "messages": b.stage_messages("selection", self.manifest, item, page),
                "schema": b.schema_for("selection", item, page), "grammar": b.selection_grammar(item, page),
                "result": result, "raw_response": json.dumps(result), "finish_reason": "stop",
                "prompt_tokens": 100, "output_tokens": 20,
                "runtime": {"vllm": "0.21.0", "transformers": "5.5.4"},
                "recorded_at": "2026-10-25T10:00:00+00:00"}

    def export(self, directory, selections=None):
        return protocol.export_run(directory, self.manifest, self.items, self.pages,
                                   self.selections if selections is None else selections)

    def test_verbatim_single_and_two_sentence_evidence_without_translation(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            summary = self.export(directory)
            with (directory / "comparison.csv").open(encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            self.assertNotIn("translation", rows[0])
            self.assertEqual(rows[0]["bar_text"], "Unchanged source.")
            self.assertEqual(rows[0]["english_text"], "Original English.  Keep spacing.")
            self.assertEqual(rows[1]["english_text"], "Original English.  Keep spacing. Second sentence.")
            self.assertEqual(summary["new_inference_calls"], 0)
            self.assertEqual(summary["generated_translations"], 0)
            self.assertEqual(summary["reused_selection_prompt_tokens"], 5300)
            sid = self.items[0]["source"]["source_id"]
            saved = json.loads(b.record_path(directory / "selections", sid).read_text())
            self.assertEqual(saved, self.selections[sid])
            self.assertEqual(set(json.loads(saved["messages"][1]["content"])),
                             {"focus_source", "bavarian_page", "english_page"})

    def test_exact_partial_unmatched_and_contradictory_partitions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            summary = self.export(directory)
            self.assertEqual(summary["output_group_counts"], {
                "exact_candidates": 1, "partial_candidates": 1, "unmatched": 50, "contradictory": 1})
            ids = []
            for group, count in summary["output_group_counts"].items():
                records = [json.loads(line) for line in (directory / (group + ".jsonl")).read_text().splitlines()]
                self.assertEqual(len(records), count)
                ids.extend(r["source"]["source_id"] for r in records)
                if group == "unmatched":
                    self.assertTrue(all(r["english_text"] is None and r["evidence"] == [] for r in records))
            self.assertEqual(len(ids), len(set(ids)))
            self.assertEqual(set(ids), set(self.selections))

    def test_invalid_or_missing_selection_rejected_before_writing(self):
        sid = self.items[0]["source"]["source_id"]
        for mutate in (lambda records: records.pop(sid),
                       lambda records: records[sid].update(finish_reason="length"),
                       lambda records: records[sid]["result"].update(evidence_ids=["invented"]),
                       lambda records: records[sid]["result"].update(translation="rewritten"),
                       lambda records: records[sid].update(messages=[])):
            records = copy.deepcopy(self.selections)
            mutate(records)
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary) / "output"
                with self.assertRaises(ValueError):
                    self.export(directory, records)
                self.assertFalse(directory.exists())

    def test_idempotence_and_tamper_detection_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "output"
            self.assertEqual(self.export(directory), self.export(directory))
            file = directory / "comparison.csv"
            file.write_text("tampered")
            with self.assertRaisesRegex(ValueError, "artifact differs"):
                self.export(directory)
            self.assertEqual(file.read_text(), "tampered")

    def test_loader_uses_selection_files_without_translation_results(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(protocol, "ROOT", Path(temporary)):
            directory = Path(temporary) / protocol.SOURCE_DIRECTORY
            b.atomic_json(directory / "run_manifest.json", self.manifest)
            b.atomic_json(directory / "inputs.json", {"items": self.items, "pages": self.pages})
            for sid, record in self.selections.items():
                b.atomic_json(b.record_path(directory / "selections", sid), record)
            (directory / "results").mkdir()
            (directory / "results" / "unreadable.json").write_text("not JSON; must never be read")
            with (patch.object(protocol, "SOURCE_RUN_ID", b.digest(self.manifest)),
                  patch.object(b, "local_run", return_value=(self.manifest, self.items, self.pages))):
                self.assertEqual(protocol.load_selections()[3], self.selections)
                (directory / "inputs.json").write_text("{}")
                with self.assertRaisesRegex(ValueError, "frozen inputs"):
                    protocol.load_selections()


if __name__ == "__main__":
    unittest.main()
