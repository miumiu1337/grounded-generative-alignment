import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import variant_b as protocol

try:
    from runners.modal import variant_b as runner
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    runner = None


class VariantBTests(unittest.TestCase):
    def setUp(self):
        self.items = [{"source": {"source_id": f"bar_{page}:v1.0.1:S{n:04d}",
            "bar_text": "A source sentence.", "bar_title": str(page), "bar_page_id": page,
            "bar_revision_id": 123}} for page, count in ((2544, 39), (59, 14)) for n in range(1, count + 1)]
        self.pages = {str(page): {"bar": "Numbered Bavarian page", "en": "FULL-ENGLISH-PAGE",
            "en_page_id": page + 1, "en_revision_id": 456,
            "candidates": [{"sentence_ids": [f"en_{page}:S1"], "text": "First evidence."},
                           {"sentence_ids": [f"en_{page}:S2"], "text": "Second evidence."},
                           {"sentence_ids": [f"en_{page}:S1", f"en_{page}:S2"], "text": "Both."}]}
            for page in (2544, 59)}
        self.manifest = {"experiment": json.loads(protocol.CONFIG.read_text()),
            "prompts": {"selection": "Select evidence.", "translation": "Translate faithfully."},
            "inputs_sha256": protocol.digest({"items": self.items, "pages": self.pages})}
        self.item, self.page = self.items[0], self.pages["2544"]

    def selection(self, ids=None, status="partial", contradiction=False):
        return {"source_id": self.item["source"]["source_id"],
            "evidence_ids": self.page["candidates"][0]["sentence_ids"] if ids is None else ids,
            "match_status": status, "contradiction": contradiction}

    def stage(self, stage, selection=None):
        result = self.selection() if stage == "selection" else {
            "source_id": self.item["source"]["source_id"], "translation": "A faithful translation."}
        return {"run_id": protocol.digest(self.manifest), "source_id": self.item["source"]["source_id"],
            "stage": stage, "messages": protocol.stage_messages(stage, self.manifest, self.item, self.page, selection),
            "schema": protocol.schema_for(stage, self.item, self.page),
            "grammar": protocol.selection_grammar(self.item, self.page) if stage == "selection" else None,
            "raw_response": json.dumps(result), "result": result, "finish_reason": "stop",
            "prompt_tokens": 100, "output_tokens": 20, "runtime": {"vllm": "0.21.0", "transformers": "5.5.4"},
            "recorded_at": "2026-10-25T10:00:00+00:00"}

    def bundle(self):
        selection = self.stage("selection")
        translation = self.stage("translation", selection["result"])
        result = {"condition": "grounded_b", "run_id": protocol.digest(self.manifest),
            "source": self.item["source"], "selection": selection["result"],
            "selection_sha256": protocol.digest(selection), "evidence": [self.page["candidates"][0]],
            "en_page_id": self.page["en_page_id"], "en_revision_id": self.page["en_revision_id"],
            "translation": translation, "output_group": "grounded_candidates", "quality_status": "unassessed"}
        source_id = self.item["source"]["source_id"]
        return {"selections": {source_id: selection}, "results": {source_id: result}}

    def test_both_stages_exclude_blind_and_a_outputs(self):
        poisoned = {**self.item, "blind": {"translation": "SECRET-DRAFT"}, "selection": "OLD-A-SELECTION"}
        selection = protocol.selection_messages(self.manifest, poisoned, self.page)
        translation = protocol.translation_messages(self.manifest, poisoned, self.page, self.selection())
        for messages in (selection, translation):
            self.assertNotIn("SECRET-DRAFT", json.dumps(messages))
            self.assertNotIn("OLD-A-SELECTION", json.dumps(messages))
        selected = json.loads(selection[1]["content"])
        self.assertEqual(set(selected), {"focus_source", "bavarian_page", "english_page"})
        translated = json.loads(translation[1]["content"])
        self.assertNotIn("FULL-ENGLISH-PAGE", json.dumps(translated))
        self.assertEqual(translated["selected_english_evidence"], [self.page["candidates"][0]])

    def test_draft_contaminated_or_changed_inputs_rejected(self):
        for mutate in (lambda items: items[0].update(blind="draft"),
                       lambda items: items[0]["source"].update(translation="draft"),
                       lambda items: items.pop()):
            items, manifest = copy.deepcopy(self.items), copy.deepcopy(self.manifest)
            mutate(items)
            manifest["inputs_sha256"] = protocol.digest({"items": items, "pages": self.pages})
            with self.assertRaises(ValueError):
                protocol.validate_inputs(manifest, items, self.pages)

    def test_same_whole_span_constraints_and_no_forced_match(self):
        for ids in (["invented"], list(reversed(self.page["candidates"][2]["sentence_ids"])), ["en_59:S1"]):
            with self.assertRaises(ValueError):
                protocol.validate_selection(self.selection(ids), self.item, self.page)
        none = self.selection([], "none")
        protocol.validate_selection(none, self.item, self.page)
        self.assertEqual(protocol.output_group(none), "unmatched")
        self.assertEqual(json.loads(protocol.translation_messages(self.manifest, self.item, self.page, none)[1]["content"])
                         ["selected_english_evidence"], [])
        grammar = protocol.selection_grammar(self.item, self.page)
        self.assertEqual(grammar, protocol.shared.selection_grammar(self.item, self.page))

    def test_invalid_stage_runtime_context_output_and_payload_rejected(self):
        for updates in ({"finish_reason": "length"}, {"runtime": {"vllm": "wrong"}},
                        {"prompt_tokens": 65000}, {"recorded_at": "2026-10-25T10:00:00"},
                        {"messages": []}, {"raw_response": '{"source_id":1,"source_id":2}'}):
            with self.assertRaises(ValueError):
                protocol.validate_stage({**self.stage("selection"), **updates}, "selection",
                                        self.manifest, self.item, self.page)

    def test_partial_resume_manifest_mismatch_and_selection_binding(self):
        saved = self.bundle()
        with tempfile.TemporaryDirectory() as temporary, patch.object(protocol, "ROOT", Path(temporary)):
            source_id = self.item["source"]["source_id"]
            partial = {"selections": saved["selections"], "results": {}}
            directory, verified = protocol.write_local(partial, self.manifest, self.items, self.pages)
            self.assertEqual(verified, partial)
            changed = copy.deepcopy(self.manifest)
            changed["prompts"]["selection"] += " changed"
            with self.assertRaises(ValueError):
                protocol.read_saved(directory, changed, self.items, self.pages)
            saved["results"][source_id]["selection_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                protocol.write_local(saved, self.manifest, self.items, self.pages)
            self.assertFalse((directory / "results").exists())

    def test_translation_cannot_predate_selection(self):
        saved = self.bundle()
        saved["results"][self.item["source"]["source_id"]]["translation"]["recorded_at"] = "2026-10-24T10:00:00+00:00"
        with tempfile.TemporaryDirectory() as temporary, patch.object(protocol, "ROOT", Path(temporary)):
            with self.assertRaisesRegex(ValueError, "predates"):
                protocol.write_local(saved, self.manifest, self.items, self.pages)
            self.assertFalse((Path(temporary) / "data").exists())

    def test_export_has_b_translation_and_separate_partitions(self):
        saved = self.bundle()
        with tempfile.TemporaryDirectory() as temporary, patch.object(protocol, "ROOT", Path(temporary)):
            directory, verified = protocol.write_local(saved, self.manifest, self.items, self.pages)
            summary = protocol.export_results(directory, verified, self.manifest)
            self.assertEqual(summary["completed"], 1)
            self.assertEqual(summary["new_prompt_tokens"], 200)
            self.assertEqual(summary["output_group_counts"], {"grounded_candidates": 1})
            self.assertNotIn("blind", (directory / "comparison.csv").read_text())
            self.assertEqual(len((directory / "grounded_candidates.jsonl").read_text().splitlines()), 1)
            self.assertEqual((directory / "unmatched.jsonl").read_text(), "")

    @unittest.skipIf(runner is None, "Install runners/modal/requirements.txt for the Modal worker test")
    def test_failed_translation_keeps_committed_selection_and_resume_reuses_it(self):
        selection, translated = self.stage("selection"), self.stage("translation", self.selection())
        fake_engine = Mock()
        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(runner, "REMOTE", Path(temporary)),
              patch.object(runner, "results_volume", Mock()), patch.object(runner, "cache_volume", Mock()),
              patch.object(runner, "version", return_value="test-version"),
              patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=Mock(return_value=fake_engine))}),
              patch.object(runner, "generate_stage", side_effect=[selection, {**translated, "finish_reason": "length"}])):
            with self.assertRaisesRegex(RuntimeError, "Invalid translation"):
                runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            directory = Path(temporary) / self.manifest["experiment"]["run_name"]
            saved = protocol.read_saved(directory, self.manifest, self.items, self.pages)
            self.assertEqual(len(saved["selections"]), 1)
            self.assertEqual(saved["results"], {})
            self.assertTrue(protocol.record_path(directory / "failures/translation", self.item["source"]["source_id"]).exists())
            with patch.object(runner, "generate_stage", return_value=translated) as generate:
                resumed = runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(generate.call_args.args[5], "translation")
            self.assertEqual(len(resumed["results"]), 1)


if __name__ == "__main__":
    unittest.main()
