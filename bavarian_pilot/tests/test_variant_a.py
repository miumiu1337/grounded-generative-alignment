import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import variant_a as protocol
try:
    import modal_variant_a as runner
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    runner = None


@unittest.skipIf(runner is None, "Install requirements-modal.txt for Variant A tests")
class VariantATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.items, cls.pages = protocol.local_run()

    def selection(self, item=None, evidence=None, status="partial", contradiction=False):
        item = item or self.items[0]
        page = self.pages[str(item["source"]["bar_page_id"])]
        ids = page["candidates"][0]["sentence_ids"] if evidence is None else evidence
        return {"source_id": item["source"]["source_id"], "evidence_ids": ids,
                "match_status": status, "contradiction": contradiction}

    def stage(self, item, stage, selection=None):
        page = self.pages[str(item["source"]["bar_page_id"])]
        value = self.selection(item) if stage == "selection" else {
            "source_id": item["source"]["source_id"], "translation": "Example refinement."}
        return {"run_id": protocol.digest(self.manifest), "source_id": item["source"]["source_id"],
                "stage": stage, "messages": protocol.stage_messages(stage, self.manifest, item, page, selection),
                "schema": protocol.schema_for(stage, item, page), "raw_response": json.dumps(value),
                "grammar": protocol.selection_grammar(item, page) if stage == "selection" else None,
                "result": value, "finish_reason": "stop", "prompt_tokens": 100, "output_tokens": 20}

    def test_all_53_blind_translations_reused_without_evaluation_pages(self):
        self.assertEqual(len(self.items), 53)
        self.assertEqual(set(self.pages), {"2544", "59"})
        self.assertEqual([len(self.pages[p]["candidates"]) for p in ("2544", "59")], [327, 1012])
        self.assertTrue(all(item["blind"]["run_id"] == self.manifest["experiment"]["blind_run_id"] for item in self.items))
        self.assertEqual([i["source"]["bar_page_id"] for i in runner.work_order(self.items)[:2]], [2544, 59])

    def test_source_and_original_draft_visible_to_selection(self):
        item = self.items[0]
        messages = protocol.selection_messages(self.manifest, item, self.pages["2544"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["focus_source"], item["source"])
        self.assertEqual(payload["blind_translation"], item["blind"]["translation"])
        self.assertEqual(payload["english_page"], self.pages["2544"]["en"])

    def test_refinement_sees_only_the_fixed_english_evidence(self):
        item = self.items[0]
        page = copy.deepcopy(self.pages["2544"])
        page["en"] = "FULL-PAGE-MUST-NOT-ENTER-REFINEMENT"
        selected = self.selection(item)
        messages = protocol.refinement_messages(self.manifest, item, page, selected)
        payload = json.loads(messages[1]["content"])
        self.assertNotIn("FULL-PAGE-MUST-NOT-ENTER-REFINEMENT", json.dumps(payload))
        self.assertEqual(payload["selection"], selected)
        self.assertEqual(payload["selected_english_evidence"], [page["candidates"][0]])

    def test_unknown_disjoint_reversed_or_wrong_article_evidence_rejected(self):
        item, page = self.items[0], self.pages["2544"]
        singles = [c["sentence_ids"][0] for c in page["candidates"] if len(c["sentence_ids"]) == 1]
        for ids in (["invented"], [singles[0], singles[-1]], [singles[1], singles[0]],
                    [self.pages["59"]["candidates"][0]["sentence_ids"][0]]):
            with self.assertRaises(ValueError):
                protocol.validate_selection(self.selection(item, ids), item, page)

    def test_decoding_grammar_enumerates_only_complete_frozen_spans(self):
        item, page = self.items[0], self.pages["2544"]
        grammar = protocol.selection_grammar(item, page)
        evidence_line = next(line for line in grammar.splitlines() if line.startswith("evidence ::="))
        literals = evidence_line.split("::=", 1)[1].strip().split(" | ")
        arrays = [json.loads(json.loads(literal)) for literal in literals]
        self.assertEqual(arrays, [span["sentence_ids"] for span in page["candidates"]])
        self.assertNotIn('"none", "contradiction": false',
                         json.loads(grammar.splitlines()[-1].split("::=", 1)[1].strip().split(" | ")[-1]))

    def test_none_and_contradictions_remain_separate(self):
        item, page = self.items[0], self.pages["2544"]
        none = self.selection(item, [], "none")
        protocol.validate_selection(none, item, page)
        self.assertEqual(protocol.output_group(none), "unmatched")
        conflict = self.selection(item, status="none", contradiction=True)
        protocol.validate_selection(conflict, item, page)
        self.assertEqual(protocol.output_group(conflict), "contradictory")
        for bad in (self.selection(item, [], "partial"), self.selection(item, status="none"),
                    self.selection(item, status="exact", contradiction=True)):
            with self.assertRaises(ValueError):
                protocol.validate_selection(bad, item, page)

    def test_malformed_json_rewritten_source_and_empty_translation_fail(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '```json\n{}\n```'):
            with self.assertRaises(ValueError):
                protocol.strict_json(text)
        for value in ({"source_id": "rewritten", "translation": "Text"},
                      {"source_id": self.items[0]["source"]["source_id"], "translation": " "}):
            with self.assertRaises(ValueError):
                protocol.validate_refinement(value, self.items[0])

    def test_saved_selection_survives_incomplete_refinement_and_changed_run_fails(self):
        item, page = self.items[0], self.pages["2544"]
        selection = self.stage(item, "selection")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            protocol.atomic_json(directory / "run_manifest.json", self.manifest)
            protocol.atomic_json(protocol.record_path(directory / "selections", item["source"]["source_id"]), selection)
            saved = protocol.read_saved(directory, self.manifest, self.items, self.pages)
            self.assertEqual(len(saved["selections"]), 1)
            self.assertEqual(saved["results"], {})
            refinement = self.stage(item, "refinement", selection["result"])
            refinement["finish_reason"] = "length"
            with self.assertRaises(ValueError):
                protocol.validate_stage(refinement, "refinement", self.manifest, item, page, selection["result"])
            changed = copy.deepcopy(self.manifest)
            changed["prompts"]["selection"] += " changed"
            with self.assertRaises(ValueError):
                protocol.read_saved(directory, changed, self.items, self.pages)

    def test_remote_invalid_bundle_is_rejected_before_local_publication(self):
        record = self.stage(self.items[0], "selection")
        record["run_id"] = "wrong"
        with tempfile.TemporaryDirectory() as temporary, patch.object(protocol, "ROOT", Path(temporary)):
            with self.assertRaises(ValueError):
                protocol.write_local({"selections": {record["source_id"]: record}, "results": {}},
                                     self.manifest, self.items, self.pages)
            self.assertFalse((Path(temporary) / "data/translations").exists())

    def test_refinement_failure_commits_selection_and_resume_skips_selection(self):
        item = self.items[0]
        selection = self.stage(item, "selection")
        refinement = self.stage(item, "refinement", selection["result"])
        truncated = {**refinement, "finish_reason": "length"}
        fake_engine = Mock()
        from types import SimpleNamespace
        fake_vllm = SimpleNamespace(LLM=Mock(return_value=fake_engine))
        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(runner, "REMOTE", Path(temporary)),
              patch.object(runner, "results_volume", Mock()),
              patch.object(runner, "cache_volume", Mock()),
              patch.object(runner, "version", return_value="test-version"),
              patch.dict("sys.modules", {"vllm": fake_vllm}),
              patch.object(runner, "generate_stage", side_effect=[selection, truncated])):
            with self.assertRaisesRegex(RuntimeError, "Invalid refinement"):
                runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            directory = Path(temporary) / self.manifest["experiment"]["run_name"]
            saved = protocol.read_saved(directory, self.manifest, self.items, self.pages)
            self.assertEqual(len(saved["selections"]), 1)
            self.assertEqual(saved["results"], {})
            self.assertTrue(protocol.record_path(directory / "failures/refinement", item["source"]["source_id"]).exists())
            with patch.object(runner, "generate_stage", return_value=refinement) as generate:
                resumed = runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(generate.call_args.args[5], "refinement")
            self.assertEqual(len(resumed["results"]), 1)


if __name__ == "__main__":
    unittest.main()
