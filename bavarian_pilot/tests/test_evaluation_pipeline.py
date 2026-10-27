import copy
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import evaluation_pipeline as protocol
try:
    from runners.modal import evaluation as runner
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    runner = None


class EvaluationPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources, cls.pages, cls.groups = protocol.corpus()
        cls.items = [{"source": source} for source in cls.sources]
        cls.config = protocol.read(protocol.ROOT / protocol.CONFIGS["b"])
        cls.config.update(split="evaluation", run_name="test_evaluation_b")
        cls.manifest = {"method": "b", "experiment": cls.config,
            "prompts": {"selection": "Select evidence", "translation": "Translate faithfully"},
            "inputs_sha256": protocol.a.digest({"items": cls.items, "pages": cls.pages})}
        cls.item = runner.work_order(cls.items, cls.pages)[0] if runner else cls.items[0]
        cls.page = cls.pages[str(cls.item["source"]["bar_page_id"])]

    def stage(self, stage, selection=None):
        if stage == "selection":
            value = {"source_id": self.item["source"]["source_id"], "evidence_ids": [],
                     "match_status": "none", "contradiction": False}
        else:
            value = {"source_id": self.item["source"]["source_id"], "translation": "A translation."}
        schema, grammar = protocol.constraints("b", stage, self.item, self.page)
        return {"run_id": protocol.a.digest(self.manifest), "source_id": self.item["source"]["source_id"],
            "stage": stage, "messages": protocol.messages("b", stage, self.manifest, self.item, self.page, selection),
            "schema": schema, "grammar": grammar, "raw_response": json.dumps(value), "result": value,
            "finish_reason": "stop", "prompt_tokens": 100, "output_tokens": 30,
            "runtime": {"vllm": "0.21.0", "transformers": "5.5.4"}, "recorded_at": "2026-10-26T12:00:00+00:00"}

    @unittest.skipIf(runner is None, "Install runners/modal/requirements.txt for cloud worker tests")
    def test_uploaded_modules_import_without_access_to_the_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            for module in runner.LOCAL_MODULES:
                source = protocol.ROOT / module
                if source.is_dir():
                    shutil.copytree(source, Path(tmp) / module)
                else:
                    shutil.copyfile(source.with_suffix(".py"), Path(tmp) / (module + ".py"))
            result = subprocess.run([sys.executable, "-I", "-B", "-c",
                "import sys; sys.path.insert(0, sys.argv[1]); import runners.modal.evaluation", tmp],
                cwd=tmp, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_evaluation_sources_exclude_development_and_match_original_spans(self):
        self.assertEqual(len(self.sources), 524)
        self.assertEqual({s["bar_page_id"] for s in self.sources}, protocol.EVALUATION_PAGES)
        for group in self.groups:
            self.assertEqual(group["targets"], self.pages[str(group["bar_page_id"])]["candidates"])
        protocol.validate_bundle(self.manifest, self.items, self.pages)

    def test_drafts_and_development_sources_cannot_enter_b(self):
        for mutate in (lambda items: items[0].update(blind={"translation": "LEAK"}),
                       lambda items: items[0]["source"].update(bar_page_id=59)):
            items = copy.deepcopy(self.items)
            mutate(items)
            manifest = {**self.manifest, "inputs_sha256": protocol.a.digest({"items": items, "pages": self.pages})}
            with self.assertRaises(ValueError):
                protocol.validate_bundle(manifest, items, self.pages)

    def test_blind_has_no_english_page_or_draft(self):
        config = protocol.read(protocol.ROOT / protocol.CONFIGS["blind"])
        config["split"] = "evaluation"
        manifest = {"method": "blind", "experiment": config, "prompts": {"blind": "Translate faithfully"},
                    "inputs_sha256": protocol.a.digest({"items": self.items, "pages": {}})}
        protocol.validate_bundle(manifest, self.items, {})
        payload = protocol.messages("blind", "blind", manifest, self.item, self.page)
        self.assertEqual(payload[1]["content"], "Bavarian text:\n" + self.item["source"]["bar_text"])
        self.assertNotIn("english_page", json.dumps(payload))
        contaminated = {**manifest, "inputs_sha256": protocol.a.digest({"items": self.items, "pages": self.pages})}
        with self.assertRaisesRegex(ValueError, "Blind payload"):
            protocol.validate_bundle(contaminated, self.items, self.pages)

    def test_invalid_finish_runtime_context_and_span_rejected(self):
        for update in ({"finish_reason": "length"}, {"runtime": {}}, {"prompt_tokens": 65536},
                       {"output_tokens": 1025}, {"recorded_at": "2026-10-26T12:00:00"}):
            with self.assertRaises(ValueError):
                protocol.validate_stage({**self.stage("selection"), **update}, "b", "selection",
                                        self.manifest, self.item, self.page)
        record = self.stage("selection")
        record["result"]["evidence_ids"] = ["en_FAKE"]
        record["result"]["match_status"] = "partial"
        record["raw_response"] = json.dumps(record["result"])
        with self.assertRaises(ValueError):
            protocol.validate_stage(record, "b", "selection", self.manifest, self.item, self.page)

    def test_saved_selection_resumes_but_orphan_translation_is_not_published(self):
        sid = self.item["source"]["source_id"]
        selection = self.stage("selection")
        translated = self.stage("translation", selection["result"])
        partial = {"selection": {sid: selection}, "translation": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            protocol.write_local(path, partial, self.manifest, self.items, self.pages)
            self.assertEqual(protocol.read_saved(path, self.manifest, self.items, self.pages), partial)
            with self.assertRaises(ValueError):
                protocol.write_local(Path(tmp) / "bad", {"selection": {}, "translation": {sid: translated}},
                                     self.manifest, self.items, self.pages)
            self.assertFalse((Path(tmp) / "bad").exists())
            changed = {**self.manifest, "prompts": {"selection": "Changed", "translation": "Changed"}}
            with self.assertRaisesRegex(ValueError, "differs"):
                protocol.read_saved(path, changed, self.items, self.pages)

    @unittest.skipIf(runner is None, "Install runners/modal/requirements.txt for cloud worker tests")
    def test_preflight_rejects_oversize_input_without_truncation(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.encode.return_value = [1] * 65000
        with self.assertRaisesRegex(RuntimeError, "no truncation"):
            runner.stage_prompt(tokenizer, "b", "selection", self.manifest, self.item, self.page)

    @unittest.skipIf(runner is None, "Install runners/modal/requirements.txt for cloud worker tests")
    def test_failure_commits_selection_and_resume_does_not_generate_it_again(self):
        selection = self.stage("selection")
        translated = self.stage("translation", selection["result"])
        sid = self.item["source"]["source_id"]
        with (tempfile.TemporaryDirectory() as tmp,
              patch.object(runner, "REMOTE", Path(tmp)),
              patch.object(runner, "results_volume", Mock()), patch.object(runner, "cache_volume", Mock()),
              patch.object(runner, "version", return_value="test"),
              patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=Mock(return_value=Mock()))}),
              patch.object(runner, "generate_stage", side_effect=[selection, {**translated, "finish_reason": "length"}])):
            with self.assertRaisesRegex(RuntimeError, "Invalid b/translation"):
                runner.run.local(self.manifest, self.items, self.pages, 1)
            path = Path(tmp) / self.config["run_name"]
            saved = protocol.read_saved(path, self.manifest, self.items, self.pages)
            self.assertEqual(set(saved["selection"]), {sid})
            self.assertEqual(saved["translation"], {})
            self.assertEqual(len(list((path / "failures/translation").glob("*.json"))), 1)
            with patch.object(runner, "generate_stage", return_value=translated) as generate:
                resumed = runner.run.local(self.manifest, self.items, self.pages, 1)
                self.assertEqual(generate.call_count, 1)
                self.assertEqual(set(resumed["translation"]), {sid})

    def test_incomplete_judgments_cannot_unmask_mapping(self):
        with (tempfile.TemporaryDirectory() as tmp, patch.object(protocol, "OUTPUT", Path(tmp)),
              patch.object(protocol, "load_freeze"), patch.object(protocol, "FREEZE", Path(tmp) / "freeze.json")):
            protocol.FREEZE.write_text("{}")
            directory = Path(tmp) / "assessment"
            directory.mkdir()
            blinded = [{"candidates": [{"case_id": "E1"}]}]
            contexts = {"56": "Bavarian context"}
            manifest = {"evaluation_freeze_sha256": protocol.sha(protocol.FREEZE),
                        "blinded_items_sha256": protocol.a.digest(blinded), "contexts_sha256": protocol.a.digest(contexts)}
            for name, value in (("run_manifest", manifest), ("blinded_items", blinded),
                                ("bavarian_contexts", contexts), ("judgments", [])):
                protocol.a.atomic_json(directory / (name + ".json"), value)
            # No private mapping exists: a missing-judgment error must precede
            # any attempt to read it or calculate method scores.
            with self.assertRaisesRegex(ValueError, "Incomplete judgments"):
                protocol.score()

    def test_scoring_exports_original_text_and_keeps_abstentions_in_yield(self):
        with (tempfile.TemporaryDirectory() as tmp, patch.object(protocol, "OUTPUT", Path(tmp)),
              patch.object(protocol, "load_freeze"), patch.object(protocol, "FREEZE", Path(tmp) / "freeze.json")):
            protocol.FREEZE.write_text("{}")
            rows = [{"output_id": method, "method": method, "source": self.item["source"],
                "target_text": text, "candidate_rank": None, "technical_status": "pass" if text else "no_target",
                "target_kind": "generated_translation"} for method, text in (
                    ("blind_translation", "Minor error"), ("grounded_b", "Faithful"), ("rerank_only", None))]
            blinded, mapping = protocol.prepare_items(rows)
            contexts = {str(self.item["source"]["bar_page_id"]): "Context"}
            judgments = [{"case_id": candidate["case_id"], "score": 4 if candidate["english"] == "Faithful" else 3,
                "issues": [] if candidate["english"] == "Faithful" else ["minor"], "note": "Test fixture"}
                for group in blinded for candidate in group["candidates"]]
            manifest = {"evaluation_freeze_sha256": protocol.sha(protocol.FREEZE),
                "blinded_items_sha256": protocol.a.digest(blinded), "contexts_sha256": protocol.a.digest(contexts),
                "method_key_sha256": protocol.a.digest(mapping), "technical_outputs_sha256": protocol.a.digest(rows)}
            directory = Path(tmp) / "assessment"
            for name, value in (("run_manifest", manifest), ("blinded_items", blinded),
                                ("bavarian_contexts", contexts), ("judgments", judgments), ("private/method_key", mapping)):
                protocol.a.atomic_json(directory / (name + ".json"), value)
            protocol.a.atomic_json(Path(tmp) / "quality/report.json", {"outputs": rows})
            with patch("builtins.print"):
                protocol.score()
            summary = protocol.read(directory / "summary.json")
            self.assertEqual(summary["methods"]["rerank_only"]["missing"], 1)
            self.assertEqual(summary["methods"]["grounded_b"]["strict_yield"], 1.0)
            self.assertEqual(summary["paired_vs_blind"]["grounded_b"], {"higher": 1})
            self.assertIn(self.item["source"]["bar_text"], (directory / "comparison.csv").read_text())


if __name__ == "__main__":
    unittest.main()
