import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

try:
    from runners.modal import blind as runner
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    runner = None


@unittest.skipIf(runner is None, "Install runners/modal/requirements.txt for Modal-specific tests")
class ModalTranslationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest, cls.sources = runner.local_run()

    def record(self, source):
        return {
            "condition": "blind_translation", "provider": "modal_self_hosted", "model": runner.MODEL,
            "model_revision": runner.REVISION, "run_id": runner.digest(self.manifest), "source": source,
            "messages": runner.messages_for(source, self.manifest["prompt"]),
            "generation": self.manifest["experiment"]["generation"],
            "translation": "Example translation.", "finish_reason": "stop", "prompt_tokens": 20, "output_tokens": 3,
        }

    def test_only_source_and_prompt_enter_model_context(self):
        source = {**self.sources[0], "english_article": "UNSEEN-ENGLISH-CONTEXT"}
        messages = runner.messages_for(source, self.manifest["prompt"])
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(messages[1]["content"], "Bavarian text:\n" + source["bar_text"])
        self.assertNotIn("UNSEEN-ENGLISH-CONTEXT", json.dumps(messages))

    def test_atomic_saved_result_resumes_and_protocol_changes_fail(self):
        source = self.sources[0]
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            runner.atomic_json(directory / "run_manifest.json", self.manifest)
            record = self.record(source)
            path = runner.record_path(directory, source["source_id"])
            runner.atomic_json(path, record)
            self.assertEqual(list(runner.read_saved(directory, self.manifest, self.sources)), [source["source_id"]])
            with self.assertRaises(ValueError):
                runner.atomic_json(path, {**record, "translation": "Changed."})
            changed = copy.deepcopy(self.manifest)
            changed["prompt"] += " Changed."
            with self.assertRaises(ValueError):
                runner.read_saved(directory, changed, self.sources)

    def test_remote_record_validated_before_any_local_write(self):
        source = self.sources[0]
        record = {**self.record(source), "model_revision": "wrong-revision"}
        with tempfile.TemporaryDirectory() as directory, patch.object(runner, "ROOT", Path(directory)):
            with self.assertRaises(ValueError):
                runner.write_local({source["source_id"]: record}, self.manifest, self.sources)
            self.assertFalse((Path(directory) / "data/translations").exists())

    def test_revision_and_source_fingerprints_are_checked(self):
        changed = copy.deepcopy(self.manifest)
        changed["experiment"]["model_revision"] = "changed"
        with self.assertRaises(ValueError):
            runner.validate_run(changed, self.sources)
        changed_sources = copy.deepcopy(self.sources)
        changed_sources[0]["bar_text"] += " Changed."
        with self.assertRaises(ValueError):
            runner.validate_run(self.manifest, changed_sources)

    def test_completed_result_committed_before_later_truncation(self):
        good = SimpleNamespace(outputs=[SimpleNamespace(text="A translation.", finish_reason="stop", token_ids=[1, 2])],
                               prompt_token_ids=[1, 2, 3])
        bad = SimpleNamespace(outputs=[SimpleNamespace(text="Truncated", finish_reason="length", token_ids=[1])],
                              prompt_token_ids=[1, 2, 3])
        engine = Mock()
        engine.generate.side_effect = [[good], [bad]]
        engine.get_tokenizer.return_value.apply_chat_template.return_value = "formatted"
        fake_vllm = SimpleNamespace(LLM=Mock(return_value=engine), SamplingParams=Mock())
        results_volume, cache_volume = Mock(), Mock()
        with (tempfile.TemporaryDirectory() as directory,
              patch.object(runner, "REMOTE_ROOT", Path(directory)),
              patch.object(runner, "results_volume", results_volume),
              patch.object(runner, "cache_volume", cache_volume),
              patch.object(runner, "version", return_value="test-version"),
              patch.dict("sys.modules", {"vllm": fake_vllm}), patch("builtins.print")):
            with self.assertRaisesRegex(RuntimeError, "Incomplete response"):
                runner.translate.local(self.manifest, self.sources, 2)
            output = Path(directory) / self.manifest["experiment"]["run_name"]
            saved = runner.read_saved(output, self.manifest, self.sources)
            self.assertEqual(list(saved), [self.sources[0]["source_id"]])
            self.assertEqual(results_volume.commit.call_count, 2)
            self.assertEqual(cache_volume.commit.call_count, 1)


if __name__ == "__main__":
    unittest.main()
