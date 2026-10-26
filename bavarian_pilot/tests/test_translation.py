import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests

from download_articles import save_json
from translate_blind import (
    MODEL, body_for, call_api, digest_json, load_api_key, load_experiment,
    load_saved, load_sources, main, parse_response, result_path,
)


def response():
    return {
        "candidates": [{"finishReason": "STOP", "content": {"parts": [
            {"thought": True, "text": "Internal reasoning"}, {"text": "Munich lies on the Isar."},
        ]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 10, "totalTokenCount": 90},
        "modelVersion": MODEL, "responseId": "example-response",
    }


class TranslationTests(unittest.TestCase):
    def setUp(self):
        self.config, self.prompt = load_experiment()
        self.source = {"source_id": "bar_59:v1.0.1:S0001", "bar_text": "Minga liegt an da Isar.",
                       "bar_title": "Example", "bar_page_id": 59, "bar_revision_id": 1}

    def test_only_bavarian_text_and_instruction_enter_request(self):
        source = {**self.source, "english_article": "ENGLISH-CONTEXT-SENTINEL", "evidence_ids": ["E1"]}
        body = body_for(source, self.config, self.prompt)
        self.assertEqual(body["contents"], [{"role": "user", "parts": [
            {"text": "Bavarian text:\nMinga liegt an da Isar."}]}])
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], self.prompt)
        self.assertNotIn("ENGLISH-CONTEXT-SENTINEL", json.dumps(body))
        self.assertNotIn("tools", body)

    def test_frozen_development_sources_only(self):
        sources = load_sources(self.config)
        self.assertEqual(len(sources), 53)
        self.assertEqual({s["bar_page_id"] for s in sources}, {59, 2544})
        self.assertTrue(all(":v1.0.1:" in s["source_id"] for s in sources))
        self.assertEqual(len({s["source_id"] for s in sources}), 53)

    def test_changed_freeze_rejected(self):
        config = {**self.config, "preprocessing_freeze_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "Frozen file changed"):
            load_sources(config)

    def test_parse_excludes_reasoning_and_records_usage(self):
        result = parse_response(response())
        self.assertEqual(result["translation"], "Munich lies on the Isar.")
        self.assertEqual(result["total_tokens"], 90)
        self.assertEqual(result["thought_tokens"], 0)
        self.assertNotIn("Internal reasoning", json.dumps(result))

    def test_malformed_truncated_blocked_and_wrong_model_results_fail(self):
        invalid = []
        for mutate in (
            lambda r: r.update(candidates=[]),
            lambda r: r["candidates"][0].update(finishReason="MAX_TOKENS"),
            lambda r: r.update(modelVersion="another-model"),
            lambda r: r["usageMetadata"].update(promptTokenCount=True),
            lambda r: r["usageMetadata"].update(totalTokenCount=1),
            lambda r: r["candidates"][0]["content"].update(parts=[{"text": " "}]),
            lambda r: r["candidates"][0]["content"].update(parts=[{"functionCall": {"name": "search"}}]),
            lambda r: r.update(usageMetadata=None),
        ):
            item = response()
            mutate(item)
            invalid.append(item)
        for item in invalid:
            with self.subTest(item=item), self.assertRaises(ValueError):
                parse_response(item)

    def test_api_timeout_and_errors_do_not_expose_key_or_retry(self):
        key = "secret-key-marker"
        session = Mock()
        session.post.side_effect = requests.Timeout(key)
        with self.assertRaises(RuntimeError) as error:
            call_api(session, key, {})
        self.assertNotIn(key, str(error.exception))
        session.post.assert_called_once()
        session.post.assert_called_with(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            headers={"x-goog-api-key": key}, json={}, timeout=(10, 180),
        )
        session.post.side_effect = None
        session.post.return_value.status_code = 429
        with self.assertRaisesRegex(RuntimeError, "HTTP 429") as error:
            call_api(session, key, {})
        self.assertNotIn(key, str(error.exception))
        session.post.return_value.json.assert_not_called()

    def test_malformed_api_json_surfaces(self):
        session = Mock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.side_effect = ValueError("raw response must not leak")
        with self.assertRaisesRegex(RuntimeError, "invalid JSON") as error:
            call_api(session, "secret", {})
        self.assertNotIn("raw response must not leak", str(error.exception))

    def test_saved_translations_resume_only_for_identical_source_and_settings(self):
        run_id = digest_json({"run": "example"})
        record = {"condition": "blind_translation", "source": self.source, "run_id": run_id,
                  "request": body_for(self.source, self.config, self.prompt), "result": parse_response(response())}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            save_json(result_path(directory, self.source["source_id"]), record)
            saved = load_saved(directory, [self.source], run_id, self.config, self.prompt)
            self.assertEqual(list(saved), [self.source["source_id"]])
            with self.assertRaises(ValueError):
                load_saved(directory, [self.source], "changed-run", self.config, self.prompt)
            changed_source = {**self.source, "bar_text": "A different sentence."}
            with self.assertRaises(ValueError):
                load_saved(directory, [changed_source], run_id, self.config, self.prompt)

    def test_api_key_loaded_without_shell_expansion(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True), patch("translate_blind.ROOT", Path(directory)):
            with self.assertRaisesRegex(ValueError, "Add GEMINI_API_KEY"):
                load_api_key()
            path = Path(directory) / ".env"
            path.write_text('GEMINI_API_KEY="example-api-key"\n')
            self.assertEqual(load_api_key(), "example-api-key")
            path.write_text('GEMINI_API_KEY=$(malicious-command)\n')
            with self.assertRaisesRegex(ValueError, "invalid characters") as error:
                load_api_key()
            self.assertNotIn("malicious-command", str(error.exception))

    def test_plan_makes_no_calls_and_execution_resumes_without_repeating_calls(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            prompt_path = root / self.config["prompt_file"]
            prompt_path.parent.mkdir(parents=True)
            prompt_path.write_text(self.prompt, encoding="utf-8")
            experiment = root / "experiment.json"
            experiment.write_text(json.dumps(self.config))
            key = "secret-key-marker"
            with (patch("translate_blind.ROOT", root), patch("translate_blind.EXPERIMENT", experiment),
                  patch("translate_blind.load_experiment", return_value=(self.config, self.prompt)),
                  patch("translate_blind.load_sources", return_value=[self.source]),
                  patch("translate_blind.load_api_key", return_value=key) as get_key,
                  patch("translate_blind.call_api", return_value=parse_response(response())) as api,
                  patch("builtins.print")):
                with patch("sys.argv", ["translate_blind.py"]):
                    main()
                get_key.assert_not_called()
                api.assert_not_called()
                self.assertFalse((root / "data/translations").exists())
                with patch("sys.argv", ["translate_blind.py", "--execute", "--interval-seconds", "0"]):
                    main()
                    main()
                api.assert_called_once()
                get_key.assert_called_once()
                output = root / "data/translations" / self.config["run_name"]
                record = json.loads(result_path(output, self.source["source_id"]).read_text())
                self.assertEqual(record["source"], self.source)
                self.assertEqual(record["result"]["translation"], "Munich lies on the Isar.")
                self.assertEqual(record["quality_status"], "unassessed")
                for path in output.glob("*.json"):
                    self.assertNotIn(key, path.read_text())


if __name__ == "__main__":
    unittest.main()
