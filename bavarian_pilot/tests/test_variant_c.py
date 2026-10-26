import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import variant_c as p
try:
    import modal_variant_c as runner
except ModuleNotFoundError as error:
    if error.name != "modal":
        raise
    runner = None


class VariantCTests(unittest.TestCase):
    def setUp(self):
        config = json.loads(p.CONFIG.read_text())
        self.pages = {str(page): {"bar": "BAVARIAN-CONTEXT", "en": "Topic (en), revision 456\nEN-WIKI-CONTEXT"}
                      for page in (2544, 59)}
        self.items = []
        for page, count in ((2544, 39), (59, 14)):
            for n in range(1, count + 1):
                source = {"source_id": f"bar_{page}:v1.0.1:S{n:04d}", "bar_text": "Original Bavarian.",
                          "bar_title": str(page), "bar_page_id": page, "bar_revision_id": 123}
                item = {"source": source, "blind": {"translation": "SECRET-DRAFT", "run_id": config["blind_run_id"],
                        "prompt_tokens": 10, "output_tokens": 5}}
                item["search"] = {"source_id": source["source_id"], "query": p.query_for(item, self.pages[str(page)]),
                    "provider": "codex_web_tool", "searched_at": "2026-10-25T08:00:00+00:00", "status": "ok",
                    "sources": [{"reference_id": "WEB1", "url": "https://example.org/source", "title": "Example source",
                        "text": "A short reference excerpt.", "retrieved_at": "2026-10-25T08:00:01+00:00",
                        "provider_ref": "test-view", "evidence_kind": "page_excerpt"}], "failures": [], "search_ref": "test-search"}
                self.items.append(item)
        self.item, self.page = self.items[0], self.pages["2544"]
        self.manifest = {"experiment": config, "prompts": {"refinement": "Refine.", "verification": "Verify."},
                         "inputs_sha256": p.digest({"items": self.items, "pages": self.pages})}

    def refined(self):
        return {"source_id": self.item["source"]["source_id"], "translation": "Refined translation.", "used_reference_ids": ["WEB1"]}

    def stage(self, stage, refinement=None):
        result = self.refined() if stage == "refinement" else {
            "source_id": self.item["source"]["source_id"], "verdict": "pass", "issues": []}
        return {"run_id": p.digest(self.manifest), "source_id": self.item["source"]["source_id"], "stage": stage,
            "messages": p.stage_messages(stage, self.manifest, self.item, self.page, refinement),
            "schema": p.schema_for(stage, self.item), "grammar": None, "raw_response": json.dumps(result), "result": result,
            "finish_reason": "stop", "prompt_tokens": 100, "output_tokens": 20,
            "runtime": {"vllm": "0.21.0", "transformers": "5.5.4"}, "recorded_at": "2026-10-25T09:00:00+00:00"}

    def test_refinement_has_draft_wiki_and_external_evidence_verifier_excludes_draft_and_english_wiki(self):
        refine = json.loads(p.stage_messages("refinement", self.manifest, self.item, self.page)[1]["content"])
        verify = json.loads(p.stage_messages("verification", self.manifest, self.item, self.page, self.refined())[1]["content"])
        self.assertEqual(refine["blind_translation"], "SECRET-DRAFT")
        self.assertIn("EN-WIKI-CONTEXT", refine["english_wikipedia_page"])
        self.assertEqual(refine["external_search"], self.item["search"])
        self.assertNotIn("SECRET-DRAFT", json.dumps(verify))
        self.assertNotIn("EN-WIKI-CONTEXT", json.dumps(verify))
        self.assertEqual(verify["focus_source"], self.item["source"])

    def test_wikimedia_private_and_credential_urls_rejected(self):
        for url in ("https://en.wikipedia.org/a", "https://commons.wikimedia.org/a", "https://wikipedia.org/a",
                    "https://127.0.0.1/a", "https://localhost/a", "http://example.org/a", "https://a:b@example.org/a"):
            with self.assertRaises(ValueError):
                p.external_url(url)

    def test_search_budget_identity_timestamps_and_status_checked(self):
        policy = self.manifest["experiment"]["search_policy"]
        p.validate_search(self.item["search"], self.item, self.page, policy)
        for changes in ({"query": "changed"}, {"searched_at": "2026-10-25T08:00:00"},
                        {"status": "no_results"}, {"sources": self.item["search"]["sources"] * 3}):
            with self.assertRaises(ValueError):
                p.validate_search({**self.item["search"], **changes}, self.item, self.page, policy)
        changed = copy.deepcopy(self.item["search"])
        changed["sources"][0]["text"] = "word " * 26
        with self.assertRaises(ValueError):
            p.validate_search(changed, self.item, self.page, policy)

    def test_tracking_variants_are_one_reference(self):
        self.assertEqual(p.external_url("https://example.org/source?trk=one#part"),
                         p.external_url("https://example.org/source?utm_source=two"))
        changed = copy.deepcopy(self.item["search"])
        changed["sources"].append({**changed["sources"][0], "reference_id": "WEB2",
                                   "url": "https://example.org/source?trk=duplicate"})
        with self.assertRaisesRegex(ValueError, "Repeated"):
            p.validate_search(changed, self.item, self.page, self.manifest["experiment"]["search_policy"])

    def test_citation_must_exist_once_and_none_is_explicit(self):
        for ids in (["invented"], ["WEB1", "WEB1"]):
            with self.assertRaises(ValueError):
                p.validate_refinement({**self.refined(), "used_reference_ids": ids}, self.item)
        no_results = {**self.item["search"], "sources": [], "status": "no_results"}
        p.validate_search(no_results, self.item, self.page, self.manifest["experiment"]["search_policy"])
        p.validate_refinement({**self.refined(), "used_reference_ids": []}, self.item)

    def test_self_check_requires_issues_when_flagged_and_has_no_rewrite_field(self):
        good = {"source_id": self.item["source"]["source_id"], "verdict": "pass", "issues": []}
        p.validate_verification(good, self.item)
        for changed in ({**good, "verdict": "flag"}, {**good, "translation": "rewritten"},
                        {**good, "issues": [{"kind": "addition", "description": "Extra claim"}]}):
            with self.assertRaises(ValueError):
                p.validate_verification(changed, self.item)

    def test_edit_log_reconstructs_final_text(self):
        for draft, final in (("A short draft.", "A refined draft!"), ("same", "same"), ("abc", "xyz more")):
            reconstructed = draft
            for edit in reversed(p.edits_for(draft, final)):
                self.assertEqual(draft[edit["draft_start"]:edit["draft_end"]], edit["old"])
                reconstructed = reconstructed[:edit["draft_start"]] + edit["new"] + reconstructed[edit["draft_end"]:]
            self.assertEqual(reconstructed, final)

    @unittest.skipIf(runner is None, "Install existing requirements-modal.txt for worker tests")
    def test_failed_verification_preserves_refinement_and_resume_reuses_it(self):
        refined, verified = self.stage("refinement"), self.stage("verification", self.refined())
        with (tempfile.TemporaryDirectory() as temporary,
              patch.object(runner, "REMOTE", Path(temporary)),
              patch.object(runner, "results_volume", Mock()), patch.object(runner, "cache_volume", Mock()),
              patch.object(runner, "version", return_value="test"),
              patch.dict("sys.modules", {"vllm": SimpleNamespace(LLM=Mock(return_value=Mock()))}),
              patch.object(runner, "generate_stage", side_effect=[refined, {**verified, "finish_reason": "length"}])):
            with self.assertRaisesRegex(RuntimeError, "Invalid verification"):
                runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            directory = Path(temporary) / self.manifest["experiment"]["run_name"]
            saved = p.read_saved(directory, self.manifest, self.items, self.pages)
            self.assertEqual(len(saved["refinements"]), 1)
            self.assertFalse(saved["results"])
            with patch.object(runner, "generate_stage", return_value=verified) as generate:
                saved = runner.run_variant.local(self.manifest, self.items, self.pages, 1)
            self.assertEqual(generate.call_count, 1)
            self.assertEqual(len(saved["results"]), 1)
            with patch.object(p, "ROOT", Path(temporary) / "local"):
                out, checked = p.write_local(saved, self.manifest, self.items, self.pages)
                summary = p.export_results(out, checked, self.manifest)
            self.assertEqual(summary["selfcheck_verdicts"], {"pass": 1})
            self.assertEqual(summary["new_prompt_tokens"], 200)
            changed = copy.deepcopy(self.manifest)
            changed["prompts"]["refinement"] += " changed"
            with self.assertRaises(ValueError):
                p.read_saved(directory, changed, self.items, self.pages)


if __name__ == "__main__":
    unittest.main()
