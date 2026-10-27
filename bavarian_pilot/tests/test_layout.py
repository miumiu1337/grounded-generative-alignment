import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import blind_records
import evaluation_pipeline


class LayoutTests(unittest.TestCase):
    def test_offline_modules_work_without_modal(self):
        script = """
import importlib.abc
import sys
class NoModal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'modal' or fullname.startswith('modal.'):
            raise ModuleNotFoundError('Modal is unavailable', name='modal')
sys.meta_path.insert(0, NoModal())
import blind_records, output_quality, semantic_evaluation, evaluation_pipeline
import variant_a, variant_b, variant_c
manifest, sources = blind_records.local_run()
assert len(sources) == 53
assert 'modal' not in sys.modules
"""
        result = subprocess.run(
            [sys.executable, "-B", "-c", script], cwd=evaluation_pipeline.ROOT,
            capture_output=True, text=True, timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_saved_manifest_still_rejects_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "run_manifest.json").write_text(json.dumps({"inputs": "original"}))
            with self.assertRaisesRegex(ValueError, "inputs or settings changed"):
                blind_records.recorded_manifest(directory, {"inputs": "changed"})

    def test_changed_runner_cannot_resume_a_historical_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "runner.py"
            path.write_text("# reorganized runner\n")
            with self.assertRaisesRegex(ValueError, "original experiment code"):
                blind_records.require_original_runner({"runner_sha256": "old"}, path)

    def test_read_only_freeze_allows_code_layout_changes(self):
        frozen = {"files_sha256": {"old.py": "old"}, "methods": {"b": {"seed": 42}}}
        current = {**frozen, "files_sha256": {"runners/modal/b.py": "new"}}
        with patch.object(evaluation_pipeline, "read", return_value=frozen), \
                patch.object(evaluation_pipeline, "sha", return_value=evaluation_pipeline.FROZEN_SHA256), \
                patch.object(evaluation_pipeline, "frozen_protocol", return_value=current):
            self.assertEqual(evaluation_pipeline.load_freeze(read_only=True), frozen)
            with self.assertRaisesRegex(ValueError, "original inference"):
                evaluation_pipeline.load_freeze()

    def test_read_only_freeze_rejects_changed_scientific_settings(self):
        frozen = {"files_sha256": {}, "methods": {"b": {"seed": 42}}}
        current = copy.deepcopy(frozen)
        current["methods"]["b"]["seed"] = 7
        with patch.object(evaluation_pipeline, "read", return_value=frozen), \
                patch.object(evaluation_pipeline, "sha", return_value=evaluation_pipeline.FROZEN_SHA256), \
                patch.object(evaluation_pipeline, "frozen_protocol", return_value=current):
            with self.assertRaisesRegex(ValueError, "settings/code/inputs changed"):
                evaluation_pipeline.load_freeze(read_only=True)

    def test_read_only_freeze_rejects_a_modified_original_record(self):
        frozen = {"files_sha256": {}}
        with patch.object(evaluation_pipeline, "read", return_value=frozen), \
                patch.object(evaluation_pipeline, "sha", return_value="changed"), \
                patch.object(evaluation_pipeline, "frozen_protocol", return_value=frozen):
            with self.assertRaisesRegex(ValueError, "original evaluation freeze changed"):
                evaluation_pipeline.load_freeze(read_only=True)
