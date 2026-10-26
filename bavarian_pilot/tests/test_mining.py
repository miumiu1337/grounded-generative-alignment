import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

try:
    import numpy as np
    import mine_baselines as mining
except ModuleNotFoundError as error:
    if error.name != "numpy":
        raise
    mining = None


@unittest.skipIf(mining is None, "Install requirements-mining.txt in the mining environment")
class MiningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(mining.EXPERIMENT.read_text())

    def vectors(self, count=4):
        return np.eye(1024, dtype=np.float32)[:count]

    def test_same_53_sources_and_only_linked_development_candidates(self):
        groups = mining.load_inputs(self.config)
        self.assertEqual([(g["bar_page_id"], len(g["sources"]), len(g["targets"])) for g in groups],
                         [(2544, 39, 327), (59, 14, 1012)])
        for group in groups:
            prefix = f"en_{group['en_page_id']}:v1.0.1:"
            self.assertTrue(all(sentence.startswith(prefix) for target in group["targets"]
                                for sentence in target["sentence_ids"]))

    def test_ratio_formula_and_invalid_vectors(self):
        vectors = self.vectors()
        margin, cosine = mining.rank(vectors * 2, vectors * 7)
        np.testing.assert_allclose(cosine, np.eye(4))
        np.testing.assert_allclose(margin, np.eye(4) * 4)
        for bad in (np.zeros((4, 1024)), np.full((4, 1024), np.nan), np.ones((4, 3))):
            with self.assertRaises(ValueError):
                mining.rank(bad, vectors)
        with self.assertRaises(ValueError):
            mining.rank(vectors, -vectors - 1)

    def test_successful_batch_survives_later_error_and_resumes(self):
        texts = ["a", "b", "c", "d"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npz"
            embed = Mock(side_effect=[self.vectors(2), RuntimeError("dependency down")])
            with self.assertRaisesRegex(RuntimeError, "dependency down"):
                mining.cached_embeddings(path, texts, "run", 2, embed)
            resume = Mock(return_value=self.vectors(2))
            actual = mining.cached_embeddings(path, texts, "run", 2, resume)
            resume.assert_called_once_with(["c", "d"])
            self.assertEqual(actual.shape, (4, 1024))
            with self.assertRaises(ValueError):
                mining.cached_embeddings(path, texts, "changed", 2, resume)
            with self.assertRaises(ValueError):
                mining.cached_embeddings(path, texts + ["new"], "run", 2, resume)

    def test_bge_checkpoint_checked_and_timeout_is_propagated(self):
        bad = {"models": [{"name": self.config["bge"]["model"], "digest": "wrong"}]}
        with patch.object(mining, "ollama_json", return_value=bad):
            with self.assertRaises(ValueError):
                mining.check_bge(self.config)
        with patch.object(mining.requests, "post", side_effect=mining.requests.Timeout()):
            with self.assertRaisesRegex(RuntimeError, "completed batches are saved"):
                mining.ollama_json(self.config, "/api/embed", {"input": ["x"]})

    def test_stable_ties_preserve_ids_and_text_without_acceptance(self):
        group = {"sources": [{"source_id": str(i), "bar_text": f"Source {i}"} for i in range(4)],
                 "en_page_id": 1, "en_revision_id": 2,
                 "targets": [{"sentence_ids": [str(i)], "text": f"Target {i}"} for i in range(4)]}
        config = {**self.config, "top_k": 3}
        records, _, _ = mining.ranked_records(group, np.ones((4, 1024)), np.ones((4, 1024)), config)
        self.assertEqual([c["sentence_ids"] for c in records[0]["candidates"]], [["0"], ["1"], ["2"]])
        self.assertEqual(records[0]["candidates"][0]["text"], "Target 0")
        self.assertTrue(all(r["acceptance_status"] == "unthresholded" for r in records))

    def test_span_overlap_not_reported_as_exact_agreement(self):
        record = {"source": {"source_id": "a", "bar_text": "x", "bar_title": "topic"},
                  "candidates": [{"sentence_ids": ["e1"], "text": "y", "margin_ratio": 1.1}],
                  "reciprocal_top1": True}
        other = copy.deepcopy(record)
        other["candidates"][0]["sentence_ids"] = ["e1", "e2"]
        summary = mining.compare({"laser_1": [record], "bge_m3": [other]}, self.config)["summary"]
        self.assertEqual(summary["exact_top1_agreement"], 0)
        self.assertEqual(summary["top1_sentence_overlap"], 1)
        self.assertNotIn("precision", summary)
        other["source"]["source_id"] = "wrong"
        with self.assertRaises(ValueError):
            mining.compare({"laser_1": [record], "bge_m3": [other]}, self.config)


if __name__ == "__main__":
    unittest.main()
