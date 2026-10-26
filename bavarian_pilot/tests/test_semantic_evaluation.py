import unittest

import semantic_evaluation as evaluation


class SemanticEvaluationTests(unittest.TestCase):
    def test_deduplication_and_hidden_methods(self):
        rows = [{"source": {"source_id": "S1", "bar_text": "Source", "bar_title": "Article", "bar_page_id": 1},
                 "method": method, "target_text": text, "candidate_rank": rank,
                 "output_id": method+str(rank), "technical_status": "pass", "target_kind": "translation"}
                for method, text, rank in (("A", "Same", None), ("B", "Same", None), ("M", "Other", 1), ("M", "Lower", 2))]
        blinded, mapping = evaluation.prepare_items(rows)
        self.assertEqual(len(mapping), 3)
        self.assertEqual(len(blinded[0]["candidates"]), 2)
        self.assertEqual(mapping[0]["case_id"], mapping[1]["case_id"])
        self.assertEqual(set(blinded[0]), {"source_id", "source_text", "context_id", "candidates"})
        self.assertEqual(evaluation.prepare_items(rows), (blinded, mapping))

    def test_missing_duplicate_and_inconsistent_judgments_rejected(self):
        good = {"case_id": "E1", "score": 4, "issues": [], "note": "Faithful."}
        evaluation.validate_judgments([good], {"E1"})
        for judgments, expected in (([], {"E1"}), ([good, good], {"E1"}),
                                    ([{**good, "score": 2}], {"E1"}),
                                    ([{**good, "score": None}], {"E1"}),
                                    ([{**good, "score": 4, "issues": ["omission"]}], {"E1"})):
            with self.assertRaises(ValueError):
                evaluation.validate_judgments(judgments, expected)

    def test_missing_outputs_reduce_yield_and_uncertainty_is_not_zero_quality(self):
        rows = [{"method": "A", "case_id": cid, "technical_status": "pass"} for cid in ("E1", None, "E2")]
        judgments = {"E1": {"score": 4, "issues": []}, "E2": {"score": None, "issues": ["uncertain"]}}
        summary = evaluation.summarize(rows, judgments)["A"]
        self.assertEqual((summary["produced"], summary["missing"], summary["uncertain"]), (2, 1, 1))
        self.assertEqual(summary["mean_fidelity_0_4"], 4)
        self.assertEqual(summary["strict_yield"], .3333)


if __name__ == "__main__":
    unittest.main()
