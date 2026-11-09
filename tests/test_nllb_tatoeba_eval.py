import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from runners import nllb_tatoeba_eval as evaluation


class TatoebaEvaluationTests(unittest.TestCase):
    def test_existing_blind_conditions_are_unchanged(self):
        self.assertEqual(evaluation.evaluation_conditions({}),
                         ('blind_finetuned',['blind_finetuned','unadapted_initialized_control']))

    def test_other_models_have_distinct_labels_and_can_reuse_existing_control(self):
        for name in ('laser','variant_a','variant_b'):
            condition=name+'_finetuned'
            self.assertEqual(evaluation.evaluation_conditions(
                {'finetuned_condition':condition,'conditions':[condition]}),(condition,[condition]))
        with self.assertRaises(ValueError):
            evaluation.evaluation_conditions({'finetuned_condition':'laser_finetuned',
                                             'conditions':['blind_finetuned']})

    def fixture(self):
        examples = [{'example_id': 'a', 'bavarian_reference': 'This is a sufficiently long example.'},
                    {'example_id': 'b', 'bavarian_reference': 'A second example contains different words.'}]
        rows = [{'example_id': row['example_id'], 'prediction': row['bavarian_reference'],
                 'length_capped': False} for row in examples]
        return examples, rows

    def test_perfect_predictions_score_100_with_recorded_signatures(self):
        examples, rows = self.fixture()
        result = evaluation.score_predictions(examples, rows)
        self.assertAlmostEqual(result['BLEU'], 100)
        self.assertAlmostEqual(result['chrF++'], 100)
        self.assertIn('version:2.5.1', result['BLEU_signature'])
        self.assertIn('nw:2', result['chrF++_signature'])

    def test_missing_and_reordered_predictions_are_rejected(self):
        examples, rows = self.fixture()
        for changed in [rows[:1], list(reversed(rows))]:
            with self.assertRaisesRegex(ValueError, 'full test set'):
                evaluation.score_predictions(examples, changed)

    def test_empty_and_capped_outputs_are_counted_not_dropped(self):
        examples, rows = self.fixture()
        rows[0]['prediction'] = ''; rows[1]['length_capped'] = True
        result = evaluation.score_predictions(examples, rows)
        self.assertEqual(result['examples'], 2)
        self.assertEqual(result['empty_predictions'], 1)
        self.assertEqual(result['length_capped_predictions'], 1)

    def test_benchmark_tampering_fails_before_inference(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'evaluate.py').write_text('fixture runner')
            rows = [{'example_id': f'test_{i:03d}', 'english': 'English source',
                     'bavarian_reference': 'Bavarian reference'} for i in range(98)]
            (root / 'test.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            protocol = {'runner_sha256': evaluation.sha(root / 'evaluate.py'),
                        'test_sha256': evaluation.sha(root / 'test.jsonl')}
            protocol['evaluation_id'] = evaluation.digest(protocol)
            (root / 'protocol.json').write_text(json.dumps(protocol))
            evaluation.validate_inputs(root)
            (root / 'test.jsonl').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Test-set checksum'):
                evaluation.validate_inputs(root)


if __name__ == '__main__':
    unittest.main()
