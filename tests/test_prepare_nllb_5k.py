import copy
import unittest

from runners.prepare_nllb_5k import select_subset


class ParameterSubsetTests(unittest.TestCase):
    def test_subset_selection_preserves_texts_and_parent_order(self):
        rows = [{'source_id': str(i), 'english': f'English {i}', 'bavarian': f'Bavarian {i}'} for i in range(20)]
        original = copy.deepcopy(rows)
        selected = select_subset(rows, 5)
        selected_ids = {r['source_id'] for r in selected}
        self.assertEqual(len(selected_ids), 5)
        self.assertEqual(selected, [r for r in rows if r['source_id'] in selected_ids])
        self.assertEqual(rows, original)

    def test_membership_is_independent_of_input_order(self):
        rows = [{'source_id': str(i)} for i in range(20)]
        self.assertEqual({r['source_id'] for r in select_subset(rows, 5)},
                         {r['source_id'] for r in select_subset(rows[::-1], 5)})

    def test_invalid_counts_and_duplicate_source_ids_are_rejected(self):
        rows = [{'source_id': str(i)} for i in range(10)]
        for count in (0, 11):
            with self.assertRaises(ValueError):
                select_subset(rows, count)
        with self.assertRaises(ValueError):
            select_subset(rows + rows[:1], 5)


if __name__ == '__main__':
    unittest.main()
