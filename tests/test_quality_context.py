import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from runners import export_quality_context as export


class QualityContextTests(unittest.TestCase):
    def test_repeated_context_round_trip_has_plain_string_schema(self):
        rows = [{'source_text':'A sentence.', 'article':'An article. '*100,
                 'english_article':'An English article.', 'paragraph':'A sentence.',
                 'legacy_source_id':None, 'heading_path':[]} for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'data.parquet'
            export.parquet(path, rows)
            self.assertEqual(pq.read_schema(path).field('article').type, pa.string())
            self.assertEqual(pq.read_schema(path).field('legacy_source_id').type, pa.string())
            self.assertEqual(pq.read_table(path).to_pylist(),rows)

    def test_changed_boundaries_never_inherit_a_revised_source_id(self):
        text='A complete original sentence.'
        common={'text':text,'title':'Title','url':'https://example.org','page_id':1,
                'revision_id':2,'text_sha256':'hash','wikidata_id':'Q1'}
        value={'bar':common,'en':common,'legacy_provenance':{'old':{'source_start':2,'source_end':10}},
               'blocks':[{'start':0,'end':len(text)}], 'spans':{},}
        value['bar']={**common,'excluded_blocks':[]}
        row={'page_id':1,'source_id':'old','bavarian':text[2:10],'english':'Old fragment.'}
        with patch.object(export,'article',return_value=value):
            result=export.enrich(row)
        self.assertEqual(result['source_text'],row['bavarian'])
        self.assertEqual(result['target_text'],row['english'])
        self.assertFalse(result['preprocessing_v5_eligible'])
        self.assertEqual(result['preprocessing_v5_reason'],'sentence_boundaries_changed')
        self.assertIsNone(result['source_id_v5'])

    def test_mismatched_original_text_is_rejected(self):
        value={'bar':{'text':'Original'},'legacy_provenance':{'old':{'source_start':0,'source_end':8}}}
        with patch.object(export,'article',return_value=value):
            with self.assertRaisesRegex(ValueError,'text/offset mismatch'):
                export.enrich({'page_id':1,'source_id':'old','bavarian':'Modified'})


if __name__=='__main__':
    unittest.main()
