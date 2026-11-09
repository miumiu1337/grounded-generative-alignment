import copy
import unittest
from runners.prepare_nllb_mining import mined_row


class MiningTrainingTests(unittest.TestCase):
    def fixture(self):
        source={'source_id':'bar_1:S1','bar_page_id':1,'bar_revision_id':2,
                'bar_title':'Topic','bar_text':'Original Bavarian.'}
        original={'source_id':source['source_id'],'bavarian':source['bar_text'],
                  'english':'Saved blind draft.','blind_checkpoint':'blind/path','blind_record_sha256':'blind-hash'}
        candidate={'sentence_ids':['en_1:S1'],'text':'Verbatim English evidence.'}
        bundle={'sources':[source],'page':{'candidates':[candidate]}}
        record={'source':source,'acceptance_status':'unthresholded','candidate_count':1,
                'reciprocal_top1':False,'candidates':[dict(candidate,rank=1,cosine=0.6,margin_ratio=1.1)]}
        return original,record,bundle

    def test_training_input_is_verbatim_top_one_and_target_is_unchanged(self):
        original,record,bundle=self.fixture()
        row=mined_row(original,record,bundle,'laser','ranking.json.gz','hash')
        self.assertEqual(row['english'],'Verbatim English evidence.')
        self.assertEqual(row['bavarian'],original['bavarian'])
        self.assertEqual(row['source_id'],original['source_id'])
        self.assertNotIn('blind_record_sha256',row)
        self.assertNotIn('blind_checkpoint',row)
        self.assertFalse(row['reciprocal_top1'])

    def test_changed_sources_and_english_spans_are_rejected(self):
        original,record,bundle=self.fixture()
        for field in ('source','candidate'):
            changed=copy.deepcopy(record)
            if field=='source':changed['source']['bar_text']='Modified source.'
            else:changed['candidates'][0]['text']='Rewritten evidence.'
            with self.assertRaises(ValueError):mined_row(original,changed,bundle,'bge_m3','ranking','hash')

    def test_missing_candidates_and_nonfinite_scores_are_rejected(self):
        original,record,bundle=self.fixture()
        for variant in ('missing','nan'):
            changed=copy.deepcopy(record)
            if variant=='missing':changed['candidates']=[]
            else:changed['candidates'][0]['cosine']=float('nan')
            with self.assertRaises(ValueError):mined_row(original,changed,bundle,'laser','ranking','hash')


if __name__=='__main__':unittest.main()
