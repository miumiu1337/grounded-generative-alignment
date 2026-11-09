import copy
import json
import unittest

from preprocessing.finewiki_pilot import digest
from parallel_data_generation import finewiki_experiments as validation, scale_protocol
from runners.prepare_nllb_variant_a import refined_row


class VariantATrainingTests(unittest.TestCase):
    def fixture(self, condition='variant_a'):
        source = {'source_id':'bar_1:S1', 'bar_text':'Original Bavarian.'}
        original = {'source_id':source['source_id'], 'bavarian':source['bar_text'],
                    'english':'Saved blind draft.'}
        page = {'bar':'Full Bavarian article.', 'en':'Full English article.',
                'candidates':[{'sentence_ids':['en_1:S1'], 'text':'English evidence.'}]}
        final_stage='refinement' if condition=='variant_a' else 'translation'
        config = {'prompts':{condition:{'selection':'Select evidence.', final_stage:'Produce English.'}}}
        selection = {'source_id':source['source_id'], 'evidence_ids':['en_1:S1'],
                     'match_status':'partial', 'contradiction':False}
        selected, refined = [], []
        for stage, result, target in [('selection',selection,selected),
                (final_stage,{'source_id':source['source_id'],'translation':'Refined English.'},refined)]:
            manifest = {'condition':condition,'stage':stage}
            dependencies={'draft':original['english']} if condition=='variant_a' else {}
            if stage==final_stage:dependencies['selection']=selection
            req = scale_protocol.request(condition,stage,source,page,config['prompts'],**dependencies)
            record = {'source':source,'run_id':digest(manifest),'model':validation.MODEL,
                      'model_revision':validation.REVISION,'condition':condition,'stage':stage,
                      'request_sha256':digest(req),'finish_reason':'stop','quality_status':'unassessed',
                      'prompt_tokens':20,'output_tokens':10,'result':result,
                      'raw_response':json.dumps(result)}
            target.extend([manifest,record,stage+'.json.gz','fixture-sha'])
        return original,source,page,config,selected,refined

    def test_saved_refinement_is_input_and_original_bavarian_is_target(self):
        args = self.fixture()
        row = refined_row(*args)
        self.assertEqual(row['english'],'Refined English.')
        self.assertEqual(row['bavarian'],args[0]['bavarian'])
        self.assertEqual(row['source_id'],args[0]['source_id'])

    def test_changed_draft_or_target_is_rejected(self):
        for field in ('english','bavarian'):
            args = self.fixture()
            args[0][field] = 'Changed text.'
            with self.assertRaises(ValueError):
                refined_row(*args)

    def test_variant_b_never_receives_the_blind_draft(self):
        args=self.fixture('variant_b')
        args[0].update(english='Different blind draft.',blind_checkpoint='blind-file',blind_record_sha256='blind-hash')
        row=refined_row(*args,condition='variant_b')
        self.assertEqual(row['english'],'Refined English.')
        self.assertEqual(row['bavarian'],args[0]['bavarian'])
        self.assertEqual(row['condition'],'variant_b')
        self.assertNotIn('blind_checkpoint',row)
        self.assertNotIn('blind_record_sha256',row)
        self.assertIn('translation_checkpoint',row)
        self.assertNotIn('refinement_checkpoint',row)

    def test_variant_a_record_cannot_be_used_for_b(self):
        args=self.fixture('variant_a')
        with self.assertRaises((KeyError,ValueError)):
            refined_row(*args,condition='variant_b')

    def test_pilot_request_cannot_replace_scale_generation_identity(self):
        args = self.fixture()
        bad = copy.deepcopy(args)
        pilot = validation.request('variant_a','selection',bad[1],bad[2],bad[3]['prompts'],
                                   draft=bad[0]['english'])
        bad[4][1]['request_sha256'] = digest(pilot)
        with self.assertRaises(ValueError):
            refined_row(*bad)
