import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

try:
    from preprocessing.datatrove_sentences import SentenceQuality,normalized_key,surface_text,COLUMNS
    from runners.curate_datatrove_parallel import select_unique,write_pairs,english_origin_reason,METHODS
    from preprocessing.article_structure import ArticleStructure
except ModuleNotFoundError as error:
    if error.name not in {'datatrove','spacy'}:raise
    raise unittest.SkipTest('Quality checks require the separate requirements-quality.txt environment') from error


class DataTroveSentenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.quality=SentenceQuality()

    def test_short_unpunctuated_captions_are_rejected_by_generic_filters(self):
        texts=['Kockums-Eisnweak (um 1900)','13 in F-sharp major']
        self.assertTrue(all(reason is not None for reason in self.quality.check(texts,'de')))

    def test_prose_passes_actual_filters_but_bullet_marker_is_not_erased(self):
        bar='De Spitalkirche St. Josef besitzt a Rokoko-Kanzel mit üppigem Schnitzdekor.'
        self.assertEqual(self.quality.check([bar],'de'),[None])
        self.assertEqual(self.quality.check(['The church has an ornate Rococo pulpit.'],'en'),[None])
        self.assertIsNotNone(self.quality.check([surface_text('- '+bar)],'de')[0])

    def test_time_first_recovered_clause_passes(self):
        text='Zwischn 4000 und 1000 v. Kr. hod si de Vawendung vo Keramik und da Ackabau entwicklt.'
        self.assertEqual(self.quality.check([text],'de'),[None])

    def test_no_verb_subject_or_capitalization_rules_are_applied(self):
        self.assertEqual(self.quality.check(['Sein Atelier und der Markt.'],'de'),[None])
        self.assertEqual(self.quality.check(['the beautiful church in the old town.'],'en'),[None])
        self.assertFalse(self.quality.config()['grammar_filters'])
        self.assertFalse(self.quality.config()['trained_language_models'])
        self.assertEqual(surface_text('da Text steht do.'),'da Text steht do.')

    def test_duplicate_keys_normalize_formatting_without_erasing_facts(self):
        self.assertEqual(normalized_key('De Kirch is do.'),normalized_key('DE  KIRCH is do!'))
        self.assertNotEqual(normalized_key('Er lebt seit 1900 do.'),normalized_key('Er lebt seit 1901 do.'))
        self.assertNotEqual(normalized_key('Da Bär is do.'),normalized_key('Da Bar is do.'))

    def test_repeated_phrases_fail(self):
        text='De Kirch steht in dem Ort und de Kirch steht in dem Ort und de Kirch steht in dem Ort.'
        self.assertIsNotNone(self.quality.check([text],'de')[0])

    def test_published_schema_has_exactly_six_string_columns(self):
        row=dict(zip(COLUMNS,('bar','en','De Kirch steht in dem Ort.','The church stands in the village.',
                             'Original Bavarian article.','Original English article.'),strict=True))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'pairs.parquet';write_pairs(path,[row])
            self.assertEqual(pq.read_schema(path).names,list(COLUMNS))
            self.assertTrue(all(field.type==pa.string() for field in pq.read_schema(path)))
            self.assertEqual(pq.read_table(path).to_pylist(),[row])
            with self.assertRaisesRegex(ValueError,'exactly six'):
                write_pairs(path,[{**row,'extra':'forbidden'}])

    def test_deduplication_reserves_keys_only_for_retained_common_rows(self):
        def item(sid,text,reason=None):
            return {'source_id':sid,'source_text':text,'english':{m:'The church stands in the village.' for m in METHODS},'reason':reason}
        kept,reasons=select_unique({'validation':[],'train':[
            item('bad','De Kirch steht in dem Ort.','bad_alignment'),
            item('good','De Kirch steht in dem Ort.'),
            item('duplicate','DE KIRCH steht in dem Ort!')]})
        self.assertEqual([r['source_id'] for r in kept['train']],['good'])
        self.assertEqual(reasons['train:duplicate_bavarian_source'],1)

    def test_validation_duplicates_are_removed_from_training(self):
        row={'source_id':'val','source_text':'De Kirch steht in dem Ort.',
             'english':{m:'The church stands in the village.' for m in METHODS},'reason':None}
        kept,_=select_unique({'validation':[row],'train':[{**row,'source_id':'train'}]})
        self.assertEqual(len(kept['validation']),1)
        self.assertEqual(kept['train'],[])

    def test_english_evidence_cannot_reuse_a_list_item_without_its_marker(self):
        first='en_1:fw8bd13e72:r2:datatrove_quality_auto_v3:S0001'
        second='en_1:fw8bd13e72:r2:datatrove_quality_auto_v3:S0002'
        a='This is a complete sentence inside a list.'
        b='This is the normal prose in the article.'
        raw='# Topic\n- '+a+'\n\n'+b+'\n'
        numbered='# Topic\n- ['+first+'] '+a+'\n\n['+second+'] '+b+'\n'
        value={'en':{'text':raw},'structures':{'en':ArticleStructure(raw)},
               'bundle':{'page':{'en':'Topic (en), revision 2\n'+numbered,
                                  'candidates':[{'sentence_ids':[first],'text':a},{'sentence_ids':[second],'text':b}]}}}
        self.assertEqual(english_origin_reason(value,[first]),'displayed_list_or_enumeration')
        self.assertIsNone(english_origin_reason(value,[second]))
        with self.assertRaisesRegex(ValueError,'Unknown saved English evidence'):
            english_origin_reason(value,['unknown'])


if __name__=='__main__':unittest.main()
