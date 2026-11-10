import unittest

from preprocessing import quality_prose as quality
from preprocessing.finewiki_pilot import REVISION


def row(text, language='bar'):
    return {'id':language+'wiki/1','in_language':language,'page_id':1,'version':2,
            'title':'Topic','url':'https://'+language+'.wikipedia.org/wiki/Topic',
            'wikidata_id':'Q1','text':text}


class ProseQualityTests(unittest.TestCase):
    def test_wolf_definition_is_rejoined_at_original_offsets(self):
        text='Da Woif oda Wuif (dt. Wolf; lat. Canis lupus) is a Raubviech aus da Familie vo de Hundsoatigen.'
        a=quality.prepare(row(text),segment=lambda text:['Da Woif oda Wuif (dt.','Wolf; lat.',text.split('lat. ',1)[1]])
        self.assertEqual(len(a['sentences']),1)
        self.assertEqual(a['sentences'][0]['text'],text)
        self.assertTrue(a['sentences'][0]['eligible_for_matching'])
        self.assertEqual(len(a['boundary_repairs']),2)

    def test_lists_are_retained_as_context_not_training_units(self):
        text='- s Natuahistorische Museum\n- Zunftspoat: Goaßlschnoizn, Oxnrennen, Sensnmahn\n'
        a=quality.prepare(row(text))
        self.assertEqual(a['sentences'],[])
        self.assertTrue(all(x['reason']=='structural_list' for x in a['excluded_blocks']))

    def test_full_paragraph_splitter_handles_titles_and_decimals(self):
        text='Da Dr. Huber lebt seit 1998 in dem Ort. Des Haus is 2.5 Meta hoch und steht nebn da Straßn.'
        a=quality.prepare(row(text))
        self.assertEqual(len(a['sentences']),2)
        self.assertTrue(all(s['eligible_for_matching'] for s in a['sentences']))

    def test_reference_section_and_subsection_are_excluded(self):
        text='# Topic\nDes is a vollständiga Satz mit gnua Wörtern.\n\n## Literatur\nAutor Huber beschreibt des Thema in dem Buach.\n### Weitere Quellen\nNo a Satz der zu de Quelln gheat.\n## Gschicht\nDes is wieda a vollständiga Satz im Haupttext.'
        a=quality.prepare(row(text))
        eligible=[s['text'] for s in a['sentences'] if s['eligible_for_matching']]
        self.assertEqual(len(eligible),2)
        self.assertTrue(any(x['reason']=='reference_section' for x in a['excluded_blocks']))

    def test_bad_brackets_and_short_fragments_are_rejected(self):
        self.assertEqual(quality.rejection_reason('Canis lupus) is a Viech aus da Familie vo de Hund.'),'unbalanced_brackets')
        self.assertIsNotNone(quality.rejection_reason('Natuahistorische Museum.'))

    def test_century_abbreviation_does_not_leave_lowercase_fragment(self):
        text='Im 20. Jhdt. haum si daun in da Hochsproch de neichn Begriffe duachgsetzt.'
        a=quality.prepare(row(text))
        self.assertEqual([s['text'] for s in a['sentences']],[text])
        self.assertTrue(a['sentences'][0]['eligible_for_matching'])
        self.assertEqual(quality.rejection_reason('haum si daun in da Hochsproch de neichn Begriffe duachgsetzt.'),
                         'lowercase_sentence_start_for_review')

    def test_alignment_failure_is_reported_and_block_not_accepted(self):
        a=quality.prepare(row('Des is a vollständiga Satz mit gnua Wörtern.'),segment=lambda text:['changed text'])
        self.assertEqual(a['sentences'],[])
        self.assertEqual(len(a['split_failures']),1)

    def test_english_wolf_parenthesis_is_not_an_orphan(self):
        text='The wolf (pl. wolves), also known as the grey wolf, is a canine native to Eurasia and North America.'
        a=quality.prepare(row(text,'en'))
        self.assertEqual(len(a['sentences']),1)
        self.assertTrue(a['sentences'][0]['eligible_for_matching'])

    def test_all_retained_spans_and_source_text_are_exact(self):
        text='# Topic\n  Des is a vollständiga Satz mit gnua Wörtern. A zweita Satz beschreibt no mehr zu dem Thema.\n'
        a=quality.prepare(row(text))
        for sentence in a['sentences']:
            self.assertEqual(text[sentence['start']:sentence['end']],sentence['text'])
            self.assertEqual(text[sentence['paragraph_start']:sentence['paragraph_end']].strip(),text.split('\n')[1].strip())


if __name__=='__main__':unittest.main()
