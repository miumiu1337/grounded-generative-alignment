import unittest

from Cleaning.paragraphs import prose_paragraphs
from preprocessing.context import context_windows
from preprocessing.article_matching import match_articles
from preprocessing.article_structure import ArticleStructure


def article(text):
    return {'id':'astwiki/1','page_id':1,'title':'Prueba','wikidata_id':'Q1','version':1,
            'url':'https://ast.wikipedia.org/wiki/Prueba','in_language':'ast','text':text}


class ArticleContextTests(unittest.TestCase):
    def test_asturian_references_are_removed_but_body_sections_remain(self):
        text='# Prueba\nLa primera oración tien información del artículu.\n\n## Referencies\nUna fonte bibliográfica con datos y detalles.\n### Notes\nOtra referencia que nun ye prosa.\n## Hestoria\nLa historia vuelve ser prosa del artículu.\n'
        paragraphs, audit = prose_paragraphs(article(text))
        self.assertEqual(len(paragraphs),2)
        self.assertEqual(paragraphs[1]['metadata']['section_path'],['Prueba','Hestoria'])
        self.assertNotEqual(paragraphs[0]['metadata']['section_id'],paragraphs[1]['metadata']['section_id'])
        self.assertTrue(any(r['reason']=='reference_section' for r in audit['removed_blocks']))

    def test_list_gaps_do_not_merge_original_paragraphs(self):
        text='# Prueba\nLa prosa anterior tien una oración completa.\n- Un elementu que nun entra nel corpus.\nLa prosa posterior tampoco debe xunise con ella.\n'
        paragraphs,_ = prose_paragraphs(article(text))
        self.assertEqual(len(paragraphs),2)
        self.assertTrue(all('elementu' not in p['text'] for p in paragraphs))
        self.assertTrue(all(text[p['metadata']['source_char_start']:p['metadata']['source_char_end']]==p['text'] for p in paragraphs))

    def test_single_rendered_newline_is_a_hard_paragraph_boundary(self):
        text='# Prueba\nPrimera oración. Segunda oración. Tercera oración.\nOtra primera oración. Otra segunda oración. Otra tercera oración.\n'
        paragraphs,_=prose_paragraphs(article(text))
        self.assertEqual(len(paragraphs),2)
        self.assertNotEqual(paragraphs[0]['paragraph_id'],paragraphs[1]['paragraph_id'])
        self.assertNotIn('Otra',paragraphs[0]['text'])

    def test_sentence_quality_can_preserve_asturian_numeric_facts(self):
        from preprocessing.datatrove_sentences import SentenceQuality
        quality=SentenceQuality(tokenizer_languages={'ast':'es','en':'en'},minimum_alpha_token_ratio=.5)
        self.assertEqual(quality.check(['Según el censu de 2010, tenía una población de 3192 habitantes.'],'es'),[None])
        self.assertEqual(quality.config()['tokenizer_languages']['ast'],'es')
        self.assertFalse(quality.config()['grammar_filters'])

    def rows(self):
        return [{'source_id':str(i),'text':str(i),'quality_reason':None,'parent_paragraph_id':'p',
                 'source_char_start':i*2,'source_char_end':i*2+1,'article_id':'astwiki/1',
                 'section_id':'s','section_path':['Prueba'],'article_title':'Prueba',
                 'page_id':1,'revision':1,'wikidata_id':'Q1'} for i in range(5)]

    def test_context_keeps_focus_target_and_exact_three_neighbours(self):
        windows,_ = context_windows(self.rows())
        self.assertEqual([r['source_id'] for r in windows],['1','2','3'])
        self.assertEqual(windows[1]['focus_text'],'2')
        self.assertEqual(windows[1]['context_text'],'1 2 3')
        self.assertEqual(windows[1]['focus_index'],1)

    def test_filtered_middle_sentence_is_not_bridged(self):
        rows=self.rows();rows[2]['quality_reason']='caption'
        windows,_ = context_windows(rows)
        self.assertEqual(windows,[])

    def test_context_cannot_cross_sections_or_paragraphs(self):
        rows=self.rows();rows[2]['section_id']='other'
        with self.assertRaises(ValueError):context_windows(rows)
        rows=self.rows();rows[2]['parent_paragraph_id']='other'
        windows,_=context_windows(rows)
        self.assertTrue(all('2' not in r['context_sentence_ids'] for r in windows))

    def test_asturian_match_reuses_latest_english_page_logic(self):
        source=[{'article_id':'astwiki/1','wikidata_id':'Q1'}]
        en=[{'page_id':2,'version':1,'wikidata_id':'Q1'},
            {'page_id':2,'version':2,'wikidata_id':'Q2'}]
        coverage,_=match_articles(source,en,source_language='ast')
        self.assertEqual(coverage[0]['status'],'no_english_match')
        with self.assertRaises(ValueError):match_articles(source,en)

    def test_asturian_reference_names_do_not_change_bavarian_default(self):
        raw='# Topic\n## Referencies\nNormal body text for this test.\n'
        self.assertIn('Normal body',ArticleStructure(raw).cleaned_text)
        self.assertNotIn('Normal body',ArticleStructure(raw,language='ast').cleaned_text)


if __name__=='__main__':unittest.main()
