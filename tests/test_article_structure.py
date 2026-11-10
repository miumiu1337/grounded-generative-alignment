import unittest

from preprocessing.article_structure import ArticleStructure,reference_heading


class ArticleStructureTests(unittest.TestCase):
    def test_reported_literatua_spelling_and_subsections_are_removed(self):
        raw='# Brasilien\nDes Land liegt in Südamerika.\n\n## Literatua\n### Biacha\nDietrich X: Brasilien heute. Frankfurt 1994.\n## Gschicht\nDes Land hod a lange Gschicht.\n'
        result=ArticleStructure(raw)
        self.assertNotIn('Dietrich',result.cleaned_text)
        self.assertNotIn('Literatua',result.cleaned_text)
        self.assertIn('lange Gschicht',result.cleaned_text)
        start=raw.index('Dietrich')
        self.assertEqual(result.reason_for_span(start,start+8),'reference_section')

    def test_bavarian_reference_heading_variants_are_matched(self):
        for title in ['Literatua','Litaradua','Litratua','Literadua','Fuassnotn','Oanzlnochweis',
                      'Literatua (Auswoi)','Quölln, Literatua und Im Netz','Literatua und Beleg']:
            with self.subTest(title=title):self.assertTrue(reference_heading(title))

    def test_complete_bullet_is_removed_instead_of_stripping_its_marker(self):
        raw='# Topic\n- De Spitalkirche besitzt a scheene Kanzel.\nDes is da normale Haupttext.\n'
        result=ArticleStructure(raw);start=raw.index('De Spitalkirche')
        self.assertEqual(result.reason_for_span(start,start+20),'displayed_list_or_enumeration')
        self.assertNotIn('Spitalkirche',result.cleaned_text)
        self.assertIn('normale Haupttext',result.cleaned_text)

    def test_displayed_enumerations_unicode_bullets_and_continuations_are_removed(self):
        raw='# Topic\n1. Nummerierter Eintrag\na) Buchstabierter Eintrag\n• Punkt mit Inhalt\n  Fortsetzung vom Punkt\n\nDes is da Haupttext.\n'
        result=ArticleStructure(raw)
        for text in ['Nummerierter','Buchstabierter','Punkt mit','Fortsetzung']:
            self.assertNotIn(text,result.cleaned_text)
        self.assertIn('Haupttext',result.cleaned_text)

    def test_unmarked_publication_entry_is_removed_without_grammar_guessing(self):
        raw='# Topic\nDietrich X (Hrsg.): Brasilien heute. Vervuert Verlag, Frankfurt 1994. ISBN 3-89354-553-0\n\nDes Land liegt in Südamerika.\n'
        result=ArticleStructure(raw)
        self.assertNotIn('ISBN',result.cleaned_text)
        self.assertEqual(result.reason_for_span(raw.index('Dietrich'),raw.index('Dietrich')+10),'bibliographic_entry')

    def test_literature_topic_and_water_source_are_not_reference_sections(self):
        raw='# Literatua\nDes is a Beitrag iwa Literatua.\n## Deitsche Literatua\nDo steht da normale Haupttext.\n## Quej\nDa Boch entspringt am Berg und fliasst ins Tal.\n'
        result=ArticleStructure(raw)
        self.assertIn('Beitrag iwa Literatua',result.cleaned_text)
        self.assertIn('normale Haupttext',result.cleaned_text)
        self.assertIn('Da Boch entspringt',result.cleaned_text)

    def test_text_about_press_or_isbn_is_not_a_publication_entry(self):
        raw='# Topic\nIn 1995, the press reported an important event.\n\nIm Joar 2005 is de ISBN 978-3-16-148410-0 ois Beispui vawendt worn.\n'
        self.assertIn('press reported',ArticleStructure(raw).cleaned_text)
        self.assertIn('ois Beispui',ArticleStructure(raw).cleaned_text)

    def test_inline_citations_removed_and_natural_enumeration_in_prose_kept(self):
        raw='# Topic\nDe Gmoa hod Biacha, Schuin und Museen.[12]\n'
        result=ArticleStructure(raw)
        self.assertIn('Biacha, Schuin und Museen.',result.cleaned_text)
        self.assertNotIn('[12]',result.cleaned_text)

    def test_any_overlap_with_an_unwanted_line_is_rejected(self):
        raw='Haupttext.\n- Listeneintrag\nWeitere Prosa.\n'
        result=ArticleStructure(raw)
        self.assertEqual(result.reason_for_span(0,len(raw)),'displayed_list_or_enumeration')
        self.assertIsNone(result.reason_for_span(0,10))

    def test_headings_inside_code_do_not_change_reference_hierarchy(self):
        raw='# Topic\nHaupttext.\n## Literatua\n```python\n# Fake title\n```\nBibliografie.\n## Gschicht\nWeitere Prosa.\n'
        result=ArticleStructure(raw)
        self.assertNotIn('Bibliografie',result.cleaned_text)
        self.assertIn('Weitere Prosa',result.cleaned_text)


if __name__=='__main__':unittest.main()
