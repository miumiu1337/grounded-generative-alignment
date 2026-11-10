import unittest

from preprocessing import retained_cleanup as clean


def fixture(text, fragments, kind=None):
    row={'id':'barwiki/1','in_language':'bar','page_id':1,'version':2,'title':'Topic',
         'url':'https://bar.wikipedia.org/wiki/Topic','wikidata_id':'Q1','text':text}
    sources=[];provenance={};previous=0
    for i,fragment in enumerate(fragments):
        start=text.index(fragment,previous);end=start+len(fragment);previous=end
        sid='old_'+str(i)
        sources.append({'source_id':sid,'bar_text':fragment})
        provenance[sid]={'source_start':start,'source_end':end}
    return clean.prepare(row,sources,provenance)['records']


class RetainedCleanupTests(unittest.TestCase):
    def test_era_fragment_recovers_verbatim_clause_and_keeps_both_records(self):
        first='Zwischn 4000 und 1000 v. Kr.'
        second='hod si de Vawendung vo Keramik und da Ackabau entwicklt.'
        text=first+' '+second
        records=fixture(text,[first,second])
        self.assertEqual(len(records),2)
        self.assertTrue(all(r['source_text']==text for r in records))
        self.assertEqual(records[0]['cleaned_unit_id'],records[1]['cleaned_unit_id'])
        self.assertEqual(records[0]['contributing_source_ids'],['old_0','old_1'])
        self.assertTrue(all(r['source_candidate_ready'] for r in records))
        self.assertTrue(all(r['translation_status']=='new_translation_required_for_recovered_context' for r in records))

    def test_music_title_is_retained_and_not_made_a_fake_sentence(self):
        text='- BWV 882 – Well-Tempered Clavier, Book 2: Prelude and Fugue No. 13 in F-sharp major'
        record=fixture(text,['13 in F-sharp major'])[0]
        self.assertFalse(record['source_candidate_ready'])
        self.assertFalse(record['source_text'].startswith('-'))
        self.assertFalse(record['source_text'].endswith('.'))
        self.assertIn('catalogue_or_work_title_requires_contextual_reconstruction',record['review_reasons'])
        self.assertEqual(record['original_source_text'],'13 in F-sharp major')

    def test_complete_bullet_is_cleaned_and_not_excluded(self):
        text='- De Spitalkirche St. Josef besitzt a Rokoko-Kanzel mit üppigem Schnitzdekor.'
        record=fixture(text,[text])[0]
        self.assertEqual(record['source_text'],text[2:])
        self.assertTrue(record['source_candidate_ready'])
        self.assertEqual(record['original_source_text'],text)

    def test_complete_bullet_does_not_absorb_neighbouring_sentences(self):
        first='- De Spitalkirche St. Josef besitzt a Rokoko-Kanzel mit üppigem Schnitzdekor.'
        text=first+' Des Altarbild stammt vo dem Maler Joseph Höttinger.'
        record=fixture(text,[first])[0]
        self.assertEqual(record['source_text'],first[2:])
        self.assertFalse(record['context_expanded'])

    def test_date_alone_stays_unresolved_instead_of_disappearing(self):
        text='Zwischn 4000 und 1000 v. Kr.'
        record=fixture(text,[text])[0]
        self.assertEqual(record['original_source_text'],text)
        self.assertFalse(record['source_candidate_ready'])
        self.assertIn('finite_verb_not_confirmed',record['review_reasons'])

    def test_time_first_clause_does_not_require_subject_first_order(self):
        text='Im Jahr 1900 hod da Huber des Haus kaaft.'
        self.assertTrue(fixture(text,[text])[0]['source_candidate_ready'])

    def test_capitalization_and_missing_period_are_recorded_repairs(self):
        record=fixture('de Kirch besitzt a scheene oide Kanzel',['de Kirch besitzt a scheene oide Kanzel'])[0]
        self.assertEqual(record['source_text'],'De Kirch besitzt a scheene oide Kanzel.')
        self.assertEqual(record['cleaning_operations'],['capitalize_initial_letter','add_declarative_period'])

    def test_question_is_not_silently_converted_to_statement(self):
        record=fixture('Is des a oide Kirch?',['Is des a oide Kirch?'])[0]
        self.assertEqual(record['source_text'],'Is des a oide Kirch?')
        self.assertFalse(record['source_candidate_ready'])

    def test_missing_provenance_is_an_error(self):
        row={'id':'barwiki/1','in_language':'bar','page_id':1,'version':2,'title':'Topic',
             'url':'https://bar.wikipedia.org/wiki/Topic','wikidata_id':'Q1','text':'Er is do.'}
        with self.assertRaisesRegex(ValueError,'IDs do not match'):
            clean.prepare(row,[{'source_id':'old','bar_text':'Er is do.'}],{})


if __name__=='__main__':
    unittest.main()
