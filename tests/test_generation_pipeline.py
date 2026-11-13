import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np

from parallel_data_generation import pipeline, wiki_protocol, wiki_selection
from parallel_data_generation.mining import mine
from parallel_data_generation.prompts import load_prompts
from parallel_data_generation.teacher import Teacher
from preprocessing.artifacts import load_config, read_json, sha, write_json


def fixture():
    source = {'source_id': 'a'*64, 'focus_text': 'Focus text.', 'context_text': 'Before. Focus text. After.',
        'focus_index': 1, 'context_sentence_ids': ['before', 'a'*64, 'after'], 'language': 'ast',
        'source_article_title': 'Xixón', 'target_article_title': 'Gijón', 'wikidata_id': 'Q1', 'section_path': ['History']}
    rows = [{'source_id': str(i), 'focus_text': f'English focus {i}.', 'context_text': f'English context {i}.',
        'context_sentence_ids': [str(i-1), str(i), str(i+1)], 'paragraph_id': 'p', 'section_id': 's',
        'source_char_start': i*10, 'section_path': ['History']} for i in range(3)]
    config = load_config(Path(__file__).resolve().parents[1]/'configs/asturian.json')
    case = {'source': source, 'blind': {'sentence_only': {'translation': 'Saved Blind.'}}}
    none = {'source_id': source['source_id'], 'evidence_ids': [], 'match_status': 'none', 'contradiction': False}
    return case, {'candidates': rows}, load_prompts(config), none


class GenerationPipelineTests(unittest.TestCase):
    def test_language_change_renders_all_prompts_and_has_exactly_five_methods(self):
        config = load_config(Path(__file__).resolve().parents[1]/'configs/asturian.json')
        config.update(source_language='fy', source_language_name='West Frisian', quality_tokenizer_language='nl')
        text = json.dumps(load_prompts(config))
        self.assertNotIn('Asturian', text)
        self.assertNotIn('Bavarian', text)
        self.assertIn('West Frisian', text)
        self.assertEqual(pipeline.METHODS, ('blind', 'mining', 'variant_a', 'variant_b', 'variant_c'))

    def test_b_has_no_blind_and_no_sentence_context_leaks_into_sentence_condition(self):
        case, article, prompts, _ = fixture()
        request = wiki_protocol.request(case, article, 'variant_b', 'sentence_only', 'selection', prompts)
        payload = json.loads(request['messages'][1]['content'])
        self.assertNotIn('blind_translation', payload)
        self.assertNotIn('source_context', payload)
        self.assertTrue(all('context' not in row for row in payload['english_candidates']))
        self.assertEqual(payload['article_topic']['target_title'], 'Gijón')
        request = wiki_protocol.request(case, article, 'variant_b', 'paragraph_context', 'selection', prompts)
        self.assertEqual(json.loads(request['messages'][1]['content'])['source_context'], case['source']['context_text'])

    def test_a_none_reuses_blind_exactly_and_training_target_is_focus_only(self):
        case, article, _, none = fixture()
        result = wiki_protocol.finish(case, article, 'variant_a', 'sentence_only', 'protocol', none, None)
        self.assertEqual(result['translation'], 'Saved Blind.')
        self.assertEqual(result['training_pair']['target_text'], 'Focus text.')
        self.assertEqual(result['training_pair']['target_article'], 'Xixón')
        self.assertEqual(result['training_pair']['target_lang'], 'ast')
        with self.assertRaises(ValueError):
            wiki_protocol.finish(case, article, 'variant_a', 'sentence_only', 'protocol', none,
                                 {'source_id': case['source']['source_id'], 'translation': 'New text.'})

    def test_chunking_covers_all_sentences_and_allowed_adjacent_pairs(self):
        _, article, _, _ = fixture()
        page = wiki_protocol.page_for(article, 'sentence_only')
        pages = wiki_selection.partition(page, lambda value: len(value['sentences']) <= 2)
        wiki_selection.coverage(page, pages)
        with self.assertRaises(wiki_selection.SelectionCapacityError):
            wiki_selection.partition(page, lambda value: False)

    def test_empty_english_pool_has_no_forced_evidence_selection(self):
        case, _, prompts, none = fixture(); generate = Mock(); fits = Mock()
        selected, trace = wiki_selection.select(case, {'candidates': []}, 'variant_b',
                                                'sentence_only', prompts, fits, generate)
        self.assertEqual(selected, none)
        self.assertEqual(trace, [])
        self.assertEqual(wiki_selection.replay(case, {'candidates': []}, 'variant_b',
            'sentence_only', prompts, trace, generate), none)
        generate.assert_not_called(); fits.assert_not_called()

    def test_mining_context_inputs_produce_focus_only_training_text(self):
        case, article, _, _ = fixture()
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, language: np.ones((len(texts), 1024), dtype=np.float32)
        result = mine([case['source']], article['candidates'], encoder, 'ast', 'paragraph_context')
        self.assertEqual(encoder.encode.call_args_list[0].args[0], [case['source']['context_text']])
        chosen = result[0]['candidates'][0]
        pair = pipeline.training_pair(case['source'], chosen['focus_text'])
        self.assertEqual(pair['source_text'], 'English focus 0.')
        self.assertEqual(pair['target_text'], 'Focus text.')

    def test_c_without_evidence_never_calls_the_teacher(self):
        case, _, prompts, _ = fixture(); generate = Mock()
        result = pipeline.refine_c(case, 'sentence_only', [], prompts, generate)
        self.assertEqual(result['translation'], 'Saved Blind.')
        generate.assert_not_called()

    def test_c_requires_exact_citation_and_acceptance_checks(self):
        case, _, prompts, _ = fixture()
        evidence = [{'reference_id': 'FW1', 'text': 'A supported translation.', 'url': 'https://example.org/'}]
        proposal = {'translation': 'Supported Blind.', 'corrections': [{'old_text': 'Saved', 'new_text': 'Supported',
            'reference_id': 'FW1', 'quote': 'invented quote'}]}
        generate = Mock(return_value=proposal)
        rejected = pipeline.refine_c(case, 'sentence_only', evidence, prompts, generate)
        self.assertEqual(rejected['translation'], 'Saved Blind.')
        self.assertEqual(rejected['decision'], 'invalid_proposal_keep_blind')
        proposal['corrections'][0]['quote'] = 'supported translation'
        generate.side_effect = [proposal, {'supported': True, 'faithful': True, 'complete': False, 'reason': 'Content omitted.'}]
        flagged = pipeline.refine_c(case, 'sentence_only', evidence, prompts, generate)
        self.assertEqual(flagged['translation'], 'Saved Blind.')
        generate.side_effect = [proposal, {'supported': True, 'faithful': True, 'complete': True, 'reason': 'Supported.'}]
        accepted = pipeline.refine_c(case, 'sentence_only', evidence, prompts, generate)
        self.assertEqual(accepted['translation'], 'Supported Blind.')

    def test_teacher_reuses_saved_raw_response_and_rejects_incomplete_output(self):
        case, _, prompts, _ = fixture()
        settings = {'quantized_file': 'model.gguf', 'context_tokens': 4096,
                    'maximum_output_tokens': 512, 'temperature': 0, 'seed': 42}
        with tempfile.TemporaryDirectory() as directory, patch('parallel_data_generation.teacher.requests.Session') as session:
            session.return_value.get.return_value = Mock(json=lambda: {'model_path': '/models/model.gguf'}, raise_for_status=lambda: None)
            teacher = Teacher('http://localhost:8080', settings, Path(directory)/'raw')
            response = {'truncated': False, 'stop_type': 'eos', 'content': '{"translation":"Saved Blind."}'}
            with patch.object(teacher, 'tokens', return_value=[1]), patch.object(teacher, 'post', return_value=response) as post:
                self.assertEqual(pipeline.blind(case, 'sentence_only', prompts, teacher.generate), 'Saved Blind.')
                self.assertEqual(pipeline.blind(case, 'sentence_only', prompts, teacher.generate), 'Saved Blind.')
                self.assertEqual(post.call_count, 1)
            with patch.object(teacher, 'tokens', return_value=[1]), patch.object(teacher, 'post', return_value={**response, 'stop_type': 'limit'}):
                with self.assertRaisesRegex(ValueError, 'Incomplete'):
                    pipeline.blind(case, 'paragraph_context', prompts, teacher.generate)

    def test_all_five_methods_keep_the_same_focus_targets_in_both_conditions(self):
        case, article, _, none = fixture(); sid = case['source']['source_id']
        config_path = Path(__file__).resolve().parents[1]/'configs/asturian.json'
        config = load_config(config_path)
        class FakeTeacher:
            def __init__(self, *args): pass
            def fits(self, request): return True
            def generate(self, stage, request):
                if stage == 'blind': return {'translation': 'Saved Blind.'}
                if stage == 'selection': return none
                return {'source_id': sid, 'translation': 'B translation.'}
        class FakeEncoder:
            def __init__(self, *args): pass
            def encode(self, texts, language): return np.ones((len(texts), 1024), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cleaned = root/'cleaned'; output = root/'generated'
            window = {**case['source'], 'article_id': 'astwiki/1', 'source_char_start': 0}
            source = {'article_id': 'astwiki/1', 'title': 'Xixón', 'wikidata_id': 'Q1', 'revision': 1, 'windows': [window]}
            target = {'article_id': 'enwiki/2', 'title': 'Gijón', 'wikidata_id': 'Q1', 'revision': 2, 'windows': article['candidates']}
            files = {}
            for name, value in [('source.json', source), ('target.json', target)]:
                write_json(cleaned/name, value); files[name] = sha(cleaned/name)
            write_json(cleaned/'manifest.json', {'status': 'complete', 'config': config, 'files': files,
                'matched_pairs': [{'article_id': 'astwiki/1', 'en_article_id': 'enwiki/2', 'en_revision_id': 2}]})
            index = root/'fineweb.sqlite'; index.write_bytes(b'retrieval mocked')
            with patch('parallel_data_generation.teacher.Teacher', FakeTeacher), \
                    patch('parallel_data_generation.mining.Encoder', FakeEncoder), \
                    patch('parallel_data_generation.fineweb.search', return_value=[]):
                for condition in pipeline.CONDITIONS:
                    for method in pipeline.METHODS:
                        result = pipeline.run(config_path, cleaned, output, method, condition, endpoint='http://mocked',
                                              fineweb_index=index if method == 'variant_c' else None)
                        self.assertEqual(result['completed'], 1, (method, condition, result))
                        self.assertEqual(result['deferred'], 0)
                        saved = read_json(output/result['protocol_id']/method/condition/(sid+'.json.gz'))
                        self.assertEqual(saved['training_pair']['target_text'], 'Focus text.')
                        self.assertEqual(set(saved['training_pair']), {'source_lang', 'target_lang', 'source_text',
                            'target_text', 'source_article', 'target_article'})


if __name__ == '__main__':
    unittest.main()
