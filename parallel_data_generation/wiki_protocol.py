"""Article evidence selection for language-configurable A and B, with focus-only targets."""

import json

from preprocessing import finewiki_pilot as data
from parallel_data_generation import variant_a as shared

METHODS = ('variant_a', 'variant_b')
CONDITIONS = ('sentence_only', 'paragraph_context')


def page_for(article, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown context condition')
    rows = sorted(article['candidates'], key=lambda row: row['source_char_start'])
    ids = {row['source_id']: f'E{index+1:04d}' for index, row in enumerate(rows)}
    sentences, spans = [], []
    for index, row in enumerate(rows):
        label = ids[row['source_id']]
        sentence = {'id': label, 'text': row['focus_text']}
        if condition == 'paragraph_context':
            sentence.update(context=row['context_text'], section_path=row['section_path'])
        sentences.append(sentence)
        spans.append({'sentence_ids': [label], 'text': row['focus_text'],
                      'original_sentence_ids': [row['source_id']]})
        if index+1 < len(rows):
            following = rows[index+1]
            neighbors = row['context_sentence_ids']; at = neighbors.index(row['source_id'])
            if (row['paragraph_id'] == following['paragraph_id']
                    and row['section_id'] == following['section_id']
                    and at+1 < len(neighbors) and neighbors[at+1] == following['source_id']):
                spans.append({'sentence_ids': [label, ids[following['source_id']]],
                    'text': row['focus_text']+' '+following['focus_text'],
                    'original_sentence_ids': [row['source_id'], following['source_id']]})
    return {'sentences': sentences, 'candidates': spans}


def input_identity(case, article, method, condition):
    if method not in METHODS or condition not in CONDITIONS:
        raise ValueError('Unknown A/B method or condition')
    value = {'source': case['source'], 'english_article': article, 'method': method, 'condition': condition}
    if method == 'variant_a':
        value['blind'] = case['blind'][condition]
    return data.digest(value)


def request(case, article, method, condition, stage, prompts, selection=None):
    if method not in METHODS or condition not in CONDITIONS:
        raise ValueError('Unknown A/B method or condition')
    final_stage = 'refinement' if method == 'variant_a' else 'translation'
    if stage not in ('selection', final_stage):
        raise ValueError('Unknown A/B stage')
    source = case['source']; page = page_for(article, condition)
    payload = {'focus_source': {'source_id': source['source_id'], 'text': source['focus_text']},
        'article_topic': {'source_title': source['source_article_title'],
                         'target_title': source['target_article_title'], 'wikidata_id': source['wikidata_id']}}
    if condition == 'paragraph_context':
        payload.update(source_context=source['context_text'], source_section_path=source['section_path'])
    if method == 'variant_a':
        payload['blind_translation'] = case['blind'][condition]['translation']
    if stage == 'selection':
        if selection is not None:
            raise ValueError('Selection must not receive an earlier selection')
        payload['english_candidates'] = page['sentences']
        payload['allowed_evidence_spans'] = [span['sentence_ids'] for span in page['candidates']]
    else:
        shared.validate_selection(selection, {'source': source}, page)
        payload['selection'] = selection
        payload['selected_english_evidence'] = shared.evidence_for(selection, page)
    item = {'source': source}
    schema = shared.schema_for('selection' if stage == 'selection' else 'refinement', item, page)
    grammar = shared.selection_grammar(item, page) if stage == 'selection' and page['candidates'] else None
    return {'messages': [{'role': 'system', 'content': prompts[method][stage]},
                         {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}],
            'schema': schema, 'grammar': grammar}


def finish(case, article, method, condition, identity, selection, generated):
    source = case['source']; page = page_for(article, condition)
    shared.validate_selection(selection, {'source': source}, page)
    if method == 'variant_a' and selection['match_status'] == 'none':
        if generated is not None:
            raise ValueError('A without supporting evidence must reuse Blind exactly')
        translation = case['blind'][condition]['translation']; decision = 'no_evidence_keep_blind'
    else:
        shared.validate_refinement(generated, {'source': source})
        translation = generated['translation']; decision = 'refined' if method == 'variant_a' else 'translated_after_selection'
    return {'source_id': source['source_id'], 'method': method, 'condition': condition,
        'protocol_id': identity, 'input_sha256': input_identity(case, article, method, condition),
        'selection': selection, 'translation': translation, 'decision': decision,
        'selected_evidence': shared.evidence_for(selection, page), 'quality_status': 'unassessed',
        'training_pair': {'source_lang': 'en', 'target_lang': source['language'], 'source_text': translation,
            'target_text': source['focus_text'], 'source_article': source['target_article_title'],
            'target_article': source['source_article_title']}}


def validate_record(record, case, article, method, condition, identity):
    expected = finish(case, article, method, condition, identity, record['selection'],
        None if method == 'variant_a' and record['selection']['match_status'] == 'none'
        else {'source_id': case['source']['source_id'], 'translation': record['translation']})
    if any(record[key] != value for key, value in expected.items()):
        raise ValueError('Saved A/B result differs from its source, evidence, or focus-only training pair')
