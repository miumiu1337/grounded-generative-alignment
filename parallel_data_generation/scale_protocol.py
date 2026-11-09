"""Six-condition article bundles using the established FineWiki prompts and validators."""

import json

from preprocessing import finewiki_pilot as data
from parallel_data_generation import finewiki_experiments as protocol

CONDITIONS = ('blind', 'laser', 'bge_m3', 'variant_a', 'variant_b', 'variant_c')
SERIALIZATION = 'shared_article_context_before_focus_source_v1'


def freeze_pair(prepared):
    pair = prepared['pair']
    bar, en = prepared['bar'], prepared['en']
    for lang, article in [('bar', bar), ('en', en)]:
        if (article['language'] != lang or article['dataset_revision'] != data.REVISION
                or article['wikidata_id'] != pair['wikidata_id']):
            raise ValueError('Prepared article differs from the frozen topic or snapshot')
        for sentence in article['sentences']:
            if article['text'][sentence['start']:sentence['end']] != sentence['text']:
                raise ValueError('Frozen sentence differs from its raw article span')
    if (bar['page_id'] != pair['page_id'] or en['page_id'] != pair['en_page_id']
            or int(bar['revision_id']) != pair['revision_id'] or int(en['revision_id']) != pair['en_revision_id']):
        raise ValueError('Prepared pair page IDs or revisions changed')
    numbered = {lang: data.numbered_page({**article, 'sentences': [s for s in article['sentences']
                            if s['eligible_for_matching']]}) for lang, article in [('bar', bar), ('en', en)]}
    page = {'bar': f"{bar['title']} (bar), revision {bar['revision_id']}\n" + numbered['bar'],
            'en': f"{en['title']} (en), revision {en['revision_id']}\n" + numbered['en'],
            'candidates': en['candidate_spans'], 'en_page_id': en['page_id'], 'en_revision_id': en['revision_id'],
            'wikidata_id': pair['wikidata_id'], 'bar_article_id': bar['dataset_id'], 'en_article_id': en['dataset_id'],
            'bar_url': bar['url'], 'en_url': en['url'], 'bar_text_sha256': bar['text_sha256'],
            'en_text_sha256': en['text_sha256'],
            'numbering': 'eligible_units_only;all_original_article_text_retained'}
    sources, provenance = [], {}
    for sentence in bar['sentences']:
        if not sentence['eligible_for_matching']:
            continue
        source = {'source_id': sentence['sentence_id'], 'bar_text': sentence['text'],
                  'bar_page_id': bar['page_id'], 'bar_title': bar['title'], 'bar_revision_id': bar['revision_id']}
        sources.append(source)
        provenance[source['source_id']] = {'dataset': data.DATASET, 'dataset_revision': data.REVISION,
            'wikidata_id': pair['wikidata_id'], 'source_article_id': bar['dataset_id'],
            'target_article_id': en['dataset_id'], 'source_url': bar['url'], 'target_url': en['url'],
            'source_revision': bar['revision_id'], 'target_revision': en['revision_id'],
            'source_segmentation': bar['sentence_splitter'], 'target_segmentation': en['sentence_splitter'],
            'source_start': sentence['start'], 'source_end': sentence['end'], 'split': 'expanded_finewiki_cohort',
            'original_pilot_role': pair['pilot_role']}
    return {'page_id': str(bar['page_id']), 'sources': sources, 'page': page, 'provenance': provenance,
            'preparation_config_sha256': prepared['config_sha256'], 'sources_sha256': data.digest(sources),
            'page_sha256': data.digest(page), 'conditions': CONDITIONS, 'serialization': SERIALIZATION}


def request(condition, stage, source, page, prompts, **dependencies):
    request = protocol.request(condition, stage, source, page, prompts, **dependencies)
    if condition == 'blind':
        return request
    messages = [dict(message) for message in request['messages']]
    for message in messages:
        if message['role'] != 'user':
            continue
        payload = json.loads(message['content'])
        first = ('bavarian_page', 'english_page', 'english_wikipedia_page')
        ordered = {key: payload[key] for key in first if key in payload}
        ordered.update({key: value for key, value in payload.items() if key not in ordered})
        message['content'] = json.dumps(ordered, ensure_ascii=False)
    return {**request, 'messages': messages}
