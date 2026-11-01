"""Apply approved development-only eligibility decisions, preserving v1 inputs."""

import copy
import hashlib

import preprocessing.finewiki_pilot as data


def prepare_revision(parent, articles, plan):
    if (plan['version'] != 2 or plan['scope'] != 'two_development_topics_only'
            or plan['dataset_revision'] != data.REVISION
            or parent['dataset_revision'] != data.REVISION
            or parent['sources_sha256'] != data.digest(parent['sources'])
            or parent['pages_sha256'] != data.digest(parent['pages'])):
        raise ValueError('Eligibility revision differs from its pinned development inputs')
    decisions, excluded = {}, {'bar': set(), 'en': set()}
    units = {}
    for article in articles:
        if (article['dataset_revision'] != data.REVISION
                or hashlib.sha256(article['text'].encode()).hexdigest() != article['text_sha256']):
            raise ValueError('Reviewed article text or dataset revision changed')
        for sentence in article['sentences']:
            if article['text'][sentence['start']:sentence['end']] != sentence['text']:
                raise ValueError('Reviewed sentence offsets changed')
            if sentence['sentence_id'] in units:
                raise ValueError('Repeated sentence ID in reviewed articles')
            units[sentence['sentence_id']] = (article, sentence)
    for decision in plan['decisions']:
        sid = decision['sentence_id']
        if sid in decisions or sid not in units or decision['action'] != 'context_only':
            raise ValueError('Repeated, unknown or unsupported eligibility decision')
        article, sentence = units[sid]
        if (decision['article_id'] != article['dataset_id'] or sentence['text'] != decision['original_text']
                or hashlib.sha256(sentence['text'].encode()).hexdigest() != decision['text_sha256']):
            raise ValueError('Eligibility decision differs from the reviewed exact text')
        decisions[sid] = decision
        excluded[article['language']].add(sid)
    if not excluded['bar'].issubset({s['source_id'] for s in parent['sources']}):
        raise ValueError('Source eligibility decisions must be development inputs')
    revised = copy.deepcopy(parent)
    revised['sources'] = [s for s in parent['sources'] if s['source_id'] not in excluded['bar']]
    if len(revised['sources']) != plan['expected_source_units']:
        raise ValueError('Revised shared source cohort differs from the approved count')
    revised['provenance'] = {s['source_id']: parent['provenance'][s['source_id']] for s in revised['sources']}
    audit = []
    for page_id, page in revised['pages'].items():
        old = page['candidates']
        page['candidates'] = [c for c in old if not excluded['en'].intersection(c['sentence_ids'])]
        if not page['candidates']:
            raise ValueError('Eligibility revision removed every English candidate')
        # Do not rebuild spans across removed units or change numbered article context.
        audit.append({'page_id': page_id, 'before_spans': len(old), 'after_spans': len(page['candidates']),
                      'single_units': sum(len(c['sentence_ids']) == 1 for c in page['candidates'])})
    revised.update(sources_sha256=data.digest(revised['sources']), pages_sha256=data.digest(revised['pages']),
                   parent_inputs_sha256=data.digest(parent), eligibility_plan_sha256=data.digest(plan),
                   eligibility_plan=plan, export_directory='huggingface_v2')
    settings = revised['settings']
    settings.update(run_name='finewiki_bar_en_development_review_v2',
                    protocol_freeze={'status': 'pending_user_approval', 'heldout_execution_allowed': False},
                    eligibility_plan_sha256=data.digest(plan))
    settings['segmentation']['development_source_units'] = len(revised['sources'])
    settings['segmentation']['development_target_units'] = sum(x['single_units'] for x in audit)
    settings['export']['rerank_acceptance'] = plan['rerank_acceptance']
    return revised, {'status': 'draft_development_revision', 'protocol_freeze': 'pending_user_approval',
                     'source_units': len(revised['sources']), 'context_only_bar': sorted(excluded['bar']),
                     'context_only_en': sorted(excluded['en']), 'candidate_counts': audit,
                     'parent_inputs_sha256': data.digest(parent), 'plan_sha256': data.digest(plan)}
