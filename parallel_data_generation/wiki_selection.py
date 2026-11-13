"""Select from every article candidate using bounded, overlapping requests."""

import json

from preprocessing import finewiki_pilot as data
from parallel_data_generation import variant_a as shared
from parallel_data_generation import wiki_protocol as protocol


class SelectionCapacityError(ValueError):
    pass


class SelectionResponseError(ValueError):
    pass


def subset(page, sentence_ids, evidence_ids=None):
    ids = set(sentence_ids)
    if len(ids) != len(sentence_ids) or not ids <= {s['id'] for s in page['sentences']}:
        raise ValueError('Unknown or duplicate English candidate IDs')
    allowed = None if evidence_ids is None else {tuple(value) for value in evidence_ids}
    result = {'sentences': [row for row in page['sentences'] if row['id'] in ids],
        'candidates': [row for row in page['candidates'] if set(row['sentence_ids']) <= ids
            and (allowed is None or tuple(row['sentence_ids']) in allowed)]}
    if allowed is not None and {tuple(row['sentence_ids']) for row in result['candidates']} != allowed:
        raise ValueError('Unknown evidence span in selection plan')
    return result


def request(case, article, method, condition, prompts, page):
    result = protocol.request(case, article, method, condition, 'selection', prompts)
    body = json.loads(result['messages'][1]['content'])
    body['english_candidates'] = page['sentences']
    body['allowed_evidence_spans'] = [row['sentence_ids'] for row in page['candidates']]
    result['messages'][1]['content'] = json.dumps(body, ensure_ascii=False)
    result['schema'] = shared.schema_for('selection', {'source': case['source']}, page)
    result['grammar'] = shared.selection_grammar({'source': case['source']}, page) if page['candidates'] else None
    return result


def partition(page, fits):
    """A boundary overlap retains every allowed pair as well as every sentence."""
    if fits(page):
        return [page]
    rows = page['sentences']
    if len(rows) <= 2:
        raise SelectionCapacityError('An individual English evidence unit exceeds the full prompt capacity; no truncation')
    at = len(rows)//2
    left = subset(page, [row['id'] for row in rows[:at+1]])
    right = subset(page, [row['id'] for row in rows[at:]])
    return partition(left, fits)+partition(right, fits)


def descriptor(page):
    return {'sentence_ids': [row['id'] for row in page['sentences']],
        'evidence_ids': [row['sentence_ids'] for row in page['candidates']]}


def coverage(pool, pages):
    if ({row['id'] for page in pages for row in page['sentences']} != {row['id'] for row in pool['sentences']}
            or {tuple(row['sentence_ids']) for page in pages for row in page['candidates']}
            != {tuple(row['sentence_ids']) for row in pool['candidates']}):
        raise ValueError('Selection chunks do not cover the complete candidate pool')


def next_pool(full, selected):
    spans = list(dict.fromkeys(tuple(row['evidence_ids']) for row in selected if row['evidence_ids']))
    ids = {sid for span in spans for sid in span}
    return subset(full, [row['id'] for row in full['sentences'] if row['id'] in ids], spans)


def select(case, article, method, condition, prompts, fits_request, generate):
    full = protocol.page_for(article, condition); pool = full; rounds = []
    if not full['candidates']:
        return {'source_id': case['source']['source_id'], 'evidence_ids': [],
                'match_status': 'none', 'contradiction': False}, []
    for round_index in range(8):
        def fits(page):
            return fits_request(request(case, article, method, condition, prompts, page))
        pages = partition(pool, fits); coverage(pool, pages)
        trace = []; selections = []
        for index, page in enumerate(pages):
            stage = 'selection' if round_index == 0 and len(pages) == 1 else f'selection_r{round_index:02d}_c{index:04d}'
            value = generate(stage, request(case, article, method, condition, prompts, page))
            try:
                shared.validate_selection(value, {'source': case['source']}, page)
            except (ValueError, KeyError, TypeError) as error:
                raise SelectionResponseError('Invalid evidence selection; original raw response retained') from error
            selections.append(value); trace.append({'stage': stage, **descriptor(page)})
        rounds.append(trace)
        unique = {}
        for value in selections:
            if value['evidence_ids']:
                unique.setdefault(data.digest(value), value)
        if len(unique) <= 1:
            selected = next(iter(unique.values())) if unique else {'source_id': case['source']['source_id'],
                'evidence_ids': [], 'match_status': 'none', 'contradiction': False}
            return selected, rounds
        new_pool = next_pool(full, list(unique.values()))
        if round_index and descriptor(new_pool) == descriptor(pool):
            raise SelectionCapacityError('Evidence adjudication cannot reduce the pool within the context bound; no candidates omitted')
        pool = new_pool
    raise SelectionCapacityError('Evidence selection reached its bounded eight rounds; no translation accepted')


def replay(case, article, method, condition, prompts, rounds, generate):
    """Verify every chunk's response and coverage without calling the teacher."""
    if rounds == [] and not protocol.page_for(article, condition)['candidates']:
        return {'source_id': case['source']['source_id'], 'evidence_ids': [],
                'match_status': 'none', 'contradiction': False}
    if not isinstance(rounds, list) or not 1 <= len(rounds) <= 8:
        raise ValueError('Invalid evidence selection trace')
    full = protocol.page_for(article, condition); pool = full
    for round_index, trace in enumerate(rounds):
        if not isinstance(trace, list) or not trace:
            raise ValueError('Empty selection round')
        pages = [subset(pool, row['sentence_ids'], row['evidence_ids']) for row in trace]
        coverage(pool, pages); unique = {}
        for index, (row, page) in enumerate(zip(trace, pages)):
            stage = 'selection' if round_index == 0 and len(pages) == 1 else f'selection_r{round_index:02d}_c{index:04d}'
            if row['stage'] != stage:
                raise ValueError('Selection trace stage differs')
            value = generate(stage, request(case, article, method, condition, prompts, page))
            shared.validate_selection(value, {'source': case['source']}, page)
            if value['evidence_ids']:
                unique.setdefault(data.digest(value), value)
        if len(unique) <= 1:
            if round_index != len(rounds)-1:
                raise ValueError('Selection trace continues after the final decision')
            return next(iter(unique.values())) if unique else {'source_id': case['source']['source_id'],
                'evidence_ids': [], 'match_status': 'none', 'contradiction': False}
        pool = next_pool(full, list(unique.values()))
    raise ValueError('Selection trace has no final adjudicated decision')
