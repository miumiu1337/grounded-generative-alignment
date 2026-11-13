"""Five methods over one frozen cohort; neighboring sentences are context only."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from parallel_data_generation import corrections, wiki_protocol, wiki_selection
from parallel_data_generation.prompts import load_prompts
from preprocessing.artifacts import load_config, read_json, safe_path, sha, write_json
from preprocessing.finewiki_pilot import digest

METHODS = ('blind', 'mining', 'variant_a', 'variant_b', 'variant_c')
CONDITIONS = ('sentence_only', 'paragraph_context')
TRANSLATION_SCHEMA = {'type': 'object', 'properties': {'translation': {'type': 'string', 'minLength': 1}},
                      'required': ['translation'], 'additionalProperties': False}


def prepare_cases(cleaned, config, maximum_sources):
    manifest = read_json(cleaned/'manifest.json')
    if manifest['status'] != 'complete' or manifest['config'] != config:
        raise ValueError('Cleaning must be complete and use the same language configuration')
    articles = {}
    for name, expected in manifest['files'].items():
        path = safe_path(cleaned, name)
        if sha(path) != expected:
            raise ValueError('Cleaned article checksum differs: '+name)
        article = read_json(path); articles[article['article_id']] = article
    cases = []; english = {}
    for pair in manifest['matched_pairs']:
        source = articles[pair['article_id']]; target = articles[pair['en_article_id']]
        if source['wikidata_id'] != target['wikidata_id'] or target['revision'] != pair['en_revision_id']:
            raise ValueError('Matched article topic or revision differs')
        # Freeze the same focus IDs and English pool for both context conditions.
        english[target['article_id']] = {'candidates': target['windows'], 'revision': target['revision'],
            'wikidata_id': target['wikidata_id'], 'title': target['title']}
        for window in source['windows']:
            if len(window['context_sentence_ids']) != 3 or window['focus_index'] != 1:
                raise ValueError('Expected a complete three-sentence window with middle focus')
            cases.append({'source': {**window, 'language': config['source_language'],
                'source_article_title': source['title'], 'target_article_title': target['title'],
                'english_match': pair}, 'blind': {}})
    # Stable preprocessing order; no translation success or model scores select the cohort.
    cases.sort(key=lambda case: (case['source']['article_id'], case['source']['source_char_start']))
    return cases[:maximum_sources], english


def focus_payload(source, condition):
    if condition not in CONDITIONS:
        raise ValueError('Unknown context condition')
    value = {'focus_sentence': source['focus_text']}
    if condition == 'paragraph_context':
        if source['focus_index'] != 1 or len(source['context_sentence_ids']) != 3:
            raise ValueError('Context must contain exactly previous, focus and next sentences')
        value.update(context=source['context_text'], section_path=source['section_path'])
    return value


def translation(value):
    if (not isinstance(value, dict) or set(value) != {'translation'}
            or not isinstance(value['translation'], str) or not value['translation'].strip()):
        raise ValueError('Malformed Blind translation')
    return value['translation']


def blind(case, condition, prompts, generate):
    request = {'messages': [{'role': 'system', 'content': prompts['blind']},
        {'role': 'user', 'content': json.dumps(focus_payload(case['source'], condition), ensure_ascii=False)}],
        'schema': TRANSLATION_SCHEMA, 'grammar': None}
    return translation(generate('blind', request))


def refine_c(case, condition, sources, prompts, generate):
    draft = case['blind'][condition]['translation']; source = case['source']
    if not sources:
        return {'translation': draft, 'decision': 'no_external_evidence', 'evidence': []}
    payload = {**focus_payload(source, condition), 'blind_translation': draft,
        'article_topic': {'source_title': source['source_article_title'], 'target_title': source['target_article_title']},
        'evidence': sources}
    def call(stage, value, schema):
        return generate(stage, {'messages': [{'role': 'system', 'content': prompts['variant_c'][stage]},
            {'role': 'user', 'content': json.dumps(value, ensure_ascii=False)}], 'schema': schema, 'grammar': None})
    proposal = call('proposal', payload, corrections.PROPOSAL_SCHEMA)
    result = {'translation': draft, 'decision': 'proposal_kept_blind', 'evidence': sources, 'proposal': proposal}
    try:
        corrections.validate_proposal(proposal, draft, sources)
    except (ValueError, KeyError, TypeError) as error:
        return {**result, 'decision': 'invalid_proposal_keep_blind', 'validation_error': str(error)}
    if proposal['translation'] == draft:
        return result
    verification = call('verification', {**payload, 'proposal': proposal}, corrections.VERIFICATION_SCHEMA)
    result['verification'] = verification
    try:
        accepted = corrections.accept(verification)
    except (ValueError, KeyError, TypeError) as error:
        return {**result, 'decision': 'invalid_verification_keep_blind', 'validation_error': str(error)}
    if accepted:
        result.update(translation=proposal['translation'], decision='supported_correction_verified')
    else:
        result['decision'] = 'verification_rejected_or_uncertain_keep_blind'
    return result


def training_pair(source, english):
    if not isinstance(english, str) or not english.strip():
        raise ValueError('Empty English training text')
    return {'source_lang': 'en', 'target_lang': source['language'], 'source_text': english,
        'target_text': source['focus_text'], 'source_article': source['target_article_title'],
        'target_article': source['source_article_title']}


def _run(config_path, cleaned, output, method, condition, maximum_sources=5000, endpoint=None, fineweb_index=None):
    if method not in METHODS or condition not in CONDITIONS or maximum_sources < 1:
        raise ValueError('Choose one of the five methods, a valid context condition and a positive cohort size')
    config = load_config(config_path); prompts = load_prompts(config)
    cases, articles = prepare_cases(cleaned, config, maximum_sources)
    if not cases:
        raise ValueError('No eligible shared source windows after cleaning')
    source_root = Path(__file__).resolve().parents[1]
    code = [*Path(__file__).parent.glob('*.py'), *Path(__file__).parent.glob('prompts/*.txt'),
            *source_root.joinpath('Cleaning').glob('*.py'), *source_root.joinpath('preprocessing').glob('*.py')]
    settings = {'config': config, 'prompts': prompts, 'cleaned_manifest_sha256': sha(cleaned/'manifest.json'),
        'source_ids': [case['source']['source_id'] for case in cases], 'code_sha256': {
            str(path.relative_to(source_root)): sha(path) for path in sorted(code)}}
    identity = digest(settings); root = output/identity
    write_json(root/'protocol.json', settings)
    if method == 'variant_c' and fineweb_index is None:
        raise ValueError('Variant C requires a completed FineWeb index')
    evidence_sha = sha(fineweb_index) if method == 'variant_c' else None
    write_json(root/method/condition/'assignment.json', {'protocol_id': identity, 'method': method,
        'condition': condition, 'evidence_index_sha256': evidence_sha, 'assigned_sources': len(cases)})
    teacher = None
    if method != 'mining':
        if endpoint is None:
            raise ValueError('Provide an already running llama.cpp --endpoint; no cloud job is launched here')
        from parallel_data_generation.teacher import Teacher
        teacher = Teacher(endpoint, config['teacher'], root/'raw')
    groups = defaultdict(list)
    for case in cases:
        groups[case['source']['english_match']['en_article_id']].append(case)
    encoder = None; completed = deferred = 0
    for article_id, group in groups.items():
        article = articles[article_id]; mining_rows = {}
        if method == 'mining' and any(not (root/method/condition/(case['source']['source_id']+'.json.gz')).exists() for case in group):
            from parallel_data_generation.mining import Encoder, mine
            if encoder is None:
                encoder = Encoder(config['mining'])
            for row in mine([case['source'] for case in group], article['candidates'], encoder,
                            config['source_language'], condition):
                mining_rows[row['source']['source_id']] = row
        for case in group:
            source = case['source']; sid = source['source_id']
            path = root/method/condition/(sid+'.json.gz')
            failure = root/'deferred'/method/condition/(sid+'.json.gz')
            if path.exists():
                saved = read_json(path)
                if saved['protocol_id'] != identity or saved['source_id'] != sid:
                    raise ValueError('Saved result belongs to another source or protocol')
                completed += 1; continue
            if failure.exists():
                deferred += 1; continue
            try:
                if method in ('blind', 'variant_a', 'variant_c'):
                    draft = blind(case, condition, prompts, teacher.generate)
                    case['blind'][condition] = {'translation': draft}
                if method == 'blind':
                    result = {'translation': draft, 'decision': 'blind_translation'}
                elif method == 'mining':
                    ranking = mining_rows[sid]
                    if not ranking['candidates']:
                        raise ValueError('No eligible English mining candidate; source retained in coverage')
                    result = {'translation': ranking['candidates'][0]['focus_text'],
                        'decision': 'top1_mining_candidate', 'ranking': ranking, 'encoder': config['mining']['encoder']}
                elif method in ('variant_a', 'variant_b'):
                    selected, trace = wiki_selection.select(case, article, method, condition, prompts,
                                                           teacher.fits, teacher.generate)
                    stage = 'refinement' if method == 'variant_a' else 'translation'
                    generated = None
                    if method == 'variant_b' or selected['match_status'] != 'none':
                        generated = teacher.generate(stage, wiki_protocol.request(case, article, method, condition,
                            stage, prompts, selection=selected))
                    result = wiki_protocol.finish(case, article, method, condition, identity, selected, generated)
                    result['selection_trace'] = trace
                else:
                    from parallel_data_generation.fineweb import search
                    evidence = search(fineweb_index, article_id, source['target_article_title']+' '+draft,
                                      config['fineweb']['sources_per_sentence'])
                    result = refine_c(case, condition, evidence, prompts, teacher.generate)
                result.update(source_id=sid, method=method, condition=condition, protocol_id=identity,
                    training_pair=training_pair(source, result['translation']), quality_status='unassessed')
                write_json(path, result); completed += 1
            except (ValueError, KeyError, TypeError) as error:
                write_json(failure, {'source_id': sid, 'method': method, 'condition': condition,
                    'error_class': type(error).__name__, 'error': str(error), 'protocol_id': identity})
                deferred += 1
                print(json.dumps({'event': 'source_deferred', 'source_id': sid, 'reason': str(error)}), flush=True)
            print(json.dumps({'method': method, 'condition': condition, 'completed': completed,
                              'deferred': deferred, 'assigned': len(cases)}), flush=True)
    summary = {'protocol_id': identity, 'method': method, 'condition': condition,
        'completed': completed, 'deferred': deferred, 'assigned': len(cases),
        'status': 'complete' if not deferred else 'complete_with_explicit_deferred_sources'}
    write_json(root/method/condition/'summary.json', summary)
    return summary


def run(config_path, cleaned, output, method, condition, maximum_sources=5000, endpoint=None, fineweb_index=None):
    import fasteners
    output.mkdir(parents=True, exist_ok=True)
    lock = fasteners.InterProcessLock(str(output/'generation.lock'))
    if not lock.acquire(blocking=False):
        raise RuntimeError('Another generation worker owns this output directory; shared Blind requests must have one writer')
    try:
        return _run(config_path, cleaned, output, method, condition, maximum_sources, endpoint, fineweb_index)
    finally:
        lock.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cleaned', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--condition', choices=CONDITIONS, default='sentence_only')
    parser.add_argument('--maximum-sources', type=int, default=5000)
    parser.add_argument('--endpoint')
    parser.add_argument('--fineweb-index', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.cleaned, args.output, args.method, args.condition,
                         args.maximum_sources, args.endpoint, args.fineweb_index)))
