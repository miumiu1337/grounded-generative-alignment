"""Reprocess the frozen article pairs without altering previous datasets or outputs."""

from collections import Counter
from datetime import datetime, timezone
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import time
import uuid

from preprocessing import finewiki_pilot as data, quality_prose as quality
from runners import scale_cohort
from runners.nllb_blind_training import sha

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'results/finewiki_scale_20261102'
DEST = CORPUS / 'preprocessing_v5_20261110'
MARKER = re.compile(r'\[(?:bar|en)_\d+:fw[0-9a-f]{8}:r\d+:(?:[a-z0-9_]+:)?S\d+\] ')


def original_text(page, language):
    raw = MARKER.sub('', page[language].split('\n', 1)[1])
    if hashlib.sha256(raw.encode()).hexdigest() != page[language+'_text_sha256']:
        raise ValueError('Reconstructed article differs from its original FineWiki checksum')
    return raw


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    payload = json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()
    if path.suffix == '.gz':
        payload = gzip.compress(payload, compresslevel=1, mtime=0)
    temporary.write_bytes(payload)
    temporary.replace(path)


def article_row(bundle, language):
    header = bundle['page'][language].split('\n', 1)[0]
    match = re.fullmatch(r'(.*) \(' + language + r'\), revision (\d+)', header)
    if not match:
        raise ValueError('Malformed frozen article header')
    pid = bundle['page_id'] if language == 'bar' else bundle['page']['en_page_id']
    return {'id':language+'wiki/'+str(pid),'in_language':language,'page_id':int(pid),
        'version':int(match[2]),'title':match[1],'url':bundle['page'][language+'_url'],
        'wikidata_id':bundle['page']['wikidata_id'],'text':original_text(bundle['page'],language)}


def process(index, destination=DEST, limit=None, reuse_english=None):
    policy = {'dataset_revision':data.REVISION,'policy':quality.POLICY,
              'code_sha256':sha(Path(quality.__file__)),'runner_sha256':sha(Path(__file__))}
    if reuse_english is not None:
        reused_config = data.read_json(reuse_english/'config.json')
        reused_summary = data.read_json(reuse_english/'summary.json')
        if reused_summary['status'] != 'complete' or reused_summary['processed_articles'] != 16306:
            raise ValueError('English reuse requires a complete previous corpus pass')
        policy['english_preprocessing_id'] = reused_config['preprocessing_id']
        policy['english_reuse'] = 'previous_structurally_screened_english_sentences;original_article_checksums_verified'
    policy['preprocessing_id'] = data.digest(policy)
    config = destination / 'config.json'
    if config.exists() and data.read_json(config) != policy:
        raise ValueError('Existing preprocessing output uses another policy/code; preserve it and use another destination')
    write(config, policy)
    priority = set()
    for split in ('train','validation'):
        path = ROOT / 'notebooks/bavarian_nllb_blind_20k_20261108/package' / (split + '.jsonl')
        priority.update(str(json.loads(line)['page_id']) for line in path.read_text().splitlines())
    index = sorted(index, key=lambda item: (str(item['bar_page_id']) not in priority, int(item['bar_page_id'])))
    if limit is not None:
        index = index[:limit]
    totals = Counter();started = time.monotonic()
    try:
        for number, item in enumerate(index, 1):
            pid = str(item['bar_page_id'])
            target = destination / 'articles' / (pid + '.json.gz')
            if target.exists():
                result = data.read_json(target)
                if result['preprocessing_id'] != policy['preprocessing_id'] or result['input_sha256'] != item['inputs_sha256']:
                    raise ValueError('Cached preprocessing identity differs: ' + pid)
            else:
                signal.alarm(120)
                source = CORPUS / 'colab_bge_full_20261105/raw' / (pid + '.json.gz')
                if sha(source) != item['inputs_sha256']:
                    raise ValueError('Frozen article checksum differs: ' + pid)
                bundle = data.read_json(source)
                if data.digest(bundle['page']) != bundle['page_sha256'] or data.digest(bundle['sources']) != bundle['sources_sha256']:
                    raise ValueError('Frozen article payload differs: ' + pid)
                result = {'page_id':pid,'preprocessing_id':policy['preprocessing_id'],
                          'input_sha256':item['inputs_sha256'],'previous_source_count':len(bundle['sources'])}
                legacy = {(p['source_start'],p['source_end']):sid for sid,p in bundle['provenance'].items()}
                for language in ('bar','en'):
                    row = article_row(bundle, language)
                    if language == 'en' and reuse_english is not None:
                        previous = data.read_json(reuse_english/'articles'/(pid+'.json.gz'))
                        if (previous['preprocessing_id'] != policy['english_preprocessing_id']
                                or previous['input_sha256'] != item['inputs_sha256']
                                or previous['en']['text'] != row['text']):
                            raise ValueError('Cached English article identity differs: '+pid)
                        article = previous['en']
                    else:
                        article = quality.prepare(row)
                    article['text'] = row['text']
                    if language == 'bar':
                        for sentence in article['sentences']:
                            sentence['legacy_source_id'] = legacy.get((sentence['start'],sentence['end']))
                    result[language] = article
                signal.alarm(0)
                write(target, result)
            totals['article_pairs'] += 1
            totals['previous_eligible_source_units'] += result['previous_source_count']
            for language in ('bar','en'):
                article = result[language]
                totals[language+'_eligible_sentences'] += article['counts'].get('eligible',0)
                totals[language+'_rejected_prose_units'] += sum(not s['eligible_for_matching'] for s in article['sentences'])
                totals[language+'_excluded_blocks'] += len(article['excluded_blocks'])
                totals[language+'_boundary_repairs'] += len(article['boundary_repairs'])
                totals[language+'_split_alignment_failures'] += len(article['split_failures'])
            totals['bar_eligible_with_unchanged_legacy_span'] += sum(s['eligible_for_matching'] and s['legacy_source_id'] is not None for s in result['bar']['sentences'])
            totals['bar_eligible_needing_new_generation'] += sum(s['eligible_for_matching'] and s['legacy_source_id'] is None for s in result['bar']['sentences'])
            if number % 50 == 0 or number == len(index):
                state = {'status':'running' if number<len(index) else 'complete',
                    'preprocessing_id':policy['preprocessing_id'],'processed_articles':number,'target_articles':len(index),
                    'elapsed_seconds':round(time.monotonic()-started,2),'counts':dict(totals),
                    'updated_at':datetime.now(timezone.utc).isoformat()}
                write(destination/'status.json',state)
                print(json.dumps(state),flush=True)
            if priority.issubset({str(x['bar_page_id']) for x in index[:number]}) and not (destination/'published_cohort_ready.json').exists():
                write(destination/'published_cohort_ready.json',{'article_pairs':len(priority),'preprocessing_id':policy['preprocessing_id']})
        write(destination/'summary.json',state)
    except BaseException as error:
        signal.alarm(0)
        write(destination/'failure.json',{'page_id':pid,'error_class':type(error).__name__,'error':str(error),
              'completed_articles':number-1,'preprocessing_id':policy['preprocessing_id']})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int)
    parser.add_argument('--destination',type=Path,default=DEST)
    parser.add_argument('--reuse-english',type=Path)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:
        raise ValueError('Article limit must be positive')
    signal.signal(signal.SIGALRM,lambda *args: (_ for _ in ()).throw(TimeoutError('Article preprocessing exceeded 120 seconds')))
    os.nice(10)
    process(scale_cohort.index(CORPUS),args.destination,args.limit,args.reuse_english)


if __name__=='__main__':main()
