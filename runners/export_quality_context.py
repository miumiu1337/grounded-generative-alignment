"""Export article context and a separately versioned, structurally screened corpus."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from preprocessing import finewiki_pilot as data

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'results/finewiki_scale_20261102'
PREPARED = CORPUS / 'preprocessing_v5_20261110'
WORK = ROOT / 'publication_work_20261110'
OUT = WORK / 'huggingface'
PACKAGES = {
    'blind': 'bavarian_nllb_blind_20k_20261108/package',
    'laser': 'bavarian_nllb_mining_20k_20261108/package/laser',
    'bge_m3': 'bavarian_nllb_mining_20k_20261108/package/bge_m3',
    'variant_a': 'bavarian_nllb_variant_a_20k_20261109/package',
    'variant_b': 'bavarian_nllb_variant_b_20k_20261109/package',
}
CONTEXT = ('article', 'english_article', 'paragraph')


def read(path):
    payload = path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix == '.gz' else payload)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(rows):
    """Dictionary encode repeated context without expanding articles in memory."""
    columns = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        if key in CONTEXT:
            unique, lookup, indices = [], {}, []
            for value in values:
                if value not in lookup:
                    lookup[value] = len(unique)
                    unique.append(value)
                indices.append(lookup[value])
            columns[key] = pa.DictionaryArray.from_arrays(pa.array(indices, type=pa.int32()), pa.array(unique, type=pa.string()))
        elif key == 'heading_path':
            columns[key] = pa.array(values, type=pa.list_(pa.string()))
        elif key in {'legacy_source_id', 'source_id_v5', 'preprocessing_v5_reason', 'wikidata_id'}:
            columns[key] = pa.array(values, type=pa.string())
        else:
            columns[key] = pa.array(values)
    return pa.table(columns)


def parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table(rows), path, compression='zstd', use_dictionary=True,
                   dictionary_pagesize_limit=64*1024*1024, store_schema=False,
                   write_page_checksum=True)
    # Read context as dictionaries to avoid allocating a full article per row.
    restored = pq.read_table(path, read_dictionary=list(CONTEXT))
    if restored.num_rows != len(rows):
        raise ValueError('Parquet row count changed: ' + str(path))
    for key in rows[0]:
        if key in CONTEXT:
            offset = 0
            for chunk in restored[key].chunks:
                unique = chunk.dictionary.to_pylist()
                for index in chunk.indices.to_pylist():
                    if unique[index] != rows[offset][key]:
                        raise ValueError('Parquet context changed: ' + str(path) + '/' + key)
                    offset += 1
        elif restored[key].to_pylist() != [row[key] for row in rows]:
            raise ValueError('Parquet values changed: ' + str(path) + '/' + key)


@lru_cache(maxsize=768)
def article(pid):
    value = read(PREPARED / 'articles' / (pid + '.json.gz'))
    raw = read(CORPUS / 'colab_bge_full_20261105/raw' / (pid + '.json.gz'))
    value['legacy_provenance'] = raw['provenance']
    for key in ('sentences', 'excluded_blocks', 'boundary_repairs', 'split_failures'):
        value['en'].pop(key, None)
    value['blocks'] = data.blocks(value['bar']['text'])
    value['spans'] = {(s['start'], s['end']): s for s in value['bar']['sentences']}
    return value


def context(value, start, end, paragraph_start, paragraph_end):
    bar, en = value['bar'], value['en']
    return {'article': bar['text'], 'article_title': bar['title'], 'article_url': bar['url'],
        'article_page_id': str(bar['page_id']), 'article_revision': str(bar['revision_id']),
        'article_text_sha256': bar['text_sha256'], 'wikidata_id': bar['wikidata_id'],
        'english_article': en['text'], 'english_article_title': en['title'],
        'english_article_url': en['url'], 'english_article_page_id': str(en['page_id']),
        'english_article_revision': str(en['revision_id']), 'english_article_text_sha256': en['text_sha256'],
        'sentence_start': start, 'sentence_end': end,
        'paragraph': bar['text'][paragraph_start:paragraph_end],
        'paragraph_start': paragraph_start, 'paragraph_end': paragraph_end}


def enrich(row):
    value = article(str(row['page_id']))
    provenance = value['legacy_provenance'][row['source_id']]
    start, end = provenance['source_start'], provenance['source_end']
    if value['bar']['text'][start:end] != row['bavarian']:
        raise ValueError('Legacy text/offset mismatch: ' + row['source_id'])
    block = next((b for b in value['blocks'] if b['start'] <= start and end <= b['end']), None)
    if block is None:
        raise ValueError('Legacy sentence crosses structural blocks: ' + row['source_id'])
    revised = value['spans'].get((start, end))
    reason = None
    if revised is not None:
        reason = revised['context_only_reason']
    else:
        excluded = next((b for b in value['bar']['excluded_blocks'] if b['start'] <= start and end <= b['end']), None)
        reason = excluded['reason'] if excluded else 'sentence_boundaries_changed'
    return {'source_lang':'bar', 'target_lang':'en', 'source_text':row['bavarian'], 'target_text':row['english'],
        'source_id':row['source_id'], **context(value,start,end,block['start'],block['end']),
        'preprocessing_v5_eligible':reason is None, 'preprocessing_v5_reason':reason,
        'source_id_v5': revised['sentence_id'] if revised is not None else None,
        'quality_status':'historical_parallel_candidate;translation_alignment_not_human_validated'}


def main():
    if OUT.exists():
        raise FileExistsError('Preserve the existing prepared release: ' + str(OUT))
    ready = read(PREPARED / 'published_cohort_ready.json')
    policy = read(PREPARED / 'config.json')
    if ready['preprocessing_id'] != policy['preprocessing_id']:
        raise ValueError('Preprocessing identity differs')
    OUT.mkdir(parents=True)
    old_summary, partitions, cohorts = {}, {}, {}
    for split in ('train', 'validation'):
        package = ROOT / 'notebooks' / PACKAGES['blind']
        cohorts[split] = [json.loads(line) for line in (package / (split+'.jsonl')).read_text().splitlines()]
        for row in cohorts[split]:
            pid = str(row['page_id'])
            if pid in partitions and partitions[pid] != split:
                raise ValueError('Historical article split overlaps')
            partitions[pid] = split
    for method, relative in PACKAGES.items():
        package = ROOT / 'notebooks' / relative
        config = 'bar_en_20k_' + method
        old_summary[method] = {}
        for split in ('train', 'validation'):
            rows = [json.loads(line) for line in (package/(split+'.jsonl')).read_text().splitlines()]
            if [r['source_id'] for r in rows] != [r['source_id'] for r in cohorts[split]]:
                raise ValueError('Matched source order differs: '+method+'/'+split)
            enriched = [enrich(row) for row in rows]
            historical = pq.read_table(ROOT/'publication_work_20261109/huggingface/data'/config/(split+'.parquet'))
            for key in ('source_lang','target_lang','source_text','target_text'):
                if historical[key].to_pylist() != [row[key] for row in enriched]:
                    raise ValueError('Historical published values changed: '+method+'/'+split+'/'+key)
            parquet(OUT/'data'/config/(split+'.parquet'), enriched)
            old_summary[method][split] = dict(Counter(row['preprocessing_v5_reason'] or 'retained_unchanged' for row in enriched))
            print('Context added and round-trip checked:',method,split,len(enriched),flush=True)
    article.cache_clear()
    # The new corpus is intentionally separate from the historical parallel data.
    # Changed spans have no valid inherited translations.
    summary = read(PREPARED/'summary.json')
    if summary['status'] != 'complete' or summary['processed_articles'] != 16306:
        raise ValueError('Full revised corpus is not complete; do not publish a partial inventory')
    inventory, counts, shards, shard_number = [], Counter(), [], 0
    def flush():
        nonlocal shard_number
        if not inventory:
            return
        name = f'data/bar_sentences_v5/corpus-{shard_number:05d}.parquet'
        parquet(OUT/name, inventory)
        shards.append({'path':name,'rows':len(inventory),'bytes':(OUT/name).stat().st_size,'sha256':sha(OUT/name)})
        inventory.clear();shard_number += 1
    for number, path in enumerate(sorted((PREPARED/'articles').glob('*.json.gz')),1):
        value = read(path)
        if value['preprocessing_id'] != policy['preprocessing_id']:
            raise ValueError('Mixed preprocessing identities')
        bar = value['bar']
        for sentence in bar['sentences']:
            if not sentence['eligible_for_matching']:
                continue
            if bar['text'][sentence['start']:sentence['end']] != sentence['text']:
                raise ValueError('Revised sentence does not match original article')
            inventory.append({'source_lang':'bar', 'source_text':sentence['text'],
                'source_id':sentence['sentence_id'], **context(value,sentence['start'],sentence['end'],sentence['paragraph_start'],sentence['paragraph_end']),
                'heading_path':sentence['heading_path'], 'legacy_source_id':sentence['legacy_source_id'],
                'legacy_span_unchanged':sentence['legacy_source_id'] is not None,
                'historical_article_partition':partitions.get(str(bar['page_id']),'unassigned'),
                'quality_status':'structurally_screened;translation_and_language_quality_not_human_validated'})
            counts['sentences'] += 1
            counts['changed_or_new_spans_requiring_generation'] += sentence['legacy_source_id'] is None
            if len(inventory) >= 4000:
                flush()
        if number%1000 == 0:
            print('Exported revised articles:',number,'sentences:',counts['sentences'],flush=True)
    flush()
    if counts['sentences'] != summary['counts']['bar_eligible_sentences']:
        raise ValueError('Revised inventory count differs from preprocessing summary')
    release = OUT/'releases/preprocessing_v5'
    write(release/'config.json',policy)
    write(release/'summary.json',summary)
    write(release/'historical_20k_screening.json',old_summary)
    write(release/'inventory.json',{'counts':dict(counts),'shards':shards,
        'translation_reuse_policy':'Only exact unchanged spans may reuse saved outputs; repaired/new spans require new translations',
        'historical_datasets':'Original 20k sentence/translation values and row order preserved; article context and v5 flags added',
        'inventory_split':'Unsplit source inventory; historical article partition is metadata, not a new evaluation protocol'})
    for relative in ('preprocessing/quality_prose.py','runners/refine_preprocessing.py',
                     'runners/export_quality_context.py','tests/test_quality_prose.py','tests/test_quality_context.py'):
        target = release/'code'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/relative).read_bytes())
    body = (WORK/'remote_README.md').read_text()
    marker = '\n---\n'
    front, body = body.split(marker,1)
    front = front.replace('  default: true\n','')
    front += ('\n- config_name: bar_sentences_v5\n  default: true\n  data_files:\n'
              '  - split: corpus\n    path: data/bar_sentences_v5/corpus-*.parquet')
    note = ('\n## Revised preprocessing and article context\n\n'
        f"`bar_sentences_v5` contains {counts['sentences']:,} structurally screened Bavarian prose sentences from all 16,306 frozen article pairs. "
        'Sentence boundaries are found within whole prose paragraphs; cuts inside brackets and after nonterminal abbreviations are merged. '
        'Lists, labels, headings, reference sections, incomplete units, unresolved markup and within-article duplicates are excluded from sentence training data. '
        'Units whose first alphabetic character is lowercase are conservatively withheld for review; this may exclude valid dialect forms. '
        'Original dialect wording is preserved as exact article substrings. This is structural screening, not a human assessment of language or translation quality.\n\n'
        'The `article` column contains the full original Bavarian article. `article_title` and `article_url` identify it. '
        '`english_article`, `english_article_title` and `english_article_url` provide the matched English article. '
        '`paragraph` provides local context; `sentence_start` and `sentence_end` are zero-based Python character offsets into `article`. '
        'Revision IDs and article text SHA-256 hashes preserve attribution and traceability. Articles remain unfiltered context, including lists and references.\n\n'
        'The five `bar_en_20k_*` configurations preserve the exact historical sentence/translation values and row order used for the original experiments. '
        'They now include the same context columns plus `preprocessing_v5_eligible` and `preprocessing_v5_reason`. '
        'Those flags assess the Bavarian source structure; they do not validate the English translation or mining alignment. '
        'The reported model scores still refer to those historical inputs.\n\n'
        'The original release file hashes in `releases/20k/manifest.json` apply to historical revision '
        f"`{read(WORK/'base.json')['revision']}`; the new context-enriched files have separate hashes in `releases/preprocessing_v5/manifest.json`.\n\n"
        f"The revised inventory has {counts['changed_or_new_spans_requiring_generation']:,} changed or new spans requiring new translations. "
        '`legacy_source_id` is present only when the original sentence span is unchanged; saved outputs must never be attached to a repaired span. '
        'The inventory has no English translation column and is not a ready-made parallel training split. '
        'Before training, assign disjoint article/topic splits, screen benchmark overlap and review translation alignment. '
        '`historical_article_partition` records only the original 20k experiment partition. '
        'The preprocessing policy, counts and screening summary are in `releases/preprocessing_v5/`.\n')
    (OUT/'README.md').write_text(front+marker+body+note)
    artifacts=[{'path':p.relative_to(OUT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size}
               for p in sorted(OUT.rglob('*')) if p.is_file()]
    write(release/'manifest.json',{'release':'preprocessing_v5_20261110','parent_revision':read(WORK/'base.json')['revision'],
        'artifacts':artifacts,'original_text_and_order_preserved':True,'parquet_round_trip_verified':True})
    print(json.dumps({'status':'prepared_and_verified','files':len(artifacts)+1,
        'bytes':sum(p['bytes'] for p in artifacts),'counts':dict(counts),'historical_screening':old_summary['blind']}),flush=True)


if __name__=='__main__':
    main()
