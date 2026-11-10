"""Publish every original source ID with proposed repairs and article context."""

from collections import Counter
from functools import lru_cache
import json
from pathlib import Path

import pyarrow.parquet as pq

from preprocessing import finewiki_pilot as data
from runners.export_quality_context import parquet, sha, write, PACKAGES

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'results/finewiki_scale_20261102'
PREPARED=CORPUS/'preprocessing_v6_20261110'
WORK=ROOT/'publication_work_20261110_retained'
OUT=WORK/'huggingface'
CONTEXT=('article','english_article','paragraph')


@lru_cache(maxsize=768)
def article(pid):
    value=data.read_json(PREPARED/'articles'/(pid+'.json.gz'))
    value['lookup']={r['source_id']:r for r in value['records']}
    return value


def context(value,record):
    bar,en=value['bar'],value['en']
    return {'article':bar['text'],'article_title':bar['title'],'article_url':bar['url'],
        'article_page_id':str(bar['page_id']),'article_revision':str(bar['version']),
        'article_text_sha256':value['article_text_sha256'],'wikidata_id':bar['wikidata_id'],
        'english_article':en['text'],'english_article_title':en['title'],'english_article_url':en['url'],
        'english_article_page_id':str(en['page_id']),'english_article_revision':str(en['version']),
        'paragraph':bar['text'][record['paragraph_start']:record['paragraph_end']]}


def main():
    summary=data.read_json(PREPARED/'summary.json')
    if summary['status']!='complete' or summary['counts']['retained_source_records']!=225700:
        raise ValueError('The complete original 225700 source IDs must be retained before publication')
    if OUT.exists():raise FileExistsError('Preserve the existing prepared update: '+str(OUT))
    OUT.mkdir(parents=True)
    partitions={};cohorts={}
    for split in ('train','validation'):
        rows=[json.loads(line) for line in (ROOT/'notebooks'/PACKAGES['blind']/(split+'.jsonl')).read_text().splitlines()]
        cohorts[split]=[r['source_id'] for r in rows]
        for row in rows:
            pid=str(row['page_id'])
            if pid in partitions and partitions[pid]!=split:raise ValueError('Historical article split overlaps')
            partitions[pid]=split
    readiness={};source_checks={}
    for method in PACKAGES:
        config='bar_en_20k_'+method
        readiness[method]={}
        for split in ('train','validation'):
            source=ROOT/'publication_work_20261110/huggingface/data'/config/(split+'.parquet')
            names=pq.read_schema(source).names
            rows=pq.read_table(source,columns=[name for name in names if name not in CONTEXT]).to_pylist()
            if [r['source_id'] for r in rows]!=cohorts[split]:raise ValueError('Historical source order differs')
            counters=Counter()
            for row in rows:
                value=article(row['article_page_id']);record=value['lookup'][row['source_id']]
                if row['source_text']!=record['original_source_text']:raise ValueError('Historical training source text changed')
                row.update({key:val for key,val in context(value,record).items() if key in CONTEXT})
                row.update(cleaned_source_text=record['source_text'],cleaned_sentence=record['cleaned_sentence'],
                    cleaned_unit_id=record['cleaned_unit_id'],cleaning_status=record['cleaning_status'],
                    source_candidate_ready=record['source_candidate_ready'],review_reasons=record['review_reasons'],
                    cleaning_operations=record['cleaning_operations'],contributing_source_ids=record['contributing_source_ids'],
                    translation_status=record['translation_status'],original_record_retained=True,
                    cleaned_sentence_start=record['cleaned_sentence_start'],cleaned_sentence_end=record['cleaned_sentence_end'])
                counters['records']+=1
                counters['changed_bavarian_targets']+=row['cleaned_source_text']!=row['source_text']
                counters['source_clause_candidates']+=record['source_candidate_ready']
                counters['requires_contextual_or_grammar_review']+=not record['source_candidate_ready']
                counters['requires_new_english_for_context_repair']+=record['context_expanded']
            parquet(OUT/'data'/config/(split+'.parquet'),rows)
            # The historical pair columns above are never changed by this export.
            readiness[method][split]=dict(counters)
            source_checks[config+'/'+split]={'historical_rows':len(rows),'source_order_preserved':True,
                                          'historical_source_and_target_columns_preserved':True}
            print('Retained historical rows and added repair columns:',method,split,len(rows),flush=True)
    article.cache_clear()
    ids=set();batch=[];shards=[];count=0
    def flush():
        if not batch:return
        name=f'data/bar_units_v6/corpus-{len(shards):05d}.parquet'
        parquet(OUT/name,batch)
        shards.append({'path':name,'rows':len(batch),'bytes':(OUT/name).stat().st_size,'sha256':sha(OUT/name)})
        batch.clear()
    for number,path in enumerate(sorted((PREPARED/'articles').glob('*.json.gz')),1):
        value=data.read_json(path)
        for record in value['records']:
            sid=record['source_id']
            if sid in ids:raise ValueError('Repeated original source ID')
            ids.add(sid)
            if value['bar']['text'][record['original_sentence_start']:record['original_sentence_end']]!=record['original_source_text']:
                raise ValueError('Original source text was lost or changed')
            if value['bar']['text'][record['cleaned_sentence_start']:record['cleaned_sentence_end']]!=record['article_span_text']:
                raise ValueError('Repaired source span does not match original article context')
            batch.append({'source_lang':'bar',**record,**context(value,record),
                'historical_article_partition':partitions.get(str(value['bar']['page_id']),'unassigned')})
            count+=1
            if len(batch)>=4000:flush()
        if number%2000==0:print('Exported original articles:',number,'retained records:',count,flush=True)
    flush()
    if count!=225700 or len(ids)!=225700:raise ValueError('Publication must preserve all 225700 original IDs')
    release=OUT/'releases/preprocessing_v6'
    write(release/'config.json',data.read_json(PREPARED/'config.json'))
    write(release/'summary.json',summary)
    import hashlib
    write(release/'source_retention.json',{'original_records':225700,'retained_records':count,'dropped_records':0,
        'unique_original_source_ids':len(ids),'sorted_source_ids_sha256':hashlib.sha256(('\n'.join(sorted(ids))+'\n').encode()).hexdigest(),
        'inventory_shards':shards,'historical_20k_checks':source_checks})
    write(release/'retraining_readiness.json',{'status':'repair_and_parallel_alignment_review_required_before_retraining',
        'direction':'English to Bavarian','base_model':'facebook/nllb-200-distilled-600M',
        'base_revision':'f8d333a098d19b4fd9a8b18f94170487ad3f821d','restart_from_original_base':True,
        'matched_conditions':['blind','laser','bge_m3','variant_a','variant_b'],
        'keep_original_source_ids_and_article_disjoint_validation':True,
        'required_steps':['Review unresolved source clauses without dropping source records',
                          'Translate or re-mine repaired complete units with changed contextual meaning',
                          'Review alignment for all five conditions and reuse valid unchanged translations',
                          'Build new versioned matched training packages',
                          'Retrain each condition from the same original base and evaluate on the untouched benchmark'],
        'duplicate_recovered_units':'Several original fragments can map to one recovered sentence; retain all provenance but define training weights before fitting',
        'gpu_training_started':False,'original_cohort_counts':readiness})
    for relative in ('preprocessing/retained_cleanup.py','runners/retain_source_cleanup.py',
                     'runners/export_retained_cleanup.py','tests/test_retained_cleanup.py'):
        path=release/'code'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/relative).read_bytes())
    card=(WORK/'remote_README.md').read_text()
    front,body=card.split('\n---\n',1)
    front=front.replace('  default: true\n','')
    front+='\n- config_name: bar_units_v6\n  default: true\n  data_files:\n  - split: corpus\n    path: data/bar_units_v6/corpus-*.parquet'
    note=('\n## Retained source units and traceable cleanup\n\n'
        '**`bar_units_v6` preserves all 225,700 original source IDs. No original source record is dropped.** '
        '`original_source_text` preserves the original text; `source_text` contains a non-generative repair candidate. '
        'The full Bavarian and matched English articles, paragraph context and original/recovered offsets accompany each record. '
        'Complete list sentences lose their leading list marker; known false abbreviation boundaries are joined using original article text. '
        'Capitalization and whitespace changes are recorded. A full stop is added only when the clause screen supplies positive verb evidence.\n\n'
        '`cleaned_sentence` is populated for clause candidates passing these checks; it is null for unresolved titles, dates, labels and other fragments. '
        '`review_reasons`, `cleaning_operations` and `cleaning_status` explain each case. '
        'The verb screen is an incomplete review aid, not a validated Bavarian syntactic parser. It does not enforce rigid SVO. '
        'Unrecognized dialect verbs may be flagged for review. No missing subject, verb or fact is invented by this pass. '
        'Records requiring contextual reconstruction remain in the inventory with their original text.\n\n'
        'Several old fragments may map to the same recovered sentence; `cleaned_unit_id` and `contributing_source_ids` record that relationship. '
        'Preserving 225,700 source records does not establish that they are 225,700 distinct grammatical sentences. '
        'The earlier `bar_sentences_v5` view is a filtered reference inventory, not the complete retained corpus.\n\n'
        'The five historical 20k configurations retain their original `source_text` and `target_text` columns for reproducibility. '
        'They now also contain `cleaned_source_text`, `cleaned_sentence` and repair/review fields. '
        'Changed contextual meaning requires new English outputs before retraining English → Bavarian; old fragment translations must not be attached to longer recovered clauses. '
        'After source and alignment review, training must restart from the original NLLB base using newly versioned matched packages. '
        'Existing scores describe the historical inputs. `releases/preprocessing_v6/retraining_readiness.json` records the remaining work.\n')
    (OUT/'README.md').write_text(front+'\n---\n'+body+note)
    artifacts=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
               for p in sorted(OUT.rglob('*')) if p.is_file()]
    write(release/'manifest.json',{'release':'retained_cleanup_v6_20261110','parent_revision':data.read_json(WORK/'base.json')['revision'],
        'original_records':225700,'dropped_records':0,'artifacts':artifacts,'parquet_round_trip_verified':True,
        'training_started':False})
    print(json.dumps({'status':'prepared_and_verified','original_records':225700,'dropped_records':0,
        'files':len(artifacts)+1,'bytes':sum(x['bytes'] for x in artifacts),'training_readiness':readiness['blind']}),flush=True)


if __name__=='__main__':main()
