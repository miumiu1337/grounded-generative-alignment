"""Filter the frozen corpus with DataTrove and export six-column parallel pairs."""

from collections import Counter
from functools import lru_cache
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from preprocessing import finewiki_pilot as data, article_structure
from preprocessing.article_structure import ArticleStructure,HEADING
from preprocessing.datatrove_sentences import SentenceQuality,COLUMNS,normalized_key,surface_text
from runners import scale_cohort
from runners.nllb_blind_training import sha,write_json as write

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'results/finewiki_scale_20261102'
INPUTS=CORPUS/'colab_bge_full_20261105/raw'
WORK=ROOT/'publication_work_20261110_structure'
OUT=WORK/'huggingface'
PACKAGES={
    'blind':'bavarian_nllb_blind_20k_20261108/package',
    'laser':'bavarian_nllb_mining_20k_20261108/package/laser',
    'bge_m3':'bavarian_nllb_mining_20k_20261108/package/bge_m3',
    'variant_a':'bavarian_nllb_variant_a_20k_20261109/package',
    'variant_b':'bavarian_nllb_variant_b_20k_20261109/package',
}
METHODS=tuple(PACKAGES)
MARKER=re.compile(r'\[((?:bar|en)_\d+:fw[0-9a-f]{8}:r\d+:(?:[a-z0-9_]+:)?S\d+)\] ')


def reference_ranges(text):
    return [(start,end) for start,end,reason in ArticleStructure(text).removed if reason=='reference_section']


def write_pairs(path,rows):
    if not rows:raise ValueError('Cannot publish an empty parallel split')
    if any(tuple(row)!=COLUMNS or any(not isinstance(value,str) or not value for value in row.values()) for row in rows):
        raise ValueError('Parallel tables must have exactly six nonempty string columns')
    columns={}
    for name in COLUMNS:
        if name.endswith('_article'):
            unique=[];lookup={};indices=[]
            for row in rows:
                text=row[name]
                if text not in lookup:lookup[text]=len(unique);unique.append(text)
                indices.append(lookup[text])
            columns[name]=pa.DictionaryArray.from_arrays(pa.array(indices,type=pa.int32()),pa.array(unique,type=pa.string()))
        else:columns[name]=pa.array([row[name] for row in rows],type=pa.string())
    path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.table(columns),path,compression='zstd',store_schema=False,
                   dictionary_pagesize_limit=64*1024*1024,write_page_checksum=True)
    schema=pq.read_schema(path)
    if schema.names!=list(COLUMNS) or any(field.type!=pa.string() for field in schema):raise ValueError('Published schema changed')
    restored=pq.read_table(path,read_dictionary=['source_article','target_article'])
    for name in COLUMNS:
        if name.endswith('_article'):
            position=0
            for chunk in restored[name].chunks:
                dictionary=chunk.dictionary.to_pylist()
                for index in chunk.indices.to_pylist():
                    if rows[position][name]!=dictionary[index]:raise ValueError('Article context changed during export')
                    position+=1
        elif restored[name].to_pylist()!=[row[name] for row in rows]:raise ValueError('Parallel text changed during export')


@lru_cache(maxsize=1)
def frozen_index():return {str(item['bar_page_id']):item for item in scale_cohort.index(CORPUS)}


def article_title(page,language):
    if language not in {'bar','en'}:raise ValueError('Unsupported article-title language')
    header=page[language].split('\n',1)[0]
    match=re.fullmatch(r'(.*) \('+language+r'\), revision \d+',header)
    if not match or not match[1].strip():raise ValueError('Malformed frozen article title')
    return match[1]


@lru_cache(maxsize=128)
def article(pid):
    pid=str(pid);path=INPUTS/(pid+'.json.gz')
    if sha(path)!=frozen_index()[pid]['inputs_sha256']:raise ValueError('Frozen input checksum differs: '+pid)
    bundle=data.read_json(path)
    if data.digest(bundle['page'])!=bundle['page_sha256'] or data.digest(bundle['sources'])!=bundle['sources_sha256']:
        raise ValueError('Frozen article payload differs: '+pid)
    value={'bundle':bundle,'structures':{},'cleaned_context':{}}
    for language in ('bar','en'):
        numbered=bundle['page'][language].split('\n',1)[1]
        text=MARKER.sub('',numbered)
        if hashlib.sha256(text.encode()).hexdigest()!=bundle['page'][language+'_text_sha256']:
            raise ValueError('Reconstructed raw article checksum differs: '+pid+'/'+language)
        value[language]={'text':text,'title':article_title(bundle['page'],language)}
        value['structures'][language]=ArticleStructure(text)
        value['cleaned_context'][language]=value['structures'][language].cleaned_text
    return value


def english_origin_reason(value,ids):
    if not ids:return None
    if 'english_units' not in value:
        numbered=value['bundle']['page']['en'].split('\n',1)[1];positions={};offset=0
        for match in MARKER.finditer(numbered):
            positions[match[1]]=match.start()-offset;offset+=len(match[0])
        units={}
        for candidate in value['bundle']['page']['candidates']:
            if len(candidate['sentence_ids'])!=1:continue
            sid=candidate['sentence_ids'][0];start=positions[sid];end=start+len(candidate['text'])
            if value['en']['text'][start:end]!=candidate['text']:raise ValueError('English span text/offset mismatch')
            units[sid]=(start,end)
        value['english_units']=units
    for sid in ids:
        if sid not in value['english_units']:raise ValueError('Unknown saved English evidence ID: '+sid)
        reason=value['structures']['en'].reason_for_span(*value['english_units'][sid])
        if reason:return reason
    return None


def output_structure_reason(text):
    if not text.strip():return 'empty_text'
    if any(HEADING.match(line) for line in text.splitlines()):return 'structural_heading'
    structure=ArticleStructure(text)
    return structure.reason_for_span(0,len(text))


def source_pass(engine,config,cohort_ids):
    marker=WORK/'source_summary.json'
    if marker.exists():
        summary=data.read_json(marker)
        if summary['curation_id']!=config['curation_id'] or sha(WORK/'source_audit.jsonl.gz')!=summary['audit_sha256']:
            raise ValueError('Previous source pass identity differs')
        return data.read_json(WORK/'selected_source_checks.json'),summary
    counters=Counter();reasons=Counter();seen={};ids=set();selected={};batch=[];started=time.monotonic()
    article_removals={'bar':Counter(),'en':Counter()}
    temporary=WORK/'source_audit.jsonl.gz.partial'
    with gzip.open(temporary,'wt',encoding='utf-8',compresslevel=1) as audit:
        def flush():
            if not batch:return
            eligible=[r for r in batch if r['reason'] is None]
            results=engine.check([r['text'] for r in eligible],'de')
            for item,reason in zip(eligible,results,strict=True):item['reason']=reason
            for item in batch:
                sid=item['source_id']
                if sid in ids:raise ValueError('Repeated original source ID in frozen inventory')
                ids.add(sid);counters['input_source_units']+=1
                item['quality_passed']=item['reason'] is None
                if item['quality_passed']:
                    counters['quality_pass_records']+=1
                    key=normalized_key(item['text'])
                    item['duplicate_of']=seen.get(key)
                    if item['duplicate_of'] is None:seen[key]=sid;counters['unique_quality_sources']+=1
                    else:counters['duplicate_sources_removed']+=1
                else:reasons[item['reason']]+=1
                if sid in cohort_ids:selected[sid]=dict(item)
                audit.write(json.dumps(item,ensure_ascii=False,separators=(',',':'))+'\n')
            batch.clear()
        for number,pid in enumerate(sorted(frozen_index(),key=int),1):
            value=article(pid);bundle=value['bundle']
            for language in ('bar','en'):article_removals[language].update(value['structures'][language].counts)
            for record in bundle['sources']:
                sid=record['source_id'];provenance=bundle['provenance'][sid]
                start,end=provenance['source_start'],provenance['source_end']
                if value['bar']['text'][start:end]!=record['bar_text']:raise ValueError('Original source span mismatch')
                reason=value['structures']['bar'].reason_for_span(start,end)
                text=surface_text(record['bar_text'])
                batch.append({'source_id':sid,'page_id':pid,'text':text,'reason':reason,'saved_translation_reusable':True})
                if len(batch)>=512:flush()
            if number%1000==0:
                write(WORK/'status.json',{'stage':'source_quality','processed_articles':number,'target_articles':16306,
                    'counts':dict(counters),'elapsed_seconds':round(time.monotonic()-started,1)})
                print('Source-quality articles:',number,'counts:',json.dumps(dict(counters)),flush=True)
        flush()
    if len(ids)!=225700 or set(selected)!=cohort_ids:raise ValueError('Full original source coverage was not verified')
    temporary.replace(WORK/'source_audit.jsonl.gz')
    summary={'curation_id':config['curation_id'],'counts':dict(counters),'rejections':dict(reasons),
             'removed_article_lines':{lang:dict(values) for lang,values in article_removals.items()},
             'audit_sha256':sha(WORK/'source_audit.jsonl.gz'),'elapsed_seconds':round(time.monotonic()-started,1)}
    write(WORK/'selected_source_checks.json',selected);write(marker,summary)
    return selected,summary


def select_unique(candidates):
    """Reserve duplicate keys only after every condition passes all checks."""
    kept={'train':[],'validation':[]};seen_source=set();seen_targets={m:set() for m in METHODS};reasons=Counter()
    for split in ('validation','train'):
        for item in candidates[split]:
            reason=item['reason'];source_key=normalized_key(item['source_text'])
            if reason is None and source_key in seen_source:reason='duplicate_bavarian_source'
            keys={m:normalized_key(text) for m,text in item['english'].items()}
            if reason is None:
                for method,key in keys.items():
                    if key in seen_targets[method]:reason='duplicate_english_'+method;break
            if reason is not None:reasons[split+':'+reason]+=1;continue
            seen_source.add(source_key)
            for method,key in keys.items():seen_targets[method].add(key)
            kept[split].append(item)
    return kept,dict(reasons)


def run():
    os.nice(10);WORK.mkdir(exist_ok=True)
    engine=SentenceQuality()
    config={'quality':engine.config(),'source_code_sha256':sha(ROOT/'preprocessing/datatrove_sentences.py'),
        'runner_sha256':sha(Path(__file__)),'structure_code_sha256':sha(Path(article_structure.__file__)),
        'dataset_revision':data.REVISION,'frozen_article_index_sha256':data.digest(list(frozen_index().values())),
        'input_segmentation':'original_frozen_datatrove_quality_auto_v3',
        'structure_policy':'prose_only;remove_reference_sections_lists_enumerations_tables_code_and_bibliographic_entries',
        'article_columns':'article_titles_only;cleaned_context_used_internally;raw_originals_archived',
        'matching':'same_Bavarian_sources_across_five_conditions;article_disjoint_validation',
        'mining_alignment_filter':'reciprocal_top1_required_for_LASER_and_BGE_training_pairs',
        'translation_reuse':'surface_changes_only;exclude_context_expansion_with_new_words',
        'columns':COLUMNS}
    config['curation_id']=data.digest(config)
    if (WORK/'config.json').exists() and data.read_json(WORK/'config.json')!=json.loads(json.dumps(config)):
        raise ValueError('Curation code or policy changed; preserve the previous outputs')
    write(WORK/'config.json',config)
    original={};cohort_ids=set()
    for method,relative in PACKAGES.items():
        original[method]={}
        for split in ('train','validation'):
            rows=[json.loads(line) for line in (ROOT/'notebooks'/relative/(split+'.jsonl')).read_text().splitlines()]
            original[method][split]=rows;cohort_ids.update(r['source_id'] for r in rows)
    for split in ('train','validation'):
        reference=original['blind'][split]
        for method in METHODS:
            if [(r['source_id'],r['bavarian']) for r in original[method][split]]!=[(r['source_id'],r['bavarian']) for r in reference]:
                raise ValueError('Original training conditions are not matched')
    selected,source_summary=source_pass(engine,config,cohort_ids)
    candidates={'train':[],'validation':[]}
    for split in ('train','validation'):
        for index,row in enumerate(original['blind'][split]):
            sid=row['source_id'];source=selected[sid]
            reason=None if source['quality_passed'] else 'source_'+source['reason']
            if reason is None and not source['saved_translation_reusable']:reason='repaired_source_needs_new_english'
            if reason is None and split=='train':
                for method in ('laser','bge_m3'):
                    if original[method][split][index]['reciprocal_top1'] is not True:
                        reason=method+'_not_reciprocal';break
            english={method:surface_text(original[method][split][index]['english']) for method in METHODS}
            if reason is None:
                value=article(str(row['page_id']))
                if not value['cleaned_context']['bar'] or not value['cleaned_context']['en']:
                    reason='empty_cleaned_article_context'
                for method in METHODS:
                    if reason is not None:break
                    saved=original[method][split][index]
                    origin=english_origin_reason(value,saved.get('english_sentence_ids',saved.get('evidence_ids',[])))
                    structural=origin or output_structure_reason(saved['english'])
                    if structural is not None:reason='english_'+method+'_'+structural
            candidates[split].append({'source_id':sid,'source_text':source['text'],'page_id':row['page_id'],
                'wikidata_id':row['wikidata_id'],'english':english,'reason':reason})
    texts=[text for rows in candidates.values() for item in rows if item['reason'] is None for text in item['english'].values()]
    print('English quality checks:',len(texts),'candidate outputs;',len(set(texts)),'unique',flush=True)
    # Smaller calls expose progress and allow completed source checks to be reused.
    unique=list(dict.fromkeys(texts))
    for start in range(0,len(unique),512):
        engine.check(unique[start:start+512],'en')
        if start%4096==0:print('English candidates checked:',min(start+512,len(unique)),'/',len(unique),flush=True)
    for rows in candidates.values():
        for item in rows:
            if item['reason'] is None:
                for method,text in item['english'].items():
                    reason=engine.cache[('en',text)]
                    if reason is not None:item['reason']='english_'+method+'_'+reason;break
    kept,rejections=select_unique(candidates)
    if not kept['train'] or not kept['validation']:raise ValueError('Quality filtering produced an empty matched split')
    for key in ('page_id','wikidata_id'):
        if {r[key] for r in kept['train']} & {r[key] for r in kept['validation']}:raise ValueError('Training/validation topics overlap')
    if OUT.exists():raise FileExistsError('Preserve the previous prepared dataset update')
    OUT.mkdir()
    configs=[]
    for method in METHODS:
        configs.append({'config_name':method,'data_files':[{'split':split,'path':f'data/{method}/{split}.parquet'} for split in ('train','validation')],
                        **({'default':True} if method=='blind' else {})})
        for split in ('train','validation'):
            rows=[]
            for item in kept[split]:
                value=article(str(item['page_id']))
                rows.append(dict(zip(COLUMNS,('bar','en',item['source_text'],item['english'][method],value['bar']['title'],value['en']['title']),strict=True)))
            write_pairs(OUT/f'data/{method}/{split}.parquet',rows)
            print('Six-column export verified:',method,split,len(rows),flush=True)
    release=OUT/'releases/structure_20261110'
    write(release/'config.json',config)
    write(release/'summary.json',{'source_corpus':source_summary,'paired_split_counts':{s:len(r) for s,r in kept.items()},
        'paired_rejections':rejections,'matched_conditions':METHODS,'unique_sources_and_targets_per_condition':True,
        'new_models_trained':False,'quality_status':'automatic_screened_candidates;semantic_alignment_not_human_validated'})
    write(release/'source_ids.json',{split:[r['source_id'] for r in rows] for split,rows in kept.items()})
    # Private review rows retain source IDs for the next versioned training packages.
    write(WORK/'matched_pairs.json',kept)
    for relative in ('preprocessing/datatrove_sentences.py','preprocessing/article_structure.py',
                     'runners/curate_datatrove_parallel.py','tests/test_datatrove_sentences.py','tests/test_article_structure.py','requirements-quality.txt'):
        target=release/'code'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
    body=(f'# Bavarian parallel pairs\n\nFive matched conditions: blind, LASER, BGE-M3, Variant A and Variant B. '
          f"Each contains {len(kept['train']):,} training pairs and the same {len(kept['validation']):,} held-out validation pairs.\n\n"
          '`source_lang` is `bar` and `target_lang` is `en`. `source_text` is Bavarian; `target_text` is the corresponding method’s English output. '
          '`source_article` and `target_article` contain only the original Bavarian and English article titles. Raw original articles and cleaned prose contexts are retained locally. '
          'These are the only six table columns. NLLB fine-tuning reverses the text columns for English → Bavarian.\n\n'
          'Filtering uses DataTrove Gopher quality, C4 quality and Gopher repetition checks configured for individual sentences. '
          'Bavarian uses German tokenization; English uses English settings. No trained grammar models, finite-verb tests, subject tests, SVO checks, capitalization requirements or forced full stops are used. '
          'Reference sections (including observed Bavarian heading variants), displayed lists, enumerations, publication entries, tables and code are removed using article structure before quality filtering. Inline numeric citation markers are removed; the text is otherwise unchanged apart from NFC and whitespace normalization. '
          'Mining spans and selected A/B evidence are checked against the cleaned English article structure too. Exact duplicate keys normalize case, whitespace and punctuation while preserving numbers and diacritics. '
          'Sources are unique across training and validation. English outputs are also unique within each condition. '
          'LASER and BGE-M3 training pairs must be reciprocal top-one matches. All five conditions retain the same source IDs.\n\n'
          f"The full source pass screened 225,700 frozen units and retained {source_summary['counts']['unique_quality_sources']:,} unique source candidates. "
          'The published parallel subsets use the intersection with existing completed outputs, keeping the original frozen unit boundaries. No source sentences are rewritten or expanded. '
          'Original articles and rejected data remain in local archives; previous Hugging Face revisions preserve historical releases.\n\n'
          'This is automatic textual and structural screening, not grammatical or human-gold translation validation. DataTrove heuristic thresholds can still reject valid material. '
          'The original Tatoeba scores and training histories in `assessment/` describe the earlier 20k inputs, not this filtered release. '
          'New comparisons must restart from the original NLLB base using the matched filtered data.\n\n'
          'FineWiki revision: `8bd13e72e6a002407649b3e898535f42ceb1aeb9`. '
          'FineWiki intentionally preserves lists and removes most, not all, reference sections. Its frozen Bavarian heading list omits common spellings such as Literatua. '
          'Policy, counts and source IDs are in `releases/structure_20261110/`. Code: '
          '[GitHub](https://github.com/miumiu1337/grounded-generative-alignment/tree/test).\n')
    metadata={'license':'cc-by-sa-4.0','language':['bar','en'],'task_categories':['translation'],'configs':configs}
    (OUT/'README.md').write_text('---\n'+yaml.safe_dump(metadata,sort_keys=False,allow_unicode=True)+'---\n\n'+body)
    artifacts=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
    write(release/'manifest.json',{'release':'datatrove_structural_quality_20261110','curation_id':config['curation_id'],
        'artifacts':artifacts,'schema':COLUMNS,'parquet_round_trip_verified':True})
    print(json.dumps({'status':'prepared_and_verified','source_counts':source_summary['counts'],
        'paired_counts':{s:len(r) for s,r in kept.items()},'files':len(artifacts)+1,'bytes':sum(a['bytes'] for a in artifacts)}),flush=True)


if __name__=='__main__':
    try:run()
    except BaseException as error:
        write(WORK/'failure.json',{'error_class':type(error).__name__,'error':str(error)});raise
