"""Apply traceable repairs while keeping all 225,700 original FineWiki units."""

from collections import Counter
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import signal
import time

from preprocessing import finewiki_pilot as data, retained_cleanup as clean
from runners import scale_cohort
from runners.nllb_blind_training import sha
from runners.refine_preprocessing import article_row, write

ROOT=Path(__file__).resolve().parents[1]
CORPUS=ROOT/'results/finewiki_scale_20261102'
DEST=CORPUS/'preprocessing_v6_20261110'


def run(destination=DEST,limit=None):
    config={'dataset':data.DATASET,'dataset_revision':data.REVISION,'policy':clean.POLICY,
            'code_sha256':sha(Path(clean.__file__)),'runner_sha256':sha(Path(__file__)),
            'paragraph_helpers_sha256':sha(Path(data.__file__)),
            'boundary_helpers_sha256':sha(Path(__import__('preprocessing.quality_prose',fromlist=['x']).__file__))}
    config['preprocessing_id']=data.digest(config)
    if (destination/'config.json').exists() and data.read_json(destination/'config.json')!=config:
        raise ValueError('Preserve outputs from the previous cleanup policy; use another destination')
    write(destination/'config.json',config)
    index=sorted(scale_cohort.index(CORPUS),key=lambda item:int(item['bar_page_id']))
    if limit is not None:index=index[:limit]
    counts=Counter();reasons=Counter();operations=Counter();seen=set();started=time.monotonic()
    try:
        for number,item in enumerate(index,1):
            pid=str(item['bar_page_id']);target=destination/'articles'/(pid+'.json.gz')
            if target.exists():
                result=data.read_json(target)
                if result['preprocessing_id']!=config['preprocessing_id'] or result['input_sha256']!=item['inputs_sha256']:
                    raise ValueError('Cached source cleanup identity differs: '+pid)
            else:
                signal.alarm(120)
                path=CORPUS/'colab_bge_full_20261105/raw'/(pid+'.json.gz')
                if sha(path)!=item['inputs_sha256']:raise ValueError('Frozen input checksum differs: '+pid)
                bundle=data.read_json(path)
                if data.digest(bundle['sources'])!=bundle['sources_sha256'] or data.digest(bundle['page'])!=bundle['page_sha256']:
                    raise ValueError('Frozen input payload differs: '+pid)
                bar=article_row(bundle,'bar');en=article_row(bundle,'en')
                result=clean.prepare(bar,bundle['sources'],bundle['provenance'])
                result.update(preprocessing_id=config['preprocessing_id'],input_sha256=item['inputs_sha256'],bar=bar,en=en)
                signal.alarm(0);write(target,result)
            if result['original_record_count']!=result['retained_record_count']:
                raise ValueError('Source records were dropped: '+pid)
            units={};ready=set()
            for record in result['records']:
                sid=record['source_id']
                if sid in seen:raise ValueError('Repeated original source ID: '+sid)
                seen.add(sid)
                if not record['original_record_retained']:raise ValueError('Original source missing: '+sid)
                counts['retained_source_records']+=1
                counts['original_source_records']+=1
                counts['source_candidate_ready_records']+=record['source_candidate_ready']
                counts['records_requiring_contextual_or_grammar_review']+=not record['source_candidate_ready']
                counts['context_recovered_records']+=record['context_expanded']
                counts['records_with_changed_text']+=record['source_text']!=record['original_source_text']
                reasons.update(record['review_reasons']);operations.update(record['cleaning_operations'])
                units[record['cleaned_unit_id']]=True
                if record['source_candidate_ready']:ready.add(record['cleaned_unit_id'])
            counts['distinct_cleaned_context_units']+=len(units)
            counts['distinct_source_clause_candidates']+=len(ready)
            counts['articles']+=1;counts['split_alignment_failures']+=len(result['split_failures'])
            if number%100==0 or number==len(index):
                status={'status':'complete' if number==len(index) else 'running',
                    'preprocessing_id':config['preprocessing_id'],'processed_articles':number,'target_articles':len(index),
                    'elapsed_seconds':round(time.monotonic()-started,2),'counts':dict(counts),
                    'review_reasons':dict(reasons),'cleaning_operations':dict(operations),
                    'at':datetime.now(timezone.utc).isoformat()}
                write(destination/'status.json',status)
                if number%1000==0 or number==len(index):print(json.dumps(status),flush=True)
        if limit is None and (counts['retained_source_records']!=225700 or len(seen)!=225700):
            raise ValueError('Full cleanup must preserve exactly 225700 unique original IDs')
        status['counts']['dropped_source_records']=counts['original_source_records']-counts['retained_source_records']
        write(destination/'summary.json',status)
    except BaseException as error:
        signal.alarm(0)
        write(destination/'failure.json',{'page_id':pid,'error_class':type(error).__name__,
             'error':str(error),'completed_articles':number-1})
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,default=DEST)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.limit is not None and args.limit<1:raise ValueError('Article limit must be positive')
    signal.signal(signal.SIGALRM,lambda *args:(_ for _ in ()).throw(TimeoutError('Article cleanup exceeded 120 seconds')))
    os.nice(10);run(args.destination,args.limit)


if __name__=='__main__':main()
