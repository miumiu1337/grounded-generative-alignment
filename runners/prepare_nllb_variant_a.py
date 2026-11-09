"""Package saved canonical A/B outputs on the frozen blind training cohort."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil
import zipfile

from preprocessing import finewiki_pilot as data
from parallel_data_generation import finewiki_experiments as validation, scale_protocol as protocol
from runners import scale_cohort
from runners.nllb_blind_training import verify_package
from runners.prepare_nllb_blind import (ROOT, CORPUS, benchmark_overlap, eligible,
                                       file_sha, load_benchmark, normalize)

BLIND = ROOT / 'notebooks/bavarian_nllb_blind_20k_20261108/package'
FROZEN_WORKER = ROOT / 'notebooks/bavarian_nllb_mining_20k_20261108/package/laser'
DEST = ROOT / 'notebooks/bavarian_nllb_variant_a_20k_20261109'


def stage_records(page_id, stage, config, config_sha, condition='variant_a'):
    records = {}
    for path in sorted((CORPUS/'outputs/records'/page_id/condition/stage).glob('*.json.gz')):
        saved = data.read_json(path)
        expected = {'condition': condition, 'stage': stage, 'config_sha256': config_sha,
                    'model': config['model'], 'revision': config['model_revision'],
                    'generation': config['generator']}
        if saved['manifest'] != expected:
            raise ValueError(condition + ' generation identity differs: ' + str(path))
        checkpoint_sha = file_sha(path)
        for record in saved['records']:
            sid = record['source']['source_id']
            if sid in records:
                raise ValueError('Duplicate canonical '+condition+' record: ' + sid)
            records[sid] = (saved['manifest'], record, str(path.relative_to(ROOT)), checkpoint_sha)
    return records


def refined_row(original, source, page, config, selected, refined, condition='variant_a'):
    if condition not in {'variant_a', 'variant_b'}:
        raise ValueError('Unsupported generation condition')
    if source['source_id'] != original['source_id'] or source['bar_text'] != original['bavarian']:
        raise ValueError(condition + ' source differs from the frozen blind cohort')
    sm, selection, spath, ssha = selected
    fm, refinement, fpath, fsha = refined
    dependencies = {'draft': original['english']} if condition == 'variant_a' else {}
    final_stage = 'refinement' if condition == 'variant_a' else 'translation'
    request = protocol.request(condition, 'selection', source, page, config['prompts'], **dependencies)
    validation.validate_record(selection, source, sm, request, page)
    request = protocol.request(condition, final_stage, source, page, config['prompts'],
                               selection=selection['result'], **dependencies)
    validation.validate_record(refinement, source, fm, request, page)
    row = {key: value for key, value in original.items()
           if key not in {'english', 'english_tokens', 'bavarian_tokens'}}
    if condition == 'variant_b':
        for key in ('blind_checkpoint', 'blind_record_sha256'):
            row.pop(key, None)
    origin = ('canonical_saved_blind_plus_wikipedia_refinement' if condition == 'variant_a'
              else 'canonical_saved_evidence_first_translation_without_blind')
    row.update(english=refinement['result']['translation'], condition=condition,
               english_origin=origin,
               selection_checkpoint=spath, selection_checkpoint_sha256=ssha,
               selection_record_sha256=data.digest(selection),
               evidence_ids=selection['result']['evidence_ids'],
               match_status=selection['result']['match_status'],
               contradiction=selection['result']['contradiction'])
    row.update({final_stage+'_checkpoint':fpath, final_stage+'_checkpoint_sha256':fsha,
                final_stage+'_record_sha256':data.digest(refinement)})
    return row


def prepare(condition='variant_a'):
    if condition not in {'variant_a','variant_b'}:
        raise ValueError('Unsupported generation condition')
    destination = DEST if condition == 'variant_a' else DEST.with_name('bavarian_nllb_variant_b_20k_20261109')
    from transformers import AutoTokenizer
    parent = verify_package(BLIND)
    verify_package(FROZEN_WORKER)
    originals = [json.loads(line) for line in (BLIND/'train.jsonl').read_text().splitlines()]
    validation = [json.loads(line) for line in (BLIND/'validation.jsonl').read_text().splitlines()]
    if len(originals) != 20000 or len(validation) != 500 or len({r['source_id'] for r in originals}) != 20000:
        raise ValueError('Frozen training cohort size or identity differs')
    package = destination/'package'
    staging = destination/'package.preparing'
    if package.exists() or staging.exists():
        raise FileExistsError(condition+' package already exists or preparation is incomplete; preserve it')
    configs = {}
    for name in ('frozen/config.json', 'cohort_v2/addendum_config.json'):
        config = data.read_json(CORPUS/name)
        configs[data.digest(config)] = config
    index = {str(row['bar_page_id']): row for row in scale_cohort.index(CORPUS)}
    tokenizer = AutoTokenizer.from_pretrained(BLIND/'tokenizer', local_files_only=True,
                                              src_lang='eng_Latn', tgt_lang='bar_Latn')
    benchmark = load_benchmark()
    validation_english = {normalize(row['english']) for row in validation}
    groups = defaultdict(list)
    for original in originals:
        groups[original['page_id']].append(original)
    rows, failures = {}, []
    for number, (pid, group) in enumerate(groups.items(), 1):
        raw = CORPUS/'colab_bge_full_20261105/raw'/(pid+'.json.gz')
        if file_sha(raw) != index[pid]['inputs_sha256']:
            raise ValueError('Frozen article checksum differs: ' + pid)
        bundle = data.read_json(raw)
        if (data.digest(bundle['sources']) != bundle['sources_sha256'] or
                data.digest(bundle['page']) != bundle['page_sha256']):
            raise ValueError('Frozen article/source digest differs: ' + pid)
        config = configs[bundle['config_sha256']]
        selected = stage_records(pid, 'selection', config, bundle['config_sha256'], condition)
        final_stage = 'refinement' if condition == 'variant_a' else 'translation'
        refined = stage_records(pid, final_stage, config, bundle['config_sha256'], condition)
        sources = {source['source_id']: source for source in bundle['sources']}
        for original in group:
            sid = original['source_id']
            if original['frozen_input_sha256'] != index[pid]['inputs_sha256']:
                raise ValueError('Training row references another article input')
            row = refined_row(original, sources[sid], bundle['page'], config, selected[sid], refined[sid], condition)
            reason = eligible(row, tokenizer)
            if reason is None and benchmark_overlap(row, benchmark):
                reason = 'tatoeba_dev_test_overlap'
            if reason is None and normalize(row['english']) in validation_english:
                reason = 'common_validation_english_overlap'
            if reason:
                failures.append({'source_id': sid, 'reason': reason})
            rows[sid] = row
        if number % 50 == 0 or number == len(groups):
            print(json.dumps({'validated_articles': number, 'target_articles': len(groups),
                              'validated_pairs': len(rows), 'technical_failures': len(failures)}), flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    if failures:
        (destination/'preparation_failures.json').write_text(json.dumps(failures, indent=2)+'\n')
        raise ValueError(condition+' failed matched-cohort checks; no training package created')
    ordered = [rows[row['source_id']] for row in originals]
    counts = Counter(normalize(row['english']) for row in ordered)
    audit = {'condition': condition, 'training_pairs': 20000, 'validation_pairs': 500,
             'same_ordered_training_source_ids_as_blind': True, 'unchanged_original_bavarian_targets': True,
             'same_validation_file_sha256_as_blind': file_sha(BLIND/'validation.jsonl'),
             'unique_english_inputs': len(counts), 'reused_english_input_rows': 20000-len(counts),
             'selection_and_final_stage_requests_validated': True,
             'blind_translations_reused': 20000 if condition == 'variant_a' else 0,
             'english_inputs_shared_with_common_validation': 0, 'over_512_tokens': 0,
             'benchmark_overlap_policy': 'Same exact/near-overlap screen against Tatoeba dev/test as blind',
             'quality_status': 'Technical/source-integrity filtering only; generated English is not human gold',
             'training_cohort_policy': 'Retain all original 20000 IDs; no quality or duplicate filtering based on test results',
             'selection_match_status': dict(Counter(row['match_status'] for row in ordered)),
             'selection_contradictions': sum(row['contradiction'] for row in ordered)}
    staging.mkdir()
    (staging/'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in ordered))
    shutil.copyfile(BLIND/'validation.jsonl', staging/'validation.jsonl')
    shutil.copytree(BLIND/'tokenizer', staging/'tokenizer')
    for name in ('train.py', 'export_helper.py'):
        shutil.copyfile(FROZEN_WORKER/name, staging/name)
    (staging/'data_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    manifest = {key: parent[key] for key in ('base_model', 'base_revision', 'direction', 'dataset_revision',
                                           'counts', 'language_policy', 'training')}
    manifest.update(condition=condition, blind_reference_run=parent['run_id'],
                    training_source_ids_sha256=data.digest([row['source_id'] for row in originals]),
                    files={p.relative_to(staging).as_posix(): file_sha(p)
                           for p in sorted(staging.rglob('*')) if p.is_file()})
    manifest['run_id'] = data.digest(manifest)
    (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    verify_package(staging)
    staging.rename(package)
    archive = destination/('nllb_'+condition+'_20k_inputs.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(package).as_posix())
    summary = {'condition': condition, 'run_id': manifest['run_id'], 'archive': str(archive),
               'archive_bytes': archive.stat().st_size, 'archive_sha256': file_sha(archive),
               'train_pairs': 20000, 'validation_pairs': 500, 'unique_english_inputs': len(counts),
               'base_initialization': 'Fresh pinned original base; never continue another method weights'}
    (destination/'preparation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    prepare()
