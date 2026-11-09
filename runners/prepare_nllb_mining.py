"""Frozen LASER/BGE top-one data for matched NLLB fine-tunes from the original base."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import shutil
import zipfile

from preprocessing import finewiki_pilot as data
from runners import scale_cohort
from runners.prepare_nllb_blind import (ROOT, CORPUS, eligible, benchmark_overlap,
                                       load_benchmark, normalize, file_sha)
from runners.nllb_blind_training import verify_package

DEST = ROOT / 'notebooks/bavarian_nllb_mining_20k_20261108'
BLIND = ROOT / 'notebooks/bavarian_nllb_blind_20k_20261108/package'
CORE = '1a5edcd85b5f795fa1c3cf925acb50cd81a301231b8de8b27f27357edf53e72d'


def mined_row(original, record, bundle, method, checkpoint, checkpoint_sha):
    expected = next(source for source in bundle['sources'] if source['source_id'] == original['source_id'])
    if record['source'] != expected or expected['bar_text'] != original['bavarian']:
        raise ValueError('Mining source differs from the frozen blind cohort')
    if record['acceptance_status'] != 'unthresholded' or record['candidate_count'] != len(bundle['page']['candidates']):
        raise ValueError('Mining is not the original unthresholded candidate inventory')
    candidates = record['candidates']
    if not candidates or len(candidates) != min(5, record['candidate_count']):
        raise ValueError('Missing top-one candidate or incomplete saved ranking')
    inventory = {tuple(candidate['sentence_ids']): candidate['text'] for candidate in bundle['page']['candidates']}
    previous = math.inf
    for rank, candidate in enumerate(candidates, 1):
        if (candidate['rank'] != rank or inventory.get(tuple(candidate['sentence_ids'])) != candidate['text']
                or not all(math.isfinite(candidate[key]) for key in ('cosine', 'margin_ratio'))
                or candidate['margin_ratio'] > previous):
            raise ValueError('Ranked English span or score differs from its frozen candidate inventory')
        previous = candidate['margin_ratio']
    top = candidates[0]
    result = {key: value for key, value in original.items()
              if key not in {'english', 'english_tokens', 'bavarian_tokens', 'blind_checkpoint', 'blind_record_sha256'}}
    result.update(english=top['text'], condition=method, english_origin='verbatim_top_one_article_span',
        english_sentence_ids=top['sentence_ids'], mining_checkpoint=checkpoint,
        mining_checkpoint_sha256=checkpoint_sha, mining_cosine=top['cosine'],
        mining_margin_ratio=top['margin_ratio'], reciprocal_top1=record['reciprocal_top1'],
        mining_acceptance='unthresholded')
    return result


def prepare():
    from transformers import AutoTokenizer
    parent = verify_package(BLIND)
    originals = [json.loads(line) for line in (BLIND / 'train.jsonl').read_text().splitlines()]
    validation_rows = [json.loads(line) for line in (BLIND / 'validation.jsonl').read_text().splitlines()]
    if len(originals) != 20000 or len(validation_rows) != 500:
        raise ValueError('Blind training or validation cohort changed')
    tokenizer = AutoTokenizer.from_pretrained(BLIND / 'tokenizer', local_files_only=True,
                                              src_lang='eng_Latn', tgt_lang='bar_Latn')
    benchmark = load_benchmark()
    index = {str(row['bar_page_id']): row for row in scale_cohort.index(CORPUS)}
    groups = defaultdict(list)
    for row in originals:
        groups[row['page_id']].append(row)
    selected = {method: {} for method in ('laser', 'bge_m3')}
    mining_hashes = {method: {} for method in selected}
    for pid, group in groups.items():
        path = CORPUS / 'colab_bge_full_20261105/raw' / (pid + '.json.gz')
        if file_sha(path) != index[pid]['inputs_sha256']:
            raise ValueError('Frozen source input checksum differs')
        bundle = data.read_json(path)
        if (data.digest(bundle['sources']) != bundle['sources_sha256']
                or data.digest(bundle['page']) != bundle['page_sha256']
                or any(row['frozen_input_sha256'] != index[pid]['inputs_sha256'] for row in group)):
            raise ValueError('Frozen source/context identity differs')
        for method in selected:
            ranking = CORPUS / 'outputs' / method / 'rankings' / (pid + '.json.gz')
            saved = data.read_json(ranking);manifest = saved['manifest']
            if (manifest['condition'] != method or manifest['core_sha256'] != CORE
                    or manifest['config_sha256'] != bundle['config_sha256']
                    or manifest['neighbors'] != 4 or manifest['top_k'] != 5 or manifest['threshold'] is not None
                    or saved['source_sha256'] != bundle['sources_sha256']
                    or saved['page_sha256'] != bundle['page_sha256']):
                raise ValueError('Saved mining protocol differs')
            if method == 'laser':
                if (manifest['model'] != 'LASER-1'
                        or manifest['asset_identity_sha256'] != '1f53121a57525b13aef6bedfe3cf307a6e4078e1cdbb80eefbddb9fde20086ab'
                        or (manifest['source_tokenizer'], manifest['target_tokenizer']) != ('de', 'en')):
                    raise ValueError('LASER model identity differs')
            elif (manifest['model'], manifest['revision']) != ('BAAI/bge-m3', '5617a9f61b028005a4858fdac845db406aefb181'):
                raise ValueError('BGE-M3 model identity differs')
            records = {record['source']['source_id']: record for record in saved['records']}
            if len(records) != len(saved['records']):
                raise ValueError('Duplicate mining source record')
            ranking_sha = file_sha(ranking);relative = str(ranking.relative_to(ROOT))
            mining_hashes[method][relative] = ranking_sha
            for original in group:
                row = mined_row(original, records[original['source_id']], bundle, method, relative, ranking_sha)
                if eligible(row, tokenizer) or benchmark_overlap(row, benchmark):
                    raise ValueError('Mined top-one pair failed technical/benchmark screening: ' + row['source_id'])
                selected[method][row['source_id']] = row
    validation_english = {normalize(row['english']) for row in validation_rows}
    package_root = DEST / 'package'
    if package_root.exists():
        raise FileExistsError('Mining training package already exists; keep the frozen identity')
    package_root.mkdir(parents=True)
    plan = {'order': ['laser', 'bge_m3'], 'blind_reference_run': parent['run_id'],
            'direction': 'en_to_bar', 'packages': {}, 'base_initialization': 'Each method starts fresh from the pinned original NLLB base; never continue blind or LASER weights'}
    for method in selected:
        rows = [selected[method][original['source_id']] for original in originals]
        package = package_root / method;package.mkdir()
        (package / 'train.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
        shutil.copyfile(BLIND / 'validation.jsonl', package / 'validation.jsonl')
        shutil.copytree(BLIND / 'tokenizer', package / 'tokenizer')
        shutil.copyfile(Path(__file__).with_name('nllb_blind_training.py'), package / 'train.py')
        shutil.copyfile(Path(__file__).with_name('nllb_local_recovery.py'), package / 'export_helper.py')
        counts = Counter(normalize(row['english']) for row in rows)
        audit = {'condition': method, 'training_pairs': 20000, 'validation_pairs': 500,
            'same_ordered_training_source_ids_as_blind': [row['source_id'] for row in rows] == [row['source_id'] for row in originals],
            'unchanged_original_bavarian_targets': all(row['bavarian'] == old['bavarian'] for row, old in zip(rows, originals)),
            'same_validation_file_sha256_as_blind': file_sha(package / 'validation.jsonl') == parent['files']['validation.jsonl'],
            'unique_english_inputs': len(counts), 'reused_english_input_rows': 20000-len(counts),
            'english_inputs_shared_with_common_validation': len(set(counts) & validation_english),
            'reciprocal_top_one_rows': sum(row['reciprocal_top1'] is True for row in rows),
            'pair_selection': 'Original unthresholded top-one; no reciprocal filtering, translation generation, reranking, or English-input deduplication',
            'duplicate_policy': 'Retain 20000 distinct Bavarian source IDs; many-to-one English retrieval is an observed property of the mining baseline',
            'validation': 'Same 500 blind-English/original-Bavarian held-out pairs for every condition; diagnostic loss only, no best-checkpoint selection',
            'benchmark_screening': 'Both sides pass the same exact/near-overlap screening against Tatoeba dev/test as the blind data; no test scores used for selection',
            'quality_status': 'Unassessed mining candidates, not human-approved parallel pairs',
            'ranking_sha256': mining_hashes[method]}
        (package / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n')
        manifest = {key: parent[key] for key in ('base_model', 'base_revision', 'direction', 'dataset_revision',
                                               'counts', 'language_policy', 'training')}
        manifest.update(condition=method, blind_reference_run=parent['run_id'],
            training_source_ids_sha256=data.digest([row['source_id'] for row in rows]),
            files={path.relative_to(package).as_posix(): file_sha(path) for path in sorted(package.rglob('*')) if path.is_file()})
        manifest['run_id'] = data.digest(manifest)
        (package / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        verify_package(package)
        plan['packages'][method] = {'run_id': manifest['run_id'], 'manifest_sha256': file_sha(package / 'manifest.json'),
            'train_pairs': 20000, 'validation_pairs': 500, 'unique_english_inputs': len(counts)}
    (package_root / 'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    archive = DEST / 'nllb_mining_20k_inputs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package_root.rglob('*')):
            if path.is_file():output.write(path, path.relative_to(package_root).as_posix())
    summary = {'archive_sha256': file_sha(archive), 'archive_bytes': archive.stat().st_size, **plan}
    (DEST / 'preparation_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    prepare()
