"""Freeze matched 5k blind-data experiments differing only in learning rate."""

import hashlib
import json
import math
from pathlib import Path
import shutil
import zipfile

from runners.nllb_blind_training import verify_package, sha

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'notebooks/bavarian_nllb_5k_20261109'
PARENT = ROOT / 'notebooks/bavarian_nllb_blind_20k_20261108/package'
RATES = {'lr_1e4': 1e-4, 'lr_3e5': 3e-5}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_subset(rows, count=5000):
    ids = [r['source_id'] for r in rows]
    if len(set(ids)) != len(ids) or not 1 <= count <= len(rows):
        raise ValueError('Duplicate sources or invalid subset size')
    chosen = set(sorted(ids, key=lambda sid: hashlib.sha256(('42|' + sid).encode()).hexdigest())[:count])
    return [r for r in rows if r['source_id'] in chosen]


def prepare():
    parent = verify_package(PARENT)
    rows = [json.loads(x) for x in (PARENT / 'train.jsonl').read_text().splitlines()]
    validation = [json.loads(x) for x in (PARENT / 'validation.jsonl').read_text().splitlines()]
    if len(rows) != 20000 or len(validation) != 500:
        raise ValueError('Parent cohort differs from the frozen 20k experiment')
    rows = select_subset(rows)
    if ({r['page_id'] for r in rows} & {r['page_id'] for r in validation}
            or {r['wikidata_id'] for r in rows} & {r['wikidata_id'] for r in validation}):
        raise ValueError('Training and validation topics overlap')
    if DEST.exists():
        raise FileExistsError('5k experiment already exists; preserve its immutable identity')
    DEST.mkdir()
    plan = {'experiment': 'blind_5k_learning_rate_comparison', 'condition': 'blind',
        'parent_run_id': parent['run_id'], 'training_pairs': 5000, 'validation_pairs': 500,
        'training_source_ids_sha256': digest([r['source_id'] for r in rows]),
        'subset_policy': 'Lowest SHA256(42|source_id), retaining original parent order',
        'order': list(RATES), 'changed_parameter': 'learning_rate',
        'expected_optimizer_steps': math.ceil(5000 / 32) * 3,
        'comparison': 'Final fixed-epoch shared validation loss; no Tatoeba test scoring or test-based selection',
        'initialization': 'Each rate starts from the pinned original base, never from another fine-tune',
        'packages': {}}
    for label, rate in RATES.items():
        package = DEST / 'package' / label
        package.mkdir(parents=True)
        (package / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        shutil.copyfile(PARENT / 'validation.jsonl', package / 'validation.jsonl')
        shutil.copytree(PARENT / 'tokenizer', package / 'tokenizer')
        shutil.copyfile(Path(__file__).with_name('nllb_blind_training.py'), package / 'train.py')
        audit = {'parent_run_id': parent['run_id'], 'training_pairs': 5000, 'validation_pairs': 500,
            'parent_train_sha256': parent['files']['train.jsonl'],
            'shared_validation_sha256': parent['files']['validation.jsonl'],
            'training_source_ids_sha256': plan['training_source_ids_sha256'],
            'texts_unchanged_from_parent': True, 'disjoint_articles_and_wikidata': True,
            'quality_status': 'Synthetic English and original Bavarian; not human-gold translations'}
        (package / 'data_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
        manifest = {k: parent[k] for k in ('base_model', 'base_revision', 'condition', 'direction',
                                           'dataset_revision', 'language_policy')}
        manifest.update(counts={'train': 5000, 'validation': 500},
            training={**parent['training'], 'learning_rate': rate, 'evaluate_final_validation': True},
            parent_run_id=parent['run_id'], experiment='blind_5k_learning_rate_comparison',
            training_source_ids_sha256=plan['training_source_ids_sha256'])
        manifest['files'] = {p.relative_to(package).as_posix(): sha(p)
                             for p in sorted(package.rglob('*')) if p.is_file()}
        manifest['run_id'] = digest(manifest)
        (package / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        verify_package(package)
        archive = DEST / (label + '.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for p in sorted(package.rglob('*')):
                if p.is_file():
                    z.write(p, p.relative_to(package).as_posix())
        plan['packages'][label] = {'run_id': manifest['run_id'], 'learning_rate': rate,
            'archive_sha256': sha(archive), 'archive_bytes': archive.stat().st_size,
            'train_sha256': manifest['files']['train.jsonl'],
            'validation_sha256': manifest['files']['validation.jsonl']}
    first, second = plan['packages'].values()
    if first['train_sha256'] != second['train_sha256'] or first['validation_sha256'] != second['validation_sha256']:
        raise ValueError('Learning-rate experiments have different input pairs')
    plan['plan_id'] = digest(plan)
    (DEST / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    print(json.dumps(plan, indent=2))


if __name__ == '__main__':
    prepare()
