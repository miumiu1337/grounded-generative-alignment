"""Prepare a frozen blind English-to-Bavarian dataset and hosted Colab notebook."""

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import unicodedata
import zipfile

from preprocessing import finewiki_pilot as data
from parallel_data_generation import finewiki_experiments as validation
from runners import scale_cohort

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / 'results/finewiki_scale_20261102'
DEST = ROOT / 'notebooks/bavarian_nllb_blind_20k_20261108'
TOKENIZER = ROOT / '.cache/nllb_training_preparation/tokenizer'
BASE_REVISION = 'f8d333a098d19b4fd9a8b18f94170487ad3f821d'


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def stable_key(value):
    return hashlib.sha256(('42|' + value).encode()).hexdigest()


def benchmark_key(text):
    return ' '.join(''.join(c if c.isalnum() else ' ' for c in normalize(text)).split())


def near_duplicate(text, reference):
    if text == reference:
        return True
    if min(len(text), len(reference)) < 30 or min(len(text), len(reference)) / max(len(text), len(reference)) < 0.9:
        return False
    left = {text[i:i+3] for i in range(len(text)-2)}
    right = {reference[i:i+3] for i in range(len(reference)-2)}
    return 2 * len(left & right) / (len(left) + len(right)) >= 0.9


def load_benchmark():
    path = ROOT / '.cache/nllb_training_preparation/tatoeba_bar_eng_v2023-09-26.tar'
    if file_sha(path) != 'd62c59a07ec4587c9e54700dfdd3df1f868c34d9384a7226e2fccba5df3ea578':
        raise ValueError('Tatoeba overlap-screening archive identity changed')
    result = {'bavarian': [], 'english': []}
    with tarfile.open(path) as archive:
        for split, count in [('dev', 12), ('test', 98)]:
            for extension, language in [('src', 'bavarian'), ('trg', 'english')]:
                member = archive.extractfile(f'data/release/v2023-09-26/bar-eng/{split}.{extension}')
                values = member.read().decode('utf-8').splitlines()
                if len(values) != count:
                    raise ValueError('Unexpected benchmark split size')
                result[language].extend(benchmark_key(value) for value in values)
    return result


def benchmark_overlap(row, benchmark):
    for language in ('bavarian', 'english'):
        key = benchmark_key(row[language])
        if any(near_duplicate(key, reference) for reference in benchmark[language]):
            return True
    return False


def eligible(row, tokenizer):
    if not row['english'].strip() or not row['bavarian'].strip():
        return 'empty'
    tokens = tokenizer(row['english'], text_target=row['bavarian'], truncation=False)
    if max(len(tokens['input_ids']), len(tokens['labels'])) > 512:
        return 'over_512_tokens'
    if tokens['input_ids'][0] != tokenizer.convert_tokens_to_ids('eng_Latn') or tokens['labels'][0] != tokenizer.convert_tokens_to_ids('bar_Latn'):
        raise ValueError('NLLB language tokens do not match the training direction')
    row['english_tokens'] = len(tokens['input_ids'])
    row['bavarian_tokens'] = len(tokens['labels'])
    return None


def load_rows(page_id, index, configs, retries):
    raw_path = CORPUS / 'colab_bge_full_20261105/raw' / (page_id + '.json.gz')
    if file_sha(raw_path) != index['inputs_sha256']:
        raise ValueError('Frozen article checksum differs: ' + page_id)
    bundle = data.read_json(raw_path)
    config = configs[index['config_sha256']]
    if (bundle['config_sha256'] != index['config_sha256']
            or data.digest(bundle['page']) != bundle['page_sha256']
            or data.digest(bundle['sources']) != bundle['sources_sha256']):
        raise ValueError('Frozen source/context identity differs: ' + page_id)
    records = {}
    for path in sorted((CORPUS / 'outputs/records' / page_id / 'blind/translation').glob('*.json.gz')):
        saved = data.read_json(path)
        for record in saved['records']:
            sid = record['source']['source_id']
            if sid in records:
                raise ValueError('Duplicate canonical blind record: ' + sid)
            records[sid] = (saved['manifest'], record, str(path.relative_to(ROOT)))
    for sid, value in retries.get(page_id, {}).items():
        if sid not in records:
            records[sid] = value
    result = []
    for source in bundle['sources']:
        saved = records.get(source['source_id'])
        if saved is None:
            continue  # Explicitly unsuccessful blind sentences are not training pairs.
        manifest, record, path = saved
        request = validation.request('blind', 'translation', source, bundle['page'], config['prompts'])
        validation.validate_record(record, source, manifest, request, bundle['page'])
        result.append({'source_id': source['source_id'], 'page_id': page_id,
            'wikidata_id': bundle['page']['wikidata_id'], 'english': record['result']['translation'],
            'bavarian': source['bar_text'], 'bavarian_title': source['bar_title'],
            'bavarian_revision': source['bar_revision_id'], 'bavarian_url': bundle['page']['bar_url'],
            'english_article_url': bundle['page']['en_url'], 'blind_record_sha256': data.digest(record),
            'blind_checkpoint': path, 'frozen_input_sha256': index['inputs_sha256']})
    return result


def prepare():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True,
                                              src_lang='eng_Latn', tgt_lang='bar_Latn')
    index = {str(row['bar_page_id']): row for row in scale_cohort.index(CORPUS)}
    configs = {}
    for name in ('frozen/config.json', 'cohort_v2/addendum_config.json'):
        config = data.read_json(CORPUS / name)
        configs[data.digest(config)] = config
    status = data.read_json(CORPUS / 'parallel_data_status_20261107.json')
    pool = set(status['common_source_ids'])
    planned = [json.loads(line)['source_id'] for line in
        (CORPUS / 'variant_c_20k_free_plan_20261107/queries_20k.jsonl').read_text().splitlines()]
    if len(pool) != 21329 or len(planned) != 20000 or not set(planned) <= pool:
        raise ValueError('The matched A/B planning pool changed')
    pilot = data.read_json(ROOT.parent / 'grounded-generative-alignment/bavarian_pilot/data/article_split.json')
    excluded_pages = {str(row['bar_page_id']) for kind in ('development', 'evaluation') for row in pilot[kind]}
    pool_pages = {sid.split(':')[0].removeprefix('bar_') for sid in pool}
    benchmark = load_benchmark()
    retries = {}
    for path in sorted((CORPUS / 'outputs/blind_retries').glob('*/records/*.json.gz')):
        saved = data.read_json(path)
        for record in saved['records']:
            source = record['source'];pid = str(source['bar_page_id'])
            retries.setdefault(pid, {})[source['source_id']] = (saved['manifest'], record, str(path.relative_to(ROOT)))
    candidates = {}
    for pid in sorted(pool_pages, key=int):
        for row in load_rows(pid, index[pid], configs, retries):
            if row['source_id'] in pool:
                candidates[row['source_id']] = row
    if set(candidates) != pool:
        raise ValueError('A shared A/B source has no valid saved blind translation')
    ordered = sorted(set(planned), key=stable_key) + sorted(pool - set(planned), key=stable_key)
    train, rejected = [], []
    seen_bar, seen_en = set(), set()
    for sid in ordered:
        row = candidates[sid]
        reason = 'pilot_article' if row['page_id'] in excluded_pages else eligible(row, tokenizer)
        if reason is None and benchmark_overlap(row, benchmark):
            reason = 'tatoeba_dev_test_overlap'
        bar_key, en_key = normalize(row['bavarian']), normalize(row['english'])
        if reason is None and (bar_key in seen_bar or en_key in seen_en):
            reason = 'duplicate_normalized_text'
        if reason:
            rejected.append({'source_id': sid, 'reason': reason})
            continue
        seen_bar.add(bar_key);seen_en.add(en_key);train.append(row)
        if len(train) == 20000:
            break
    if len(train) != 20000:
        DEST.mkdir(parents=True, exist_ok=True)
        data.write_json(DEST / 'preparation_shortfall.json', {'eligible': len(train),
            'exclusions': dict(Counter(row['reason'] for row in rejected))}, replace=True)
        raise ValueError('Fewer than 20000 eligible shared sources; inspect preparation_shortfall.json')
    train_qids = {row['wikidata_id'] for row in train}
    validation_rows = []
    validation_rejected = Counter()
    for pid in sorted(set(index) - pool_pages - excluded_pages, key=stable_key):
        for row in sorted(load_rows(pid, index[pid], configs, retries), key=lambda row: stable_key(row['source_id'])):
            reason = eligible(row, tokenizer)
            if reason is None and benchmark_overlap(row, benchmark):
                reason = 'tatoeba_dev_test_overlap'
            bk, ek = normalize(row['bavarian']), normalize(row['english'])
            if row['wikidata_id'] in train_qids:
                reason = 'training_topic'
            if reason is None and (bk in seen_bar or ek in seen_en):
                reason = 'duplicate_normalized_text'
            if reason:
                validation_rejected[reason] += 1
                continue
            validation_rows.append(row);seen_bar.add(bk);seen_en.add(ek)
            if len(validation_rows) == 500:
                break
        if len(validation_rows) == 500:
            break
    if len(validation_rows) != 500:
        raise ValueError('Could not construct the 500-pair disjoint validation set')
    package = DEST / 'package'
    if package.exists():
        raise FileExistsError('Prepared package already exists; preserve its frozen identity')
    package.mkdir(parents=True)
    for split, rows in [('train', train), ('validation', validation_rows)]:
        (package / (split + '.jsonl')).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    shutil.copytree(TOKENIZER, package / 'tokenizer')
    shutil.copyfile(Path(__file__).with_name('nllb_blind_training.py'), package / 'train.py')
    audit = {'train_pairs': len(train), 'validation_pairs': len(validation_rows),
        'train_articles': len({row['page_id'] for row in train}),
        'validation_articles': len({row['page_id'] for row in validation_rows}),
        'disjoint_articles': not ({row['page_id'] for row in train} & {row['page_id'] for row in validation_rows}),
        'disjoint_wikidata': not (train_qids & {row['wikidata_id'] for row in validation_rows}),
        'excluded_pilot_pages': sorted(excluded_pages, key=int), 'training_exclusions': rejected,
        'validation_exclusions': dict(validation_rejected),
        'reused_provisional_c_ids': len({row['source_id'] for row in train} & set(planned)),
        'replacement_ids_from_ab_reserve': sorted({row['source_id'] for row in train} - set(planned)),
        'benchmark_overlap_status': 'Both sides screened against Tatoeba v2023-09-26 dev (12) and test (98); no benchmark scoring or training-split use',
        'benchmark_overlap_policy': 'Unicode casefold/alphanumeric/whitespace equality; near duplicate at character-trigram Dice >=0.9 with length ratio >=0.9 and both at least 30 characters',
        'benchmark_archive_sha256': 'd62c59a07ec4587c9e54700dfdd3df1f868c34d9384a7226e2fccba5df3ea578',
        'quality_status': 'Technical/source-integrity filtering only; generated English is not human gold',
        'comparison_status': 'First blind run; freeze these IDs for later methods. Final six-way QC may require a matched rerun.',
        'validation_status': 'Held-out synthetic blind English with original Bavarian targets; monitoring only, not independent benchmark quality'}
    (package / 'data_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    manifest = {'base_model': 'facebook/nllb-200-distilled-600M', 'base_revision': BASE_REVISION,
        'condition': 'blind', 'direction': 'en_to_bar', 'dataset_revision': data.REVISION,
        'counts': {'train': 20000, 'validation': 500},
        'language_policy': 'bar_Latn added token; initialize its embedding from deu_Latn; unchanged SentencePiece vocabulary',
        'training': {'epochs': 3, 'micro_batch': 2, 'gradient_accumulation': 16,
                     'learning_rate': 1e-4, 'seed': 42, 'optimizer': 'adafactor',
                     'maximum_tokens': 512, 'truncation': False, 'precision': 'fp16_amp',
                     'checkpoint_selection': 'final_fixed_epoch; validation loss is diagnostic'},
        'files': {path.relative_to(package).as_posix(): file_sha(path) for path in sorted(package.rglob('*')) if path.is_file()}}
    manifest['run_id'] = data.digest(manifest)
    (package / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    from runners.nllb_blind_training import verify_package
    verify_package(package)
    archive = DEST / 'nllb_blind_20k_inputs.zip'
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as output:
        for path in sorted(package.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(package).as_posix())
    summary = {'run_id': manifest['run_id'], 'train_pairs': 20000, 'validation_pairs': 500,
               'archive_sha256': file_sha(archive), 'archive_bytes': archive.stat().st_size,
               'reused_provisional_c_ids': audit['reused_provisional_c_ids'],
               'replacement_ids_from_ab_reserve': len(audit['replacement_ids_from_ab_reserve'])}
    (DEST / 'preparation_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    prepare()
