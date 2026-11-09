"""Inspect/seal local training saves and export bounded backup parts; no GPU inference."""

import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import zipfile
import uuid


def load_worker(path):
    spec = importlib.util.spec_from_file_location('nllb_recovery_worker', path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def prepare_local(worker_path, inputs, drive):
    worker = load_worker(worker_path)
    manifest = worker.verify_package(inputs)
    root = drive / manifest['run_id'][:16]
    local = Path('/content/nllb_training_work') / manifest['run_id'][:16]
    print('Colab-local free GiB:', round(shutil.disk_usage('/content').free / 1024**3, 2), flush=True)
    print('Mounted Drive free GiB:', round(shutil.disk_usage(drive).free / 1024**3, 2), flush=True)
    latest = root / 'latest_checkpoint.json'
    print('Last committed Drive step:', json.loads(latest.read_text())['global_step'] if latest.exists() else None, flush=True)
    with Path('/content/nllb_blind_training.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        candidates = []
        for directory in local.glob('checkpoint-*'):
            state_path = directory / 'trainer_state.json'
            if state_path.exists():
                state = json.loads(state_path.read_text())
                candidates.append((state['global_step'], directory))
        if not candidates:
            print('No local save remains. Resume will use the verified Drive checkpoint.', flush=True)
            return None
        step, directory = max(candidates, key=lambda value: value[0])
        if step > math.ceil(manifest['counts']['train'] / (manifest['training']['micro_batch'] *
                manifest['training']['gradient_accumulation'])) * manifest['training']['epochs']:
            raise ValueError('Local checkpoint exceeds this experiment step budget')
        print('Validating local checkpoint at step', step, '(hashing weights; no model loading).', flush=True)
        receipt = worker.seal_local_checkpoint(directory, manifest['run_id'])
        worker.write_json(root / 'local_recovery.json', {'run_id': manifest['run_id'],
            'global_step': step, 'local_checkpoint': str(directory),
            'recovery_code_sha256': worker.sha(Path(__file__)),
            'policy': 'Keep existing Drive checkpoint. New training saves stay on volatile Colab disk; download before runtime deletion.'})
        print('Local checkpoint verified. Resume step:', receipt['global_step'], flush=True)
        return directory


def export_parts(worker_path, inputs, destination=Path('/content/nllb_backup_parts')):
    worker = load_worker(worker_path)
    manifest = worker.verify_package(inputs)
    local = Path('/content/nllb_training_work') / manifest['run_id'][:16]
    candidates = []
    for directory in local.glob('checkpoint-*'):
        if (directory / 'complete.json').exists():
            receipt = json.loads((directory / 'complete.json').read_text())
            candidates.append((receipt['global_step'], directory))
    if not candidates:
        raise FileNotFoundError('No sealed local checkpoint; run recovery inspection first')
    step, checkpoint = max(candidates, key=lambda value: value[0])
    receipt = worker.verify_checkpoint(checkpoint, manifest['run_id'])
    folder = destination / manifest['run_id'][:16] / f'step-{step}'
    folder.mkdir(parents=True, exist_ok=True)
    name = f'nllb_blind_{manifest["run_id"][:16]}_step{step}'
    index = folder / (name + '.backup.json')
    if index.exists():
        record = json.loads(index.read_text())
        for part in record['parts']:
            if worker.sha(folder / part['name']) != part['sha256']:
                raise ValueError('Existing local backup part differs; preserve it for inspection')
        return index
    total = sum(path.stat().st_size for path in checkpoint.rglob('*') if path.is_file())
    if shutil.disk_usage(folder).free < 2 * total + 512 * 1024**2:
        raise OSError('Not enough Colab-local space for split backup; preserve the checkpoint')
    archive = folder / (name + '.zip')
    print('Preparing resumable backup for step', step, flush=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_STORED) as output:
        for path in sorted(checkpoint.rglob('*')):
            if path.is_file():
                output.write(path, path.relative_to(checkpoint).as_posix())
    archive_sha = worker.sha(archive)
    archive_bytes = archive.stat().st_size
    parts = []
    with archive.open('rb') as source:
        number = 0
        while source.tell() < archive_bytes:
            number += 1
            part = folder / (name + f'.part{number:03d}')
            remaining = min(512 * 1024**2, archive_bytes - source.tell())
            digest = hashlib.sha256()
            with part.open('wb') as output:
                while remaining:
                    chunk = source.read(min(4 * 1024**2, remaining))
                    if not chunk:
                        raise EOFError('Unexpected end while splitting backup')
                    output.write(chunk);digest.update(chunk);remaining -= len(chunk)
            parts.append({'name': part.name, 'bytes': part.stat().st_size, 'sha256': digest.hexdigest()})
            print('Backup part ready:', part.name, flush=True)
    worker.write_json(index, {'run_id': manifest['run_id'], 'global_step': step,
        'archive_sha256': archive_sha, 'archive_bytes': archive_bytes, 'parts': parts,
        'checkpoint_receipt': receipt, 'manifest': manifest})
    # Keep the original checkpoint and split parts. Only the redundant local ZIP is removed.
    archive.unlink()
    return index


def export_single(worker_path, inputs, destination=Path('/content/nllb_model_exports')):
    """One lossless compressed ZIP; preserve original weight precision and resumable state."""
    worker = load_worker(worker_path)
    manifest = worker.verify_package(inputs)
    local = Path('/content/nllb_training_work') / manifest['run_id'][:16]
    candidates = []
    for directory in local.glob('checkpoint-*'):
        if (directory / 'complete.json').exists():
            receipt = json.loads((directory / 'complete.json').read_text())
            candidates.append((receipt['global_step'], directory))
    if not candidates:
        raise FileNotFoundError('No sealed checkpoint available to export')
    step, checkpoint = max(candidates, key=lambda value: value[0])
    receipt = worker.verify_checkpoint(checkpoint, manifest['run_id'])
    folder = destination / manifest['run_id'][:16]
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / f'nllb_{manifest["condition"]}_{manifest["run_id"][:16]}_step{step}.zip'
    marker = archive.with_suffix('.export.json')
    if archive.exists() and marker.exists():
        saved = json.loads(marker.read_text())
        if saved['run_id'] != manifest['run_id'] or worker.sha(archive) != saved['archive_sha256']:
            raise ValueError('Existing single-file export differs; preserve it for inspection')
        return archive
    files = [checkpoint / name for name in receipt['files']]
    if shutil.disk_usage(folder).free < sum(path.stat().st_size for path in files) + 512 * 1024**2:
        raise OSError('Not enough Colab-local space for one model ZIP; original checkpoint is preserved')
    temporary = archive.with_suffix('.zip.part')
    print('Creating one lossless model ZIP; no quantization or model inference.', flush=True)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as output:
        for path in sorted(files):
            print('Archiving:', path.name, flush=True)
            output.write(path, path.relative_to(checkpoint).as_posix())
        output.write(checkpoint / 'complete.json', 'complete.json')
        output.writestr('training_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    with zipfile.ZipFile(temporary) as saved:
        if saved.testzip() is not None:
            raise ValueError('Single model archive failed its CRC check')
    temporary.replace(archive)
    worker.write_json(marker, {'run_id': manifest['run_id'], 'global_step': step,
        'archive_sha256': worker.sha(archive), 'archive_bytes': archive.stat().st_size,
        'checkpoint_receipt': receipt})
    print('ONE_MODEL_ZIP:', archive, 'bytes:', archive.stat().st_size,
          'sha256:', json.loads(marker.read_text())['archive_sha256'], flush=True)
    return archive


def restore_single(worker_path, inputs, archive, local_base=Path('/content/nllb_training_work')):
    """Restore an own-run single ZIP after integrity checks; never overwrite another save."""
    worker = load_worker(worker_path)
    manifest = worker.verify_package(inputs)
    local = Path(local_base) / manifest['run_id'][:16]
    local.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        info = source.infolist()
        names = [member.filename for member in info]
        if len(names) != len(set(names)) or len(names) > 128:
            raise ValueError('Duplicate or excessive backup members')
        for name in ('training_manifest.json', 'complete.json'):
            if source.getinfo(name).file_size > 256 * 1024:
                raise ValueError('Oversized backup metadata')
        if json.loads(source.read('training_manifest.json')) != manifest:
            raise ValueError('Backup belongs to another method or frozen training run')
        receipt = json.loads(source.read('complete.json'))
        if receipt['run_id'] != manifest['run_id'] or type(receipt['global_step']) is not int:
            raise ValueError('Backup checkpoint identity differs')
        if receipt['global_step'] < 1:
            raise ValueError('Backup has no completed optimizer step')
        if set(names) != set(receipt['files']) | {'complete.json', 'training_manifest.json'}:
            raise ValueError('Unexpected checkpoint backup contents')
        total = sum(member.file_size for member in info)
        if total > 16 * 1024**3 or shutil.disk_usage(local).free < total + 512 * 1024**2:
            raise OSError('Backup exceeds available Colab-local storage')
        for member in info:
            relative = Path(member.filename)
            if (relative.is_absolute() or '..' in relative.parts
                    or (member.external_attr >> 16) & 0o170000 == 0o120000):
                raise ValueError('Unsafe backup entry')
        destination = local / f'checkpoint-{receipt["global_step"]}'
        if destination.exists():
            old = worker.verify_checkpoint(destination, manifest['run_id'])
            if old['files'] != receipt['files']:
                raise ValueError('Existing local checkpoint differs; preserve both copies')
            return destination
        staging = local / ('.restore-' + uuid.uuid4().hex)
        staging.mkdir()
        for name in receipt['files']:
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            actual = hashlib.sha256()
            with source.open(name) as original, target.open('wb') as output:
                for chunk in iter(lambda: original.read(4 * 1024**2), b''):
                    actual.update(chunk);output.write(chunk)
            if actual.hexdigest() != receipt['files'][name]:
                raise ValueError('Restored checkpoint checksum mismatch: ' + name)
        worker.write_json(staging / 'complete.json', receipt)
        worker.verify_checkpoint(staging, manifest['run_id'])
        os.replace(staging, destination)
    print('Restored verified checkpoint:', destination, flush=True)
    return destination
