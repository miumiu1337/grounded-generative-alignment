"""Run a frozen NLLB worker with explicit persistent Modal Volume commits."""

import importlib.util
import json
from pathlib import Path
import zipfile


def load_worker(package):
    spec = importlib.util.spec_from_file_location('frozen_modal_nllb_worker', package/'train.py')
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    return worker


def unpack_inputs(archive_path, destination, expected_sha, expected_run, sha,
                  expected_condition='variant_a'):
    if expected_condition not in {'blind','variant_a','variant_b','bge_m3'}:
        raise ValueError('Unsupported Modal training condition')
    if sha(archive_path) != expected_sha:
        raise ValueError('Modal training input archive checksum differs')
    with zipfile.ZipFile(archive_path) as archive:
        total = 0
        seen = set()
        for item in archive.infolist():
            relative = Path(item.filename)
            total += item.file_size
            if (relative.is_absolute() or '..' in relative.parts or item.filename in seen
                    or (item.external_attr >> 16) & 0o170000 == 0o120000
                    or item.file_size > 100*1024**2 or total > 512*1024**2):
                raise ValueError('Unsafe or oversized Modal training input archive')
            seen.add(item.filename)
        destination.mkdir(parents=True, exist_ok=True)
        archive.extractall(destination)
    worker = load_worker(destination)
    manifest = worker.verify_package(destination)
    if manifest['run_id'] != expected_run or manifest['condition'] != expected_condition:
        raise ValueError('Modal training package is not the authorized condition/run')
    return worker, manifest


def run_training(worker, package, runs, seconds, volume):
    manifest = worker.verify_package(package)
    root = runs/manifest['run_id'][:16]
    root.mkdir(parents=True, exist_ok=True)
    original_commit = worker.commit_checkpoint
    original_event = worker.event
    original_write = worker.write_json

    def commit(local, target, run_id, step):
        saved = original_commit(local, target, run_id, step)
        volume.commit()
        # Verify the committed file view in a later container before reporting success to the user.
        original_event(target, 'modal_volume_checkpoint_committed', global_step=step,
                       directory=saved.name)
        return saved

    def write(path, value):
        if Path(path).name == 'status.json':
            value = {**value, 'checkpoint_storage': 'modal_volume', 'platform': 'modal'}
        return original_write(path, value)

    def event(target, kind, **fields):
        renamed = {'drive_checkpoint_started': 'modal_checkpoint_copy_started',
                   'drive_checkpoint_verified': 'modal_checkpoint_verified'}
        return original_event(target, renamed.get(kind, kind), **fields)

    worker.commit_checkpoint = commit
    worker.write_json = write
    worker.event = event
    worker.write_json(root/'execution_amendments/modal_storage_v1.json', {
        'run_id': manifest['run_id'], 'platform': 'modal',
        'change': 'Frozen training worker; copy verified full checkpoints to a Modal Volume and explicitly commit before training advances',
        'model_data_optimizer_schedule': 'unchanged', 'executed_wrapper_sha256': worker.sha(Path(__file__))})
    archived = root/'execution_amendments'/('modal_wrapper_'+worker.sha(Path(__file__))+'.py')
    if not archived.exists():
        archived.write_bytes(Path(__file__).read_bytes())
    try:
        worker.run(package, runs, seconds, checkpoint_storage='drive')
    finally:
        # Propagate commit failures, including when persisting failure telemetry.
        volume.commit()
    status = json.loads((root/'status.json').read_text())
    latest = json.loads((root/'latest_checkpoint.json').read_text())
    worker.verify_checkpoint(root/'checkpoints'/latest['directory'], manifest['run_id'])
    return {'status': status, 'checkpoint': latest}
