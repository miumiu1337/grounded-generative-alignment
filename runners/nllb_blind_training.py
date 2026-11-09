"""Colab-only NLLB training with verified, resumable Drive checkpoints."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import time
import uuid


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def event(root, kind, **fields):
    value = {'at': now(), 'event': kind, **fields}
    with (root / 'events.jsonl').open('a') as stream:
        stream.write(json.dumps(value) + '\n')
        stream.flush()
    print(json.dumps(value), flush=True)


def verify_package(package):
    manifest = json.loads((package / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(package / relative) != expected:
            raise ValueError('Input checksum mismatch: ' + name)
    identity = {key: value for key, value in manifest.items() if key != 'run_id'}
    actual = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    if actual != manifest['run_id']:
        raise ValueError('Training manifest identity changed')
    return manifest


def verify_checkpoint(directory, run_id):
    receipt = json.loads((directory / 'complete.json').read_text())
    if receipt['run_id'] != run_id or type(receipt['global_step']) is not int:
        raise ValueError('Checkpoint belongs to another experiment')
    names = set(receipt['files'])
    if not {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'config.json', 'rng_state.pth'} <= names:
        raise ValueError('Checkpoint lacks resumable training state')
    if not any(name.endswith('.safetensors') for name in names):
        raise ValueError('Checkpoint has no safe model weights')
    for name, expected in receipt['files'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(directory / relative) != expected:
            raise ValueError('Checkpoint integrity failure: ' + name)
    state = json.loads((directory / 'trainer_state.json').read_text())
    if state['global_step'] != receipt['global_step']:
        raise ValueError('Checkpoint step differs from its receipt')
    return receipt


def commit_checkpoint(local, root, run_id, step):
    """Commit marker is written last; interrupted copies never become resumable."""
    files = [path for path in local.rglob('*') if path.is_file() and path.name != 'complete.json']
    size = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(root).free < size + 512 * 1024**2:
        raise OSError('Insufficient Drive space for the next checkpoint; keep the previous checkpoint and free space')
    destination = root / 'checkpoints' / (f'step-{step:07d}-' + uuid.uuid4().hex[:12])
    destination.mkdir(parents=True)
    hashes = {}
    for original in files:
        name = original.relative_to(local).as_posix()
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = sha(original)
        shutil.copyfile(original, target)
        if sha(target) != expected:
            raise ValueError('Drive checkpoint copy failed verification: ' + name)
        hashes[name] = expected
    write_json(destination / 'complete.json', {'run_id': run_id, 'global_step': step,
               'committed_at': now(), 'files': hashes})
    receipt = verify_checkpoint(destination, run_id)
    write_json(root / 'latest_checkpoint.json', {'directory': destination.name, **receipt})
    # Only this run's older committed checkpoints are removed, after a verified replacement.
    for old in destination.parent.iterdir():
        if old == destination or not old.is_dir() or not (old / 'complete.json').exists():
            continue
        marker = json.loads((old / 'complete.json').read_text())
        if marker.get('run_id') == run_id and marker.get('global_step', step) < step:
            shutil.rmtree(old)
    return destination


def seal_local_checkpoint(directory, run_id):
    """Seal a finished Trainer save, including the save whose Drive copy failed."""
    required = {'trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'config.json', 'rng_state.pth'}
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob('*')
             if p.is_file() and p.name != 'complete.json'}
    if not required <= set(files) or not any(name.endswith('.safetensors') for name in files):
        raise ValueError('Local checkpoint is incomplete; preserve it and use the verified Drive copy')
    if any(p.stat().st_size == 0 for p in files.values()):
        raise ValueError('Local checkpoint contains an empty file')
    state = json.loads((directory / 'trainer_state.json').read_text())
    step = state['global_step']
    if type(step) is not int or step < 1 or directory.name != f'checkpoint-{step}':
        raise ValueError('Local checkpoint step/name mismatch')
    for name, path in files.items():
        if path.is_symlink():
            raise ValueError('Checkpoint symlinks are not allowed')
        if name.endswith('.safetensors'):
            # Inspect the tensor header and data bounds without loading weights into RAM/GPU.
            with path.open('rb') as stream:
                length = int.from_bytes(stream.read(8), 'little')
                if not 2 <= length <= 16 * 1024**2:
                    raise ValueError('Malformed safetensors header')
                header = json.loads(stream.read(length))
            available = path.stat().st_size - 8 - length
            tensors = [v for k, v in header.items() if k != '__metadata__']
            if not tensors or any(not 0 <= v['data_offsets'][0] <= v['data_offsets'][1] <= available for v in tensors):
                raise ValueError('Truncated or malformed safetensors checkpoint')
    write_json(directory / 'complete.json', {'run_id': run_id, 'global_step': step,
        'committed_at': now(), 'storage': 'volatile_colab_local',
        'files': {name: sha(path) for name, path in files.items()}})
    return verify_checkpoint(directory, run_id)


def restore_checkpoint(root, local, run_id, prefer_local=False):
    candidates = []
    for directory in (root / 'checkpoints').glob('step-*'):
        marker = directory / 'complete.json'
        if not marker.exists():
            event(root, 'incomplete_checkpoint_ignored', directory=directory.name)
            continue
        receipt = json.loads(marker.read_text())
        if receipt['run_id'] != run_id:
            raise ValueError('Foreign checkpoint in this run directory')
        candidates.append((receipt['global_step'], directory))
    if prefer_local:
        sealed = []
        for directory in local.glob('checkpoint-*'):
            if (directory / 'complete.json').exists():
                marker = json.loads((directory / 'complete.json').read_text())
                if marker['run_id'] != run_id:
                    raise ValueError('Local checkpoint belongs to another experiment')
                sealed.append((marker['global_step'], directory))
        if sealed:
            step, directory = max(sealed, key=lambda value: value[0])
            if not candidates or step >= max(value[0] for value in candidates):
                verify_checkpoint(directory, run_id)
                event(root, 'local_checkpoint_restored', global_step=step, storage='volatile_colab_local')
                return directory
    if not candidates:
        return None
    step, original = max(candidates, key=lambda value: value[0])
    event(root, 'checkpoint_verification_started', global_step=step)
    receipt = verify_checkpoint(original, run_id)
    destination = local / f'checkpoint-{step}'
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in receipt['files'].items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(target) != expected:
            shutil.copyfile(original / name, target)
        if sha(target) != expected:
            raise ValueError('Restored checkpoint failed verification: ' + name)
    event(root, 'checkpoint_restored', global_step=step)
    return destination


def run(package, drive, seconds, checkpoint_storage='drive'):
    if checkpoint_storage not in {'drive', 'local'}:
        raise ValueError('Unknown checkpoint storage policy')
    manifest = verify_package(package)
    root = drive / manifest['run_id'][:16]
    root.mkdir(parents=True, exist_ok=True)
    local = Path('/content/nllb_training_work') / manifest['run_id'][:16]
    previous_status = root / 'status.json'
    if previous_status.exists() and json.loads(previous_status.read_text()).get('status') == 'training_complete':
        latest = json.loads((root / 'latest_checkpoint.json').read_text())
        verify_checkpoint(root / 'checkpoints' / latest['directory'], manifest['run_id'])
        print('Training already complete; verified saved model. No additional training started.', flush=True)
        return
    if checkpoint_storage == 'local' and previous_status.exists() and json.loads(previous_status.read_text()).get('status') == 'training_complete_local':
        latest_local = json.loads((root / 'latest_local_checkpoint.json').read_text())
        directory = local / f"checkpoint-{latest_local['global_step']}"
        if directory.exists():
            verify_checkpoint(directory, manifest['run_id'])
            print('Training complete on Colab local disk. Download the model before disconnecting.', flush=True)
            return
    write_json(root / 'manifest.json', manifest)
    status = lambda value, **fields: write_json(root / 'status.json', {'at': now(), 'status': value,
        'run_id': manifest['run_id'], **fields})
    status('preflight')
    try:
        if checkpoint_storage == 'local':
            code_sha = sha(Path(__file__))
            write_json(root / 'execution_amendments' / (code_sha + '.json'), {
                'at': now(), 'run_id': manifest['run_id'], 'executed_code_sha256': code_sha,
                'change': 'Checkpoint storage only: use sealed Colab-local saves; keep existing Drive checkpoint unchanged',
                'training_parameters': 'unchanged', 'durability': 'Local progress requires download before runtime deletion'})
            archived = root / 'execution_amendments' / (code_sha + '.py')
            if not archived.exists():
                archived.write_bytes(Path(__file__).read_bytes())
        import torch
        import transformers
        import accelerate
        from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                                  Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback, set_seed)
        if not torch.cuda.is_available():
            raise RuntimeError('A hosted Colab GPU is required. CPU/Mac training is disabled')
        if transformers.__version__ != '4.57.6' or accelerate.__version__ != '1.12.0':
            raise RuntimeError('Run the pinned dependency cell before training')
        if int(torch.__version__.split('.')[0]) < 2 or (
                int(torch.__version__.split('.')[0]) == 2 and int(torch.__version__.split('.')[1]) < 6):
            raise RuntimeError('PyTorch >= 2.6 is required to load the official base checkpoint securely')
        if not 60 <= seconds <= 10 * 3600:
            raise ValueError('Session budget must be between one minute and ten hours')
        settings = manifest['training']
        set_seed(settings['seed'])
        torch.set_num_threads(2)
        event(root, 'training_preflight', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
              transformers=transformers.__version__, accelerate=accelerate.__version__,
              direction=manifest['direction'], train_pairs=manifest['counts']['train'],
              validation_pairs=manifest['counts']['validation'])
        local.mkdir(parents=True, exist_ok=True)
        resume = restore_checkpoint(root, local, manifest['run_id'], prefer_local=checkpoint_storage == 'local')
        tokenizer = AutoTokenizer.from_pretrained(package / 'tokenizer', local_files_only=True,
                                                  src_lang='eng_Latn', tgt_lang='bar_Latn')
        model_source = str(resume) if resume else manifest['base_model']
        event(root, 'model_loading', resumed=bool(resume))
        model = AutoModelForSeq2SeqLM.from_pretrained(model_source,
            **({} if resume else {'revision': manifest['base_revision']}))
        bar_id, german_id = [tokenizer.convert_tokens_to_ids(value) for value in ('bar_Latn', 'deu_Latn')]
        if not resume:
            model.resize_token_embeddings(max(model.config.vocab_size, len(tokenizer)), mean_resizing=False)
            with torch.no_grad():
                model.get_input_embeddings().weight[bar_id].copy_(model.get_input_embeddings().weight[german_id])
                output = model.get_output_embeddings()
                if output.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
                    output.weight[bar_id].copy_(output.weight[german_id])
        model.config.use_cache = False
        model.generation_config.forced_bos_token_id = bar_id
        model.generation_config.max_length = 512
        features = {}
        for split in ('train', 'validation'):
            rows = [json.loads(line) for line in (package / (split + '.jsonl')).read_text().splitlines()]
            features[split] = []
            for row in rows:
                encoded = tokenizer(row['english'], text_target=row['bavarian'], truncation=False)
                if max(len(encoded['input_ids']), len(encoded['labels'])) > 512:
                    raise ValueError('Overlength input escaped preparation; no truncation permitted')
                if encoded['input_ids'][0] != tokenizer.convert_tokens_to_ids('eng_Latn') or encoded['labels'][0] != bar_id:
                    raise ValueError('Language-token encoding differs from the frozen preparation')
                features[split].append(encoded)
        started = time.monotonic()
        interrupted = {'requested': False}
        def stop_requested(signum, frame):
            if interrupted['requested']:
                raise KeyboardInterrupt()
            interrupted['requested'] = True
            print('Stop requested: finishing the current optimizer step and saving a checkpoint.', flush=True)
        signal.signal(signal.SIGINT, stop_requested)
        signal.signal(signal.SIGTERM, stop_requested)

        class Progress(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                event(root, 'training_metrics', global_step=state.global_step, metrics=logs or {})
            def on_step_end(self, args, state, control, **kwargs):
                if state.global_step == 1:
                    control.should_save = True
                if interrupted['requested'] or time.monotonic() - started >= seconds:
                    control.should_save = True
                    control.should_training_stop = True
                return control
            def on_save(self, args, state, control, **kwargs):
                if checkpoint_storage == 'local':
                    directory = local / f'checkpoint-{state.global_step}'
                    receipt = seal_local_checkpoint(directory, manifest['run_id'])
                    write_json(root / 'latest_local_checkpoint.json', receipt)
                    status('training', global_step=state.global_step, maximum_steps=state.max_steps,
                           checkpoint_storage='volatile_colab_local', local_checkpoint=str(directory))
                    event(root, 'local_checkpoint_verified', global_step=state.global_step,
                          warning='Download before runtime deletion; existing Drive backup unchanged')
                    return
                event(root, 'drive_checkpoint_started', global_step=state.global_step)
                saved = commit_checkpoint(local / f'checkpoint-{state.global_step}', root,
                                          manifest['run_id'], state.global_step)
                status('training', global_step=state.global_step, maximum_steps=state.max_steps,
                       committed_checkpoint=saved.name)
                event(root, 'drive_checkpoint_verified', global_step=state.global_step)

        args = Seq2SeqTrainingArguments(output_dir=str(local), num_train_epochs=settings['epochs'],
            per_device_train_batch_size=settings['micro_batch'],
            gradient_accumulation_steps=settings['gradient_accumulation'],
            per_device_eval_batch_size=1, learning_rate=settings['learning_rate'],
            optim='adafactor', lr_scheduler_type='linear', warmup_ratio=0.06,
            fp16=True, gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': False},
            eval_strategy='steps', eval_steps=100, save_strategy='steps', save_steps=100,
            save_total_limit=2, logging_steps=10, logging_first_step=True,
            seed=settings['seed'], data_seed=settings['seed'], report_to='none',
            dataloader_num_workers=0, predict_with_generate=False,
            save_safetensors=True, load_best_model_at_end=False, push_to_hub=False)
        trainer = Seq2SeqTrainer(model=model, args=args, train_dataset=features['train'],
            eval_dataset=features['validation'], processing_class=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
            callbacks=[Progress()])
        status('training', resumed_from=resume.name if resume else None)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        step = trainer.state.global_step
        complete = step >= trainer.state.max_steps
        if complete and settings.get('evaluate_final_validation', False):
            metrics = trainer.evaluate(metric_key_prefix='final_validation')
            write_json(root / 'final_validation.json', {
                'run_id': manifest['run_id'], 'global_step': step,
                'validation_pairs': manifest['counts']['validation'],
                'interpretation': 'Held-out synthetic validation loss; not external benchmark quality',
                'metrics': metrics})
        local_checkpoint = local / f'checkpoint-{step}'
        if not local_checkpoint.exists():
            trainer.save_model(str(local_checkpoint))
            raise RuntimeError('Final resumable checkpoint missing; local model preserved; inspect Trainer logs')
        if checkpoint_storage == 'local':
            seal_local_checkpoint(local_checkpoint, manifest['run_id'])
            status('training_complete_local' if complete else 'paused_local_resumable', global_step=step,
                   maximum_steps=trainer.state.max_steps, checkpoint_storage='volatile_colab_local',
                   model_location=str(local_checkpoint), backup_required=True,
                   warning='Download this checkpoint before deleting/disconnecting the runtime')
            event(root, 'training_complete_local' if complete else 'paused_local_resumable', global_step=step)
            return
        latest = json.loads((root / 'latest_checkpoint.json').read_text())
        if latest['global_step'] != step:
            commit_checkpoint(local_checkpoint, root, manifest['run_id'], step)
        status('training_complete' if complete else 'paused_resumable', global_step=step,
               maximum_steps=trainer.state.max_steps, validation='synthetic held-out loss; not benchmark quality',
               model_location=json.loads((root / 'latest_checkpoint.json').read_text())['directory'])
        event(root, 'training_complete' if complete else 'paused_resumable', global_step=step)
    except BaseException as error:
        status('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
               error_class=type(error).__name__, error=str(error))
        event(root, 'training_failed', error_class=type(error).__name__, error=str(error))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--drive', type=Path, required=True)
    parser.add_argument('--seconds', type=int, default=36000)
    parser.add_argument('--checkpoint-storage', choices=['drive', 'local'], default='drive')
    args = parser.parse_args()
    import fcntl
    lock_path = Path('/content/nllb_blind_training.lock')
    with lock_path.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.inputs, args.drive, args.seconds, args.checkpoint_storage)


if __name__ == '__main__':
    main()
