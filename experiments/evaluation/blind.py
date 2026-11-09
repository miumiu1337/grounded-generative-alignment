"""Fixed, checkpointed English-to-Bavarian evaluation; inference is Colab GPU only."""

import argparse
import csv
from datetime import datetime, timezone
import gc
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def validate_inputs(inputs):
    protocol = json.loads((inputs / 'protocol.json').read_text())
    if digest({k: v for k, v in protocol.items() if k != 'evaluation_id'}) != protocol['evaluation_id']:
        raise ValueError('Evaluation protocol identity differs')
    if sha(inputs / 'test.jsonl') != protocol['test_sha256']:
        raise ValueError('Test-set checksum differs')
    if sha(inputs / 'evaluate.py') != protocol['runner_sha256']:
        raise ValueError('Evaluation runner checksum differs')
    examples = [json.loads(line) for line in (inputs / 'test.jsonl').read_text().splitlines()]
    if len(examples) != 98 or len({row['example_id'] for row in examples}) != 98:
        raise ValueError('Expected all 98 unique Tatoeba test examples')
    if any(not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', row['example_id']) for row in examples):
        raise ValueError('Unsafe benchmark example identifier')
    if any(not row['english'].strip() or not row['bavarian_reference'].strip() for row in examples):
        raise ValueError('Empty benchmark source/reference')
    return protocol, examples


def score_predictions(examples, rows):
    from sacrebleu.metrics import BLEU, CHRF
    if len(rows) != len(examples) or any(row['example_id'] != example['example_id']
                                        for example, row in zip(examples, rows)):
        raise ValueError('Predictions must preserve the full test set and its order')
    hypotheses = [row['prediction'] for row in rows]
    references = [[example['bavarian_reference'] for example in examples]]
    bleu = BLEU(tokenize='13a', lowercase=False, smooth_method='exp', effective_order=False)
    chrf = CHRF(char_order=6, word_order=2, beta=2, lowercase=False)
    result = {'examples': len(rows), 'BLEU': bleu.corpus_score(hypotheses, references).score,
              'chrF++': chrf.corpus_score(hypotheses, references).score,
              'BLEU_signature': str(bleu.get_signature()), 'chrF++_signature': str(chrf.get_signature()),
              'empty_predictions': sum(not text.strip() for text in hypotheses),
              'length_capped_predictions': sum(row['length_capped'] for row in rows)}
    return result


def run(inputs, output, seconds=7200):
    protocol, examples = validate_inputs(inputs)
    root = output / protocol['evaluation_id'][:16]
    root.mkdir(parents=True, exist_ok=True)
    save(root / 'protocol.json', protocol)
    def status(state, **fields):
        value = {'at': datetime.now(timezone.utc).isoformat(), 'status': state,
                 'evaluation_id': protocol['evaluation_id'], **fields}
        save(root / 'status.json', value)
        print(json.dumps(value), flush=True)
    started = time.monotonic()
    try:
        if not 60 <= seconds <= 7200:
            raise ValueError('Evaluation session budget must be 60..7200 seconds')
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
        if not torch.cuda.is_available():
            raise RuntimeError('Use the hosted Colab GPU; local Mac/CPU inference is disabled')
        if version('transformers') != '4.57.6' or version('sacrebleu') != '2.5.1':
            raise RuntimeError('Run the pinned evaluation dependency cell')
        torch.set_num_threads(2)
        set_seed(42)
        checkpoint = Path(protocol['checkpoint_path'])
        receipt = protocol['checkpoint_receipt']
        for name, expected in receipt['files'].items():
            relative = Path(name)
            if relative.is_absolute() or '..' in relative.parts or sha(checkpoint / relative) != expected:
                raise ValueError('Final trained checkpoint missing or changed: ' + name)
        state = json.loads((checkpoint / 'trainer_state.json').read_text())
        if state['global_step'] != 1875 or receipt['global_step'] != 1875:
            raise ValueError('Evaluation requires the final step-1875 checkpoint')
        runtime = {'torch': torch.__version__, 'transformers': version('transformers'),
                   'sacrebleu': version('sacrebleu'), 'gpu': torch.cuda.get_device_name(0),
                   'precision': 'float16', 'seed': 42}
        save(root / 'runtime.json', runtime)
        summaries = {}
        for condition in ('blind_finetuned', 'unadapted_initialized_control'):
            directory = root / condition
            directory.mkdir(exist_ok=True)
            saved_rows = {}
            for example in examples:
                path = directory / (example['example_id'] + '.json')
                if path.exists():
                    row = json.loads(path.read_text())
                    if (row['evaluation_id'] != protocol['evaluation_id'] or row['condition'] != condition
                            or row['example_sha256'] != digest(example) or not isinstance(row['prediction'], str)
                            or type(row['length_capped']) is not bool):
                        raise ValueError('Cached prediction identity changed')
                    saved_rows[example['example_id']] = row
            missing = [example for example in examples if example['example_id'] not in saved_rows]
            model = tokenizer = None
            if missing:
                status('loading_model', condition=condition, completed_examples=len(saved_rows))
                tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True,
                    src_lang='eng_Latn', tgt_lang='bar_Latn')
                if condition == 'blind_finetuned':
                    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.float16,
                                                                 use_safetensors=True, local_files_only=True)
                else:
                    model = AutoModelForSeq2SeqLM.from_pretrained(protocol['base_model'],
                        revision=protocol['base_revision'], torch_dtype=torch.float16)
                    model.resize_token_embeddings(max(model.config.vocab_size, len(tokenizer)), mean_resizing=False)
                    bar, german = [tokenizer.convert_tokens_to_ids(v) for v in ('bar_Latn', 'deu_Latn')]
                    with torch.no_grad():
                        model.get_input_embeddings().weight[bar].copy_(model.get_input_embeddings().weight[german])
                        if model.get_output_embeddings().weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
                            model.get_output_embeddings().weight[bar].copy_(model.get_output_embeddings().weight[german])
                model = model.to('cuda').eval()
                model.config.use_cache = True
                for offset in range(0, len(missing), 4):
                    if time.monotonic() - started >= seconds:
                        status('paused_resumable', condition=condition, completed_examples=len(saved_rows))
                        return
                    batch = missing[offset:offset+4]
                    encoded = tokenizer([row['english'] for row in batch], truncation=False,
                                        padding=True, return_tensors='pt')
                    if encoded['input_ids'].shape[1] > 512:
                        raise ValueError('Benchmark input exceeds 512 tokens; no truncation allowed')
                    with torch.inference_mode():
                        sequences = model.generate(**encoded.to('cuda'),
                            forced_bos_token_id=tokenizer.convert_tokens_to_ids('bar_Latn'),
                            **protocol['generation'])
                    texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
                    for example, text, sequence in zip(batch, texts, sequences.cpu().tolist()):
                        tokens = [token for token in sequence if token != tokenizer.pad_token_id]
                        row = {'evaluation_id': protocol['evaluation_id'], 'condition': condition,
                            'example_id': example['example_id'], 'example_sha256': digest(example),
                            'english': example['english'], 'bavarian_reference': example['bavarian_reference'],
                            'prediction': text.strip(),
                            'length_capped': len(tokens) - 1 >= protocol['generation']['max_new_tokens']}
                        save(directory / (example['example_id'] + '.json'), row)
                        saved_rows[example['example_id']] = row
                    status('evaluating', condition=condition, completed_examples=len(saved_rows), total_examples=98)
            ordered = [saved_rows[example['example_id']] for example in examples]
            summaries[condition] = score_predictions(examples, ordered)
            save(root / (condition + '_metrics.json'), summaries[condition])
            with (root / (condition + '_predictions.csv')).open('w', newline='') as stream:
                fields = ['example_id', 'english', 'bavarian_reference', 'prediction', 'length_capped']
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows({key: row[key] for key in fields} for row in ordered)
            del model, tokenizer
            gc.collect(); torch.cuda.empty_cache()
        save(root / 'summary.json', {'evaluation_id': protocol['evaluation_id'], 'results': summaries,
            'interpretation': '98-example external test; small sample and spelling variation limit precision. No tuning on these scores. The unadapted control has the same German-initialized Bavarian token, not native Bavarian support.'})
        status('complete', results=summaries)
    except BaseException as error:
        status('failed', error_class=type(error).__name__, error=str(error))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import fcntl
    with Path('/content/nllb_tatoeba_eval.lock').open('a') as lock, Path('/content/nllb_blind_training.lock').open('a') as training_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(training_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args.inputs, args.output)
