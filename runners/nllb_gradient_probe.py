"""Diagnostic forward/backward pass in a separate GPU process before training."""

import json
import math
from importlib.metadata import version


def probe(package, manifest):
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq, set_seed
    import torch.nn.functional as F

    if not torch.cuda.is_available() or version('transformers') != '4.57.6':
        raise RuntimeError('The probe requires the pinned hosted CUDA environment')
    torch.set_num_threads(2)
    set_seed(manifest['training']['seed'])
    tokenizer = AutoTokenizer.from_pretrained(package / 'tokenizer', local_files_only=True,
                                              src_lang='eng_Latn', tgt_lang='bar_Latn')
    model = AutoModelForSeq2SeqLM.from_pretrained(manifest['base_model'], revision=manifest['base_revision'])
    bar, german, english = [tokenizer.convert_tokens_to_ids(x) for x in ('bar_Latn', 'deu_Latn', 'eng_Latn')]
    model.resize_token_embeddings(max(model.config.vocab_size, len(tokenizer)), mean_resizing=False)
    with torch.no_grad():
        model.get_input_embeddings().weight[bar].copy_(model.get_input_embeddings().weight[german])
        output = model.get_output_embeddings()
        if output.weight.data_ptr() != model.get_input_embeddings().weight.data_ptr():
            output.weight[bar].copy_(output.weight[german])
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.train().cuda()
    rows = [json.loads(x) for x in (package / 'train.jsonl').read_text().splitlines()][:2]
    features = [tokenizer(r['english'], text_target=r['bavarian'], truncation=False) for r in rows]
    batch = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)(features)
    if not (batch['input_ids'][:, 0].eq(english).all() and batch['labels'][:, 0].eq(bar).all()):
        raise ValueError('Source/target language tokens are reversed')
    if not (batch['decoder_input_ids'][:, 0].eq(model.config.decoder_start_token_id).all()
            and batch['decoder_input_ids'][:, 1].eq(bar).all()):
        raise ValueError('Teacher-forcing decoder shift differs from the Bavarian labels')
    pad = tokenizer.pad_token_id
    for i, feature in enumerate(features):
        length = len(feature['labels'])
        if batch['labels'][i, length:].ne(-100).any():
            raise ValueError('Target padding contributes to the loss')
    batch = {k: v.cuda() for k, v in batch.items()}
    scaler = torch.amp.GradScaler('cuda')
    for attempt in range(8):
        set_seed(manifest['training']['seed'])
        model.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            result = model(**batch)
            expected = F.cross_entropy(result.logits.float().reshape(-1, result.logits.shape[-1]),
                                       batch['labels'].reshape(-1), ignore_index=-100)
        if not torch.isfinite(result.loss) or not torch.allclose(result.loss, expected, atol=1e-4, rtol=1e-4):
            raise ValueError('Model loss is nonfinite or differs from masked target cross-entropy')
        scale = scaler.get_scale()
        scaler.scale(result.loss).backward()
        # No optimizer step: this diagnostic model is discarded. Actual runs
        # initialize their own model and RNG from the original base afterward.
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.stack([g.detach().float().norm(2) for g in gradients]).norm(2) / scale
        value = float(norm)
        if math.isfinite(value) and value > 0:
            return {'status': 'forward_backward_checks_passed',
                'training_run_id': manifest['run_id'], 'sampled_source_ids': [r['source_id'] for r in rows],
                'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__,
                'transformers': version('transformers'), 'loss': float(result.loss.detach()),
                'masked_cross_entropy_matches': True, 'decoder_shift_matches': True,
                'padding_ignored': True, 'finite_nonzero_gradient_norm': value,
                'gradient_scale': scale, 'scale_backoffs': attempt,
                'actual_training_weights_modified': False}
        scaler.update(new_scale=scale / 2)
        del result, expected, gradients, norm
    raise RuntimeError('No finite nonzero gradients after eight FP16 scale attempts')
