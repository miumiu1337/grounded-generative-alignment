"""One mining method, with selectable BGE-M3 or LASER-1 sentence embeddings."""

import hashlib
from pathlib import Path

import numpy as np

from parallel_data_generation.scale_mining import rank_group


class Encoder:
    def __init__(self, settings):
        self.settings = settings
        if settings['encoder'] == 'bge_m3':
            import torch
            from transformers import AutoModel, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(settings['bge_model'], revision=settings['bge_revision'], token=False)
            self.model = AutoModel.from_pretrained(settings['bge_model'], revision=settings['bge_revision'],
                torch_dtype=torch.float16 if settings['device'] == 'cuda' else torch.float32,
                attn_implementation='sdpa', token=False).to(settings['device']).eval()
        elif settings['encoder'] == 'laser':
            paths = [Path(settings['laser_model_directory'])/name for name in
                     ('93langs.fcodes', '93langs.fvocab', 'bilstm.93langs.2018-12-26.pt')]
            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
            # Check the original LASER-1 asset receipt before deserializing its checkpoint.
            receipt = hashlib.sha256(('1.1.2|'+'|'.join(hashes)+'|bar=de|en=en').encode()).hexdigest()
            if receipt != '1f53121a57525b13aef6bedfe3cf307a6e4078e1cdbb80eefbddb9fde20086ab':
                raise ValueError('LASER-1 assets differ from the pinned original baseline')
            from laserembeddings import Laser
            self.model = Laser(*map(str, paths), embedding_options={
                'cpu': settings['device'] == 'cpu', 'max_sentences': 16})
        else:
            raise ValueError('Choose mining.encoder=bge_m3 or laser')

    def encode(self, texts, language):
        if not texts:
            return np.empty((0, 1024), dtype=np.float32)
        if self.settings['encoder'] == 'laser':
            code = 'en' if language == 'en' else self.settings['laser_tokenizer_language']
            return self.model.embed_sentences(texts, lang=code)
        import torch
        output = []
        for start in range(0, len(texts), 8):
            features = self.tokenizer(texts[start:start+8], padding=False, truncation=False)
            if max(map(len, features['input_ids'])) > 8192:
                raise ValueError('BGE-M3 input exceeds8192 tokens; no truncation')
            # Bound padding memory as well as sentence count.
            offset = 0
            while offset < len(features['input_ids']):
                end = offset+1
                while end < len(features['input_ids']) and (end-offset+1)*max(
                        len(ids) for ids in features['input_ids'][offset:end+1]) <= 4096:
                    end += 1
                rows = [{key: value[i] for key, value in features.items()} for i in range(offset, end)]
                batch = self.tokenizer.pad(rows, padding=True, return_tensors='pt').to(self.settings['device'])
                with torch.inference_mode():
                    values = self.model(**batch).last_hidden_state[:, 0].float()
                    norms = values.norm(dim=1, keepdim=True)
                    if not torch.isfinite(values).all() or (norms <= 0).any():
                        raise ValueError('Invalid BGE-M3 embedding')
                    output.extend((values/norms).cpu().numpy())
                offset = end
        return np.asarray(output, dtype=np.float32)


def mine(sources, targets, encoder, language, condition='sentence_only'):
    """Context affects ranking inputs; outputs retain only the English focus."""
    if condition not in ('sentence_only', 'paragraph_context'):
        raise ValueError('Unknown context condition')
    field = 'focus_text' if condition == 'sentence_only' else 'context_text'
    x = encoder.encode([source[field] for source in sources], language)
    y = encoder.encode([target[field] for target in targets], 'en')
    return rank_group(sources, targets, x, y)
