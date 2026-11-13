"""Clean matched FineWiki articles and export sentences plus three-sentence windows."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from datatrove.data import Document

from Cleaning.paragraphs import prose_paragraphs, sentence_records
from Cleaning.quality import SentenceQuality, normalized_key
from Cleaning.segmentation import ProseSentenceSegmentation, SaTCPU
from preprocessing.artifacts import load_config, read_json, safe_path, sha, write_json
from preprocessing.context import context_windows


def clean_article(article, segment, quality, tokenizer_language, seen, headings=()):
    paragraphs, audit = prose_paragraphs(article, reference_headings=headings)
    documents = (Document(text=p['text'], id=p['paragraph_id'], metadata=p['metadata']) for p in paragraphs)
    rows = sentence_records(segment(documents), {article['id']: article}, quality, tokenizer_language)
    for row in rows:
        if row['quality_reason'] is None:
            key = normalized_key(row['text'])
            if key in seen:
                row.update(quality_reason='duplicate_sentence', duplicate_of=seen[key])
            else:
                seen[key] = row['source_id']
    windows, reasons = context_windows(rows)
    quality.cache.clear()
    return {'article_id': article['id'], 'title': article['title'], 'revision': article['version'],
            'wikidata_id': article['wikidata_id'], 'raw_text_sha256': hashlib.sha256(article['text'].encode()).hexdigest(),
            'structure_audit': audit, 'sentences': rows, 'windows': windows, 'context_reasons': reasons}


def run(config, articles, model_directory, output):
    config = load_config(config)
    inventory = read_json(articles/'manifest.json')
    if inventory['config'] != config or inventory['status'] != 'complete':
        raise ValueError('Article retrieval must be complete and use the same configuration')
    segment = ProseSentenceSegmentation(SaTCPU(model_directory, threads=2, threshold=config['sat_threshold']))
    source = config['source_language']
    quality = SentenceQuality({source: config['quality_tokenizer_language'], 'en': 'en'},
                              minimum_alpha_token_ratio=config['minimum_alpha_token_ratio'])
    settings = {'config': config, 'input_manifest_sha256': sha(articles/'manifest.json'),
                'segmenter': segment.identity, 'quality': quality.config(),
                'context': 'Exactly previous, focus, next within one paragraph/section; focus_index=1',
                'code_sha256': {p.name: sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))}}
    write_json(output/'config.json', settings)
    seen = {source: {}, 'en': {}}; counts = Counter(); files = {}
    for item in inventory['articles']:
        original_path = safe_path(articles, item['file'])
        if sha(original_path) != item['sha256']:
            raise ValueError('Original article checksum differs: ' + item['file'])
        original = read_json(original_path)
        language = original['in_language']
        target = output/'articles'/(language+'_'+str(original['page_id'])+'.json.gz')
        if target.exists():
            cleaned = read_json(target)
            if cleaned['raw_text_sha256'] != hashlib.sha256(original['text'].encode()).hexdigest():
                raise ValueError('Saved cleaned article has different source text')
            for row in cleaned['sentences']:
                if row['quality_reason'] is None:
                    key = normalized_key(row['text'])
                    if key in seen[language]:
                        raise ValueError('Saved deduplication order differs')
                    seen[language][key] = row['source_id']
        else:
            tokenizer = config['quality_tokenizer_language'] if language == source else 'en'
            headings = config['reference_headings'] if language == source else ()
            cleaned = clean_article(original, segment, quality, tokenizer, seen[language], headings)
            write_json(target, cleaned)
        files[str(target.relative_to(output))] = sha(target)
        counts[language+'_articles'] += 1
        counts[language+'_sentences'] += sum(r['quality_reason'] is None for r in cleaned['sentences'])
        counts[language+'_windows'] += len(cleaned['windows'])
        print(json.dumps({'article': original['id'], 'counts': dict(counts)}), flush=True)
    result = {'status': 'complete', 'config': config, 'files': files, 'counts': dict(counts),
              'matched_pairs': inventory['pairs'], 'cleaning_config_sha256': sha(output/'config.json')}
    write_json(output/'manifest.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'articles', 'model-directory', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.articles, args.model_directory, args.output)))
