"""Offline Variant C retrieval from a pinned FineWeb shard; no search API."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlsplit

from parallel_data_generation.variant_c import external_url
from preprocessing.artifacts import load_config, read_json, safe_path, sha

MIRRORS = ('wikiwand.com', 'wikimili.com', 'wikizero.com', 'en-academic.com',
           'enacademic.com', 'wikipedia-on-ipfs.org', 'thefullwiki.org', 'wikinews.org')


def canonical_url(url):
    if isinstance(url, str) and url.startswith('http://'):
        url = 'https://'+url[7:]
    result = external_url(url)
    host = urlsplit(result).hostname
    if any(host == domain or host.endswith('.'+domain) for domain in MIRRORS):
        raise ValueError('Wikipedia mirror or excluded Wikimedia evidence')
    return result


def terms(query):
    return sorted(set(re.findall(r'[^\W_]+', query.casefold())))[:64]


def build_index(config, topics, shard, output):
    """Literal topic pool followed by BM25; retrieval is not a support judgment."""
    import ahocorasick
    import pyarrow.parquet as pq
    import spacy
    from Cleaning.structure import ArticleStructure
    from Cleaning.quality import SentenceQuality, surface_text

    policy = config['fineweb']
    if sha(shard) != policy['sha256']:
        raise ValueError('FineWeb shard checksum differs from the pinned sample')
    identity = {'policy': policy, 'topics': topics, 'code_sha256': sha(__file__)}
    if output.exists():
        with sqlite3.connect(output) as db:
            if json.loads(db.execute('SELECT value FROM metadata').fetchone()[0])['identity'] != identity:
                raise ValueError('Existing FineWeb index uses different topics or settings')
        return
    variants = {}
    for topic in topics:
        for title in (topic['title'], re.sub(r'\s*\([^)]*\)\s*', ' ', topic['title'])):
            key = ' '.join(title.casefold().split())
            if len(key) >= 3:
                variants.setdefault(key, set()).add(topic['article_id'])
    if not variants:
        raise ValueError('No usable topic title variants')
    matcher = ahocorasick.Automaton()
    for title, ids in variants.items():
        matcher.add_word(title, (len(title), tuple(sorted(ids))))
    matcher.make_automaton()
    nlp = spacy.blank('en'); nlp.add_pipe('sentencizer')
    quality = SentenceQuality({'en': 'en'}, minimum_alpha_token_ratio=.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.partial')
    if temporary.exists():
        raise FileExistsError('An incomplete FineWeb index remains; inspect it or choose a new output path')
    db = sqlite3.connect(temporary)
    counts = Counter(); urls = set(); retained = scanned = passages = 0
    try:
        db.execute('CREATE VIRTUAL TABLE excerpts USING fts5(text, tokenize="porter unicode61")')
        db.execute('CREATE TABLE provenance(id INTEGER PRIMARY KEY, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE topics(topic TEXT, passage INTEGER, PRIMARY KEY(topic,passage))')
        for batch in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=['id','text','url','date','dump'], use_threads=False):
            for row in batch.to_pylist():
                scanned += 1
                text = row['text']
                if not isinstance(text, str) or not text.strip() or len(text) > 131072:
                    continue
                normalized = ' '.join(text.casefold().split()); matched = set()
                for end, (length, ids) in matcher.iter(normalized):
                    begin = end-length+1
                    if (begin and normalized[begin-1].isalnum()) or (end+1 < len(normalized) and normalized[end+1].isalnum()):
                        continue
                    matched.update(ids)
                matched = {topic for topic in matched if counts[topic] < policy['maximum_documents_per_topic']}
                if not matched:
                    continue
                try:
                    url = canonical_url(row['url'])
                except (TypeError, ValueError):
                    counts['rejected_url'] += 1
                    continue
                if url in urls:
                    continue
                size = len(text.encode())
                if retained+size > policy['maximum_retained_text_bytes']:
                    raise OSError('FineWeb retained-text budget exceeded; partial index preserved')
                urls.add(url); retained += size
                counts.update(matched)
                structure = ArticleStructure(text, language='en'); origin = 0
                for line in text.splitlines(keepends=True):
                    end = origin+len(line)
                    if not line.strip() or structure.reason_for_span(origin, end):
                        origin = end; continue
                    for sentence in nlp(line).sents:
                        excerpt = surface_text(sentence.text)
                        if (len(excerpt.split()) > policy['maximum_excerpt_words']
                                or quality.check([excerpt], 'en')[0]):
                            continue
                        passages += 1
                        metadata = {'provider': 'fineweb', 'document_id': str(row['id']), 'url': url,
                            'original_url': row['url'], 'title': 'FineWeb: '+str(urlsplit(url).hostname),
                            'text': excerpt, 'crawl_date': str(row['date']), 'dump': row['dump'],
                            'dataset': 'HuggingFaceFW/fineweb', 'revision': policy['revision'], 'shard': policy['shard'],
                            'document_text_sha256': hashlib.sha256(text.encode()).hexdigest(),
                            'raw_start': origin+sentence.start_char, 'raw_end': origin+sentence.end_char}
                        db.execute('INSERT INTO excerpts(rowid,text) VALUES(?,?)', (passages, excerpt))
                        db.execute('INSERT INTO provenance VALUES(?,?)', (passages, json.dumps(metadata, ensure_ascii=False)))
                        db.executemany('INSERT INTO topics VALUES(?,?)', [(topic, passages) for topic in sorted(matched)])
                    origin = end
                quality.cache.clear()
            db.commit()
        db.execute('CREATE TABLE metadata(value TEXT NOT NULL)')
        db.execute('INSERT INTO metadata VALUES(?)', (json.dumps({'identity': identity, 'passages': passages,
            'rows_scanned': scanned, 'counts': dict(counts), 'scope': 'One pinned FineWeb sample shard; not exhaustive web coverage.'}),))
        db.commit()
    finally:
        db.close()
    os.replace(temporary, output)


def search(index, article_id, query, limit=2):
    match = ' OR '.join('"'+term.replace('"', '""')+'"' for term in terms(query))
    if not match:
        return []
    # mode=ro prevents a misspelled index path from creating an empty database.
    with sqlite3.connect(Path(index).absolute().as_uri()+'?mode=ro', uri=True) as db:
        rows = db.execute('SELECT p.payload,bm25(excerpts) AS score FROM excerpts '
            'JOIN provenance p ON p.id=excerpts.rowid JOIN topics t ON t.passage=excerpts.rowid '
            'WHERE excerpts MATCH ? AND t.topic=? ORDER BY score,excerpts.rowid', (match, article_id))
        result = []; urls = set()
        for payload, score in rows:
            row = json.loads(payload)
            if canonical_url(row['url']) in urls:
                continue
            urls.add(row['url'])
            result.append({**row, 'reference_id': 'FW'+str(len(result)+1), 'retrieval_score': score})
            if len(result) == limit:
                break
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cleaned', 'shard', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = read_json(args.cleaned/'manifest.json')
    if manifest['status'] != 'complete' or manifest['config'] != config:
        raise ValueError('FineWeb topics must come from completed cleaning with the same configuration')
    topics = []
    for name, expected in manifest['files'].items():
        path = safe_path(args.cleaned, name)
        if sha(path) != expected:
            raise ValueError('FineWeb topic article checksum differs')
        article = read_json(path)
        if article['article_id'].startswith('enwiki/'):
            topics.append({'article_id': article['article_id'], 'title': article['title']})
    build_index(config, topics, args.shard, args.output)
