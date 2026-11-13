"""Identify comparable articles and retrieve exact rows from pinned FineWiki shards."""

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_url

from preprocessing.article_matching import match_articles
from preprocessing.artifacts import load_config, read_json, sha, write_json
from preprocessing.finewiki_pilot import DATASET, validate_article
from preprocessing.finewiki_range import RangeReader

META = ['id', 'page_id', 'wikidata_id', 'version', 'title', 'in_language']
COLUMNS = META + ['text', 'url']


def index_shard(shard, revision, directory):
    path = directory/(Path(shard).stem+'.json.gz')
    identity = {'dataset': DATASET, 'revision': revision, 'shard': shard}
    if path.exists():
        saved = read_json(path)
        if saved['lookup'] != identity:
            raise ValueError('Metadata cache belongs to another FineWiki shard/revision')
        return path
    rows = []
    url = hf_hub_url(DATASET, shard, repo_type='dataset', revision=revision)
    with RangeReader(url, deadline_seconds=600) as stream:
        parquet = pq.ParquetFile(stream, pre_buffer=False)
        for group in range(parquet.num_row_groups):
            for index, row in enumerate(parquet.read_row_group(group, columns=META, use_threads=False).to_pylist()):
                rows.append({**row, 'shard_path': shard, 'row_group': group, 'row_index': index})
    write_json(path, {'lookup': identity, 'rows': rows})
    return path


def iter_metadata(paths):
    for path in paths:
        yield from read_json(path)['rows']


def latest_sources(rows, language):
    selected = {}
    for row in rows:
        if (row['in_language'] != language or type(row['version']) is not int or row['version'] <= 0
                or type(row['page_id']) is not int or row['page_id'] <= 0
                or row['id'] != f"{language}wiki/{row['page_id']}"):
            raise ValueError('Invalid source article metadata')
        prior = selected.get(row['page_id'])
        if prior is None or row['version'] > prior['version']:
            selected[row['page_id']] = row
        elif row['version'] == prior['version'] and row['wikidata_id'] != prior['wikidata_id']:
            raise ValueError('Conflicting latest source topic metadata')
    return sorted(selected.values(), key=lambda row: row['page_id'])


def retrieve_rows(lookups, revision, output):
    groups = defaultdict(list)
    for row in lookups:
        groups[(row['shard_path'], row['row_group'])].append(row)
    articles = {}
    for (shard, group), rows in sorted(groups.items()):
        missing = []
        for lookup in rows:
            path = output/'articles'/(lookup['id'].replace('/', '_')+'.json.gz')
            if path.exists():
                article = read_json(path)
                if any(article[k] != lookup[k] for k in META):
                    raise ValueError('Saved article metadata differs from its pinned lookup')
                validate_article(article, lookup['in_language'])
                articles[article['id']] = {'file': str(path.relative_to(output)), 'sha256': sha(path)}
            else:
                missing.append(lookup)
        if not missing:
            continue
        url = hf_hub_url(DATASET, shard, repo_type='dataset', revision=revision)
        with RangeReader(url, deadline_seconds=600) as stream:
            table = pq.ParquetFile(stream, pre_buffer=False).read_row_group(group, columns=COLUMNS, use_threads=False)
            for lookup in missing:
                index = lookup['row_index']
                if type(index) is not int or not 0 <= index < table.num_rows:
                    raise ValueError('Article row index is outside the pinned row group')
                article = table.slice(index, 1).to_pylist()[0]
                if any(article[k] != lookup[k] for k in META):
                    raise ValueError('Retrieved article differs from its page/revision/topic lookup')
                validate_article(article, lookup['in_language'])
                path = output/'articles'/(article['id'].replace('/', '_')+'.json.gz')
                write_json(path, article)
                articles[article['id']] = {'file': str(path.relative_to(output)), 'sha256': sha(path)}
        print(json.dumps({'event': 'article_group_saved', 'shard': shard, 'row_group': group,
                          'saved_articles': len(articles)}), flush=True)
    return articles


def run(config_path, output, maximum_articles=None):
    config = load_config(config_path)
    if maximum_articles is not None and maximum_articles < 1:
        raise ValueError('Maximum article pairs must be positive')
    write_json(output/'config.json', {'config': config, 'maximum_articles': maximum_articles, 'seed': 42})
    source = config['source_language']; revision = config['finewiki_revision']
    info = HfApi().repo_info(DATASET, repo_type='dataset', revision=revision, token=False, timeout=30)
    if info.sha != revision:
        raise ValueError('FineWiki resolved to a different revision')
    indices = {}
    for language in (source, 'en'):
        shards = sorted(row.rfilename for row in info.siblings
                        if row.rfilename.startswith(f'data/{language}wiki/') and row.rfilename.endswith('.parquet'))
        if not shards:
            raise ValueError('No FineWiki shards for language ' + language)
        indices[language] = []
        for shard in shards:
            path = index_shard(shard, revision, output/'metadata'/language)
            indices[language].append(path)
            print(json.dumps({'event': 'metadata_indexed', 'language': language, 'shard': shard}), flush=True)
    sources = latest_sources(iter_metadata(indices[source]), source)
    wanted = {row['wikidata_id'] for row in sources if row['wikidata_id']}
    pages = {row['page_id'] for row in iter_metadata(indices['en']) if row['wikidata_id'] in wanted}
    # Include every revision of candidate pages, including revisions whose topic changed.
    english = [row for row in iter_metadata(indices['en']) if row['page_id'] in pages]
    coverage, report = match_articles([{'article_id': row['id'], 'wikidata_id': row['wikidata_id']}
                                      for row in sources], english, source_language=source)
    write_json(output/'matching.json.gz', {'coverage': coverage, 'report': report})
    pairs = [row for row in coverage if row['status'] == 'matched']
    if maximum_articles is not None and len(pairs) > maximum_articles:
        pairs = sorted(random.Random(42).sample(pairs, maximum_articles), key=lambda row: row['article_id'])
    source_by_id = {row['id']: row for row in sources}
    english_locations = {(row['shard_path'], row['row_group'], row['row_index']): row for row in english}
    selected = []
    for pair in pairs:
        selected.append(source_by_id[pair['article_id']])
        selected.append(english_locations[(pair['en_shard_path'], pair['en_row_group'], pair['en_row_index'])])
    selected = list({row['id']: row for row in selected}.values())
    articles = retrieve_rows(selected, revision, output)
    result = {'status': 'complete', 'config': config, 'pairs': pairs, 'pair_count': len(pairs),
              'articles': [articles[key] for key in sorted(articles)],
              'metadata_files': {str(p.relative_to(output)): sha(p) for group in indices.values() for p in group},
              'selection': {'maximum_articles': maximum_articles, 'seed': 42},
              'integrity': 'Pinned commit and exact page/revision/topic/row lookup; selected article checksums. Remote whole-shard hashes were not computed.',
              'interpretation': 'Comparable article pairs; shared topic does not establish sentence equivalence.'}
    write_json(output/'manifest.json', result)
    return {'status': result['status'], 'article_pairs': len(pairs), 'retrieved_articles': len(articles)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--maximum-articles', type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.maximum_articles)))
