import copy
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from parallel_data_generation.fineweb import build_index, canonical_url, search
from preprocessing.artifacts import load_config, sha


class FineWebRetrievalTests(unittest.TestCase):
    def test_index_uses_archived_passages_and_excludes_wikipedia_mirrors(self):
        config = load_config(Path(__file__).resolve().parents[1]/'configs/asturian.json')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); shard = root/'fineweb.parquet'; index = root/'passages.sqlite'
            rows = [{'id': 'page1', 'url': 'http://example.org/city?utm_source=test',
                'text': 'Gijón stands on the northern coast of Spain. The city has an important commercial port.',
                'date': '2024-01-01', 'dump': 'fixture'},
                {'id': 'page2', 'url': 'https://en.wikipedia.org/wiki/Gijon',
                'text': 'Gijón has an important commercial port on the coast.', 'date': '2024-01-01', 'dump': 'fixture'}]
            pq.write_table(pa.Table.from_pylist(rows), shard)
            config = copy.deepcopy(config); config['fineweb']['sha256'] = sha(shard)
            topics = [{'article_id': 'enwiki/1', 'title': 'Gijón'}]
            build_index(config, topics, shard, index)
            evidence = search(index, 'enwiki/1', 'Gijón commercial port')
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]['url'], 'https://example.org/city')
            self.assertEqual(evidence[0]['provider'], 'fineweb')
            self.assertIn(evidence[0]['text'], rows[0]['text'])
            self.assertEqual(search(index, 'enwiki/2', 'Gijón port'), [])
            build_index(config, topics, shard, index)
            with self.assertRaises(ValueError):
                build_index(config, [{'article_id': 'enwiki/2', 'title': 'Different'}], shard, index)

    def test_no_wikimedia_mirrors_or_private_ip_evidence(self):
        for url in ('https://wikiwand.com/topic', 'https://en.wikipedia.org/wiki/Topic',
                    'https://127.0.0.1/private', 'https://service.local/private'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                canonical_url(url)


if __name__ == '__main__':
    unittest.main()
