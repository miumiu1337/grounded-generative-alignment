import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from Cleaning.run import clean_article
from Cleaning.quality import SentenceQuality
from Cleaning.paragraphs import prose_paragraphs
from Cleaning.segmentation import ProseSentenceSegmentation
from preprocessing.artifacts import read_json, safe_path, write_json
from preprocessing.prepare_articles import latest_sources


def article(text, page=1):
    return {'id': f'astwiki/{page}', 'page_id': page, 'title': 'Prueba', 'wikidata_id': 'Q1',
            'version': 1, 'text': text, 'url': 'https://ast.wikipedia.org/wiki/Prueba', 'in_language': 'ast'}


class CleaningPipelineTests(unittest.TestCase):
    def test_explicit_captions_custom_reference_headings_and_bullets_are_removed(self):
        raw = article('# Prueba\nA normal sentence stays in the article.\n'
            '![Museum caption](museum.jpg)\n- A list item stays out of the prose.\n'
            '## Custom sources\nA bibliographic sentence stays out of the prose.\n'
            '## History\nAnother normal sentence stays in the article.\n')
        paragraphs, audit = prose_paragraphs(raw, reference_headings=['Custom sources'])
        self.assertEqual(len(paragraphs), 2)
        self.assertIn('explicit_image_or_caption_markup', audit['removed_lines'])
        self.assertTrue(all('caption' not in p['text'] and 'bibliographic' not in p['text'] for p in paragraphs))

    def test_deduplication_keeps_rejections_visible_and_never_bridges_them(self):
        texts = ['The first sentence contains useful article information.',
                 'The second sentence contains different article information.',
                 'The third sentence describes another historical event.']
        class Backend:
            identity = {'backend': 'test_fixture'}
            def split(self, values):
                return [texts for _ in values]
        segment = ProseSentenceSegmentation(Backend())
        quality = SentenceQuality({'ast': 'es'}, minimum_alpha_token_ratio=.5)
        seen = {}
        first = clean_article(article(' '.join(texts)), segment, quality, 'es', seen)
        second = clean_article(article(' '.join(texts), page=2), segment, quality, 'es', seen)
        self.assertEqual(len(first['windows']), 1)
        self.assertEqual(first['windows'][0]['focus_text'], texts[1])
        self.assertEqual(second['windows'], [])
        self.assertTrue(all(row['quality_reason'] == 'duplicate_sentence' for row in second['sentences']))

    def test_drive_fallback_is_immutable_and_json_tuple_roundtrip_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'record.json.gz'
            with patch('preprocessing.artifacts.os.link', side_effect=OSError(errno.EPERM, 'not supported')):
                with self.assertWarns(RuntimeWarning):
                    write_json(path, {'bounds': (1, 2)})
            write_json(path, {'bounds': (1, 2)})
            self.assertEqual(read_json(path), {'bounds': [1, 2]})
            with self.assertRaises(FileExistsError):
                write_json(path, {'bounds': (1, 3)})

    def test_source_matching_discards_obsolete_topic_and_reports_conflicts(self):
        row = {'id': 'astwiki/1', 'page_id': 1, 'wikidata_id': 'Q1', 'version': 1, 'in_language': 'ast'}
        current = {**row, 'version': 2, 'wikidata_id': 'Q2'}
        self.assertEqual(latest_sources([row, current], 'ast'), [current])
        with self.assertRaises(ValueError):
            latest_sources([current, {**current, 'wikidata_id': 'Q3'}], 'ast')

    def test_manifest_paths_cannot_escape_the_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in ('../secret', '/tmp/secret'):
                with self.assertRaises(ValueError):
                    safe_path(directory, name)
            root = Path(directory)/'root'; root.mkdir()
            (root/'link').symlink_to(root.parent, target_is_directory=True)
            with self.assertRaises(ValueError):
                safe_path(root, 'link/secret')


if __name__ == '__main__':
    unittest.main()
