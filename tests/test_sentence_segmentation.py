import unittest
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from unittest.mock import patch

from datatrove.data import Document
from Cleaning.segmentation import ProseSentenceSegmentation,SaTCPU,sentence_spans


class Backend:
    identity={'model':'test_only','revision':'1'}
    def __init__(self,outputs):self.outputs=outputs;self.calls=[]
    def split(self,texts):self.calls.append(texts);return self.outputs[:len(texts)]


class SegmentationTests(unittest.TestCase):
    def test_spacing_and_whitespace_only_outputs_are_not_boundary_differences(self):
        text='One sentence.\n\nAnother sentence.  '
        self.assertEqual(sentence_spans(text,['One sentence.','\n\n','Another sentence.  ']),
                         sentence_spans(text,['One sentence.\n\n','Another sentence.']))

    def test_cpu_model_is_lazy_and_can_be_sent_to_pipeline_workers(self):
        with TemporaryDirectory() as tmp:
            manifest={'model_id':'test','model_revision':'1','tokenizer_id':'test','tokenizer_revision':'1',
                'files':{'model/model_optimized.onnx':'0'*64}}
            (Path(tmp)/'manifest.json').write_text(json.dumps(manifest))
            with patch('Cleaning.segmentation.version',return_value='test'):
                backend=SaTCPU(tmp)
            self.assertIsNone(backend.model)
            backend.model=lambda:'not pickleable'
            restored=pickle.loads(pickle.dumps(backend))
            self.assertIsNone(restored.model)
            self.assertEqual(restored.identity,backend.identity)

    def test_exact_offsets_unicode_and_article_metadata(self):
        text='  Däiz liegt an da Isar. des is a zwoata Satz.  '
        backend=Backend([['Däiz liegt an da Isar. ','des is a zwoata Satz.']])
        step=ProseSentenceSegmentation(backend)
        doc=Document(text=text,id='paragraph',metadata={'language':'bar','article_title':'Däiz','source_char_start':80})
        rows=list(step([doc]))
        self.assertEqual(len(rows),2)
        for row in rows:
            m=row.metadata;start=m['source_char_start']-80;end=m['source_char_end']-80
            self.assertEqual(row.text,text[start:end]);self.assertEqual(m['article_title'],'Däiz')
        self.assertTrue(rows[1].text.startswith('des'))
        self.assertEqual(len({r.id for r in rows}),2)
        self.assertEqual([r.id for r in step([doc])],[r.id for r in rows])

    def test_text_loss_and_rewriting_are_errors(self):
        for parts in [['First sentence.'],['Changed text.','Second sentence.']]:
            step=ProseSentenceSegmentation(Backend([parts]))
            with self.assertRaisesRegex(ValueError,'paragraph p'):
                list(step([Document(text='First sentence. Second sentence.',id='p')]))

    def test_incomplete_batch_and_structural_input_are_errors(self):
        with self.assertRaisesRegex(ValueError,'coverage'):
            list(ProseSentenceSegmentation(Backend([]))([Document(text='Text.',id='p')]))
        with self.assertRaisesRegex(ValueError,'non-prose'):
            list(ProseSentenceSegmentation(Backend([['Item.']]))([Document(text='Item.',id='p',metadata={'block_kind':'list'})]))

    def test_bounded_batches_do_not_require_language_specific_grammar(self):
        backend=Backend([['texte']]);step=ProseSentenceSegmentation(backend,batch_size=1)
        docs=[Document(text='texte',id=str(i),metadata={'language':lang}) for i,lang in enumerate(['bar','en','fy'])]
        rows=list(step(docs));self.assertEqual(len(rows),3);self.assertEqual(len(backend.calls),3)
        self.assertEqual([r.metadata['language'] for r in rows],['bar','en','fy'])


if __name__=='__main__':unittest.main()
