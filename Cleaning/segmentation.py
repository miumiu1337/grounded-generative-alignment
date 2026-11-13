"""Batched sentence segmentation of prose blocks, with exact source offsets."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from datatrove.data import Document
from datatrove.pipeline.base import PipelineStep
from preprocessing.finewiki_pilot import align_segments


def sentence_spans(text,values):
    """Align model outputs and trim whitespace edges without changing words."""
    spans=[]
    for start,end in align_segments(text,values):
        while start<end and text[start].isspace():start+=1
        while end>start and text[end-1].isspace():end-=1
        if start<end:spans.append((start,end))
    return spans


class SaTCPU:
    """Load pinned local ONNX/tokenizer artifacts; no GPU or implicit download."""
    def __init__(self,directory,threads=2,threshold=.25):
        self.directory=Path(directory)
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        if type(threads) is not int or not 1<=threads<=4:raise ValueError('Use 1–4 CPU threads')
        if type(threshold) not in (int,float) or not 0<threshold<1:
            raise ValueError('SaT threshold must be between zero and one')
        self.threshold=threshold
        self.threads=threads;self.model=None
        self.identity={k:self.manifest[k] for k in ('model_id','model_revision','tokenizer_id','tokenizer_revision')}
        self.identity.update(backend='sat_onnx_cpu',wtpsplit=version('wtpsplit'),onnxruntime=version('onnxruntime'),
            model_sha256=self.manifest['files']['model/model_optimized.onnx'],cpu_threads=threads,
            threshold=threshold,block_size=512,stride=64,split_on_input_newlines=False,max_length=None)

    def __getstate__(self):
        state=dict(self.__dict__);state['model']=None;return state

    def _load(self):
        for name,expected in self.manifest['files'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe segmenter artifact path')
            with (self.directory/relative).open('rb') as stream:
                actual=hashlib.file_digest(stream,'sha256').hexdigest()
            if actual!=expected:raise ValueError('Sentence-model artifact checksum differs: '+name)
        import onnxruntime as ort
        from wtpsplit import SaT
        options=ort.SessionOptions();options.intra_op_num_threads=self.threads;options.inter_op_num_threads=1
        self.model=SaT(str(self.directory/'model'),tokenizer_name_or_path=str(self.directory/'tokenizer'),
            from_pretrained_kwargs={'local_files_only':True},ort_providers=['CPUExecutionProvider'],
            ort_kwargs={'sess_options':options})

    def split(self,texts):
        if self.model is None:self._load()
        return self.model.split(texts,batch_size=2,threshold=self.threshold,block_size=512,stride=64,
            split_on_input_newlines=False,max_length=None,strip_whitespace=False)


class ProseSentenceSegmentation(PipelineStep):
    """Input documents are structurally screened prose paragraphs, not sentences.

    Structure/caption/reference removal belongs upstream; quality and duplicate
    filtering belong downstream. No capitalization, SVO or verb rules apply.
    """
    name='Prose sentence segmentation'
    type='Sentence segmentation'

    def __init__(self,backend,batch_size=16):
        super().__init__()
        if type(batch_size) is not int or not 1<=batch_size<=128:raise ValueError('Invalid paragraph batch size')
        self.backend=backend;self.batch_size=batch_size
        identity=getattr(backend,'identity',None)
        if not isinstance(identity,dict) or not identity:raise ValueError('Record the segmenter identity')
        self.identity=dict(identity)
        self.policy_id=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]

    def _batch(self,documents):
        with self.track_time():
            outputs=list(self.backend.split([doc.text for doc in documents]))
        if len(outputs)!=len(documents):raise ValueError('Segmenter returned incomplete paragraph coverage')
        completed=[]
        for doc,values in zip(documents,outputs,strict=True):
            if not isinstance(values,list) or any(not isinstance(s,str) for s in values):
                raise ValueError('Malformed sentence output for paragraph '+doc.id)
            try:spans=sentence_spans(doc.text,values)
            except ValueError as error:
                self.stat_update('alignment_failures')
                raise ValueError('Sentence alignment failed for paragraph '+doc.id+': '+str(error)) from error
            origin=doc.metadata.get('source_char_start',0)
            if type(origin) is not int or origin<0:raise ValueError('Invalid paragraph source offset: '+doc.id)
            for start,end in spans:
                absolute_start,absolute_end=origin+start,origin+end
                sentence_id=hashlib.sha256(f'{doc.id}|{self.policy_id}|{start}|{end}'.encode()).hexdigest()
                metadata={**doc.metadata,'parent_paragraph_id':doc.id,'paragraph_char_start':origin,
                    'sentence_start_in_paragraph':start,'sentence_end_in_paragraph':end,
                    'source_char_start':absolute_start,'source_char_end':absolute_end,
                    'segmentation_id':self.policy_id}
                completed.append(Document(text=doc.text[start:end],id=sentence_id,metadata=metadata))
            self.stat_update('paragraphs')
        self.stat_update('sentences',value=len(completed))
        yield from completed

    def run(self,data,rank=0,world_size=1):
        pending=[]
        for doc in data:
            if doc.metadata.get('block_kind','prose')!='prose':
                raise ValueError('Remove structural non-prose blocks before segmentation: '+doc.id)
            if not isinstance(doc.id,str) or not doc.id:raise ValueError('A paragraph ID is required')
            if not isinstance(doc.text,str):raise ValueError('Paragraph text must be a string: '+doc.id)
            if not doc.text.strip():self.stat_update('empty_paragraphs');continue
            pending.append(doc)
            if len(pending)==self.batch_size:
                yield from self._batch(pending);pending=[]
        if pending:yield from self._batch(pending)
