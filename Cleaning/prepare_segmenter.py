"""Download only the public pinned CPU segmentation artifacts."""

import argparse
import hashlib
import json
from pathlib import Path

MODEL='segment-any-text/sat-3l-sm'
MODEL_REVISION='137da054051ad9f1eac42025f758db4ac9f22535'
TOKENIZER='FacebookAI/xlm-roberta-base'
TOKENIZER_REVISION='e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'


def prepare(destination):
    from huggingface_hub import snapshot_download,HfApi
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    snapshot_download(MODEL,revision=MODEL_REVISION,local_dir=destination/'model',
        allow_patterns=['config.json','model_optimized.onnx'],token=False,max_workers=2)
    snapshot_download(TOKENIZER,revision=TOKENIZER_REVISION,local_dir=destination/'tokenizer',
        allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model'],
        token=False,max_workers=2)
    if not (destination/'model/model_optimized.onnx').is_file():raise ValueError('Sentence model download incomplete')
    files={}
    for directory in ('model','tokenizer'):
        for path in sorted((destination/directory).iterdir()):
            if not path.is_file():continue
            with path.open('rb') as stream:files[path.relative_to(destination).as_posix()]=hashlib.file_digest(stream,'sha256').hexdigest()
    info=HfApi().model_info(MODEL,revision=MODEL_REVISION,token=False)
    manifest={'model_id':MODEL,'model_revision':MODEL_REVISION,'tokenizer_id':TOKENIZER,
        'tokenizer_revision':TOKENIZER_REVISION,'files':files,
        'model_card_languages':info.card_data.get('language',[]),'execution':'CPUExecutionProvider',
        'language_note':'A language listed in the model card is not a validation of our dialect or Wikipedia domain.'}
    target=destination/'manifest.json'
    if target.exists() and json.loads(target.read_text())!=manifest:raise ValueError('Existing pinned model manifest differs')
    target.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'model':MODEL,'revision':MODEL_REVISION,'directory':str(destination),'downloaded_files':len(files)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--destination',type=Path,required=True)
    prepare(parser.parse_args().destination)
