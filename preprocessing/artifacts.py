"""Checked JSON artifacts for local filesystems and mounted Drive."""

import errno
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import warnings


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_path(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Unsafe artifact path')
    path = Path(root)/name
    if not path.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Artifact path escapes its root')
    return path


def read_json(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    """Never replace a different completed artifact; remove interrupted writes."""
    path = Path(path)
    content = json.dumps(value, ensure_ascii=False, sort_keys=True).encode()
    if path.exists():
        if read_json(path) != json.loads(content):
            raise FileExistsError('Existing artifact differs: ' + str(path))
        return
    if str(path).endswith('.gz'):
        content = gzip.compress(content, mtime=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content); stream.flush(); os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        try:
            os.link(temporary, path)
        except OSError as error:
            if error.errno not in (errno.EPERM, errno.EOPNOTSUPP, errno.ENOTSUP):
                raise
            warnings.warn('Hard links unavailable; using exclusive artifact creation.', RuntimeWarning)
            with path.open('xb') as destination:
                try:
                    destination.write(content); destination.flush(); os.fsync(destination.fileno())
                except BaseException:
                    path.unlink(missing_ok=True)
                    raise
    finally:
        temporary.unlink(missing_ok=True)


def load_config(path):
    value = read_json(path)
    for name in ('source_language', 'source_language_name', 'quality_tokenizer_language'):
        if not isinstance(value.get(name), str) or not value[name].strip():
            raise ValueError('Configure ' + name)
    import re
    if not re.fullmatch(r'[a-z]{2,3}', value['source_language']) or value['source_language'] == 'en':
        raise ValueError('Use a supported non-English FineWiki language code')
    if value.get('methods') != ['blind', 'mining', 'variant_a', 'variant_b', 'variant_c']:
        raise ValueError('The protocol has exactly five methods')
    if value['mining']['encoder'] not in ('bge_m3', 'laser'):
        raise ValueError('Mining encoder must be bge_m3 or laser')
    if not re.fullmatch(r'[0-9a-f]{40}', value['finewiki_revision']):
        raise ValueError('Pin FineWiki to a commit SHA')
    if value.get('context_sentences') != 3:
        raise ValueError('This comparison requires three-sentence context windows')
    threshold = value.get('sat_threshold')
    if type(threshold) not in (int,float) or not 0 < threshold < 1:
        raise ValueError('Configure a SaT threshold between zero and one')
    return value
