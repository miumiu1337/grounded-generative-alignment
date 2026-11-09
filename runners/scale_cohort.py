"""FineWiki cohort extension with immutable parent processing identities."""

from pathlib import Path
from preprocessing import finewiki_pilot as data

PARENT_SHA256 = '0abb5c480924e0f6a9e1a9fe43fa635a0985dec9c02610feab30344af8924920'


def is_addendum(config):
    if config.get('purpose') != 'finewiki_cohort_addendum':
        return False
    parent = config.get('parent_config', {})
    if (data.digest(parent) != PARENT_SHA256 or config.get('preparation_completed_pairs') != 20
            or config.get('combined_article_pairs') != 16306):
        raise ValueError('Addendum does not extend the authorized parent cohort by exactly 20 pairs')
    for key in ('dataset_revision', 'conditions', 'generator', 'model', 'model_revision',
                'runtime', 'prompts', 'mining', 'reranking', 'protocol_sha256', 'stage_protocol_sha256'):
        if config[key] != parent[key]:
            raise ValueError('Addendum changed a processing setting: ' + key)
    return True


def index(root, include_addendum=True):
    root = Path(root)
    base = data.read_json(root / 'frozen/article_index.json.gz')
    if (len(base) != 16286 or len({r['bar_page_id'] for r in base}) != 16286
            or any(r['config_sha256'] != PARENT_SHA256 for r in base)):
        raise ValueError('Parent article index changed')
    if not include_addendum or not (root / 'cohort_v2/freeze_completion.json').exists():
        return base
    completion = data.read_json(root / 'cohort_v2/freeze_completion.json')
    extra = data.read_json(root / 'cohort_v2/addendum_index.json.gz')
    manifest = data.read_json(root / 'cohort_v2/manifest.json')
    config = data.read_json(root / 'cohort_v2/addendum_config.json')
    if (completion['status'] != 'complete' or len(extra) != 20
            or not is_addendum(config) or completion['config_sha256'] != data.digest(config)
            or completion['cohort_manifest_sha256'] != data.digest(manifest)
            or any(r['config_sha256'] != data.digest(config) for r in extra)
            or completion['addendum_index_sha256'] != data.digest(extra)
            or {r['bar_page_id'] for r in extra} != set(manifest['additional_page_ids'])
            or {r['bar_page_id'] for r in extra} & {r['bar_page_id'] for r in base}):
        raise ValueError('FineWiki addendum coverage or identity changed')
    return base + extra


def config_path(root, rows):
    root = Path(root)
    hashes = {r['config_sha256'] for r in rows}
    if len(hashes) != 1:
        raise ValueError('A processing batch mixes parent and addendum configurations')
    for path in (root / 'frozen/config.json', root / 'cohort_v2/addendum_config.json'):
        if path.exists() and data.digest(data.read_json(path)) in hashes:
            return path
    raise ValueError('Unknown article processing configuration')


def input_root(config):
    if is_addendum(config):
        return '/finewiki_scale_20261102_v1/cohort_addendum_v2/generation_v1'
    return '/finewiki_scale_20261102_v1/generation_v1'
