"""Render editable system prompts without changing source or evidence text."""

import argparse
import json
from pathlib import Path
from string import Template

from preprocessing.artifacts import load_config

DIRECTORY = Path(__file__).with_name('prompts')


def load_prompts(config):
    def read(name):
        return Template((DIRECTORY/(name+'.txt')).read_text()).substitute(
            source_language_name=config['source_language_name']).strip()
    return {'blind': read('blind'),
            'variant_a': {stage: read('variant_a_'+stage) for stage in ('selection', 'refinement')},
            'variant_b': {stage: read('variant_b_'+stage) for stage in ('selection', 'translation')},
            'variant_c': {stage: read('variant_c_'+stage) for stage in ('proposal', 'verification')}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    print(json.dumps(load_prompts(load_config(parser.parse_args().config)), ensure_ascii=False, indent=2))
