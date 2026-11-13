"""Schema-constrained calls to an already running llama.cpp server."""

from urllib.parse import urlsplit

import requests

from parallel_data_generation.variant_a import strict_json
from preprocessing.artifacts import read_json, write_json
from preprocessing.finewiki_pilot import digest


class Teacher:
    def __init__(self, endpoint, settings, output):
        url = urlsplit(endpoint)
        if (url.username or url.password or url.query or url.fragment
                or url.scheme not in ('http', 'https')
                or url.scheme == 'http' and url.hostname not in ('localhost', '127.0.0.1', '::1')):
            raise ValueError('Use localhost HTTP or a trusted HTTPS teacher endpoint without URL credentials')
        self.endpoint = endpoint.rstrip('/'); self.settings = settings; self.output = output
        self.session = requests.Session()
        response = self.session.get(self.endpoint+'/props', timeout=(5, 20))
        response.raise_for_status()
        properties = response.json()
        from pathlib import Path
        if Path(properties.get('model_path', '')).name != settings['quantized_file']:
            raise ValueError('Teacher model filename differs; verify the configured checkpoint on the server')
        write_json(output.parent/'teacher_properties'/(digest(properties)+'.json'), properties)

    def post(self, route, value):
        # No automatic generation retries: an interrupted request may have completed.
        response = self.session.post(self.endpoint+route, json=value, timeout=(5, 300))
        response.raise_for_status()
        return response.json()

    def tokens(self, request):
        prompt = self.post('/apply-template', {'messages': request['messages'],
            'chat_template_kwargs': {'enable_thinking': False}})['prompt']
        if any(message['content'] not in prompt for message in request['messages']):
            raise ValueError('Chat template changed or omitted supplied text')
        return self.post('/tokenize', {'content': prompt, 'add_special': False, 'parse_special': True})['tokens']

    def fits(self, request):
        return len(self.tokens(request))+self.settings['maximum_output_tokens'] <= self.settings['context_tokens']

    def generate(self, stage, request):
        identity = digest({'stage': stage, 'request': request, 'teacher': self.settings})
        path = self.output/(identity+'.json.gz')
        if path.exists():
            saved = read_json(path)
            if saved['request'] != request or saved['request_id'] != identity:
                raise ValueError('Cached teacher request differs')
            response = saved['response']
        else:
            tokens = self.tokens(request)
            if len(tokens)+self.settings['maximum_output_tokens'] > self.settings['context_tokens']:
                raise ValueError('Full teacher request exceeds the context limit; no truncation')
            payload = {'prompt': tokens, 'n_predict': self.settings['maximum_output_tokens'],
                'temperature': self.settings['temperature'], 'seed': self.settings['seed'],
                'cache_prompt': False, 'stream': False, 'return_tokens': True}
            payload['grammar' if request.get('grammar') else 'json_schema'] = request.get('grammar') or request['schema']
            response = self.post('/completion', payload)
            write_json(path, {'request_id': identity, 'stage': stage, 'request': request,
                              'prompt_tokens': len(tokens), 'response': response})
        if response.get('truncated') or response.get('stop_type') != 'eos':
            raise ValueError('Incomplete teacher response; raw response retained')
        return strict_json(response['content'])
