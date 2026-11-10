"""Versioned, conservative prose preparation with exact original-text offsets."""

from collections import Counter
from functools import lru_cache
import hashlib
import re
import unicodedata

from preprocessing import finewiki_pilot as data

METHOD = 'prose_quality_v5'
POLICY = {
    'method': METHOD, 'splitter': 'pysbd==0.3.4;whole_structural_prose_block;clean_false',
    'language_rules': {'bar': 'de', 'en': 'en'},
    'boundary_repair': 'merge splits inside brackets and after nonterminal abbreviations',
    'scope': 'complete_prose_only;lists_headings_references_and_markup_context_only',
    'minimum_words': 6, 'minimum_characters': 35, 'maximum_characters': 1500,
    'maximum_block_characters': 50000,
    'sentence_start': 'first_alphabetic_character_must_be_uppercase;otherwise_context_only',
    'text_policy': 'exact_original_substrings;no_dialect_normalization_or_generated_repairs',
    'quality_status': 'structurally_screened;not_human_validated_translation_or_language_quality',
}
WORDS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
TERMINAL = re.compile(r'''[.!?…](?:[\s"'»”’)\]]|\[\d+(?:[-,–]\d+)*\])*$''')
ABBREVIATION = re.compile(r'\b(?:dt|lat|engl|franz|bair|boar|deu|Dr|Prof|St|Mr|Mrs|Ms|Nr|No|Jhdt|Jhd|Jh)\.$', re.I)
REFERENCE = re.compile(r'^(?:literatur|litaratur|liataratur|liataratua|quellen|quelln|einzelnachweise|'
    r'einzelnachweis|eizlnoochweis|anmerkungen|references|notes|notes and references|sources|'
    r'bibliography|further reading|external links|weblinks|weblenken|weblenkn)$', re.I)


@lru_cache(maxsize=2)
def splitter(language):
    import pysbd
    return pysbd.Segmenter(language={'bar': 'de', 'en': 'en'}[language], clean=False)


def normalized_heading(text):
    text = re.sub(r'^#{1,6}\s*|\s*#*$', '', text.strip()).strip()
    return ''.join(c for c in unicodedata.normalize('NFD', text.casefold()) if not unicodedata.combining(c))


def balanced(text):
    stack = []
    pairs = {')': '(', ']': '['}
    for char in text:
        if char in '([':
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack


def repair_spans(text, spans):
    """Remove false cuts without adding, deleting, or reordering characters."""
    cuts = {end for _, end in spans[:-1]}
    levels, stack, repairs = {}, [], []
    for index, char in enumerate(text, 1):
        if char in '([':
            stack.append(char)
        elif char in ')]' and stack and stack[-1] == {')':'(',']':'['}[char]:
            stack.pop()
        if index in cuts:
            levels[index] = bool(stack)
    output = []
    for start, end in spans:
        if output:
            boundary = output[-1][1]
            previous = text[output[-1][0]:boundary].strip()
            reason = ('inside_brackets' if levels.get(boundary) else
                      'nonterminal_abbreviation' if ABBREVIATION.search(previous) else None)
            if reason:
                repairs.append({'offset': boundary, 'reason': reason})
                output[-1] = (output[-1][0], end)
                continue
        output.append((start, end))
    return output, repairs


def rejection_reason(text):
    if not balanced(text):
        return 'unbalanced_brackets'
    first_letter = next((character for character in text if character.isalpha()), None)
    if first_letter is not None and first_letter.islower():
        return 'lowercase_sentence_start_for_review'
    if re.search(r'\{\{|\}\}|\[\[|\]\]|</?[A-Za-z][^>]*>|https?://|\|', text):
        return 'unresolved_markup_url_or_table'
    if not TERMINAL.search(text.strip()) or ABBREVIATION.search(text.strip()):
        return 'incomplete_sentence_terminal'
    if len(WORDS.findall(text)) < POLICY['minimum_words'] or len(text.strip()) < POLICY['minimum_characters']:
        return 'short_or_label_like_unit'
    if len(text) > POLICY['maximum_characters']:
        return 'overlong_unit_for_review'
    return None


def prepare(row, *, segment=None):
    data.validate_article(row, row['in_language'])
    raw = row['text'];language = row['in_language']
    if language not in {'bar','en'}:
        raise ValueError('Unsupported language for the frozen prose policy')
    blocks = data.blocks(raw)
    sentences, excluded, repairs, failures = [], [], [], []
    heading_stack, seen = [], set()
    for number, block in enumerate(blocks, 1):
        if block['kind'] == 'heading':
            match = re.match(r'^(#{1,6})\s+(.*?)\s*#*$', block['text'].strip())
            if match:
                level, title = len(match[1]), match[2]
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title, bool(REFERENCE.fullmatch(normalized_heading(title)))))
        reason = ('reference_section' if any(x[2] for x in heading_stack) else
                  'structural_' + block['kind'] if block['kind'] != 'prose' else None)
        if reason:
            excluded.append({'start':block['start'],'end':block['end'],'reason':reason})
            continue
        if len(block['text']) > POLICY['maximum_block_characters']:
            excluded.append({'start':block['start'],'end':block['end'],'reason':'overlong_block_for_review'})
            continue
        values = segment(block['text']) if segment else splitter(language).segment(block['text'])
        try:
            spans = data.align_segments(block['text'], values)
        except ValueError as error:
            failures.append({'paragraph_id':number,'error':str(error)})
            excluded.append({'start':block['start'],'end':block['end'],'reason':'split_alignment_failure'})
            continue
        spans, changed = repair_spans(block['text'], spans)
        repairs.extend({**item,'offset':item['offset']+block['start']} for item in changed)
        for start, end in spans:
            start += block['start'];end += block['start']
            text = raw[start:end]
            reason = rejection_reason(text)
            key = ' '.join(unicodedata.normalize('NFC',text).casefold().split())
            if reason is None and key in seen:
                reason = 'duplicate_within_article'
            seen.add(key)
            sid = f"{language}_{row['page_id']}:fw{data.REVISION[:8]}:r{row['version']}:{METHOD}:{start}-{end}"
            record = {'sentence_id':sid,'text':text,'start':start,'end':end,'paragraph_id':number,
                'paragraph_start':block['start'],'paragraph_end':block['end'],
                'heading_path':[x[1] for x in heading_stack],
                'eligible_for_matching':reason is None,'context_only_reason':reason}
            if raw[start:end] != record['text']:
                raise ValueError('Prepared sentence changed original article text')
            sentences.append(record)
    return {'method':METHOD,'policy':POLICY,'language':language,'page_id':row['page_id'],
        'revision_id':row['version'],'title':row['title'],'url':row['url'],'wikidata_id':row['wikidata_id'],
        'text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'sentences':sentences,'excluded_blocks':excluded,'boundary_repairs':repairs,
        'split_failures':failures,'counts':dict(Counter(
            'eligible' if s['eligible_for_matching'] else s['context_only_reason'] for s in sentences))}
