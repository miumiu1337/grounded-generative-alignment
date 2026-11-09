"""Lossless automatic preparation for the expanded FineWiki article cohort."""

from collections import Counter
import hashlib
import re

from preprocessing import finewiki_pilot as data

METHOD = 'datatrove_quality_auto_v3'
POLICY = {
    'method': METHOD, 'base_splitter': 'datatrove', 'secondary_splitter': 'pysbd',
    'language_rules': {'bar': 'de', 'en': 'en'},
    'text_policy': 'unchanged; exact source offsets; context-only text retained',
    'boundary_policy': 'PySBD subdivides DataTrove units; merge isolated abbreviation units within their block',
    'secondary_alignment_failure': 'retain baseline span; record failure and flag the unit',
    'eligibility_policy': 'exclude explicit reference sections, labels, navigation, URL-only units and clear name-list fragments',
    'warnings_are_exclusions': False, 'quality_status': 'automatic;not_human_or_gold_reviewed',
}
REFERENCE_HEADING = re.compile(
    r'^(?:literatur|litaratur|liataratur|quellen|einzelnachweise|references|notes|'
    r'notes and references|sources|bibliography|further reading|external links|weblinks|weblenken)$', re.I)
ABBREVIATION = re.compile(r'^(?:franz|engl|dt|bzw|ca|dr|prof|st|bearb|hrsg)\.$', re.I)
LABEL = re.compile(r'^(?:sg\.?\)?|pl\.?\)?|fem\.|masc\.|beispui:?)$', re.I)
CALENDAR = re.compile(r'^(?:\d{1,4}er|\d{1,2}\.\s*(?:joarhundad|joarhundert|jahrhundert|century)|joareskalenda)$', re.I)
VERB = re.compile(r'\b(?:is|isch|ist|san|sind|sei|war|woar|wor|wird|werd|wead|hot|hod|hat|'
                  r'hom|ham|haben|liegt|liegg|lebt|starb|isn|are|was|were|has|have)\b', re.I)


def context_reason(text, block, reference_section):
    stripped = re.sub(r'^(?:[-*+]\s+|\d+[.)]\s+)', '', text.strip())
    if reference_section:
        return 'reference_section'
    if LABEL.fullmatch(stripped) or (stripped.endswith(':') and len(stripped.split()) <= 8):
        return 'standalone_label_or_list_introduction'
    if CALENDAR.fullmatch(stripped):
        return 'calendar_navigation_label'
    if re.fullmatch(r'[^\W\d_]+', stripped, re.UNICODE):
        return 'isolated_word_label'
    if re.fullmatch(r'https?://\S+', stripped):
        return 'url_only'
    if text.count('|') >= 2 or ('|' in text and sum(c.isalpha() for c in text) < 25):
        return 'navigation_or_table_fragment'
    if block['kind'] == 'list' and not VERB.search(stripped):
        words = re.findall(r'[^\W\d_]+', stripped, re.UNICODE)
        if words and len(words) <= 4 and all(w[0].isupper() for w in words):
            return 'short_name_list_entry'
        if re.fullmatch(r'[^.!?\n]+\([*†]?\s*\d{3,4}[^)]*\)[^.!?\n]*', stripped):
            return 'name_and_dates_list_entry'
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) >= 3 and all(re.search(r'\(.*\d{3,4}.*\)', line) for line in lines):
        return 'unmarked_name_and_dates_list'
    return None


def prepare(row, *, baseline=None, secondary=None):
    """Return separate automatic artifacts; never alter the input article."""
    article = (baseline or data.prepare_article)(row, 'datatrove')
    if article['text'] != row['text']:
        raise ValueError('Baseline preprocessing changed original text')
    raw = row['text']
    blocks = data.blocks(raw)
    split = secondary or (lambda text: data.sentence_values(text, 'pysbd', row['in_language']))
    units, secondary_failures = [], []
    for original in article['sentences']:
        if raw[original['start']:original['end']] != original['text']:
            raise ValueError('Baseline offsets differ from original text')
        values = split(original['text'])
        try:
            spans = data.align_segments(original['text'], values)
        except ValueError as error:
            # Some wiki formatting is not preserved by PySBD. The validated
            # baseline span remains usable, with the failure made explicit.
            secondary_failures.append({'baseline_id': original['sentence_id'], 'error': str(error)})
            spans = [(0, len(original['text']))]
        for start, end in spans:
            start, end = start + original['start'], end + original['start']
            if (units and units[-1]['paragraph_id'] == original['paragraph_id']
                    and ABBREVIATION.fullmatch(raw[units[-1]['start']:units[-1]['end']].strip())):
                units[-1]['end'] = end
                units[-1]['baseline_ids'].append(original['sentence_id'])
            else:
                units.append({'start': start, 'end': end, 'paragraph_id': original['paragraph_id'],
                              'baseline_ids': [original['sentence_id']]})
    heading_stack, reference_blocks = [], set()
    for index, block in enumerate(blocks, 1):
        if block['kind'] == 'heading':
            match = re.match(r'^(#{1,6})\s+(.*?)\s*#*\s*$', block['text'].strip())
            if match:
                level, title = len(match[1]), match[2]
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, bool(REFERENCE_HEADING.fullmatch(title))))
        if any(is_reference for _, is_reference in heading_stack):
            reference_blocks.add(index)
    sentences, seen, per_block = [], Counter(), Counter()
    for unit in units:
        start, end = unit['start'], unit['end']
        text = raw[start:end]
        number = per_block[unit['paragraph_id']]
        per_block[unit['paragraph_id']] += 1
        if not any(c.isalpha() for c in text):
            article['excluded_units'].append({'start': start, 'end': end, 'text': text,
                                              'reason': 'nonalphabetic_secondary_unit'})
            continue
        block = blocks[unit['paragraph_id'] - 1]
        if not block['start'] <= start < end <= block['end']:
            raise ValueError('Automatic unit escaped its structural block')
        reason = context_reason(text, block, unit['paragraph_id'] in reference_blocks)
        flags = []
        if any(s['baseline_id'] in unit['baseline_ids'] for s in secondary_failures):
            flags.append('secondary_split_alignment_failed;baseline_boundary_retained')
        if len(text) > 1000:
            flags.append('over_1000_characters')
        if len(text) < 10:
            flags.append('under_10_characters')
        if '\n' in text:
            flags.append('multiline')
        if seen[text]:
            flags.append('duplicate_within_article')
        seen[text] += 1
        sid = f"{row['in_language']}_{row['page_id']}:fw{data.REVISION[:8]}:r{row['version']}:{METHOD}:S{len(sentences)+1:04d}"
        sentence = {**unit, 'sentence_id': sid, 'text': text, 'segment_index': number,
                    'heading_path': [], 'eligible_for_matching': reason is None,
                    'context_only_reason': reason, 'quality_flags': flags}
        sentences.append(sentence)
    candidates = []
    for index, sentence in enumerate(sentences):
        if not sentence['eligible_for_matching']:
            continue
        candidates.append({'sentence_ids': [sentence['sentence_id']], 'text': sentence['text']})
        if index + 1 < len(sentences):
            following = sentences[index + 1]
            if (following['eligible_for_matching'] and following['paragraph_id'] == sentence['paragraph_id']
                    and following['segment_index'] == sentence['segment_index'] + 1):
                candidates.append({'sentence_ids': [sentence['sentence_id'], following['sentence_id']],
                                   'text': raw[sentence['start']:following['end']]})
    baseline_spans = {(s['start'], s['end']) for s in article['sentences']}
    new_spans = {(s['start'], s['end']) for s in sentences}
    changes = {'removed_unit_spans': sorted(baseline_spans - new_spans),
               'added_unit_spans': sorted(new_spans - baseline_spans)}
    article.update(sentences=sentences, candidate_spans=candidates, sentence_splitter=METHOD,
                   boundary_review='automatic_unreviewed', preparation_policy=POLICY,
                   secondary_split_failures=secondary_failures,
                   baseline_boundaries=sorted(baseline_spans), boundary_changes=changes)
    if article['text_sha256'] != hashlib.sha256(raw.encode()).hexdigest():
        raise ValueError('Original text checksum differs')
    return article
