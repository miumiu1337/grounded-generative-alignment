"""Three immediate sentences within one section and rendered paragraph."""

from collections import Counter
import hashlib

def context_windows(rows, *, require_full_window=True):
    """Only immediate neighbours in one original paragraph can supply context."""
    groups = {}
    for row in rows:
        groups.setdefault(row['parent_paragraph_id'], []).append(row)
    output, counts = [], Counter()
    for group in groups.values():
        group.sort(key=lambda r: r['source_char_start'])
        if len({r['source_id'] for r in group}) != len(group):
            raise ValueError('Repeated sentence IDs in a paragraph')
        if len({(r['article_id'], r['section_id']) for r in group}) != 1:
            raise ValueError('A paragraph crosses an article or section boundary')
        if any(a['source_char_end'] > b['source_char_start'] for a, b in zip(group, group[1:])):
            raise ValueError('Overlapping sentence offsets')
        for i, focus in enumerate(group):
            members = group[max(0, i-1):i+2]
            reason = focus['quality_reason']
            if reason:
                counts['focus_failed_quality'] += 1
                continue
            if require_full_window and len(members) != 3:
                counts['not_three_sentence_context'] += 1
                continue
            if any(r['quality_reason'] for r in members):
                counts['neighbour_failed_quality'] += 1
                continue
            output.append({'source_id': focus['source_id'], 'focus_text': focus['text'],
                'context_text': ' '.join(r['text'] for r in members),
                'context_sentence_ids': [r['source_id'] for r in members],
                'focus_index': next(j for j, r in enumerate(members) if r['source_id'] == focus['source_id']),
                'paragraph_id': focus['parent_paragraph_id'], 'section_id': focus['section_id'],
                'section_path': focus['section_path'], 'article_id': focus['article_id'],
                'article_title': focus['article_title'], 'page_id': focus['page_id'],
                'revision': focus['revision'], 'wikidata_id': focus['wikidata_id'],
                'source_char_start': focus['source_char_start'], 'source_char_end': focus['source_char_end'],
                'context_char_start': members[0]['source_char_start'],
                'context_char_end': members[-1]['source_char_end']})
    return output, dict(counts)


def text_identity(text):
    from Cleaning.quality import normalized_key
    return hashlib.sha256(normalized_key(text).encode()).hexdigest()
