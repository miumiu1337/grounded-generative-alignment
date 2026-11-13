"""Sentence targets and bounded context from intact, section-aware prose blocks."""

import re

from preprocessing import finewiki_pilot as data
from Cleaning.structure import ArticleStructure, HEADING

POLICY = 'section_paragraph_context_v1'
POTENTIAL_TERMINATOR=re.compile(r'''[.!?]|['"](?=\s|$)''')


def could_supply_three_sentences(text):
    """Cheap upper-bound screen, not a claim that punctuation is a sentence."""
    from Cleaning.structure import CITATIONS
    return len(POTENTIAL_TERMINATOR.findall(CITATIONS.sub('',text)))>=3


def prose_paragraphs(article,reference_headings=()):
    """Return exact original spans. Removed lines never join surrounding prose."""
    language = article['in_language']
    data.validate_article(article, language)
    structure = ArticleStructure(article['text'], language=language,reference_headings=reference_headings)
    sections = []
    accepted, rejected = [], []
    for block in data.blocks(article['text']):
        if block['kind'] == 'heading':
            heading = HEADING.match(block['text'].strip())
            if heading is None:
                raise ValueError('Unrecognized heading in structured article')
            level = len(heading[1])
            while sections and sections[-1]['level'] >= level:
                sections.pop()
            sections.append({'level': level, 'title': heading[2], 'start': block['start']})
            continue
        if block['kind'] != 'prose':
            reason = structure.reason_for_span(block['start'], block['end']) or block['kind']
            rejected.append({'start':block['start'],'end':block['end'],'reason':reason})
            continue
        # FineWiki's renderer also uses single newlines for original wiki
        # paragraphs. Keep every such boundary; do not join the entire section.
        start = block['start']
        for line in block['text'].splitlines(keepends=True):
            end = start + len(line)
            reason = structure.reason_for_span(start,end)
            if reason:
                rejected.append({'start':start,'end':end,'reason':reason})
            elif line.strip():
                identity = f"{article['id']}:r{article['version']}:{start}-{end}"
                accepted.append({'paragraph_id':identity,'text':line,
                    'metadata':{'language':language,'article_id':article['id'],
                        'page_id':article['page_id'],'revision':article['version'],
                        'article_title':article['title'],'wikidata_id':article['wikidata_id'],
                        'section_path':[s['title'] for s in sections],
                        'section_id':data.digest([article['id'],article['version'],sections]),
                        'source_char_start':start,'source_char_end':end,'block_kind':'prose'}})
            start = end
    return accepted, {'removed_blocks': rejected, 'removed_lines': structure.counts}


def sentence_records(documents, original_articles, quality, tokenizer_language):
    """Keep failed units visible so context cannot bridge filtered sentences."""
    from Cleaning.quality import surface_text
    rows = []
    for doc in documents:
        metadata = doc.metadata
        raw = original_articles[metadata['article_id']]['text']
        start, end = metadata['source_char_start'], metadata['source_char_end']
        if raw[start:end] != doc.text:
            raise ValueError('Segmented sentence differs from original article offsets')
        text = surface_text(doc.text)
        reason = quality.check([text], tokenizer_language)[0]
        rows.append({'source_id': doc.id, 'text': text, 'raw_text': doc.text,
                     'quality_reason': reason, **metadata})
    return rows
