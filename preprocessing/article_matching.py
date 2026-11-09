"""Resolve same-topic article links from pinned snapshot metadata, never sentence equivalence."""

from collections import Counter, defaultdict
import re


def match_articles(sources, english_rows):
    """English rows must include all revisions of every page matching any wanted topic.

    The metadata scan performs this two-pass selection so an obsolete Wikidata ID
    cannot survive simply because a newer revision has a different topic ID.
    """
    source_ids, source_topics, by_page = set(), defaultdict(list), defaultdict(list)
    for source in sources:
        aid, qid = source['article_id'], source['wikidata_id']
        if aid in source_ids or not re.fullmatch(r'barwiki/[1-9][0-9]*', aid):
            raise ValueError('Invalid or repeated Bavarian article ID')
        if qid is not None and (not isinstance(qid, str) or not re.fullmatch(r'Q[1-9][0-9]*', qid)):
            raise ValueError('Invalid Bavarian Wikidata ID')
        source_ids.add(aid)
        if qid is not None:
            source_topics[qid].append(aid)
    for row in english_rows:
        if (type(row['page_id']) is not int or row['page_id'] <= 0 or type(row['version']) is not int or row['version'] <= 0
                or row['wikidata_id'] is not None and not re.fullmatch(r'Q[1-9][0-9]*', row['wikidata_id'])):
            raise ValueError('Invalid English article metadata')
        by_page[row['page_id']].append(row)
    topics, conflicts = defaultdict(list), []
    for page_id, rows in by_page.items():
        latest = max(row['version'] for row in rows)
        current = [r for r in rows if r['version'] == latest]
        qids = {r['wikidata_id'] for r in current}
        if len(qids) != 1:
            conflicts.append({'page_id': page_id, 'revision': latest, 'wikidata_ids': sorted(qids, key=lambda q: q or '')})
            continue
        chosen = min(current, key=lambda r: (r.get('shard_path', ''), r.get('row_group', 0), r.get('row_index', 0)))
        if chosen['wikidata_id'] is not None:
            topics[chosen['wikidata_id']].append(chosen)
    conflicted_topics = {qid for c in conflicts for qid in c['wikidata_ids'] if qid is not None}
    coverage = []
    for source in sources:
        qid = source['wikidata_id']
        candidates = topics.get(qid, [])
        status = ('missing_source_wikidata_id' if qid is None else
                  'ambiguous_source_topic' if len(source_topics[qid]) > 1 else
                  'conflicting_english_metadata' if qid in conflicted_topics else
                  'no_english_match' if not candidates else
                  'ambiguous_english_topic' if len(candidates) > 1 else 'matched')
        result = {**source, 'status': status, 'english_candidate_ids': sorted(f'enwiki/{r["page_id"]}' for r in candidates),
                  'en_article_id': None, 'en_page_id': None, 'en_revision_id': None,
                  'en_shard_path': None, 'en_row_group': None, 'en_row_index': None,
                  'en_revision_url': None}
        if status == 'matched':
            row = candidates[0]
            result.update(en_article_id=f'enwiki/{row["page_id"]}', en_page_id=row['page_id'], en_revision_id=row['version'],
                          en_shard_path=row.get('shard_path'), en_row_group=row.get('row_group'), en_row_index=row.get('row_index'),
                          en_revision_url=f'https://en.wikipedia.org/w/index.php?oldid={row["version"]}')
        coverage.append(result)
    return coverage, {'source_articles': len(sources), 'status_counts': dict(Counter(r['status'] for r in coverage)),
                      'conflicting_latest_english_metadata': conflicts,
                      'interpretation': 'Shared-topic article links;not_sentence_alignment_or_gold_translation_pairs'}
