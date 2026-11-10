"""Keep every frozen source unit and propose traceable, non-generative repairs."""

from bisect import bisect_right
import hashlib
import re
import unicodedata

from preprocessing import finewiki_pilot as data, quality_prose as prose

METHOD = 'retained_cleanup_v6'
POLICY = {
    'method': METHOD,
    'coverage': 'one_output_record_per_original_source_id;never_drop_a_source_record',
    'repairs': ['recover_whole_sentence_from_original_paragraph',
                'merge_false_abbreviation_or_bracket_boundaries',
                'strip_leading_list_markers', 'normalize_whitespace_and_unicode_NFC',
                'capitalize_initial_letter', 'add_period_only_with_clause_evidence'],
    'grammar': 'heuristic_clause_review;not_a_Bavarian_parser;no_rigid_SVO_requirement',
    'unresolved': 'retain_original_and_cleaned_candidate_with_review_reasons;never_invent_missing_facts',
    'translation_reuse': 'context_expansion_requires_new_translation;formatting_changes_require_pair_check',
    'training': 'rebuild_matched_pairs_after_source_repair_and_alignment_review;do_not_resume_old_weights',
}
LIST = re.compile(r'^\s*(?:(?:[-*+•–—]|\d+[.)])\s+)+')
TOKENS = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
ERA = re.compile(r'\b(?:v|n)\.\s*(?:Kr|Chr)\.$', re.I)
OTHER_ABBREVIATION = re.compile(r'\b(?:Kr|Chr|Jhrdt|Nr|No|Op|Bd|ca|vgl|bzw)\.$', re.I)
CATALOGUE = re.compile(r'^(?:BWV\s*\d|Opus\s*\d|Op\.\s*\d|KV\s*\d|\d+\s+in\s+[A-Ga-g](?:[ -]|$))',re.I)
# These are positive clues for review, not an exhaustive dialect morphology.
# No word is assumed to be a finite verb merely because it ends in -en or -t.
FINITE_FORMS = frozenset('''is ist iss san sand sind seind sein senn sen bist bin samma seids
war woa wor woar woarn warn waren wär waar warad warads
hod hot hat hom ham haum habm hobm haben hob håt håbn håm hobn hosd host hamds
wead werd wird wean wern werdn werden wurde wurden worn woan wurd
ka ko kann konn können kenna kinn kinnt kriagt kriagn
muass muaß muss müssn müssen derf darf deaf deafa soll sollen soi soin
wui will woin wollen mog mag mecht möchte
gibt gibts gibds gib gheat ghert gehört ghern gehören
liegt liegn liegen liegt lebt lem lebt leben wohnt wohnen
steht steh stehts stehn stengan stehen stend
heißt heisst hoaßt hoasst haasst heißt heißen hoassn heissen
besitzt besitzen besteht bestehen findet finden enthält enthalten
gehört gehören bildet bilden zeigt zeigen nennt nennen beschreibt beschreiben
bleibt bleiben beginnt beginnen endet enden entstand entstond entstandn
zählt zählen umfasst umfosst umfasst umfasstn umfasstn
stammt stammen reicht reichen reicht reichn regnt regnts regnet
schlaft schläft schlaftn schaut schaun arbeitet arbeiten
braucht brauchen verwendet verwenden bedeutet bedeit bedeutetn
führt führen gführt führt gilt gelten passt passen macht machen mocht mochn
kommt kommen kimmt kemman kummt kema zählt zähltz
starb stirbt stiabt stirbn wurde stengan haum homd hods hotts
'''.split())


def normalize(text):
    operations = []
    value = unicodedata.normalize('NFC', text)
    if value != text:
        operations.append('unicode_NFC')
    changed = LIST.sub('', value)
    if changed != value:
        operations.append('remove_leading_list_marker')
    value = ' '.join(changed.split())
    if value != changed:
        operations.append('normalize_whitespace')
    if value and value[0].islower():
        value = value[0].upper()+value[1:]
        operations.append('capitalize_initial_letter')
    return value, operations


def verb_clues(text):
    return sorted({token for token in TOKENS.findall(text.casefold()) if token in FINITE_FORMS})


def review_reasons(text, kind):
    reasons = []
    words = TOKENS.findall(text)
    clues = verb_clues(text)
    if not text or not text[0].isalpha() or not text[0].isupper():
        reasons.append('needs_capital_letter_start')
    if not text.endswith('.'):
        reasons.append('needs_declarative_sentence_ending')
    if ERA.search(text) or prose.ABBREVIATION.search(text) or OTHER_ABBREVIATION.search(text):
        reasons.append('ends_in_abbreviation_not_confirmed_sentence_end')
    if not clues:
        reasons.append('finite_verb_not_confirmed')
    if len(words)<2:
        reasons.append('clause_structure_not_confirmed')
    if not prose.balanced(text):
        reasons.append('unbalanced_brackets')
    if CATALOGUE.search(text) and not clues:
        reasons.append('catalogue_or_work_title_requires_contextual_reconstruction')
    if re.search(r'https?://|\{\{|\}\}|\[\[|\]\]|</?[A-Za-z][^>]*>|\|',text):
        reasons.append('unresolved_markup_url_or_table')
    if kind not in {'prose','list'}:
        reasons.append('structural_'+kind+'_requires_contextual_reconstruction')
    if len(text)>1500:
        reasons.append('long_unit_requires_review')
    return reasons, clues


def candidate_spans(raw, block):
    """Repair only within one block; keep labels whole rather than slicing titles."""
    if block['kind']!='prose' or len(block['text'])>50000:
        return [(block['start'],block['end'])], []
    try:
        spans=data.align_segments(block['text'],prose.splitter('bar').segment(block['text']))
    except ValueError as error:
        return [(block['start'],block['end'])], ['split_alignment_failed: '+str(error)]
    spans, repairs = prose.repair_spans(block['text'],spans)
    merged=[]
    for start,end in spans:
        if merged:
            previous=block['text'][merged[-1][0]:merged[-1][1]].strip()
            following=block['text'][start:end].lstrip()
            partial_era = re.search(r'\b(?:v|n)\.$',previous,re.I) and re.match(r'(?:Kr|Chr)\.',following,re.I)
            abbreviation=ERA.search(previous) or partial_era or (
                OTHER_ABBREVIATION.search(previous) and following and (following[0].islower() or following[0].isdigit()))
            if abbreviation:
                repairs.append({'offset':start,'reason':'era_or_catalogue_abbreviation'})
                merged[-1]=(merged[-1][0],end)
                continue
        merged.append((start,end))
    return [(start+block['start'],end+block['start']) for start,end in merged], repairs


def prepare(row, sources, provenance):
    data.validate_article(row,'bar')
    original_ids=[s['source_id'] for s in sources]
    if len(set(original_ids))!=len(original_ids) or set(original_ids)!=set(provenance):
        raise ValueError('Frozen source/provenance IDs do not match exactly')
    raw=row['text'];blocks=data.blocks(raw)
    starts=[block['start'] for block in blocks]
    spans={};failures=[]
    for i,block in enumerate(blocks):
        spans[i],repairs=candidate_spans(raw,block)
        failures.extend({'block_start':block['start'],'error':r} for r in repairs if isinstance(r,str))
    records=[]
    for source in sources:
        sid=source['source_id'];position=provenance[sid]
        start,end=position['source_start'],position['source_end']
        if raw[start:end]!=source['bar_text']:
            raise ValueError('Source text differs from its exact original article span: '+sid)
        i=bisect_right(starts,start)-1
        if i<0 or end>blocks[i]['end']:
            raise ValueError('Original source crosses structural blocks: '+sid)
        block=blocks[i]
        overlaps=[span for span in spans[i] if span[0]<end and start<span[1]]
        covering=(min(x[0] for x in overlaps),max(x[1] for x in overlaps)) if overlaps else (start,end)
        own_text,_=normalize(source['bar_text'])
        own_reasons,_=review_reasons(own_text,block['kind'])
        boundary_problems={'needs_declarative_sentence_ending','ends_in_abbreviation_not_confirmed_sentence_end','unbalanced_brackets'}
        originally_lowercase=bool(source['bar_text'].lstrip() and source['bar_text'].lstrip()[0].islower())
        if not boundary_problems.intersection(own_reasons) and not originally_lowercase:
            # A complete old sentence must not absorb neighbouring sentences
            # merely because a list line contains several clauses.
            covering=(start,end)
        if covering[0]>start or covering[1]<end:
            raise ValueError('Recovered span does not cover its complete original source: '+sid)
        cleaned,operations=normalize(raw[covering[0]:covering[1]])
        context_expanded=covering!=(start,end)
        if context_expanded:
            operations.insert(0,'recover_original_context_span')
        reasons,clues=review_reasons(cleaned,block['kind'])
        # Do not disguise dates, titles, questions or exclamations as statements.
        if (cleaned and not re.search(r'[.!?…]$',cleaned) and clues
                and all(reason=='needs_declarative_sentence_ending' for reason in reasons)):
            cleaned+='.';operations.append('add_declarative_period')
            reasons,clues=review_reasons(cleaned,block['kind'])
        if any(f['block_start']==block['start'] for f in failures):
            reasons.append('sentence_split_alignment_failed_requires_review')
        unit_id=f"bar_{row['page_id']}:fw{data.REVISION[:8]}:r{row['version']}:{METHOD}:{covering[0]}-{covering[1]}"
        records.append({'source_id':sid,'original_source_text':source['bar_text'],
            'original_sentence_start':start,'original_sentence_end':end,
            'source_text':cleaned,'cleaned_unit_id':unit_id,
            'cleaned_sentence':cleaned if not reasons else None,
            'article_span_text':raw[covering[0]:covering[1]],
            'cleaned_sentence_start':covering[0],'cleaned_sentence_end':covering[1],
            'paragraph_start':block['start'],'paragraph_end':block['end'],'block_kind':block['kind'],
            'cleaning_operations':operations,'review_reasons':reasons,'finite_verb_clues':clues,
            'source_candidate_ready':not reasons,
            'cleaning_status':'candidate_complete_clause' if not reasons else 'needs_contextual_or_grammar_review',
            'grammar_validation':'heuristic_only;not_human_or_parser_validated',
            'translation_status':('new_translation_required_for_recovered_context' if context_expanded else
                                  'saved_pair_requires_formatting_and_alignment_check' if cleaned!=source['bar_text'] else
                                  'unchanged_source;saved_translation_alignment_unassessed'),
            'context_expanded':context_expanded,'original_record_retained':True,
            'new_facts_generated':False})
    if [r['source_id'] for r in records]!=original_ids:
        raise ValueError('Cleanup lost or reordered original source records')
    groups={}
    for record in records:
        groups.setdefault(record['cleaned_unit_id'],[]).append(record['source_id'])
    for record in records:
        record['contributing_source_ids']=groups[record['cleaned_unit_id']]
    return {'method':METHOD,'page_id':row['page_id'],'records':records,'split_failures':failures,
            'article_text_sha256':hashlib.sha256(raw.encode()).hexdigest(),
            'original_record_count':len(sources),'retained_record_count':len(records)}
