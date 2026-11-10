"""Prose extraction using visible article structure, without grammar judgments."""

from bisect import bisect_right
from collections import Counter
import re
import unicodedata

from preprocessing import finewiki_pilot as data

METHOD='article_structure_v8'
REFERENCE_NAMES=frozenset('''literatur|literatua|litaradua|litaratua|literadua|litratua|litaratur|litradur|litradua|
literdua|literatu|literatuar|literatuu|literata|literratur|litteratua|literaradtua|
primealiteratua|primealiteradtua|primarliteratur|primarliteratua|sekundealiteratua|sekundealitaratua|
sekundealiteratur|sekundarliteratur|sekundarliteratua|foaschungsliteratua|fochliteratua|
einzelnachweise|einzelnachweis|einzelnachwas|einzlnochweise|oanzlnochweis|oanzlnochweise|
oanzlnachweise|oanzlnachweisa|oanzlnachweislistn|oanzlne nochweis|aanzlnochweis|anzlnochweis|aanzlbeleg|
nochweise|nachweise|beleg|belege|anmerkungen|onmeakunga|onmerkunga|fuassnotn|fuasnotn|fuassnodn|fuassnoten|fussnoten|
references|notes|sources|bibliography|further reading|external links|notes and references|see also|
weblinks|weblink|webseitn|links|link|im netz|links im netz|links in netz|links im intanet|links in netz|
intanet-link|netverweis|netzverweis|externe links|exteane links|weidalesn|zum lesn|schau aa|siehe auch
'''.replace('\n','').split('|'))
AMBIGUOUS_SOURCES=frozenset('quelle quellen quelln quej quejn quooin quoin quoon quon quolln primarquelln'.split())
HEADING=re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$')
LIST_MARKER=re.compile(r'^\s*(?:[-*+•◦‣▪∙–—]\s+|(?:\(?\d{1,3}\)|\d{1,3}[.)])\s+|(?:\(?[A-Za-z]\)|[A-Za-z][.)])\s+|[ivxlcdmIVXLCDM]{1,8}[.)]\s+|[;:]\s+)')
INLINE_ENUMERATION=re.compile(r'(?:^|[;:]\s*)\(?\d{1,2}[.)]\s+')
CITATIONS=re.compile(r'\[(?:\d+(?:[-,–]\d+)*|citation needed|edit)\]',re.I)
IDENTIFIER=re.compile(r'\b(?:ISBN(?:-1[03])?|ISSN)\s*[: ]?\s*[0-9Xx][0-9Xx -]{5,}|\bdoi\s*:|https?://(?:dx\.)?doi\.org/',re.I)
BIBLIOGRAPHIC_META=re.compile(r'\b(?:Verlag|Press|Hrsg\.|Hg\.|Aufl\.|Auflage|Jahrgang|Band\s+\d|Bd\.\s*\d|[SPp]\.\s*\d|pp\.\s*\d)',re.I)
YEAR=re.compile(r'\b(?:1[5-9]|20)\d{2}\b')


def heading_key(title):
    text=unicodedata.normalize('NFKC',title).casefold().replace('ß','ss')
    text=''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))
    text=re.sub(r'[*_`]+','',text)
    return ' '.join(text.split()).strip(' :.;')


def bibliographic_entry(text):
    """Require publication metadata, rather than guessing from a person's name."""
    if IDENTIFIER.search(text) and BIBLIOGRAPHIC_META.search(text):return True
    if IDENTIFIER.search(text) and YEAR.search(text) and re.match(r'[^.!?\n]{2,100}:\s+\S',text):return True
    pages=re.search(r'\b(?:[SPp]\.\s*\d|pp\.\s*\d|Seiten\s+\d)',text,re.I)
    publication=re.search(r'\b(?:Hrsg\.|Hg\.|Verlag|University Press|Aufl\.|Auflage)',text,re.I)
    return bool(YEAR.search(text) and pages and publication)


def reference_heading(title,body=''):
    key=heading_key(title)
    if key in REFERENCE_NAMES:return True
    # Parenthetical qualifiers and common "literature about ..." labels are
    # bibliographic; headings such as "Deitsche Literatua" remain ordinary body.
    base=re.sub(r'\s*\([^)]*\)\s*$','',key)
    if base in REFERENCE_NAMES:return True
    if re.match(r'^(?:literatua|literatur|litaratur|litratua)\s+(?:iwa|iban|uber|zum thema|zum|zum lesn)\b',key):return True
    if re.match(r'^(?:weidane|weitere|ejtane|verwendete)\s+(?:literatur|literatua|litaradua|litratua)$',key):return True
    parts=re.split(r'\s+(?:und|and|&)\s+|\s*[,/]\s*',key)
    if len(parts)>1 and all(part in REFERENCE_NAMES or part in AMBIGUOUS_SOURCES for part in parts):return True
    if key in AMBIGUOUS_SOURCES:
        return bibliographic_entry(body) or bool(re.search(r'https?://|www\.',body,re.I))
    return False


class ArticleStructure:
    def __init__(self,text):
        self.text=text
        lines=[];position=0
        for value in text.splitlines(keepends=True):
            lines.append({'start':position,'end':position+len(value),'text':value,'heading':HEADING.match(value.strip('\r\n'))})
            position+=len(value)
        fence=None
        for line in lines:
            marker=re.match(r'^(`{3,}|~{3,})',line['text'].strip())
            if fence is not None:
                line['heading']=None
                if marker and marker[1][0]==fence:fence=None
            elif marker:fence=marker[1][0];line['heading']=None
        headings=[(i,len(line['heading'][1]),line['heading'][2]) for i,line in enumerate(lines) if line['heading']]
        reference_flags={};section_ends={}
        for j,(i,level,title) in enumerate(headings):
            end=next((following for following,next_level,_ in headings[j+1:] if next_level<=level),len(lines))
            section_ends[i]=end
            body=''.join(line['text'] for line in lines[i+1:end])
            reference_flags[i]=level>1 and reference_heading(title,body)
        reasons=[None]*len(lines);stack=[];fence=None;list_indent=None
        for i,line in enumerate(lines):
            value=line['text'];stripped=value.strip();heading=line['heading']
            if fence is not None:
                reasons[i]='code_block'
                if re.match(re.escape(fence)+r'{3,}',stripped):fence=None
                continue
            marker=re.match(r'^(`{3,}|~{3,})',stripped)
            if marker:fence=marker[1][0];reasons[i]='code_block';continue
            if heading:
                level=len(heading[1])
                while stack and stack[-1][0]>=level:stack.pop()
                stack.append((level,reference_flags[i]));list_indent=None
            if any(reference for _,reference in stack):reasons[i]='reference_section';continue
            if heading:continue
            if not stripped:continue
            indent=len(value)-len(value.lstrip(' \t'))
            if LIST_MARKER.match(value):
                reasons[i]='displayed_list_or_enumeration';list_indent=indent;continue
            if list_indent is not None and indent>list_indent:
                reasons[i]='list_continuation';continue
            list_indent=None
            if stripped.startswith('|') or re.fullmatch(r'\s*:?-{3,}:?(?:\s*\|\s*:?-{3,}:?)*\s*',stripped):
                reasons[i]='table';continue
            if len(INLINE_ENUMERATION.findall(stripped))>=2:reasons[i]='displayed_enumeration';continue
            if not any(c.isalnum() for c in stripped):reasons[i]='markup';continue
        # Publication entries can survive extraction without a recognized heading.
        for block in data.blocks(text):
            if block['kind']=='prose' and bibliographic_entry(block['text']):
                for i,line in enumerate(lines):
                    if line['start']<block['end'] and block['start']<line['end'] and reasons[i] is None:
                        reasons[i]='bibliographic_entry'
        self.removed=[(line['start'],line['end'],reason) for line,reason in zip(lines,reasons) if reason is not None]
        self.ends=[end for _,end,_ in self.removed]
        self.counts=dict(Counter(reason for _,_,reason in self.removed))
        kept_prose=[i for i,line in enumerate(lines) if reasons[i] is None and line['text'].strip() and not line['heading']]
        output=[];gap=False
        for i,line in enumerate(lines):
            if reasons[i] is not None:gap=True;continue
            if line['heading'] and not any(i<j<section_ends[i] for j in kept_prose):continue
            if gap and output and not output[-1].endswith('\n\n'):output.append('\n')
            output.append(CITATIONS.sub('',line['text']));gap=False
        self.cleaned_text=''.join(output).strip()

    def reason_for_span(self,start,end):
        if not 0<=start<end<=len(self.text):raise ValueError('Article span is outside its original text')
        index=bisect_right(self.ends,start)
        if index<len(self.removed):
            left,right,reason=self.removed[index]
            if left<end and start<right:return reason
        return None
