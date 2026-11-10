"""Sentence-level DataTrove quality checks and conservative clause screening."""

from functools import lru_cache
from importlib.metadata import version
import re
import unicodedata

from datatrove.data import Document
from datatrove.pipeline.filters import C4QualityFilter, GopherQualityFilter, GopherRepetitionFilter
from datatrove.utils.text import TextNormConfig, simplify_text
import spacy

from preprocessing.quality_prose import balanced

COLUMNS=('source_lang','target_lang','source_text','target_text','source_article','target_article')
METHOD='datatrove_sentence_quality_v7'
QUALITY=dict(min_doc_words=5,max_doc_words=120,min_avg_word_length=2,max_avg_word_length=15,
             max_symbol_word_ratio=.1,max_bullet_lines_ratio=.9,max_ellipsis_lines_ratio=.3,
             max_non_alpha_words_ratio=.7,min_stop_words=None)
C4=dict(min_num_sentences=-1,min_words_per_line=5,max_word_length=80,
        remove_citations=False,filter_no_terminal_punct=True,
        filter_javascript=False,filter_policy=False)
REPETITION=dict(top_n_grams=(),dup_n_grams=((5,.15),(6,.14),(7,.13),(8,.12),(9,.11),(10,.10)))
NORM=TextNormConfig(lowercase=True,norm_whitespace=True,remove_punctuation=True,
                    norm_unicode_diacritics=False,norm_numbers=False)
ERA_OR_ABBREVIATION=re.compile(r'\b(?:(?:v|n)\.\s*(?:Kr|Chr)|dt|lat|engl|Dr|Prof|St|Nr|No|Jhdt|Jhd|Jh|Bd|Op)\.$',re.I)
NONFINITE_START=re.compile(r'^(?:BWV\s*\d|Opus\s*\d|Op\.\s*\d|KV\s*\d|\d+\s+in\s+[A-Ga-g])',re.I)
# Positive dialect auxiliary clues supplement the German model. Infinitives,
# possessive "sein", and isolated participles are deliberately not clues here.
BAVARIAN_FINITE=frozenset('''is isch san sand samma senn sen bist bin seids
hod hot hom ham haum hobm håt håm hobn hosd host hamds hods
woa wor woar warn warads warad wead werd wean wern ka ko konn
muass muaß derf deaf soi wui mog mecht gibts gibds gheat ghert
kimmt kummt kriagt'''.split())
SUBJECT_WORDS=frozenset('de da des d er sie es i du mia mir ma se si wea wer'.split())


def normalized_key(text):
    return simplify_text(unicodedata.normalize('NFC',text).casefold(),config=NORM)


def surface_text(text):
    text=re.sub(r'^\s*(?:(?:[-*+•–—]|\d+[.)])\s+)+','',text)
    text=' '.join(unicodedata.normalize('NFC',text).split())
    if text and text[0].islower():text=text[0].upper()+text[1:]
    return text


def surface_reason(text):
    if not text or not text[0].isalpha() or not text[0].isupper():return 'capital_letter_start_required'
    if not text.endswith('.') or ERA_OR_ABBREVIATION.search(text):return 'complete_sentence_ending_required'
    if not balanced(text):return 'unbalanced_brackets'
    if len(text)>1500:return 'overlong_sentence'
    if re.search(r'https?://|\{\{|\}\}|\[\[|\]\]|</?[A-Za-z][^>]*>|\|',text):return 'markup_url_or_table'
    if NONFINITE_START.search(text):return 'catalogue_or_work_title'
    return None


def clause_reason(doc,language):
    finite=[t for t in doc if t.pos_ in {'VERB','AUX'} and
            ('Fin' in t.morph.get('VerbForm') or t.tag_ in {'VVFIN','VAFIN','VMFIN','VBD','VBP','VBZ','MD'})]
    subjects=[t for t in doc if t.dep_ in {'sb','ep','nsubj','nsubjpass','csubj','expl'}]
    if language=='de':
        dialect=[t for t in doc if t.text.casefold() in BAVARIAN_FINITE]
        has_subject=bool(subjects) or bool(dialect and any(
            t.text.casefold() in SUBJECT_WORDS or t.pos_ in {'NOUN','PROPN','PRON'} for t in doc if t not in dialect))
        if not finite and not dialect:return 'finite_predicate_not_confirmed'
    else:
        has_subject=bool(subjects)
        if not finite:return 'finite_predicate_not_confirmed'
    if not has_subject:return 'subject_not_confirmed'
    return None


class SentenceQuality:
    def __init__(self):
        self.filters={language:[GopherQualityFilter(language=language,**QUALITY),
                                C4QualityFilter(language=language,**C4),
                                GopherRepetitionFilter(language=language,**REPETITION)]
                      for language in ('de','en')}
        self.models={language:spacy.load(model,exclude=['ner','lemmatizer']) for language,model in
                     [('de','de_core_news_sm'),('en','en_core_web_sm')]}
        self.cache={}

    def config(self):
        return {'method':METHOD,'datatrove':version('datatrove'),'spacy':version('spacy'),
            'de_core_news_sm':version('de-core-news-sm'),'en_core_web_sm':version('en-core-web-sm'),
            'tokenizer_languages':{'bar':'de','en':'en'},'gopher_quality':QUALITY,'c4_quality':C4,
            'gopher_repetition':REPETITION,'deduplication':{
                'casefold':True,'punctuation_and_whitespace_normalized':True,
                'numbers_and_diacritics_preserved':True,'source_unique':True,
                'target_unique_within_each_condition':True},
            'clause_screen':'finite_predicate_and_subject;German_model_with_explicit_Bavarian_auxiliary_clues',
            'quality_limit':'automatic structural/textual screening;not human-gold semantic alignment'}

    def check(self,texts,language):
        if language not in self.filters:raise ValueError('Unsupported quality-filter language')
        pending=[]
        for text in dict.fromkeys(texts):
            key=(language,text)
            if key in self.cache:continue
            reason=surface_reason(text)
            if reason is None:
                doc=Document(text=text,id='quality-candidate')
                for step in self.filters[language]:
                    decision=step.filter(doc)
                    passed,detail=decision if isinstance(decision,tuple) else (decision,None)
                    if not passed:
                        reason=detail or type(step).__name__;break
                    if not doc.text:
                        reason='datatrove_empty_after_filter';break
                    if doc.text!=text:raise ValueError('DataTrove unexpectedly changed candidate text')
            if reason is not None:self.cache[key]=reason
            else:pending.append(text)
        for doc,text in zip(self.models[language].pipe(pending,batch_size=64,n_process=1),pending,strict=True):
            self.cache[(language,text)]=clause_reason(doc,language)
        return [self.cache[(language,text)] for text in texts]
