"""DataTrove text-quality checks; no POS, clause or dialect grammar classifier."""

from importlib.metadata import version
import unicodedata

from datatrove.data import Document
from datatrove.pipeline.filters import C4QualityFilter, GopherQualityFilter, GopherRepetitionFilter
from datatrove.utils.text import TextNormConfig, simplify_text

from Cleaning.structure import CITATIONS,LIST_MARKER,bibliographic_entry

COLUMNS=('source_lang','target_lang','source_text','target_text','source_article','target_article')
METHOD='datatrove_structural_quality_v8'
QUALITY=dict(min_doc_words=5,max_doc_words=120,min_avg_word_length=2,max_avg_word_length=15,
             max_symbol_word_ratio=.1,max_bullet_lines_ratio=.9,max_ellipsis_lines_ratio=.3,
             max_non_alpha_words_ratio=.7,min_stop_words=None)
C4=dict(min_num_sentences=-1,min_words_per_line=5,max_word_length=80,
        remove_citations=False,filter_no_terminal_punct=True,
        filter_javascript=False,filter_policy=False)
REPETITION=dict(top_n_grams=(),dup_n_grams=((5,.15),(6,.14),(7,.13),(8,.12),(9,.11),(10,.10)))
NORM=TextNormConfig(lowercase=True,norm_whitespace=True,remove_punctuation=True,
                    norm_unicode_diacritics=False,norm_numbers=False)


def normalized_key(text):
    return simplify_text(unicodedata.normalize('NFC',text).casefold(),config=NORM)


def surface_text(text):
    return ' '.join(unicodedata.normalize('NFC',CITATIONS.sub('',text)).split())


class SentenceQuality:
    def __init__(self,tokenizer_languages=None,minimum_alpha_token_ratio=.7):
        self.tokenizer_languages=dict(tokenizer_languages or {'bar':'de','en':'en'})
        if any(not isinstance(k,str) or not isinstance(v,str) or not k or not v
               for k,v in self.tokenizer_languages.items()):
            raise ValueError('Record explicit source and quality-tokenizer language codes')
        if type(minimum_alpha_token_ratio) not in (float,int) or not 0 < minimum_alpha_token_ratio <= 1:
            raise ValueError('Alphabetic-token ratio must be greater than zero and at most one')
        # DataTrove's parameter is misleadingly named max_non_alpha_words_ratio;
        # its implementation is a minimum fraction of tokens containing letters.
        self.quality_settings={**QUALITY,'max_non_alpha_words_ratio':minimum_alpha_token_ratio}
        self.filters={language:[GopherQualityFilter(language=language,**self.quality_settings),
                                C4QualityFilter(language=language,**C4),
                                GopherRepetitionFilter(language=language,**REPETITION)]
                      for language in sorted(set(self.tokenizer_languages.values()))}
        self.cache={}

    def config(self):
        return {'method':METHOD,'datatrove':version('datatrove'),'spacy_tokenizer':version('spacy'),
            'tokenizer_languages':self.tokenizer_languages,'gopher_quality':self.quality_settings,'c4_quality':C4,
            'gopher_repetition':REPETITION,'deduplication':{
                'casefold':True,'punctuation_and_whitespace_normalized':True,
                'numbers_and_diacritics_preserved':True,'source_unique':True,
                'target_unique_within_each_condition':True},
            'grammar_filters':False,'trained_language_models':False,
            'text_changes':'whitespace_and_NFC_normalization;remove_inline_numeric_citation_markers',
            'quality_limit':'automatic structural/textual screening;not grammatical or semantic validation'}

    def check(self,texts,language):
        if language not in self.filters:raise ValueError('Unsupported quality-filter language')
        for text in dict.fromkeys(texts):
            key=(language,text)
            if key in self.cache:continue
            reason=('empty_text' if not text else
                    'displayed_list_or_enumeration' if LIST_MARKER.match(text) else
                    'bibliographic_entry' if bibliographic_entry(text) else None)
            if reason is None:
                doc=Document(text=text,id='quality-candidate')
                for step in self.filters[language]:
                    decision=step.filter(doc)
                    passed,detail=decision if isinstance(decision,tuple) else (decision,None)
                    if not passed:reason=detail or type(step).__name__;break
                    if not doc.text:reason='datatrove_empty_after_filter';break
                    if doc.text!=text:raise ValueError('DataTrove unexpectedly changed candidate text')
            self.cache[key]=reason
        return [self.cache[(language,text)] for text in texts]
