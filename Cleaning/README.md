# Cleaning

The current order is:

1. Extract prose paragraphs using the rendered article structure. Exclude reference sections, displayed lists/enumerations, tables, code and explicit image/caption markup. Keep removal offsets and reasons.
2. Segment each retained paragraph with the pinned SaT 3l-sm ONNX model. Align every output to the original text; missing or changed characters are errors. Each rendered newline is a hard paragraph boundary.
3. Normalize whitespace and Unicode NFC, and remove numeric inline citation markers. Keep the raw sentence and original character offsets separately.
4. Apply DataTrove Gopher quality, C4 and repetition filters using the explicitly configured tokenizer language. The Asturian example uses Spanish tokenization; this is a proxy, not Asturian grammatical validation.
5. Deduplicate accepted focuses within each language across the processed articles. Comparison keys casefold and normalize punctuation/whitespace, preserving numbers and diacritics. Keep duplicates in the audit with their original source IDs and the retained ID.
6. Build windows from immediate neighbors. Require exactly three accepted sentences in the same paragraph and section; the middle sentence is the focus. Never skip a rejected sentence to assemble a window.

`structure.py`, `segmentation.py`, `quality.py` and `run.py` implement these steps; `preprocessing/context.py` constructs the windows. Rejected sentences remain in each article's audit and are not training pairs.

Unmarked captions, missing article structure and imperfect segmentation can survive automatic checks. These filters do not establish semantic completeness or grammatical correctness. Review a sample when changing language or settings. No LLM prompt is used for cleaning: SaT predicts boundaries, and DataTrove applies text filters.

`sat_threshold` is configurable and remains 0.25 to describe the current Asturian setup. In the small CPU check, 0.25 still split the reported Zhuang sentence after “den 4.”; 0.5 preserved that boundary and kept the three-sentence Asturian fixture split correctly. This is a diagnostic on two examples, not evidence that 0.5 is generally better. Review boundaries before changing the production setting.

References: [SaT implementation](https://github.com/segment-any-text/wtpsplit), [DataTrove](https://github.com/huggingface/datatrove), [FineWiki](https://huggingface.co/datasets/HuggingFaceFW/finewiki).
