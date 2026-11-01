"""Blind translation prompt and pinned model for the FineWiki pilot."""

MODEL = "google/gemma-4-26B-A4B-it"
REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"


def messages_for(source, prompt):
    return [{"role": "system", "content": prompt},
            {"role": "user", "content": f"Bavarian text:\n{source['bar_text']}"}]
