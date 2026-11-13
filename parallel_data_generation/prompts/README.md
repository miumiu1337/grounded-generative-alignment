# Prompts

These templates come from the current experiments, with the language name replaced by `${source_language_name}`. Edit the text files, then render the complete prompts with:

```bash
python -m parallel_data_generation.prompts --config configs/asturian.json
```

| Method | System prompts |
| --- | --- |
| Blind | [Translation](blind.txt) |
| Mining | No LLM prompt; select `bge_m3` or `laser` in the configuration |
| A | [Evidence selection](variant_a_selection.txt), [refinement](variant_a_refinement.txt) |
| B | [Evidence selection](variant_b_selection.txt), [translation](variant_b_translation.txt) |
| C | [Correction proposal](variant_c_proposal.txt), [verification](variant_c_verification.txt) |

The user message contains `focus_sentence`, with `context` only for the context condition. A/B use `focus_source` and numbered English candidates; optional source and English contexts are omitted from sentence-only requests. Article titles identify the shared topic. B never receives Blind. A/C reuse the Blind output for the same focus and context condition.

Only the focus is translated. The three-sentence window is context, and must not become a three-sentence training target. The separate A/B evidence-selection protocol allows one English sentence or two immediately adjacent English sentences.

Prompt contents and code hashes are recorded in each generated protocol. Changing a prompt creates a different run identity. Earlier Bavarian 20k prompts are in `experiments/generation/frozen/config.json` for interpreting the original experiments.
