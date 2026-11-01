"""Export selected existing English evidence verbatim, without translation."""

from parallel_data_generation.variant_a import evidence_for, validate_selection


def selected_target(selection, source, page):
    validate_selection(selection, {'source': source}, page)
    if selection['match_status'] == 'none' or selection['contradiction']:
        return None
    evidence = evidence_for(selection, page)
    return evidence[0]['text'] if evidence else None
