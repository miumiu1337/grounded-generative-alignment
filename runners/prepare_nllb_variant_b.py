"""Validate saved evidence-first Variant B translations on the matched 20k cohort."""

from runners.prepare_nllb_variant_a import prepare


if __name__ == '__main__':
    prepare('variant_b')
