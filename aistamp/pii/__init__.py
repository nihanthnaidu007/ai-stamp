"""PII detection, confidence scoring, and redaction.

Public surface:

- ``scan_text`` / ``scan_prompt_and_response`` — regex + validator scanning
  with overlap arbitration, allowlists, locale packs, and optional NER.
- ``redact_text`` / ``redact_prompt_and_response`` — replace matched spans
  with a placeholder (union semantics, never leaks overlapping spans).
- ``BUILT_IN_PATTERNS`` / ``PatternConfig`` / ``load_patterns_from_yaml`` —
  pattern definitions.
- ``LOCALE_PACKS`` / ``get_locale_patterns`` — opt-in INDIA and EU bundles.
- ``NERConfig`` — configurable spaCy label set, severity mapping, and model.
"""

from aistamp.pii.locales import LOCALE_PACKS, get_locale_patterns
from aistamp.pii.ner import NERConfig
from aistamp.pii.patterns import (
    BUILT_IN_PATTERNS,
    PatternConfig,
    load_patterns_from_yaml,
)
from aistamp.pii.redaction import redact_prompt_and_response, redact_text
from aistamp.pii.scanner import scan_prompt_and_response, scan_text

__all__ = [
    "BUILT_IN_PATTERNS",
    "LOCALE_PACKS",
    "NERConfig",
    "PatternConfig",
    "get_locale_patterns",
    "load_patterns_from_yaml",
    "redact_prompt_and_response",
    "redact_text",
    "scan_prompt_and_response",
    "scan_text",
]
