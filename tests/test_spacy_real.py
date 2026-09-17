"""Real-model spaCy NER test.

Runs only when both the spaCy library and the ``en_core_web_sm`` model are
installed (``pip install ai-stamp[nlp] && python -m spacy download
en_core_web_sm``); otherwise the module skips, matching how the library
itself degrades when NER is unavailable.
"""

from __future__ import annotations

import pytest

pytest.importorskip(
    "spacy", reason="spaCy is not installed (pip install ai-stamp[nlp])"
)
pytest.importorskip(
    "en_core_web_sm",
    reason="spaCy model en_core_web_sm not installed "
    "(python -m spacy download en_core_web_sm)",
)

pytestmark = pytest.mark.spacy


def test_scan_text_with_real_model_detects_person_spans() -> None:
    from aistamp import scan_text

    text = "Albert Einstein met Marie Curie in Berlin."
    matches = scan_text(text, use_spacy=True)

    person_matches = [m for m in matches if m.pattern_name == "PERSON"]
    assert person_matches, f"expected PERSON entities, got {matches}"
    for match in person_matches:
        assert 0 <= match.start < match.end <= len(text)
        assert text[match.start : match.end] in {
            "Albert Einstein",
            "Marie Curie",
        }
        # NER matches are persisted with the MEDIUM severity used by the scanner.
        assert match.severity.value == "MEDIUM"
