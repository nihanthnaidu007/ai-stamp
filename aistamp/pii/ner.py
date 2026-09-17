"""Optional spaCy NER support with cached model loading.

The spaCy model is loaded once per process (module-level LRU cache) instead
of once per scan. When ``use_spacy=True`` but spaCy or the model is missing,
scanning degrades *loudly*: a WARN-level message with install instructions
is logged and the scan continues with regex-only results.

Installing the default model::

    pip install spacy
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from aistamp.models import PIIMatch, PIISeverity

logger = logging.getLogger("aistamp.pii")


@dataclass(frozen=True)
class NERConfig:
    """Configuration for NER scanning.

    ``severity_by_label`` maps spaCy entity labels to severities; labels
    missing from the mapping (but present in ``labels``) default to MEDIUM.
    The mapping contents are mutable even though the config is frozen —
    treat it as read-only.
    """

    model_name: str = "en_core_web_sm"
    labels: frozenset[str] = frozenset({"PERSON", "ORG"})
    severity_by_label: Mapping[str, PIISeverity] = field(
        default_factory=lambda: {
            "PERSON": PIISeverity.MEDIUM,
            "ORG": PIISeverity.MEDIUM,
        }
    )
    confidence: float = 0.85


@cache
def load_nlp(model_name: str) -> Any:
    """Load and cache a spaCy model for the lifetime of the process."""
    import spacy

    return spacy.load(model_name)


def scan_with_ner(text: str, config: NERConfig) -> list[PIIMatch]:
    """Scan ``text`` with spaCy NER; degrade loudly to no results."""
    try:
        nlp = load_nlp(config.model_name)
        doc = nlp(text)
    except Exception as exc:  # ImportError (no spaCy) / OSError (no model)
        logger.warning(
            "spaCy NER requested but unavailable; continuing with regex-only "
            "results (%s). Install with: pip install spacy && python -m spacy "
            "download %s",
            exc,
            config.model_name,
        )
        return []

    matches: list[PIIMatch] = []
    for ent in doc.ents:
        label = str(ent.label_)
        if label not in config.labels:
            continue
        matches.append(
            PIIMatch(
                pattern_name=label,
                severity=config.severity_by_label.get(label, PIISeverity.MEDIUM),
                start=int(ent.start_char),
                end=int(ent.end_char),
                redacted_snippet="",
                confidence=config.confidence,
            )
        )
    return matches
