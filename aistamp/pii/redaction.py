"""PII redaction.

``redact_text`` is the public scrubbing primitive: it replaces matched
spans in text with a placeholder. Pass ``matches=None`` to scan-and-redact
in one call, or supply matches from ``scan_text`` to redact against a
known match set.

Redaction uses *union* semantics: overlapping spans are merged before
substitution, so no fragment of any matched value survives — even when
the caller passes raw, overlapping match sets (``resolve_overlaps=False``).
"""

from __future__ import annotations

from collections.abc import Sequence

from aistamp.models import PIIMatch
from aistamp.pii.ner import NERConfig
from aistamp.pii.patterns import PatternConfig
from aistamp.pii.scanner import scan_text


def _merge_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def redact_text(
    text: str,
    matches: Sequence[PIIMatch] | None = None,
    placeholder: str = "[REDACTED]",
) -> str:
    """Return ``text`` with every matched span replaced by ``placeholder``.

    With ``matches=None`` the text is scanned first (equivalent to
    ``redact_text(text, scan_text(text))``). Match offsets outside the text
    are clamped and empty spans dropped, so stale or hostile match sets
    cannot corrupt the output.
    """
    if not isinstance(text, str):
        raise TypeError(f"redact_text expected str, got {type(text).__name__}")

    if matches is None:
        matches = scan_text(text)

    length = len(text)
    spans = [
        (max(0, min(match.start, length)), max(0, min(match.end, length)))
        for match in matches
    ]
    spans = [(start, end) for start, end in spans if end > start]

    parts: list[str] = []
    cursor = 0
    for start, end in _merge_spans(spans):
        if cursor < start:
            parts.append(text[cursor:start])
        parts.append(placeholder)
        cursor = end
    if cursor < length:
        parts.append(text[cursor:length])
    return "".join(parts)


def redact_prompt_and_response(
    prompt: str,
    response: str,
    placeholder: str = "[REDACTED]",
    extra_patterns: list[PatternConfig] | None = None,
    use_spacy: bool = False,
    *,
    allowlist: Sequence[str] | None = None,
    resolve_overlaps: bool = True,
    locale: str | None = None,
    ner_config: NERConfig | None = None,
) -> tuple[str, str]:
    """Scan-and-redact a prompt/response pair in one call.

    Returns ``(redacted_prompt, redacted_response)``. Accepts the same
    scan options as ``scan_prompt_and_response``.
    """
    redacted_prompt = redact_text(
        prompt,
        scan_text(
            prompt,
            extra_patterns,
            use_spacy,
            allowlist=allowlist,
            resolve_overlaps=resolve_overlaps,
            locale=locale,
            ner_config=ner_config,
        ),
        placeholder,
    )
    redacted_response = redact_text(
        response,
        scan_text(
            response,
            extra_patterns,
            use_spacy,
            allowlist=allowlist,
            resolve_overlaps=resolve_overlaps,
            locale=locale,
            ner_config=ner_config,
        ),
        placeholder,
    )
    return redacted_prompt, redacted_response


__all__ = [
    "redact_prompt_and_response",
    "redact_text",
]
