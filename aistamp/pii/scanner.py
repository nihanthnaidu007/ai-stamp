"""PII scanning.

PII detection in ai-stamp provides best-effort coverage using regex patterns
and optional NER. It is not a substitute for certified DLP tooling in
regulated environments.
"""

from __future__ import annotations

import ipaddress
import logging
import re

from aistamp.models import SEVERITY_RANK, PIIMatch, PIIResult, PIISeverity
from aistamp.pii.patterns import BUILT_IN_PATTERNS, PatternConfig

logger = logging.getLogger("aistamp.pii")

# Pre-compile built-in patterns once at module load. extra_patterns are
# compiled per call since callers may pass different sets.
_COMPILED_BUILT_INS: list[tuple[PatternConfig, re.Pattern[str]]] = [
    (p, re.compile(p.pattern)) for p in BUILT_IN_PATTERNS
]


def _make_redacted_snippet(
    text: str,
    start: int,
    end: int,
    redaction_spans: list[tuple[int, int]],
    context: int = 20,
) -> str:
    snippet_start = max(0, start - context)
    snippet_end = min(len(text), end + context)
    parts: list[str] = []
    cursor = snippet_start
    for span_start, span_end in sorted(redaction_spans):
        if span_end <= snippet_start or span_start >= snippet_end:
            continue
        clipped_start = max(span_start, snippet_start)
        clipped_end = min(span_end, snippet_end)
        if cursor < clipped_start:
            parts.append(text[cursor:clipped_start])
        parts.append("[REDACTED]")
        cursor = max(cursor, clipped_end)
    if cursor < snippet_end:
        parts.append(text[cursor:snippet_end])
    prefix = "..." if snippet_start > 0 else ""
    suffix = "..." if snippet_end < len(text) else ""
    return f"{prefix}{''.join(parts)}{suffix}"


def _passes_validation(config: PatternConfig, value: str) -> bool:
    if config.name == "CREDIT_CARD":
        digits = [int(char) for char in value if char.isdigit()]
        checksum = 0
        parity = len(digits) % 2
        for index, digit in enumerate(digits):
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            checksum += digit
        return len(digits) == 16 and checksum % 10 == 0
    if config.name == "IP_ADDRESS":
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return False
    return True


def _scan_with_spacy(text: str) -> list[PIIMatch]:
    try:
        import spacy

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        matches: list[PIIMatch] = []
        for ent in doc.ents:
            if ent.label_ not in {"PERSON", "ORG"}:
                continue
            matches.append(
                PIIMatch(
                    pattern_name=ent.label_,
                    severity=PIISeverity.MEDIUM,
                    start=ent.start_char,
                    end=ent.end_char,
                    redacted_snippet="",
                )
            )
        return matches
    except Exception as e:
        logger.debug("spaCy NER unavailable or failed: %s", e)
        return []


def scan_text(
    text: str,
    extra_patterns: list[PatternConfig] | None = None,
    use_spacy: bool = False,
) -> list[PIIMatch]:
    if not isinstance(text, str):
        raise TypeError(f"scan_text expected str, got {type(text).__name__}")

    if text == "":
        return []

    compiled_pairs: list[tuple[PatternConfig, re.Pattern[str]]] = list(
        _COMPILED_BUILT_INS
    )
    if extra_patterns:
        compiled_pairs.extend((p, re.compile(p.pattern)) for p in extra_patterns)

    matches: list[PIIMatch] = []
    for config, compiled in compiled_pairs:
        for m in compiled.finditer(text):
            if not _passes_validation(config, m.group()):
                continue
            matches.append(
                PIIMatch(
                    pattern_name=config.name,
                    severity=config.severity,
                    start=m.start(),
                    end=m.end(),
                    redacted_snippet="",
                )
            )

    if use_spacy:
        try:
            matches.extend(_scan_with_spacy(text))
        except Exception:
            logger.debug("_scan_with_spacy raised unexpectedly; skipping NER results.")

    spans = [(match.start, match.end) for match in matches]
    return [
        match.model_copy(
            update={
                "redacted_snippet": _make_redacted_snippet(
                    text, match.start, match.end, spans
                )
            }
        )
        for match in matches
    ]


def scan_prompt_and_response(
    prompt: str,
    response: str,
    extra_patterns: list[PatternConfig] | None = None,
    use_spacy: bool = False,
) -> PIIResult:
    prompt_matches = scan_text(prompt, extra_patterns, use_spacy)
    response_matches = scan_text(response, extra_patterns, use_spacy)

    all_matches = prompt_matches + response_matches
    if not all_matches:
        highest_severity: PIISeverity | None = None
    else:
        highest_severity = max(
            (m.severity for m in all_matches),
            key=lambda s: SEVERITY_RANK[s],
        )

    return PIIResult(
        prompt_matches=prompt_matches,
        response_matches=response_matches,
        highest_severity=highest_severity,
        match_count=len(prompt_matches) + len(response_matches),
    )
