"""PII scanning.

PII detection in ai-stamp provides best-effort coverage using regex patterns,
per-type validators, and optional NER. It is not a substitute for certified
DLP tooling in regulated environments.

v2 behaviour:

- **Overlap arbitration** — when patterns compete for the same span, the
  longest match wins (ties broken by severity, then confidence, then
  pattern name) and results are ordered by position, so overlapping
  findings are never double-counted. Pass ``resolve_overlaps=False`` for
  the 0.1.x raw-union semantics.
- **Confidence** — every match carries ``confidence`` in ``(0.0, 1.0]``
  from its per-type validator (see ``aistamp.pii.validators`` for the
  documented scale); patterns without a validator use their
  ``PatternConfig.confidence`` base value.
- **Allowlists** — global (``allowlist=``) and per-pattern
  (``PatternConfig.allowlist``) exemptions. Entries starting with
  ``regex:`` are fullmatched as regular expressions; other entries match
  exact values.
- **Locale packs** — pass ``locale="INDIA"`` or ``locale="EU"`` to add
  regional identifier coverage (see ``aistamp.pii.locales``).
- **Optional NER** — ``use_spacy=True`` uses a process-cached spaCy model
  and degrades loudly (WARN) when unavailable (see ``aistamp.pii.ner``).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from functools import lru_cache

from aistamp.models import SEVERITY_RANK, PIIMatch, PIIResult, PIISeverity
from aistamp.pii.locales import get_locale_patterns
from aistamp.pii.ner import NERConfig, scan_with_ner
from aistamp.pii.patterns import BUILT_IN_PATTERNS, PatternConfig
from aistamp.pii.validators import VALIDATOR_REJECTED_CONFIDENCE, VALIDATORS

logger = logging.getLogger("aistamp.pii")

_REGEX_ALLOWLIST_PREFIX = "regex:"


@lru_cache(maxsize=512)
def _compile_pattern(config: PatternConfig) -> re.Pattern[str]:
    """Compile a pattern once per config, including caller-supplied ones."""
    return re.compile(config.pattern)


@lru_cache(maxsize=512)
def _regex_allowlist_entry_matches(entry: str, value: str) -> bool:
    return re.fullmatch(entry, value) is not None


def _is_allowlisted(value: str, entries: Sequence[str]) -> bool:
    for entry in entries:
        if entry.startswith(_REGEX_ALLOWLIST_PREFIX):
            if _regex_allowlist_entry_matches(
                entry[len(_REGEX_ALLOWLIST_PREFIX) :], value
            ):
                return True
        elif entry == value:
            return True
    return False

def _reject_shadowed_names(
    extras: Sequence[PatternConfig], reserved: set[str]
) -> None:
    """Reject extras that reuse a reserved pattern name.

    A shadowing extra silently races the original during arbitration and
    splits per-pattern allowlist/config lookups, so names taken by built-in
    patterns, the active locale pack, or an earlier extra are refused.
    """
    for extra in extras:
        if extra.name in reserved:
            raise ValueError(
                f"Extra pattern {extra.name!r} collides with a built-in, "
                "locale-pack, or other extra pattern name; use a distinct "
                "name."
            )
        reserved.add(extra.name)


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


def _dedupe_matches(matches: list[PIIMatch]) -> list[PIIMatch]:
    seen: set[tuple[int, int, str]] = set()
    deduped: list[PIIMatch] = []
    for match in matches:
        key = (match.start, match.end, match.pattern_name)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def _resolve_overlapping_matches(matches: list[PIIMatch]) -> list[PIIMatch]:
    """Longest-match-wins arbitration with deterministic tie-breaking.

    Candidates are ranked by span length (desc), position, severity
    (desc), confidence (desc), then pattern name; a match is kept unless
    it overlaps an already-kept one. The kept set is returned in
    positional order.
    """
    ranked = sorted(
        matches,
        key=lambda m: (
            -(m.end - m.start),
            m.start,
            -SEVERITY_RANK[m.severity],
            -m.confidence,
            m.pattern_name,
        ),
    )
    kept: list[PIIMatch] = []
    for match in ranked:
        if any(
            match.start < kept_match.end and kept_match.start < match.end
            for kept_match in kept
        ):
            continue
        kept.append(match)
    return sorted(kept, key=lambda m: (m.start, m.end, m.pattern_name))


def _scan_with_spacy(text: str, ner_config: NERConfig) -> list[PIIMatch]:
    """Delegates to aistamp.pii.ner; kept here as the NER entry point."""
    return scan_with_ner(text, ner_config)


def scan_text(
    text: str,
    extra_patterns: list[PatternConfig] | None = None,
    use_spacy: bool = False,
    *,
    allowlist: Sequence[str] | None = None,
    resolve_overlaps: bool = True,
    locale: str | None = None,
    ner_config: NERConfig | None = None,
) -> list[PIIMatch]:
    """Scan ``text`` for PII and return the surviving matches.

    Results are ordered by position with no overlapping spans (unless
    ``resolve_overlaps=False``, which restores the 0.1.x raw-union
    semantics).

    Raises:
        ValueError: If an ``extra_patterns`` entry reuses the name of a
            built-in pattern, a pattern from the active locale pack, or
            another extra pattern.
    """
    if not isinstance(text, str):
        raise TypeError(f"scan_text expected str, got {type(text).__name__}")

    if text == "":
        return []

    configs: list[PatternConfig] = list(BUILT_IN_PATTERNS)
    if locale is not None:
        configs.extend(get_locale_patterns(locale))
    if extra_patterns:
        _reject_shadowed_names(extra_patterns, reserved={c.name for c in configs})
        configs.extend(extra_patterns)

    matches: list[PIIMatch] = []
    for config in configs:
        compiled = _compile_pattern(config)
        for found in compiled.finditer(text):
            value = found.group()
            if config.allowlist and _is_allowlisted(value, config.allowlist):
                continue
            validator = VALIDATORS.get(config.name)
            if validator is not None:
                confidence = validator(value)
                if confidence is None:
                    # Fail closed (audit P1-4): a rejected candidate stays a
                    # match at reduced confidence so redaction still covers
                    # the span and it cannot leak raw into persisted
                    # redacted_snippet context of neighboring matches.
                    confidence = VALIDATOR_REJECTED_CONFIDENCE
            else:
                confidence = config.confidence
            matches.append(
                PIIMatch(
                    pattern_name=config.name,
                    severity=config.severity,
                    start=found.start(),
                    end=found.end(),
                    redacted_snippet="",
                    confidence=confidence,
                )
            )

    if use_spacy:
        matches.extend(_scan_with_spacy(text, ner_config or NERConfig()))

    if allowlist:
        entries = list(allowlist)
        matches = [
            match
            for match in matches
            if not _is_allowlisted(text[match.start : match.end], entries)
        ]

    matches = _dedupe_matches(matches)
    if resolve_overlaps:
        matches = _resolve_overlapping_matches(matches)

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
    *,
    allowlist: Sequence[str] | None = None,
    resolve_overlaps: bool = True,
    locale: str | None = None,
    ner_config: NERConfig | None = None,
) -> PIIResult:
    prompt_matches = scan_text(
        prompt,
        extra_patterns,
        use_spacy,
        allowlist=allowlist,
        resolve_overlaps=resolve_overlaps,
        locale=locale,
        ner_config=ner_config,
    )
    response_matches = scan_text(
        response,
        extra_patterns,
        use_spacy,
        allowlist=allowlist,
        resolve_overlaps=resolve_overlaps,
        locale=locale,
        ner_config=ner_config,
    )

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
