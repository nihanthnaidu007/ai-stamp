"""Opt-in locale pattern packs.

Locale packs extend the built-in (US-centric) patterns with regional
identifiers. Pass ``locale="INDIA"`` or ``locale="EU"`` to ``scan_text`` /
``scan_prompt_and_response``, or import the tuples directly as
``extra_patterns``. Built-in patterns always remain active; a locale pack
only *adds* coverage.

Newer pattern types (everything not in ``PIIType``) use plain string names
so the 0.1.x enum stays untouched for existing consumers.
"""

from __future__ import annotations

from aistamp.models import PIISeverity
from aistamp.pii.patterns import PatternConfig

INDIA_PATTERNS: tuple[PatternConfig, ...] = (
    PatternConfig(
        name="AADHAAR",
        pattern=r"\b(?:\d{4}[\s-]?){2}\d{4}\b",
        severity=PIISeverity.HIGH,
        description="India Aadhaar 12-digit identifiers (Verhoeff checksum)",
        locale="INDIA",
    ),
    PatternConfig(
        name="PAN",
        pattern=r"\b[A-Z]{5}\d{4}[A-Z]\b",
        severity=PIISeverity.HIGH,
        description="India Permanent Account Numbers",
        locale="INDIA",
    ),
    PatternConfig(
        name="INDIA_MOBILE",
        pattern=r"\b(?:\+91[\s-]?|0)?[6-9]\d{9}\b",
        severity=PIISeverity.MEDIUM,
        description="India mobile numbers (+91 or trunk-0 prefixed, or bare "
        "10-digit numbers starting 6-9)",
        locale="INDIA",
    ),
)

EU_PATTERNS: tuple[PatternConfig, ...] = (
    PatternConfig(
        name="IBAN",
        pattern=r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]){8,30}\b",
        severity=PIISeverity.HIGH,
        description="International Bank Account Numbers (mod-97 checksum)",
        locale="EU",
    ),
    PatternConfig(
        name="NINO",
        pattern=r"\b(?!BG|GB|NK|TN|ZZ)[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]?\b",
        severity=PIISeverity.HIGH,
        description="UK National Insurance numbers",
        locale="EU",
    ),
    PatternConfig(
        name="STEUER_ID",
        pattern=r"\b\d{11}\b",
        severity=PIISeverity.HIGH,
        description="German tax identification numbers (structural rules)",
        locale="EU",
    ),
)

LOCALE_PACKS: dict[str, tuple[PatternConfig, ...]] = {
    "INDIA": INDIA_PATTERNS,
    "EU": EU_PATTERNS,
}


def get_locale_patterns(name: str) -> tuple[PatternConfig, ...]:
    """Return the pattern pack for a locale name (case-insensitive)."""
    key = name.upper()
    try:
        return LOCALE_PACKS[key]
    except KeyError:
        raise ValueError(
            f"Unknown locale pack {name!r}. Available: "
            f"{', '.join(sorted(LOCALE_PACKS))}"
        ) from None
