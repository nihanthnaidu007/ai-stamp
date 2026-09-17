from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aistamp.models import PIISeverity, PIIType


@dataclass(frozen=True)
class PatternConfig:
    """A single PII pattern definition.

    ``locale`` tags the pack a pattern came from (None for built-ins).
    ``confidence`` is the base confidence assigned to matches of this
    pattern when no per-type validator applies; per-type validators
    (see aistamp.pii.validators) override it. ``version`` tracks pattern
    definition revisions. ``allowlist`` entries exempt matched values from
    this pattern only: entries starting with ``regex:`` are fullmatched as
    regular expressions, anything else is an exact value match.
    """

    name: str
    pattern: str
    severity: PIISeverity
    description: str = ""
    locale: str | None = None
    confidence: float = 1.0
    version: int = 1
    allowlist: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError(
                f"Pattern '{self.name}' confidence must be in (0.0, 1.0], "
                f"got {self.confidence}."
            )
        if self.version < 1:
            raise ValueError(
                f"Pattern '{self.name}' version must be >= 1, got {self.version}."
            )
        try:
            re.compile(self.pattern)
        except re.error as e:
            raise ValueError(
                f"Pattern '{self.name}' has invalid regex: {e}"
            ) from e


BUILT_IN_PATTERNS: list[PatternConfig] = [
    PatternConfig(
        name=PIIType.EMAIL.value,
        pattern=r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        severity=PIISeverity.MEDIUM,
        description="Email addresses",
    ),
    PatternConfig(
        name=PIIType.PHONE_US.value,
        pattern=r"\b(?:\+1[\s.\-]?)?\(?\d{3}\)?[\s.\-]?\d{3}[\s.\-]?\d{4}\b",
        severity=PIISeverity.MEDIUM,
        description="US phone numbers in common formats",
    ),
    PatternConfig(
        name=PIIType.SSN.value,
        pattern=r"\b\d{3}-\d{2}-\d{4}\b",
        severity=PIISeverity.HIGH,
        description="US Social Security Numbers in XXX-XX-XXXX format",
    ),
    PatternConfig(
        name=PIIType.CREDIT_CARD.value,
        pattern=r"\b(?:\d{4}[\s\-]?){3}\d{4}\b",
        severity=PIISeverity.HIGH,
        description="Luhn-valid 16-digit card numbers with optional separators",
    ),
    PatternConfig(
        name=PIIType.CREDIT_CARD.value,
        pattern=r"\b\d{4}[\s\-]?\d{6}[\s\-]?\d{5}\b",
        severity=PIISeverity.HIGH,
        description="Luhn-valid 15-digit American Express card numbers",
    ),
    PatternConfig(
        name=PIIType.API_KEY.value,
        pattern=(
            r"\b(?:"
            r"sk-(?:proj|svcacct|admin)-[A-Za-z0-9_-]{20,}"
            r"|sk-ant-[A-Za-z0-9_-]{20,}"
            r"|sk-[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|gh[pousr]_[A-Za-z0-9]{20,}"
            r"|github_pat_[A-Za-z0-9_]{20,}"
            r"|(?:Bearer\s+)[A-Za-z0-9\-._~+/]{20,}"
            r")"
        ),
        severity=PIISeverity.HIGH,
        description="Common API key formats: OpenAI sk-/sk-proj-, Anthropic "
        "sk-ant-, AWS AKIA, GitHub ghp_/gho_/ghu_/ghs_/ghr_ and "
        "github_pat_, raw Bearer tokens",
    ),
    PatternConfig(
        name="JWT",
        pattern=r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}",
        severity=PIISeverity.HIGH,
        description="JSON Web Tokens (three base64url segments)",
    ),
    PatternConfig(
        name=PIIType.IP_ADDRESS.value,
        pattern=r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
        severity=PIISeverity.LOW,
        description="IPv4 addresses",
    ),
    PatternConfig(
        name=PIIType.IP_ADDRESS.value,
        pattern=r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{0,4}:){2,7}[0-9A-Fa-f]{0,4}(?![0-9A-Fa-f:])",
        severity=PIISeverity.LOW,
        description="IPv6 addresses",
    ),
]


_VALID_SEVERITIES = {"HIGH", "MEDIUM", "LOW"}


def load_patterns_from_yaml(path: str | Path) -> list[PatternConfig]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Pattern file not found: {path}")

    with p.open("r") as f:
        data = yaml.safe_load(f) or {}

    if "patterns" not in data:
        raise ValueError("YAML must contain a top-level 'patterns' key.")

    raw_patterns = data["patterns"] or []
    result: list[PatternConfig] = []
    seen_names: set[str] = set()

    for entry in raw_patterns:
        if "name" not in entry:
            raise ValueError("Pattern entry is missing required 'name' field.")
        if "pattern" not in entry:
            raise ValueError(
                f"Pattern entry '{entry['name']}' is missing required 'pattern' field."
            )
        if "severity" not in entry:
            raise ValueError(
                f"Pattern entry '{entry['name']}' is missing required 'severity' field."
            )

        severity_raw = str(entry["severity"]).upper()
        if severity_raw not in _VALID_SEVERITIES:
            raise ValueError(
                f"Pattern '{entry['name']}' has invalid severity '{entry['severity']}'."
                " Must be one of: HIGH, MEDIUM, LOW."
            )

        name = str(entry["name"])
        if name in seen_names:
            raise ValueError(
                f"Pattern file defines duplicate pattern name '{name}'."
            )
        seen_names.add(name)

        pattern_str = entry["pattern"]
        try:
            re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Pattern '{name}' has invalid regex: {e}") from e

        locale = entry.get("locale")
        confidence = entry.get("confidence", 1.0)
        if not isinstance(confidence, (int, float)) or isinstance(
            confidence, bool
        ):
            raise ValueError(
                f"Pattern '{name}' confidence must be a number, got "
                f"{confidence!r}."
            )
        version = entry.get("version", 1)
        if not isinstance(version, int) or isinstance(version, bool):
            raise ValueError(
                f"Pattern '{name}' version must be an integer, got {version!r}."
            )
        allowlist_raw = entry.get("allowlist", [])
        if not isinstance(allowlist_raw, list) or not all(
            isinstance(item, str) for item in allowlist_raw
        ):
            raise ValueError(
                f"Pattern '{name}' allowlist must be a list of strings."
            )

        result.append(
            PatternConfig(
                name=name,
                pattern=pattern_str,
                severity=PIISeverity(severity_raw),
                description=entry.get("description", ""),
                locale=locale if locale is None else str(locale),
                confidence=float(confidence),
                version=version,
                allowlist=tuple(allowlist_raw),
            )
        )

    return result
