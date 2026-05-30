from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from aistamp.models import PIISeverity, PIIType


@dataclass(frozen=True)
class PatternConfig:
    name: str
    pattern: str
    severity: PIISeverity
    description: str = ""


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
        name=PIIType.API_KEY.value,
        pattern=r"\b(?:sk-[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|(?:Bearer\s+)[A-Za-z0-9\-._~+/]{20,})\b",
        severity=PIISeverity.HIGH,
        description="Common API key formats: OpenAI sk-, AWS AKIA, Bearer tokens",
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

        pattern_str = entry["pattern"]
        try:
            re.compile(pattern_str)
        except re.error as e:
            raise ValueError(f"Pattern '{entry['name']}' has invalid regex: {e}") from e

        result.append(
            PatternConfig(
                name=entry["name"],
                pattern=pattern_str,
                severity=PIISeverity(severity_raw),
                description=entry.get("description", ""),
            )
        )

    return result
