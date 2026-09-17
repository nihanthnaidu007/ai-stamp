"""Per-type PII validators and confidence scoring.

Every validator receives the raw matched text and returns a confidence in
``(0.0, 1.0]`` — how likely the span is a *true* instance of the pattern
type — or ``None`` when the strongest available check for the type fails
(bad checksums, structurally impossible values). Rejected candidates are
kept as matches at ``VALIDATOR_REJECTED_CONFIDENCE`` (0.25) rather than
dropped: redaction must fail closed, so a span shaped like the pattern
type is still detected and redacted even when validation says it is not
genuine, and it cannot leak raw into persisted ``redacted_snippet``
context of neighboring matches.

Confidence 1.0 means the value passed the
strongest available check for its type (Luhn, Verhoeff, mod-97, range or
structure rules); lower values mean the match is plausible but unverified.

Per-type scale (pattern name -> behaviour):

- ``CREDIT_CARD``: Luhn checksum + length. 16 digits, or 15 digits when
  the number starts with 34/37 (American Express). Bad checksum or
  length -> rejected; valid -> 1.0.
- ``SSN``: US issuance range rules. Areas 000, 666 and 900-999, group
  00 and serial 0000 were never issued and are rejected. Valid range
  -> 1.0.
- ``PHONE_US``: NANP plausibility. Area codes starting with 0/1 or
  ending in 11 (service codes) are rejected — this is what removes
  10-digit account-number false positives. A valid area code with an
  implausible exchange (starts with 0/1 or N11) scores 0.5; the
  fictional 555-01XX range scores 0.7; otherwise 1.0.
- ``EMAIL``: domain plausibility. Single-character local parts or
  single-character domain labels score 0.7; otherwise 1.0.
- ``IP_ADDRESS``: must parse via ``ipaddress`` (rejected otherwise).
  Globally-routable addresses score 1.0; private, loopback, link-local
  and documentation ranges score 0.7.
- ``API_KEY``: vendor-prefixed keys (sk-, sk-proj-, sk-ant-, AKIA,
  ghp_/gho_/ghu_/ghs_/ghr_, github_pat_) score 1.0; raw Bearer tokens
  score 0.9 (weaker evidence).
- ``AADHAAR``: Verhoeff checksum. Failure -> rejected; pass -> 1.0.
- ``PAN``: structure. Fourth character not a known holder-type code
  scores 0.6; otherwise 1.0.
- ``IBAN``: mod-97 checksum. Failure -> rejected; pass -> 1.0.
- ``STEUER_ID``: 11 digits. When exactly one digit occurs exactly twice
  among the first ten (and no digit occurs more often) the match scores
  1.0; otherwise 0.5. Matches are never dropped: the official German
  check-digit algorithm is intentionally not enforced.

Patterns whose name has no entry in ``VALIDATORS`` (custom patterns,
NER labels, JWT) fall back to their ``PatternConfig.confidence`` base
value.
"""

from __future__ import annotations

import ipaddress
import re
from collections import Counter
from collections.abc import Callable

# Validators are keyed by pattern name so custom patterns named after a
# built-in type get the same validation for free.
Validator = Callable[[str], "float | None"]


# ---------------------------------------------------------------------------
# Verhoeff checksum (Aadhaar)
#
# Tables derived from the dihedral group D5 rather than copied: elements are
# labelled 0-9 as (rotation a = n mod 5, reflection b = n // 5) with the
# product (a1, b1) * (a2, b2) = ((a1 + (-1)^b1 * a2) mod 5, (b1 + b2) mod 2).
# Anchors: the inverse row matches the published table, and the check digit
# of "236" is 3 (so "2363" validates).
# ---------------------------------------------------------------------------
_VERHOEFF_D: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
    (1, 2, 3, 4, 0, 6, 7, 8, 9, 5),
    (2, 3, 4, 0, 1, 7, 8, 9, 5, 6),
    (3, 4, 0, 1, 2, 8, 9, 5, 6, 7),
    (4, 0, 1, 2, 3, 9, 5, 6, 7, 8),
    (5, 9, 8, 7, 6, 0, 4, 3, 2, 1),
    (6, 5, 9, 8, 7, 1, 0, 4, 3, 2),
    (7, 6, 5, 9, 8, 2, 1, 0, 4, 3),
    (8, 7, 6, 5, 9, 3, 2, 1, 0, 4),
    (9, 8, 7, 6, 5, 4, 3, 2, 1, 0),
)
_VERHOEFF_INV: tuple[int, ...] = (0, 4, 3, 2, 1, 5, 6, 7, 8, 9)
_VERHOEFF_SIGMA: tuple[int, ...] = (1, 5, 7, 6, 2, 8, 3, 0, 9, 4)


def _verhoeff_permutations() -> tuple[tuple[int, ...], ...]:
    rows: list[tuple[int, ...]] = [tuple(range(10))]
    for _ in range(7):
        rows.append(tuple(_VERHOEFF_SIGMA[v] for v in rows[-1]))
    return tuple(rows)


_VERHOEFF_P: tuple[tuple[int, ...], ...] = _verhoeff_permutations()


def _verhoeff_valid(digits: str) -> bool:
    checksum = 0
    for index, char in enumerate(reversed(digits)):
        checksum = _VERHOEFF_D[checksum][_VERHOEFF_P[index % 8][int(char)]]
    return checksum == 0


def _digits_only(value: str) -> str:
    return re.sub(r"\D", "", value)


def _luhn_ok(digits: str) -> bool:
    checksum = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        digit = int(char)
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _validate_credit_card(value: str) -> float | None:
    digits = _digits_only(value)
    if len(digits) == 15:
        if digits[:2] not in {"34", "37"} or not _luhn_ok(digits):
            return None
        return 1.0
    if len(digits) == 16 and _luhn_ok(digits):
        return 1.0
    return None


def _validate_ssn(value: str) -> float | None:
    digits = _digits_only(value)
    if len(digits) != 9:
        return None
    area, group, serial = digits[:3], digits[3:5], digits[5:]
    if area in {"000", "666"} or area[0] == "9":
        return None
    if group == "00" or serial == "0000":
        return None
    return 1.0


def _validate_phone_us(value: str) -> float | None:
    digits = _digits_only(value)
    if len(digits) == 11 and digits.startswith("1"):
        digits = digits[1:]
    if len(digits) != 10:
        return None
    area, exchange = digits[:3], digits[3:6]
    if area[0] in "01" or area[1:] == "11":
        return None
    if exchange[0] in "01" or exchange[1:] == "11":
        return 0.5
    if exchange == "555" and digits[6:8] == "01":
        return 0.7
    return 1.0


def _validate_email(value: str) -> float | None:
    local, _, domain = value.rpartition("@")
    if not local or not domain:
        return None
    if len(local) == 1 or len(domain.split(".")[0]) == 1:
        return 0.7
    return 1.0


def _validate_ip_address(value: str) -> float | None:
    try:
        parsed = ipaddress.ip_address(value.strip())
    except ValueError:
        return None
    return 1.0 if parsed.is_global else 0.7


# Longer, more specific prefixes first is unnecessary for startswith, but
# keeping the full vendor list explicit documents what is recognised.
_API_KEY_VENDOR_PREFIXES: tuple[str, ...] = (
    "sk-proj-",
    "sk-svcacct-",
    "sk-admin-",
    "sk-ant-",
    "sk-",
    "AKIA",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
)


def _validate_api_key(value: str) -> float | None:
    if value.startswith("Bearer"):
        return 0.9
    if any(value.startswith(prefix) for prefix in _API_KEY_VENDOR_PREFIXES):
        return 1.0
    return 1.0


def _validate_aadhaar(value: str) -> float | None:
    digits = _digits_only(value)
    if len(digits) != 12:
        return None
    return 1.0 if _verhoeff_valid(digits) else None


# PAN holder-type codes (4th character): P personal, C company, H Hindu
# undivided family, F firm, A association, T trust, B body of individuals,
# L local authority, J artificial juridical person, G government.
_PAN_HOLDER_TYPES = frozenset("PCHFATBLJG")


def _validate_pan(value: str) -> float | None:
    if len(value) != 10:
        return None
    if value[3] not in _PAN_HOLDER_TYPES:
        return 0.6
    return 1.0


def _validate_iban(value: str) -> float | None:
    compact = value.replace(" ", "").upper()
    if not 15 <= len(compact) <= 34:
        return None
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", compact):
        return None
    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(char, 36)) for char in rearranged)
    return 1.0 if int(numeric) % 97 == 1 else None


def _validate_steuer_id(value: str) -> float | None:
    if len(value) != 11 or not value.isdigit():
        return None
    counts = Counter(value[:10])
    exactly_one_pair = sum(1 for n in counts.values() if n == 2) == 1
    no_higher_repeats = all(n <= 2 for n in counts.values())
    return 1.0 if exactly_one_pair and no_higher_repeats else 0.5


#: Confidence assigned to candidates whose per-type validator returned
#: ``None``. Kept below every validator pass value (lowest is 0.5) so
#: downstream policy can filter rejected-looking spans, but non-zero so
#: fail-closed redaction still covers them.
VALIDATOR_REJECTED_CONFIDENCE: float = 0.25

VALIDATORS: dict[str, Validator] = {
    "CREDIT_CARD": _validate_credit_card,
    "SSN": _validate_ssn,
    "PHONE_US": _validate_phone_us,
    "EMAIL": _validate_email,
    "IP_ADDRESS": _validate_ip_address,
    "API_KEY": _validate_api_key,
    "AADHAAR": _validate_aadhaar,
    "PAN": _validate_pan,
    "IBAN": _validate_iban,
    "STEUER_ID": _validate_steuer_id,
}
