from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class PIISeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


# Numeric ranking for PIISeverity comparisons (LOW < MEDIUM < HIGH).
# Shared by pii.scanner (computing highest_severity) and policy.engine
# (comparing record severity against rule thresholds).
SEVERITY_RANK: dict[PIISeverity, int] = {
    PIISeverity.LOW: 0,
    PIISeverity.MEDIUM: 1,
    PIISeverity.HIGH: 2,
}


class PIIType(str, Enum):
    EMAIL = "EMAIL"
    PHONE_US = "PHONE_US"
    SSN = "SSN"
    CREDIT_CARD = "CREDIT_CARD"
    API_KEY = "API_KEY"
    IP_ADDRESS = "IP_ADDRESS"


class RecordStatus(str, Enum):
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    ERROR = "ERROR"
    # Write-ahead state: persisted before the provider call so a crash
    # mid-call still leaves an evidence trail. Finalized after the call.
    PENDING = "PENDING"


class PolicyAction(str, Enum):
    ALLOW = "ALLOW"
    WARN = "WARN"
    BLOCK = "BLOCK"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PIIMatch(_FrozenModel):
    pattern_name: str
    severity: PIISeverity
    start: int
    end: int
    redacted_snippet: str


class PIIResult(_FrozenModel):
    prompt_matches: list[PIIMatch]
    response_matches: list[PIIMatch]
    highest_severity: PIISeverity | None
    match_count: int


class PolicyDecision(_FrozenModel):
    action: PolicyAction
    rule_name: str | None
    reason: str | None


class ProvenanceRecord(_FrozenModel):
    content_id: str
    app_id: str
    feature_id: str
    user_id: str
    model: str
    prompt_hash: str
    response_hash: str | None
    prompt_tokens: int | None
    response_tokens: int | None
    latency_ms: float | None
    timestamp: datetime
    status: RecordStatus
    pii_result: PIIResult | None
    policy_decision: PolicyDecision | None
    # --- Pinned v0.2.0 shared fields (storage migration 0002) ---
    # Identity of the signing key used for the HMAC signature, enabling
    # key rotation without losing verifiability of older records.
    key_id: str = "default"
    # Signature algorithm identifier so verification stays deterministic
    # as algorithms evolve (e.g. future asymmetric signing).
    sig_algo: str = "HMAC-SHA256"
    # Schema version of the record payload itself.
    record_version: int = 1
    # Optional hash-chain link to the previous record's content hash.
    prev_hash: str | None = None
    # Monotonic sequence within a scope (e.g. one capture session), for
    # ordering records that share a timestamp.
    scope_sequence: int | None = None
    # Error taxonomy fields, populated when status == ERROR.
    error_type: str | None = None
    error_message: str | None = None


class VerificationResult(_FrozenModel):
    content_id: str
    verified: bool
    hash_match: bool
    hmac_valid: bool
    drift_detected: bool
    original_hash: str
    current_hash: str


class QueryFilters(BaseModel):
    """Audit query filters with validated, deterministic pagination.

    ``limit`` is clamped to ``[1, MAX_LIMIT]`` and ``offset`` must be
    non-negative, so a malformed audit request can neither exhaust memory
    (huge limit) nor desync paging (negative values).

    Two pagination modes are supported:

    - Offset paging (``limit``/``offset``) for small volumes.
    - Keyset paging for large volumes: pass ``cursor`` (an opaque string
      from ``AuditReport.next_cursor``) or the typed pair
      ``after_timestamp``/``after_id``. Keyset paging is O(1) per page
      and cannot skip or repeat rows while data is inserted. ``cursor``
      and the typed pair are mutually exclusive.
    """

    model_config = ConfigDict(frozen=True)

    MAX_LIMIT: ClassVar[int] = 1000

    content_id: str | None = None
    user_id: str | None = None
    app_id: str | None = None
    feature_id: str | None = None
    model: str | None = None
    status: RecordStatus | None = None
    pii_severity: PIISeverity | None = None
    policy_decision: PolicyAction | None = None
    from_dt: datetime | None = None
    to_dt: datetime | None = None
    after_timestamp: datetime | None = None
    after_id: int | None = None
    cursor: str | None = None
    limit: int = 100
    offset: int = 0

    @field_validator("limit")
    @classmethod
    def _clamp_limit(cls, value: int) -> int:
        if value < 1:
            raise ValueError("limit must be >= 1")
        return min(value, cls.MAX_LIMIT)

    @field_validator("offset")
    @classmethod
    def _reject_negative_offset(cls, value: int) -> int:
        if value < 0:
            raise ValueError("offset must be >= 0")
        return value

    @model_validator(mode="after")
    def _validate_pagination(self) -> QueryFilters:
        if self.cursor is not None and (
            self.after_timestamp is not None or self.after_id is not None
        ):
            raise ValueError(
                "cursor and after_timestamp/after_id are mutually exclusive"
            )
        if (self.after_timestamp is None) != (self.after_id is None):
            raise ValueError(
                "keyset pagination requires both after_timestamp and after_id"
            )
        return self


class AuditReport(_FrozenModel):
    records: list[ProvenanceRecord]
    total_count: int
    generated_at: datetime
    filters_applied: dict[str, Any]
    # Opaque keyset cursor for the next page; None when no more results.
    # Feed it back as QueryFilters.cursor.
    next_cursor: str | None = None
