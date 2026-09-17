from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class SignatureStatus(str, Enum):
    """Outcome of the signature check inside a VerificationResult.

    - VALID: a stored signature exists and verifies under the record's key.
    - INVALID: a stored signature exists but does not verify (tamper suspected).
    - UNSIGNED: no signature was stored for the record.
    - UNKNOWN_KEY: the record's key_id is absent from the provided keyring, so
      signature validity could not be established.
    """

    VALID = "VALID"
    INVALID = "INVALID"
    UNSIGNED = "UNSIGNED"
    UNKNOWN_KEY = "UNKNOWN_KEY"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class PIIMatch(_FrozenModel):
    pattern_name: str
    severity: PIISeverity
    start: int
    end: int
    redacted_snippet: str
    # Confidence that the matched span is a true PII instance, in (0.0, 1.0].
    # 1.0 = checksum-validated or strongly structured; lower values mean the
    # match is plausible but unverified; 0.25 = the per-type validator
    # rejected the value, kept fail-closed so redaction still covers the
    # span. See aistamp.pii.validators for the per-type scale.
    confidence: float = 1.0


class PIIResult(_FrozenModel):
    prompt_matches: list[PIIMatch]
    response_matches: list[PIIMatch]
    highest_severity: PIISeverity | None
    match_count: int


class PolicyDecision(_FrozenModel):
    action: PolicyAction
    rule_name: str | None
    reason: str | None
    # Decision transparency (v0.2): the concrete record-side values that fired,
    # the trace of rules that matched during evaluation, and when the decision
    # was made. Defaults keep 0.1.x constructors and persisted JSON valid.
    matched_conditions: dict[str, Any] = Field(default_factory=dict)
    evaluated_rules: list[str] = Field(default_factory=list)
    decided_at: datetime | None = None


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
    # --- v0.2 tamper-evidence envelope (pinned shared interface; storage
    # migration 0002 backfills key_id='default', sig_algo='HMAC-SHA256',
    # record_version=1 for pre-existing rows) ---
    key_id: str = "default"
    sig_algo: str = "HMAC-SHA256"
    # v2 is the secure default: its canonical payload binds the envelope and
    # chain fields (key_id, sig_algo, record_version, prev_hash,
    # scope_sequence), so rewriting chain links without the key fails record
    # verification. v1 is the byte-pinned 0.1.x payload — set explicitly only
    # when reproducing legacy signatures. Audit P0-1 (art_y6PXlLmn).
    record_version: int = 2
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
    # None when the stored record has no response hash (was the "" sentinel in 0.1.x).
    original_hash: str | None
    current_hash: str
    # v0.2 envelope fields: which key the record claims, and the signature outcome.
    # signature_status stays None only when constructed outside verify_record().
    key_id: str | None = None
    signature_status: SignatureStatus | None = None
    # Which canonical payload version the signature was verified against. For
    # v2 records the envelope and chain fields are inside the HMAC, so chain
    # relinking shows up here as SignatureStatus.INVALID without verify_chain.
    record_version: int = 1


class ChainIssueKind(str, Enum):
    """Structural defect detected while walking a hash chain.

    MISSING covers gaps and dangling predecessors; REORDERED covers link and
    sequence inconsistencies (including duplicated positions).
    """

    MISSING = "MISSING"
    REORDERED = "REORDERED"


class ChainIssue(_FrozenModel):
    kind: ChainIssueKind
    sequence: int | None
    content_id: str | None
    detail: str


class ChainVerificationResult(_FrozenModel):
    app_id: str
    feature_id: str
    valid: bool
    records_checked: int
    unchained_records: int
    issues: list[ChainIssue]
    # Security audit P1-5: chain breaks that the purge journal explains as
    # legitimate retention deletions. Anchored gaps produce no issues; this
    # counts them so audits still see that the chain has purge-shaped holes.
    anchored_gaps: int = 0


class ChainLink(_FrozenModel):
    """Link data a writer stamps onto the next record in a scope's chain."""

    scope_sequence: int
    prev_hash: str | None


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
    # Evidence rule (security audit P1-3): PENDING write-ahead records are
    # crash-interrupted, incomplete evidence. query() excludes them unless
    # the caller opts in here or filters status=PENDING explicitly, so
    # reports and exports never present incomplete evidence as complete.
    include_pending: bool = False

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
        if self.offset > 0 and (
            self.cursor is not None
            or self.after_timestamp is not None
            or self.after_id is not None
        ):
            raise ValueError(
                "offset paging cannot be combined with keyset pagination "
                "(cursor or after_timestamp/after_id)"
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


class PurgeAnchor(_FrozenModel):
    """One retention-purge event: proof that records were legitimately removed.

    purge() writes an anchor in the same transaction as the deletes, listing
    the chain positions (prev_hash values) the purge removed — retention and
    the hash chain stop cancelling each other out (security audit P1-5).
    ``signature`` is reserved for the signing layer; the store never holds
    the HMAC key.
    """

    id: int
    purged_before: datetime
    purged_count: int
    deleted_prev_hashes: list[str | None]
    anchor_created_at: datetime
    signature: str | None = None
