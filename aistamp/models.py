from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict


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


class VerificationResult(_FrozenModel):
    content_id: str
    verified: bool
    hash_match: bool
    hmac_valid: bool
    drift_detected: bool
    original_hash: str
    current_hash: str


@dataclass
class QueryFilters:
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
    limit: int = 100
    offset: int = 0


class AuditReport(_FrozenModel):
    records: list[ProvenanceRecord]
    total_count: int
    generated_at: datetime
    filters_applied: dict[str, Any]
