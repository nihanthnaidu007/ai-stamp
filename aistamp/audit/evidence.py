"""Evidence packs: one-call compliance hand-off for a single record.

An evidence pack assembles everything a compliance reviewer needs to
re-verify one piece of AI-generated content: the full record, the stored
HMAC signature, the verification verdict, the PII detail, and the policy
trace that explains why the system allowed, warned, or blocked it.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from aistamp.audit.exporter import (
    SignatureVerdict,
    pii_type_counts,
    record_to_dict,
    signature_verdict,
)
from aistamp.models import ProvenanceRecord

EVIDENCE_VERSION = 1


def build_evidence_pack(
    record: ProvenanceRecord,
    stored_signature: str | None,
    secret_key: str | None,
) -> dict[str, Any]:
    """Assemble a JSON-ready evidence pack for one provenance record.

    ``signature_verdict`` is computed from the record as read back from the
    store, so an exported pack proves both what was stored and that the
    stored signature does (or does not) verify against the current secret.
    """
    verdict: SignatureVerdict = signature_verdict(
        record, stored_signature, secret_key
    )

    policy_trace: dict[str, Any] | None = None
    if record.policy_decision is not None:
        d = record.policy_decision
        policy_trace = {
            "action": d.action.value,
            "rule_name": d.rule_name,
            "reason": d.reason,
            "matched_conditions": d.matched_conditions,
            "evaluated_rules": d.evaluated_rules,
            "decided_at": d.decided_at.isoformat() if d.decided_at else None,
        }

    return {
        "evidence_version": EVIDENCE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content_id": record.content_id,
        "record": record_to_dict(record),
        "hmac_signature": stored_signature,
        "verification": {
            "algorithm": "HMAC-SHA256",
            "signature_verdict": verdict,
        },
        "pii_detail": {
            "match_count": record.pii_result.match_count if record.pii_result else 0,
            "highest_severity": (
                record.pii_result.highest_severity.value
                if record.pii_result and record.pii_result.highest_severity
                else None
            ),
            "type_counts": pii_type_counts(record),
            "prompt_matches": (
                [m.model_dump(mode="json") for m in record.pii_result.prompt_matches]
                if record.pii_result
                else []
            ),
            "response_matches": (
                [
                    m.model_dump(mode="json")
                    for m in record.pii_result.response_matches
                ]
                if record.pii_result
                else []
            ),
        },
        "policy_trace": policy_trace,
    }
