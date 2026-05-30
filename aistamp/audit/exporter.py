from __future__ import annotations

import csv
import io
import json
from typing import Any

from aistamp.models import AuditReport, ProvenanceRecord, QueryFilters
from aistamp.store.backend import StoreBackend


class AuditExporter:
    """
    Query the provenance store and export results as JSON or CSV.
    """

    def __init__(self, backend: StoreBackend) -> None:
        self._backend = backend

    def query(self, filters: QueryFilters) -> AuditReport:
        return self._backend.query(filters)

    def to_json(self, report: AuditReport, indent: int = 2) -> str:
        data = {
            "generated_at": report.generated_at.isoformat(),
            "total_count": report.total_count,
            "filters_applied": report.filters_applied,
            "records": [self._record_to_dict(r) for r in report.records],
        }
        return json.dumps(data, indent=indent, default=str)

    def to_csv(self, report: AuditReport) -> str:
        output = io.StringIO()
        fieldnames = [
            "content_id",
            "app_id",
            "feature_id",
            "user_id",
            "model",
            "status",
            "prompt_hash",
            "response_hash",
            "prompt_tokens",
            "response_tokens",
            "latency_ms",
            "timestamp",
            "pii_match_count",
            "pii_highest_severity",
            "policy_action",
            "policy_rule_name",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in report.records:
            writer.writerow(self._record_to_csv_row(record))
        return output.getvalue()

    def _record_to_dict(self, record: ProvenanceRecord) -> dict[str, Any]:
        pii = record.pii_result.model_dump(mode="json") if record.pii_result else None
        policy = (
            record.policy_decision.model_dump(mode="json")
            if record.policy_decision
            else None
        )
        return {
            "content_id": record.content_id,
            "app_id": record.app_id,
            "feature_id": record.feature_id,
            "user_id": record.user_id,
            "model": record.model,
            "status": record.status.value,
            "prompt_hash": record.prompt_hash,
            "response_hash": record.response_hash,
            "prompt_tokens": record.prompt_tokens,
            "response_tokens": record.response_tokens,
            "latency_ms": record.latency_ms,
            "timestamp": (record.timestamp.isoformat() if record.timestamp else None),
            "pii_result": pii,
            "policy_decision": policy,
        }

    def _record_to_csv_row(self, record: ProvenanceRecord) -> dict[str, Any]:
        pt = record.prompt_tokens if record.prompt_tokens is not None else ""
        rt = record.response_tokens if record.response_tokens is not None else ""
        lat = record.latency_ms if record.latency_ms is not None else ""
        pii_count = record.pii_result.match_count if record.pii_result else 0
        pii_sev = (
            record.pii_result.highest_severity.value
            if (record.pii_result and record.pii_result.highest_severity)
            else ""
        )
        policy_action = (
            record.policy_decision.action.value if record.policy_decision else ""
        )
        policy_rule = (
            record.policy_decision.rule_name
            if (record.policy_decision and record.policy_decision.rule_name)
            else ""
        )
        return {
            "content_id": record.content_id,
            "app_id": record.app_id,
            "feature_id": record.feature_id,
            "user_id": record.user_id,
            "model": record.model,
            "status": record.status.value,
            "prompt_hash": record.prompt_hash,
            "response_hash": record.response_hash or "",
            "prompt_tokens": pt,
            "response_tokens": rt,
            "latency_ms": lat,
            "timestamp": (record.timestamp.isoformat() if record.timestamp else ""),
            "pii_match_count": pii_count,
            "pii_highest_severity": pii_sev,
            "policy_action": policy_action,
            "policy_rule_name": policy_rule,
        }
