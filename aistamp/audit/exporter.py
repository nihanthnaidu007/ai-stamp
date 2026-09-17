from __future__ import annotations

import csv
import hmac as hmac_lib
import io
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, replace
from typing import IO, Any, Literal

from aistamp.audit.manifest import ExportManifest
from aistamp.fingerprint import RecordNotFoundError, sign_record
from aistamp.models import (
    AuditReport,
    PIISeverity,
    PIIType,
    PolicyAction,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.store.backend import StoreBackend

SignatureVerdict = Literal["VALID", "INVALID", "MISSING", "UNVERIFIED"]

# Number of records fetched per backend query while streaming an export.
_STREAM_PAGE_SIZE = 500

_CSV_BASE_FIELDNAMES = (
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
    "hmac_signature",
    "signature_verdict",
)
_CSV_PII_TYPE_FIELDNAMES = tuple(
    f"pii_{t.value.lower()}" for t in PIIType
) + ("pii_other",)
_CSV_FIELDNAMES = _CSV_BASE_FIELDNAMES + _CSV_PII_TYPE_FIELDNAMES


@dataclass(frozen=True)
class ExportRecord:
    """A record paired with its stored signature and verification verdict."""

    record: ProvenanceRecord
    hmac_signature: str | None
    signature_verdict: SignatureVerdict


def record_to_dict(record: ProvenanceRecord) -> dict[str, Any]:
    """Public record serialization for exports and CLI output."""
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


def pii_type_counts(record: ProvenanceRecord) -> dict[str, int]:
    """Count PII matches by pattern name across prompt and response matches."""
    if record.pii_result is None:
        return {}
    counts: dict[str, int] = {}
    for match in (
        record.pii_result.prompt_matches + record.pii_result.response_matches
    ):
        key = match.pattern_name.upper()
        counts[key] = counts.get(key, 0) + 1
    return counts


def signature_verdict(
    record: ProvenanceRecord,
    stored_signature: str | None,
    secret_key: str | None,
) -> SignatureVerdict:
    """Classify the integrity of a record's stored HMAC signature.

    - ``MISSING``: no signature was stored with the record.
    - ``UNVERIFIED``: a signature exists but no secret key was provided to
      verify it (exports must not silently imply a check that never ran).
    - ``VALID`` / ``INVALID``: compared against a freshly computed signature.
    """
    if stored_signature is None:
        return "MISSING"
    if secret_key is None:
        return "UNVERIFIED"
    expected = sign_record(record, secret_key)
    valid = hmac_lib.compare_digest(stored_signature, expected)
    return "VALID" if valid else "INVALID"


class AuditExporter:
    """
    Query the provenance store and export results as JSON or CSV.

    When constructed with a ``secret_key``, exports carry a per-record
    ``signature_verdict`` (VALID/INVALID); without one, records with a
    stored signature are reported as UNVERIFIED.
    """

    def __init__(self, backend: StoreBackend, secret_key: str | None = None) -> None:
        self._backend = backend
        self._secret_key = secret_key

    def query(self, filters: QueryFilters) -> AuditReport:
        return self._backend.query(filters)

    # ------------------------------------------------------------------
    # Streaming access
    # ------------------------------------------------------------------

    def iter_signed_records(self, filters: QueryFilters) -> Iterator[ExportRecord]:
        """Yield ExportRecord instances page by page (bounded memory).

        Signatures are fetched per record via ``backend.get`` because the
        current StoreBackend.query does not expose stored signatures; swap
        that out behind this method when the backend grows a native way to
        return them.
        """
        fetched = 0
        offset = filters.offset
        while True:
            page_size = min(_STREAM_PAGE_SIZE, filters.limit - fetched)
            if page_size <= 0:
                return
            page = self._backend.query(
                replace(filters, limit=page_size, offset=offset)
            )
            if not page.records:
                return
            for record in page.records:
                sig = self._signature_for(record)
                yield ExportRecord(
                    record=record,
                    hmac_signature=sig,
                    signature_verdict=signature_verdict(
                        record, sig, self._secret_key
                    ),
                )
            fetched += len(page.records)
            offset += len(page.records)
            if len(page.records) < page_size:
                return

    def iter_records(self, filters: QueryFilters) -> Iterator[ProvenanceRecord]:
        """Stream records matching the filters without loading the full page set."""
        for exported in self.iter_signed_records(filters):
            yield exported.record

    def _signature_for(self, record: ProvenanceRecord) -> str | None:
        result = self._backend.get(record.content_id)
        return result[1] if result is not None else None

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def to_json(
        self,
        report: AuditReport,
        indent: int = 2,
        manifest: ExportManifest | None = None,
    ) -> str:
        buffer = io.StringIO()
        self._write_json_stream(
            buffer,
            generated_at=report.generated_at.isoformat(),
            total_count=report.total_count,
            filters_applied=report.filters_applied,
            manifest=manifest,
            export_records=self._from_report(report),
        )
        return buffer.getvalue()

    def to_json_file(
        self,
        fp: IO[str],
        filters: QueryFilters,
        manifest: ExportManifest | None = None,
    ) -> int:
        """Stream the export as compact JSON to an open file handle.

        Same document shape as ``to_json`` (written compactly so memory
        stays bounded for large exports). Returns the number of records.
        """
        total = self._total_count(filters)
        return self._write_json_stream(
            fp,
            generated_at=total["generated_at"],
            total_count=total["count"],
            filters_applied=total["filters_applied"],
            manifest=manifest,
            export_records=self.iter_signed_records(filters),
        )

    def _total_count(self, filters: QueryFilters) -> dict[str, Any]:
        # A limit-0 query returns no rows but the full count; filters_applied
        # is rebuilt from the caller's original filters, not the probe.
        probe = self._backend.query(replace(filters, limit=0))
        return {
            "generated_at": probe.generated_at.isoformat(),
            "count": probe.total_count,
            "filters_applied": self._filters_applied(filters),
        }

    @staticmethod
    def _filters_applied(filters: QueryFilters) -> dict[str, Any]:
        """Mirror of the backend's filters_applied semantics for streaming exports."""
        return {
            k: (
                v.value
                if isinstance(v, (RecordStatus, PIISeverity, PolicyAction))
                else v
            )
            for k, v in asdict(filters).items()
            if v is not None
        }

    def _from_report(self, report: AuditReport) -> Iterator[ExportRecord]:
        for record in report.records:
            sig = self._signature_for(record)
            yield ExportRecord(
                record=record,
                hmac_signature=sig,
                signature_verdict=signature_verdict(record, sig, self._secret_key),
            )

    def _write_json_stream(
        self,
        fp: IO[str],
        *,
        generated_at: str,
        total_count: int,
        filters_applied: dict[str, Any],
        manifest: ExportManifest | None,
        export_records: Iterator[ExportRecord],
    ) -> int:
        count = 0
        fp.write("{")
        fp.write(json.dumps("generated_at") + ": " + json.dumps(generated_at) + ", ")
        fp.write(json.dumps("total_count") + ": " + json.dumps(total_count) + ", ")
        fp.write(
            json.dumps("filters_applied")
            + ": "
            + json.dumps(filters_applied, default=str)
            + ", "
        )
        if manifest is not None:
            fp.write(
                json.dumps("manifest")
                + ": "
                + json.dumps(manifest.to_dict(), default=str)
                + ", "
            )
        fp.write('"records": [')
        first = True
        for exported in export_records:
            if not first:
                fp.write(", ")
            first = False
            fp.write(json.dumps(self._signed_record_dict(exported), default=str))
            count += 1
        fp.write("]}")
        return count

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def to_csv(self, report: AuditReport) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        for exported in self._from_report(report):
            writer.writerow(self._record_to_csv_row(exported))
        return output.getvalue()

    def to_csv_file(self, fp: IO[str], filters: QueryFilters) -> int:
        """Stream the export as CSV to an open file handle. Returns record count."""
        writer = csv.DictWriter(fp, fieldnames=_CSV_FIELDNAMES)
        writer.writeheader()
        count = 0
        for exported in self.iter_signed_records(filters):
            writer.writerow(self._record_to_csv_row(exported))
            count += 1
        return count

    # ------------------------------------------------------------------
    # Record serialization
    # ------------------------------------------------------------------

    def _signed_record_dict(self, exported: ExportRecord) -> dict[str, Any]:
        data = record_to_dict(exported.record)
        data["hmac_signature"] = exported.hmac_signature
        data["signature_verdict"] = exported.signature_verdict
        data["pii_type_counts"] = pii_type_counts(exported.record)
        return data

    def _record_to_csv_row(self, exported: ExportRecord) -> dict[str, Any]:
        record = exported.record
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
        row: dict[str, Any] = {
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
            "hmac_signature": exported.hmac_signature or "",
            "signature_verdict": exported.signature_verdict,
        }
        counts = pii_type_counts(record)
        known = {t.value for t in PIIType}
        for t in PIIType:
            row[f"pii_{t.value.lower()}"] = counts.get(t.value, 0)
        row["pii_other"] = sum(
            n for name, n in counts.items() if name not in known
        )
        return row

    def signed_record_dict(
        self, record: ProvenanceRecord, hmac_signature: str | None
    ) -> dict[str, Any]:
        """Public record dict enriched with signature and verification status."""
        exported = ExportRecord(
            record=record,
            hmac_signature=hmac_signature,
            signature_verdict=signature_verdict(
                record, hmac_signature, self._secret_key
            ),
        )
        return self._signed_record_dict(exported)

    def _record_to_dict(self, record: ProvenanceRecord) -> dict[str, Any]:
        # Deprecated 0.1.x private alias — kept so existing callers that
        # reached into the exporter keep working. Use record_to_dict().
        return record_to_dict(record)

    # ------------------------------------------------------------------
    # Evidence pack
    # ------------------------------------------------------------------

    def evidence_pack(self, content_id: str) -> dict[str, Any]:
        """One-call compliance hand-off for a single record.

        Assembles the record, its stored signature, the verification
        verdict, PII detail, and the policy trace. Raises
        RecordNotFoundError when the content_id is unknown.
        """
        from aistamp.audit.evidence import build_evidence_pack

        result = self._backend.get(content_id)
        if result is None:
            raise RecordNotFoundError(content_id)
        record, stored_signature = result
        return build_evidence_pack(
            record=record,
            stored_signature=stored_signature,
            secret_key=self._secret_key,
        )
