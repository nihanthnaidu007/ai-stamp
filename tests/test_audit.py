from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone

from aistamp.audit import AuditExporter
from aistamp.fingerprint import generate_content_id
from aistamp.models import (
    AuditReport,
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.store import SQLiteBackend


def _make_record(
    *,
    user_id: str = "u",
    model: str = "gpt-4o",
    pii_result: PIIResult | None = None,
    policy_decision: PolicyDecision | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="a",
        feature_id="f",
        user_id=user_id,
        model=model,
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=pii_result,
        policy_decision=policy_decision,
    )


# ---------------------------------------------------------------------------
# Group 1 — AuditExporter.query
# ---------------------------------------------------------------------------


def test_query_returns_audit_report(sqlite_backend: SQLiteBackend) -> None:
    # AuditExporter.query must return an AuditReport instance.
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(QueryFilters())
    assert isinstance(report, AuditReport)


def test_query_empty_store_returns_empty_report(sqlite_backend: SQLiteBackend) -> None:
    # Querying an empty store must return AuditReport with records=[] and total_count=0.
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(QueryFilters())
    assert report.records == []
    assert report.total_count == 0


def test_query_returns_written_record(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # After writing one record, query must return it.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(QueryFilters())
    assert report.total_count == 1


def test_query_filters_by_user_id(sqlite_backend: SQLiteBackend) -> None:
    # QueryFilters(user_id=...) must return only records for that user.
    sqlite_backend.write(_make_record(user_id="alice"), hmac_signature="h")
    sqlite_backend.write(_make_record(user_id="bob"), hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(QueryFilters(user_id="alice"))
    assert len(report.records) == 1


def test_query_total_count_reflects_pre_pagination_size(
    sqlite_backend: SQLiteBackend,
) -> None:
    # total_count must be the total matching records, not just the page size.
    for _ in range(5):
        sqlite_backend.write(_make_record(user_id="multi"), hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(QueryFilters(user_id="multi", limit=2))
    assert report.total_count == 5
    assert len(report.records) == 2


def test_query_filters_by_pii_severity(sqlite_backend: SQLiteBackend) -> None:
    high = PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=PIISeverity.HIGH,
        match_count=1,
    )
    sqlite_backend.write(_make_record(pii_result=high), hmac_signature="h")
    sqlite_backend.write(_make_record(), hmac_signature="h")
    report = sqlite_backend.query(QueryFilters(pii_severity=PIISeverity.HIGH))
    assert report.total_count == 1


def test_query_filters_by_policy_decision(sqlite_backend: SQLiteBackend) -> None:
    blocked = PolicyDecision(
        action=PolicyAction.BLOCK,
        rule_name="deny",
        reason="blocked",
    )
    sqlite_backend.write(_make_record(policy_decision=blocked), hmac_signature="h")
    sqlite_backend.write(_make_record(), hmac_signature="h")
    report = sqlite_backend.query(QueryFilters(policy_decision=PolicyAction.BLOCK))
    assert report.total_count == 1


# ---------------------------------------------------------------------------
# Group 2 — AuditExporter.to_json
# ---------------------------------------------------------------------------


def test_to_json_returns_valid_json_string(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # to_json must return a string that parses as valid JSON without error.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    s = exporter.to_json(exporter.query(QueryFilters()))
    json.loads(s)


def test_to_json_contains_total_count(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # Parsed JSON must contain a "total_count" key with the correct integer value.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    assert data["total_count"] == 1


def test_to_json_contains_records_array(sqlite_backend: SQLiteBackend) -> None:
    # Parsed JSON must contain a "records" array.
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    assert isinstance(data["records"], list)


def test_to_json_record_has_expected_keys(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # Each record in the JSON array must have content_id, app_id, model, status fields.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    record = data["records"][0]
    for key in ("content_id", "app_id", "model", "status"):
        assert key in record


def test_to_json_status_is_string_not_enum(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # Record status in JSON must be a string ("COMPLETED"), not an enum object.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    assert data["records"][0]["status"] == "COMPLETED"


def test_to_json_handles_empty_report(sqlite_backend: SQLiteBackend) -> None:
    # to_json on an empty AuditReport must return valid JSON with records=[].
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    assert data["records"] == []


# ---------------------------------------------------------------------------
# Group 3 — AuditExporter.to_csv
# ---------------------------------------------------------------------------


def test_to_csv_returns_string(sqlite_backend: SQLiteBackend) -> None:
    # to_csv must return a string.
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    assert isinstance(out, str)


def test_to_csv_first_row_is_header(sqlite_backend: SQLiteBackend) -> None:
    # The first line of the CSV must be the header row containing "content_id".
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    first_line = out.splitlines()[0]
    assert "content_id" in first_line


def test_to_csv_has_correct_number_of_rows(
    sqlite_backend: SQLiteBackend,
) -> None:
    # A report with N records must produce N+1 rows (header + N data rows).
    for _ in range(3):
        sqlite_backend.write(_make_record(), hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    rows = list(csv.reader(io.StringIO(out)))
    assert len(rows) == 4


def test_to_csv_record_fields_are_correct(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # The content_id column must match the record's content_id.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    reader = csv.DictReader(io.StringIO(out))
    row = next(reader)
    assert row["content_id"] == sample_provenance_record.content_id


def test_to_csv_handles_null_optional_fields(sqlite_backend: SQLiteBackend) -> None:
    # A record with response_hash=None must not raise. Null fields become empty strings.
    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="a",
        feature_id="f",
        user_id="u",
        model="gpt-4o",
        prompt_hash="a" * 64,
        response_hash=None,
        prompt_tokens=None,
        response_tokens=None,
        latency_ms=None,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )
    sqlite_backend.write(record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    reader = csv.DictReader(io.StringIO(out))
    row = next(reader)
    assert row["response_hash"] == ""
    assert row["latency_ms"] == ""


def test_to_csv_pii_match_count_is_zero_for_no_pii(
    sqlite_backend: SQLiteBackend,
) -> None:
    # A record with pii_result=None must have pii_match_count=0 in CSV output.
    sqlite_backend.write(_make_record(), hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    reader = csv.DictReader(io.StringIO(out))
    row = next(reader)
    assert row["pii_match_count"] == "0"


def test_to_csv_pii_highest_severity_empty_for_no_pii(
    sqlite_backend: SQLiteBackend,
) -> None:
    # A record with pii_result=None must have pii_highest_severity="" in CSV output.
    sqlite_backend.write(_make_record(), hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    reader = csv.DictReader(io.StringIO(out))
    row = next(reader)
    assert row["pii_highest_severity"] == ""


# ---------------------------------------------------------------------------
# Group 4 — round-trip: write, query, export
# ---------------------------------------------------------------------------


def test_write_query_json_export_roundtrip(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # Write a record, query it, export to JSON, parse JSON, confirm content_id matches.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    data = json.loads(exporter.to_json(exporter.query(QueryFilters())))
    assert data["records"][0]["content_id"] == sample_provenance_record.content_id


def test_write_query_csv_export_roundtrip(
    sqlite_backend: SQLiteBackend, sample_provenance_record: ProvenanceRecord
) -> None:
    # Export a written record and confirm content_id in the first CSV data row.
    sqlite_backend.write(sample_provenance_record, hmac_signature="h")
    exporter = AuditExporter(sqlite_backend)
    out = exporter.to_csv(exporter.query(QueryFilters()))
    reader = csv.DictReader(io.StringIO(out))
    row = next(reader)
    assert row["content_id"] == sample_provenance_record.content_id
