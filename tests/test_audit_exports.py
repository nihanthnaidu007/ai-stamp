"""Export verification tests: signature-bearing exports, streaming, evidence
packs, manifests, and retention."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from aistamp.audit import (
    AuditExporter,
    build_export_manifest,
    enforce_retention,
    record_to_dict,
)
from aistamp.config import Config
from aistamp.fingerprint import generate_content_id, hash_content, sign_record
from aistamp.models import (
    PIIMatch,
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.store import SQLiteBackend

_SECRET = "test-secret-key-for-aistamp-unit-tests-32chars"


def _sample_record() -> ProvenanceRecord:
    """A minimal signed-ready record with one EMAIL PII hit."""
    return ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="export_app",
        feature_id="export_feature",
        user_id="export_user",
        model="gpt-4o",
        prompt_hash=hash_content("prompt"),
        response_hash=hash_content("response"),
        prompt_tokens=5,
        response_tokens=6,
        latency_ms=10.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=PIIResult(
            prompt_matches=[
                PIIMatch(
                    pattern_name="EMAIL",
                    severity=PIISeverity.MEDIUM,
                    start=0,
                    end=18,
                    redacted_snippet="[REDACTED]",
                )
            ],
            response_matches=[],
            highest_severity=PIISeverity.MEDIUM,
            match_count=1,
        ),
        policy_decision=PolicyDecision(
            action=PolicyAction.WARN,
            rule_name="r",
            reason="because",
        ),
    )


def _write_signed(
    backend: SQLiteBackend,
    record: ProvenanceRecord,
    secret: str = _SECRET,
) -> str:
    hmac_sig = sign_record(record, secret)
    backend.write(record, hmac_sig)
    return hmac_sig


def _file_backend(tmp_path: Path) -> SQLiteBackend:
    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'audit.db'}")
    backend.create_tables()
    return backend


def _all_filters() -> QueryFilters:
    return QueryFilters(limit=100, offset=0)


# ---------------------------------------------------------------------------
# JSON / CSV exports carry signatures that re-verify
# ---------------------------------------------------------------------------


def test_to_json_carries_signature_and_valid_verdict(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
) -> None:
    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    data = json.loads(exporter.to_json(report))

    assert data["records"][0]["hmac_signature"] == signed_record_in_store["hmac"]
    assert data["records"][0]["signature_verdict"] == "VALID"


def test_to_csv_carries_signature_and_verdict(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
) -> None:
    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    rows = list(csv.DictReader(io.StringIO(exporter.to_csv(report))))

    assert len(rows) == 1
    assert rows[0]["hmac_signature"] == signed_record_in_store["hmac"]
    assert rows[0]["signature_verdict"] == "VALID"


def test_to_csv_has_per_pii_type_breakdown_columns(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
) -> None:
    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    rows = list(csv.DictReader(io.StringIO(exporter.to_csv(report))))

    # The sample record carries one EMAIL match.
    assert rows[0]["pii_email"] == "1"
    assert rows[0]["pii_ssn"] == "0"
    assert rows[0]["pii_other"] == "0"


def test_tampered_record_exports_as_invalid(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    record = _sample_record()
    stored = _write_signed(backend, record)

    # Tamper directly in the DB, bypassing the API.
    db_path = tmp_path / "audit.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE provenance_records SET model = 'tampered-model' "
            "WHERE content_id = ?",
            (record.content_id,),
        )

    exporter = AuditExporter(backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    data = json.loads(exporter.to_json(report))
    assert data["records"][0]["hmac_signature"] == stored
    assert data["records"][0]["signature_verdict"] == "INVALID"


def test_unsigned_record_exports_as_missing(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    record = _sample_record()
    backend.write(record, None)

    exporter = AuditExporter(backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    data = json.loads(exporter.to_json(report))
    assert data["records"][0]["hmac_signature"] is None
    assert data["records"][0]["signature_verdict"] == "MISSING"


def test_unverifiable_without_secret_key(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
) -> None:
    exporter = AuditExporter(sqlite_backend)
    report = exporter.query(_all_filters())
    data = json.loads(exporter.to_json(report))
    assert data["records"][0]["signature_verdict"] == "UNVERIFIED"


def test_wrong_secret_reads_invalid(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    record = _sample_record()
    _write_signed(backend, record, secret=_SECRET)

    exporter = AuditExporter(backend, secret_key="a-different-secret-key-32-chars!!")
    report = exporter.query(_all_filters())
    data = json.loads(exporter.to_json(report))
    assert data["records"][0]["signature_verdict"] == "INVALID"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_iter_records_streams_lazily(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    for i in range(3):
        record = _sample_record().model_copy(update={"user_id": f"u{i}"})
        _write_signed(backend, record)

    exporter = AuditExporter(backend, secret_key=_SECRET)
    stream = exporter.iter_records(_all_filters())
    first = next(iter(stream))
    assert first.user_id == "u0"
    assert len(list(exporter.iter_records(_all_filters()))) == 3


def test_to_json_file_matches_in_memory_output(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    _write_signed(backend, _sample_record())

    exporter = AuditExporter(backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    in_memory = exporter.to_json(report)

    out = tmp_path / "export.json"
    with out.open("w") as fp:
        exporter.to_json_file(fp, _all_filters())
    # generated_at is stamped per call — compare payload equality instead of
    # byte equality.
    file_data = json.loads(out.read_text())
    memory_data = json.loads(in_memory)
    file_data.pop("generated_at")
    memory_data.pop("generated_at")
    assert file_data == memory_data
    assert file_data["records"][0]["signature_verdict"] == "VALID"


def test_to_csv_file_matches_in_memory_output(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    _write_signed(backend, _sample_record())

    exporter = AuditExporter(backend, secret_key=_SECRET)
    report = exporter.query(_all_filters())
    in_memory = exporter.to_csv(report)

    out = tmp_path / "export.csv"
    with out.open("w") as fp:
        written = exporter.to_csv_file(fp, _all_filters())
    assert written == 1
    file_rows = list(csv.DictReader(io.StringIO(out.read_text())))
    memory_rows = list(csv.DictReader(io.StringIO(in_memory)))
    assert file_rows == memory_rows
    assert file_rows[0]["signature_verdict"] == "VALID"


# ---------------------------------------------------------------------------
# record_to_dict (public) and audit-JSON parity
# ---------------------------------------------------------------------------


def test_record_to_dict_is_public_and_matches_exporter(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
) -> None:
    record = signed_record_in_store["record"]
    d = record_to_dict(record)
    assert d["content_id"] == record.content_id

    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    signed = exporter.signed_record_dict(record, signed_record_in_store["hmac"])
    assert signed["hmac_signature"] == signed_record_in_store["hmac"]
    assert signed["signature_verdict"] == "VALID"
    assert signed["pii_type_counts"] == {"EMAIL": 1}


# ---------------------------------------------------------------------------
# Evidence packs
# ---------------------------------------------------------------------------


def test_evidence_pack_is_complete_and_verifiable(
    signed_record_in_store: dict,
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    pack = exporter.evidence_pack(sample_provenance_record.content_id)

    assert pack["evidence_version"] == 1
    assert pack["content_id"] == sample_provenance_record.content_id
    assert pack["record"]["content_id"] == sample_provenance_record.content_id
    assert pack["hmac_signature"] == signed_record_in_store["hmac"]
    assert pack["verification"]["signature_verdict"] == "VALID"
    assert pack["pii_detail"]["match_count"] == 1
    assert pack["pii_detail"]["type_counts"] == {"EMAIL": 1}
    assert pack["policy_trace"] is not None
    assert pack["policy_trace"]["action"] == "WARN"

    # Independent re-verification: the write-time signing API reproduces the
    # signature carried in the pack.
    assert (
        sign_record(sample_provenance_record, _SECRET)
        == pack["hmac_signature"]
    )


def test_evidence_pack_unknown_id_raises(
    sqlite_backend: SQLiteBackend,
) -> None:
    from aistamp.fingerprint import RecordNotFoundError

    exporter = AuditExporter(sqlite_backend, secret_key=_SECRET)
    with pytest.raises(RecordNotFoundError):
        exporter.evidence_pack("no-such-content-id")


# ---------------------------------------------------------------------------
# Export manifest
# ---------------------------------------------------------------------------


def test_manifest_records_file_digests_and_version(tmp_path: Path) -> None:
    patterns = tmp_path / "patterns.yaml"
    patterns.write_text("patterns: []\n")
    policy = tmp_path / "policy.yaml"
    policy.write_text("rules: []\n")

    manifest = build_export_manifest(pattern_files=[patterns], policy_file=policy)
    data = manifest.to_dict()

    assert data["aistamp_version"]
    assert data["manifest_version"] == 1
    assert len(data["pattern_files"]) == 1
    assert data["pattern_files"][0]["path"].endswith("patterns.yaml")
    assert (
        data["pattern_files"][0]["sha256"]
        == hashlib.sha256(patterns.read_bytes()).hexdigest()
    )
    assert (
        data["policy_file"]["sha256"]
        == hashlib.sha256(policy.read_bytes()).hexdigest()
    )


def test_manifest_digest_changes_when_file_changes(tmp_path: Path) -> None:
    policy = tmp_path / "policy.yaml"
    policy.write_text("rules: []\n")
    before = build_export_manifest(policy_file=policy).to_dict()
    policy.write_text("rules:\n  - name: new\n    action: ALLOW\n")
    after = build_export_manifest(policy_file=policy).to_dict()

    assert before["policy_file"]["sha256"] != after["policy_file"]["sha256"]


# ---------------------------------------------------------------------------
# Retention
# ---------------------------------------------------------------------------


def test_retention_dry_run_then_delete(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    old_ts = datetime.now(timezone.utc) - timedelta(days=100)
    for i in range(2):
        record = _sample_record().model_copy(
            update={"user_id": f"u{i}", "timestamp": old_ts}
        )
        _write_signed(backend, record)

    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    assert enforce_retention(db_url, older_than_days=90, dry_run=True) == 2
    assert enforce_retention(db_url, older_than_days=90, dry_run=False) == 2
    assert enforce_retention(db_url, older_than_days=90, dry_run=True) == 0
    assert len(list(backend.query(_all_filters()).records)) == 0


def test_retention_scopes_to_app_id(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    old_ts = datetime.now(timezone.utc) - timedelta(days=100)
    for app in ("keep_app", "purge_app"):
        record = _sample_record().model_copy(
            update={"app_id": app, "timestamp": old_ts}
        )
        _write_signed(backend, record)

    db_url = f"sqlite:///{tmp_path / 'audit.db'}"
    deleted = enforce_retention(
        db_url, older_than_days=90, app_id="purge_app", dry_run=False
    )
    assert deleted == 1
    remaining = backend.query(_all_filters())
    assert remaining.total_count == 1
    assert remaining.records[0].app_id == "keep_app"


def test_retention_rejects_nonpositive_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="older_than_days"):
        enforce_retention(
            f"sqlite:///{tmp_path / 'audit.db'}", older_than_days=0
        )


def test_retention_accepts_config_database_url(tmp_path: Path) -> None:
    # The CLI passes config.database_url through the same helper; smoke-test
    # that path with a Config-built URL.
    config = Config(
        secret_key=_SECRET,
        database_url=f"sqlite:///{tmp_path / 'audit.db'}",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()
    old_ts = datetime.now(timezone.utc) - timedelta(days=100)
    _write_signed(
        backend,
        _sample_record().model_copy(update={"timestamp": old_ts}),
    )

    assert (
        enforce_retention(
            config.database_url, older_than_days=90, dry_run=True
        )
        == 1
    )
