"""Integration gate: alembic-managed (production) schema vs the ORM envelope.

The 0.1.x compatibility promise covers the documented production path:
``aistamp migrate`` builds the database from the packaged alembic
migrations, after which ORM writes (``persist_record`` -> ``backend.write``)
and reads must succeed. The v0.2 envelope columns (key_id, sig_algo,
record_version, prev_hash, scope_sequence) exist in the ORM on this branch
but reach the migration schema only via the storage track's migration 0002,
so this test skips with an explicit merge-gate reason until that revision
is on the branch, then asserts the full write/verify round-trip on a
migrated (not ``create_tables()``) database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, inspect
from typer.testing import CliRunner

from aistamp.cli.main import app
from aistamp.fingerprint import (
    generate_content_id,
    hash_content,
    sign_record,
    verify_record,
)
from aistamp.models import ProvenanceRecord, RecordStatus, SignatureStatus
from aistamp.store import SQLiteBackend

runner = CliRunner()

# Columns the tamper-evidence envelope requires in the migrated schema.
_PINNED_COLUMNS = {
    "key_id",
    "sig_algo",
    "record_version",
    "prev_hash",
    "scope_sequence",
}


def _migrate(tmp_path: Path) -> str:
    """Run the documented production path: ``aistamp migrate --config <yaml>``."""
    db_path = tmp_path / "prod.db"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"secret_key: {'x' * 32}\ndatabase_url: sqlite:///{db_path}\n"
    )
    result = runner.invoke(app, ["migrate", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    return f"sqlite:///{db_path}"


def test_migrated_schema_supports_envelope_round_trip(tmp_path: Path) -> None:
    url = _migrate(tmp_path)

    engine = create_engine(url)
    columns = {col["name"] for col in inspect(engine).get_columns("provenance_records")}
    engine.dispose()
    missing = _PINNED_COLUMNS - columns
    if missing:
        pytest.skip(
            "merge-gated behind the storage track's migration 0002 — "
            f"missing columns: {sorted(missing)}"
        )

    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="app",
        feature_id="feat",
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("prompt"),
        response_hash=hash_content("response"),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )
    backend = SQLiteBackend(url)
    key = "k" * 32
    backend.write(record, sign_record(record, key))
    result = verify_record(record.content_id, "response", backend, key)
    assert result.verified is True
    assert result.key_id == "default"
    assert result.signature_status == SignatureStatus.VALID
