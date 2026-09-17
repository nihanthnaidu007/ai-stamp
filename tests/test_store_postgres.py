from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url

from aistamp.models import ProvenanceRecord, QueryFilters, RecordStatus
from aistamp.store.async_backend import AsyncPostgreSQLBackend
from aistamp.store.backend import PostgreSQLBackend

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "aistamp" / "alembic.ini"

_PINNED_COLUMNS = (
    "key_id",
    "sig_algo",
    "record_version",
    "prev_hash",
    "scope_sequence",
    "error_type",
    "error_message",
)

POSTGRES_URL = os.environ.get("AISTAMP_TEST_POSTGRES_URL", "")
ASYNC_POSTGRES_URL = (
    POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://")
    if POSTGRES_URL
    else ""
)

pytestmark = pytest.mark.skipif(
    not POSTGRES_URL,
    reason=(
        "AISTAMP_TEST_POSTGRES_URL not set — skipping PostgreSQL integration tests"
    ),
)


def _make_record(**overrides) -> ProvenanceRecord:
    base = dict(
        content_id=str(uuid.uuid4()),
        app_id="pg_test",
        feature_id="f",
        user_id="u",
        model="gpt-4o",
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )
    base.update(overrides)
    return ProvenanceRecord(**base)


# --- sync PostgreSQLBackend ------------------------------------------------


def test_postgres_backend_create_tables() -> None:
    # create_tables() must run without error against a live PostgreSQL instance.
    backend = PostgreSQLBackend(POSTGRES_URL)
    backend.create_tables()


def test_postgres_backend_write_and_get() -> None:
    # Write a record and get it back. Assert content_id matches.
    backend = PostgreSQLBackend(POSTGRES_URL)
    backend.create_tables()
    record = _make_record()
    backend.write(record, "pg_hmac")
    result = backend.get(record.content_id)
    assert result is not None
    fetched, hmac = result
    assert fetched.content_id == record.content_id
    assert hmac == "pg_hmac"


def test_postgres_backend_query() -> None:
    # Write 3 records. Query with no filters. Assert total_count >= 3.
    backend = PostgreSQLBackend(POSTGRES_URL)
    backend.create_tables()
    marker = f"pg_query_{uuid.uuid4().hex[:8]}"
    for _ in range(3):
        backend.write(_make_record(user_id=marker), "h")
    report = backend.query(QueryFilters(user_id=marker))
    assert report.total_count >= 3


def test_postgres_backend_connection_pooling() -> None:
    # Write 10 records in a loop to exercise connection pool.
    backend = PostgreSQLBackend(POSTGRES_URL, pool_size=3, max_overflow=2)
    backend.create_tables()
    marker = f"pg_pool_{uuid.uuid4().hex[:8]}"
    for _ in range(10):
        backend.write(_make_record(user_id=marker), "h")
    report = backend.query(QueryFilters(user_id=marker))
    assert report.total_count == 10


# --- async AsyncPostgreSQLBackend ------------------------------------------


@pytest_asyncio.fixture
async def async_pg_backend():
    backend = AsyncPostgreSQLBackend(ASYNC_POSTGRES_URL)
    await backend.create_tables()
    yield backend
    await backend._engine.dispose()


@pytest.mark.asyncio
async def test_async_postgres_backend_write_and_get(async_pg_backend) -> None:
    # Async write and get roundtrip against live PostgreSQL.
    record = _make_record()
    await async_pg_backend.write(record, "async_pg_hmac")
    result = await async_pg_backend.get(record.content_id)
    assert result is not None
    fetched, hmac = result
    assert fetched.content_id == record.content_id
    assert hmac == "async_pg_hmac"


@pytest.mark.asyncio
async def test_async_postgres_backend_query(async_pg_backend) -> None:
    # Async query returns correct total_count.
    marker = f"async_pg_q_{uuid.uuid4().hex[:8]}"
    for _ in range(3):
        await async_pg_backend.write(_make_record(user_id=marker), "h")
    report = await async_pg_backend.query(QueryFilters(user_id=marker))
    assert report.total_count == 3


# --- storage v2: pinned columns, JSONB/GIN, keyset, purge, migration --------


def test_postgres_pinned_columns_jsonb_and_gin() -> None:
    backend = PostgreSQLBackend(POSTGRES_URL, pool_size=3, max_overflow=2)
    backend.create_tables()
    try:
        inspector = inspect(backend._engine)
        columns = {
            c["name"]: c["type"] for c in inspector.get_columns("provenance_records")
        }
        for name in _PINNED_COLUMNS:
            assert name in columns
        assert type(columns["pii_result"]).__name__ == "JSONB"
        assert type(columns["policy_decision"]).__name__ == "JSONB"

        with backend._engine.connect() as conn:
            index_defs = dict(
                conn.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes "
                        "WHERE tablename = 'provenance_records'"
                    )
                ).all()
            )
        assert "USING gin" in index_defs["ix_provenance_records_pii_result_gin"]
        assert "ix_provenance_records_policy_decision_gin" in index_defs
        assert "ix_provenance_records_app_id_timestamp" in index_defs
        assert (
            "(app_id, timestamp)"
            in index_defs["ix_provenance_records_app_id_timestamp"]
        )
        assert (
            "(user_id, timestamp)"
            in index_defs["ix_provenance_records_user_id_timestamp"]
        )
        assert "ix_provenance_records_feature_id" in index_defs
    finally:
        backend.close()


def test_postgres_deterministic_keyset_walk() -> None:
    backend = PostgreSQLBackend(POSTGRES_URL, pool_size=3, max_overflow=2)
    backend.create_tables()
    try:
        marker = f"pg_ks_{uuid.uuid4().hex[:8]}"
        base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
        for i in range(20):
            backend.write(
                _make_record(
                    user_id=marker, timestamp=base_ts + timedelta(minutes=i % 3)
                ),
                "h",
            )

        seen: list[str] = []
        cursor: str | None = None
        while True:
            filters = (
                QueryFilters(user_id=marker, cursor=cursor, limit=7)
                if cursor
                else QueryFilters(user_id=marker, limit=7)
            )
            report = backend.query(filters)
            seen.extend(r.content_id for r in report.records)
            if report.next_cursor is None:
                break
            cursor = report.next_cursor
        assert len(seen) == len(set(seen)) == 20

        first = [
            r.content_id
            for r in backend.query(QueryFilters(user_id=marker, limit=7)).records
        ]
        second = [
            r.content_id
            for r in backend.query(QueryFilters(user_id=marker, limit=7)).records
        ]
        assert first == second
    finally:
        backend.close()


def test_postgres_purge_and_write_ahead() -> None:
    backend = PostgreSQLBackend(POSTGRES_URL, pool_size=3, max_overflow=2)
    backend.create_tables()
    try:
        now = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
        marker = f"pg_purge_{uuid.uuid4().hex[:8]}"
        old_ids: list[str] = []
        for _ in range(2):
            record = _make_record(user_id=marker, timestamp=now - timedelta(days=30))
            old_ids.append(record.content_id)
            backend.write(record, None)

        pending = _make_record(
            user_id=marker, status=RecordStatus.PENDING, timestamp=now
        )
        backend.write(pending, None)
        assert backend.get(pending.content_id) is not None

        assert backend.purge(7, now=now) >= 2
        assert backend.get(old_ids[0]) is None
        # P1-5: the purge must be journaled so chain-linked deletion is
        # detectable — an unanchored chain gap is tampering evidence.
        anchors = backend.list_purge_anchors()
        assert anchors, "every purge must write a chain anchor"
        assert sum(anchor.purged_count for anchor in anchors) >= 2

        final = pending.model_copy(update={"status": RecordStatus.COMPLETED})
        backend.finalize(pending.content_id, final, "h")
        got_final, hmac = backend.get(pending.content_id) or (None, None)
        assert got_final is not None
        assert got_final.status == RecordStatus.COMPLETED
        assert hmac == "h"
    finally:
        backend.close()


def test_postgres_migration_up_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Migration 0002 up/down against a throwaway PostgreSQL database."""
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    base_url = make_url(POSTGRES_URL)
    db_name = f"aistamp_mig_{uuid.uuid4().hex[:10]}"
    admin_url = str(base_url.set(database="postgres"))
    mig_url = str(base_url.set(database=db_name))

    admin_engine = create_engine(admin_url, isolation_level="AUTOCOMMIT")
    with admin_engine.connect() as conn:
        conn.execute(text(f'CREATE DATABASE "{db_name}"'))
    inspection_engine = create_engine(mig_url)

    def _index_defs() -> dict[str, str]:
        with inspection_engine.connect() as conn:
            return dict(
                conn.execute(
                    text(
                        "SELECT indexname, indexdef FROM pg_indexes "
                        "WHERE tablename = 'provenance_records'"
                    )
                ).all()
            )

    try:
        cfg = Config(str(ALEMBIC_INI))
        cfg.set_main_option("sqlalchemy.url", mig_url)
        command.upgrade(cfg, "head")

        columns = {
            c["name"]: c["type"]
            for c in inspect(inspection_engine).get_columns("provenance_records")
        }
        for name in _PINNED_COLUMNS:
            assert name in columns
        assert type(columns["pii_result"]).__name__ == "JSONB"
        index_defs = _index_defs()
        assert "USING gin" in index_defs["ix_provenance_records_pii_result_gin"]
        assert "ix_provenance_records_policy_decision_gin" in index_defs
        assert "ix_provenance_records_app_id_timestamp" in index_defs
        assert "ix_provenance_records_user_id_timestamp" in index_defs
        assert "ix_provenance_records_feature_id" in index_defs

        command.downgrade(cfg, "0001")
        columns_after = {
            c["name"]
            for c in inspect(inspection_engine).get_columns("provenance_records")
        }
        assert not (set(_PINNED_COLUMNS) & columns_after)
        index_defs_after = _index_defs()
        for dropped in (
            "ix_provenance_records_app_id_timestamp",
            "ix_provenance_records_user_id_timestamp",
            "ix_provenance_records_feature_id",
            "ix_provenance_records_pii_result_gin",
            "ix_provenance_records_policy_decision_gin",
        ):
            assert dropped not in index_defs_after

        command.upgrade(cfg, "head")  # re-upgrade after downgrade must work
    finally:
        inspection_engine.dispose()
        with admin_engine.connect() as conn:
            conn.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :db"
                ),
                {"db": db_name},
            )
            conn.execute(text(f'DROP DATABASE "{db_name}"'))
        admin_engine.dispose()
