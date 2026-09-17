"""Storage v2: deterministic queries, keyset pagination, retention,
write-ahead audit, buffered writes, lifecycle, migration 0002."""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import text

from aistamp.audit.exporter import AuditExporter
from aistamp.models import ProvenanceRecord, QueryFilters, RecordStatus
from aistamp.store import AsyncBufferedWriter, BufferedWriter, SQLiteBackend
from aistamp.store.async_backend import AsyncSQLiteBackend
from aistamp.store.schema import ProvenanceRecordORM

ALEMBIC_INI = Path(__file__).resolve().parents[1] / "aistamp" / "alembic.ini"


def _make_record(**overrides: Any) -> ProvenanceRecord:
    base = dict(
        content_id=str(uuid.uuid4()),
        app_id="v2_app",
        feature_id="f1",
        user_id="u1",
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


@pytest.fixture
def file_backend(tmp_path: Path) -> SQLiteBackend:
    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'v2.db'}")
    backend.create_tables()
    yield backend
    backend.close()


# --- Deterministic ordering -------------------------------------------------


def test_query_orders_by_timestamp_then_id(file_backend: SQLiteBackend) -> None:
    base_ts = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    inserted: list[tuple[str, datetime, int]] = []  # (content_id, timestamp, seq)
    for i in range(6):
        # Two distinct timestamps, three records each — ties break by id.
        ts = base_ts - timedelta(minutes=i % 2)
        record = _make_record(timestamp=ts)
        file_backend.write(record, None)
        inserted.append((record.content_id, ts, i))

    first = [r.content_id for r in file_backend.query(QueryFilters()).records]
    second = [r.content_id for r in file_backend.query(QueryFilters()).records]
    assert first == second, "repeated identical queries must return identical order"

    # Expectation: (timestamp asc, insertion order); insertion order equals
    # id order because rows are inserted sequentially.
    expected = [
        content_id
        for content_id, _ts, _seq in sorted(inserted, key=lambda p: (p[1], p[2]))
    ]
    assert [
        r.content_id for r in file_backend.query(QueryFilters()).records
    ] == expected


def test_query_is_stable_across_sessions(tmp_path: Path) -> None:
    """A brand-new backend instance must produce the same page for the
    same filters — the offset-pagination determinism regression."""
    url = f"sqlite:///{tmp_path / 'stable.db'}"
    backend = SQLiteBackend(url)
    backend.create_tables()
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    for i in range(15):
        backend.write(_make_record(timestamp=base_ts + timedelta(seconds=i)), None)

    page_one = backend.query(QueryFilters(limit=5, offset=0)).records
    backend.close()

    reopened = SQLiteBackend(url)
    reopened_page_one = reopened.query(QueryFilters(limit=5, offset=0)).records
    reopened.close()
    assert [r.content_id for r in page_one] == [r.content_id for r in reopened_page_one]


# --- Keyset pagination -------------------------------------------------------


def test_keyset_walk_has_no_skips_or_repeats(file_backend: SQLiteBackend) -> None:
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    for i in range(25):
        file_backend.write(
            _make_record(timestamp=base_ts + timedelta(minutes=i % 4)), None
        )

    seen: list[str] = []
    pages = 0
    cursor: str | None = None
    while True:
        filters = (
            QueryFilters(cursor=cursor, limit=10) if cursor else QueryFilters(limit=10)
        )
        report = file_backend.query(filters)
        assert report.total_count == 25, "total_count must ignore the keyset cursor"
        seen.extend(r.content_id for r in report.records)
        pages += 1
        next_cursor = report.next_cursor
        if next_cursor is None:
            break
        cursor = next_cursor

    assert pages == 3, "25 records / limit 10 => 3 pages"
    assert len(seen) == len(set(seen)) == 25, "no skips, no repeats"


def test_keyset_typed_pair_matches_cursor_walk(file_backend: SQLiteBackend) -> None:
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    for i in range(12):
        file_backend.write(_make_record(timestamp=base_ts + timedelta(seconds=i)), None)

    first = file_backend.query(QueryFilters(limit=7))
    assert first.next_cursor is not None
    last_record = first.records[-1]
    with file_backend._engine.connect() as conn:
        last_id = conn.execute(
            text("SELECT id FROM provenance_records WHERE content_id = :c"),
            {"c": last_record.content_id},
        ).scalar_one()

    via_cursor = file_backend.query(QueryFilters(cursor=first.next_cursor, limit=7))
    via_typed = file_backend.query(
        QueryFilters(
            after_timestamp=last_record.timestamp,
            after_id=int(last_id),
            limit=7,
        )
    )
    assert [r.content_id for r in via_cursor.records] == [
        r.content_id for r in via_typed.records
    ]


def test_next_cursor_absent_when_no_more_rows(file_backend: SQLiteBackend) -> None:
    for _ in range(3):
        file_backend.write(_make_record(), None)
    report = file_backend.query(QueryFilters(limit=1000))
    assert report.next_cursor is None
    assert len(report.records) == 3


def test_malformed_cursor_raises_value_error(file_backend: SQLiteBackend) -> None:
    for _ in range(3):
        file_backend.write(_make_record(), None)
    with pytest.raises(ValueError, match="Malformed pagination cursor"):
        file_backend.query(QueryFilters(cursor="garbage"))


# --- QueryFilters validation -------------------------------------------------


def test_limit_clamped_to_max() -> None:
    assert QueryFilters(limit=5000).limit == QueryFilters.MAX_LIMIT
    assert QueryFilters().limit == 100


@pytest.mark.parametrize("bad_limit", [0, -1, -100])
def test_non_positive_limit_rejected(bad_limit: int) -> None:
    with pytest.raises(ValidationError):
        QueryFilters(limit=bad_limit)


def test_negative_offset_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryFilters(offset=-1)


def test_cursor_and_after_pair_mutually_exclusive() -> None:
    ts = datetime(2026, 9, 17, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        QueryFilters(cursor="2026-09-17T00:00:00|1", after_timestamp=ts, after_id=1)


def test_after_pair_requires_both_fields() -> None:
    ts = datetime(2026, 9, 17, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        QueryFilters(after_timestamp=ts)
    with pytest.raises(ValidationError):
        QueryFilters(after_id=3)


def test_query_filters_is_frozen() -> None:
    filters = QueryFilters()
    with pytest.raises(ValidationError):
        filters.limit = 5  # type: ignore[misc]


# --- Schema: pinned columns and indexes ---------------------------------------


def test_pinned_columns_in_orm_metadata() -> None:
    columns = {c.name for c in ProvenanceRecordORM.__table__.columns}
    assert {
        "key_id",
        "sig_algo",
        "record_version",
        "prev_hash",
        "scope_sequence",
        "error_type",
        "error_message",
    } <= columns


def test_composite_and_feature_indexes_in_metadata() -> None:
    index_names = {i.name for i in ProvenanceRecordORM.__table__.indexes}
    assert "ix_provenance_records_app_id_timestamp" in index_names
    assert "ix_provenance_records_user_id_timestamp" in index_names
    assert "ix_provenance_records_feature_id" in index_names
    assert "ix_provenance_records_pii_result_gin" in index_names
    assert "ix_provenance_records_policy_decision_gin" in index_names


def test_pinned_column_defaults(file_backend: SQLiteBackend) -> None:
    record = _make_record()  # no pinned fields passed
    file_backend.write(record, "hmac")
    fetched, _ = file_backend.get(record.content_id) or (None, None)
    assert fetched is not None
    assert fetched.key_id == "default"
    assert fetched.sig_algo == "HMAC-SHA256"
    # v2 is the secure default on fresh records: its canonical payload binds
    # the envelope and chain fields into the HMAC (tamper-evidence P0-1 fix).
    # Pre-existing rows are backfilled to 1 by migration 0002 — that path is
    # pinned separately in test_migrate_compat.py.
    assert fetched.record_version == 2
    assert fetched.prev_hash is None
    assert fetched.scope_sequence is None
    assert fetched.error_type is None
    assert fetched.error_message is None


def test_pinned_columns_roundtrip(file_backend: SQLiteBackend) -> None:
    record = _make_record(
        key_id="key-2026-09",
        sig_algo="HMAC-SHA256",
        record_version=1,
        prev_hash="c" * 64,
        scope_sequence=42,
        error_type="ProviderTimeout",
        error_message="upstream timed out after 30s",
    )
    file_backend.write(record, "hmac")
    fetched, hmac = file_backend.get(record.content_id) or (None, None)
    assert hmac == "hmac"
    assert fetched is not None
    assert fetched == record


# --- SQLite production posture -----------------------------------------------


def test_sqlite_wal_mode(tmp_path: Path) -> None:
    db_path = tmp_path / "wal.db"
    backend = SQLiteBackend(f"sqlite:///{db_path}")
    backend.create_tables()
    try:
        raw = sqlite3.connect(db_path)
        try:
            mode = raw.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            raw.close()
        assert mode == "wal"
    finally:
        backend.close()


def test_sqlite_busy_timeout(tmp_path: Path) -> None:
    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'busy.db'}", busy_timeout_ms=1234)
    backend.create_tables()
    try:
        with backend._engine.connect() as conn:
            timeout = conn.exec_driver_sql("PRAGMA busy_timeout").scalar()
        assert timeout == 1234
    finally:
        backend.close()


def test_sqlite_writes_from_multiple_threads(tmp_path: Path) -> None:
    """check_same_thread=False + WAL: pooled connections must survive
    cross-thread reuse without 'SQLite objects created in a thread' errors."""
    import threading

    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'threads.db'}")
    backend.create_tables()
    try:
        errors: list[Exception] = []

        def stamp() -> None:
            try:
                for _ in range(10):
                    backend.write(_make_record(), None)
            except Exception as exc:  # noqa: BLE001 - collected and asserted
                errors.append(exc)

        threads = [threading.Thread(target=stamp) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        assert backend.query(QueryFilters(limit=1000)).total_count == 40
    finally:
        backend.close()


# --- Write-ahead crash-safe audit --------------------------------------------


def test_write_ahead_pending_then_finalize(file_backend: SQLiteBackend) -> None:
    pending = _make_record(status=RecordStatus.PENDING, response_hash=None)
    file_backend.write(pending, None)

    got_pending = file_backend.get(pending.content_id)
    assert got_pending is not None and got_pending[0].status == RecordStatus.PENDING

    final = pending.model_copy(
        update={"status": RecordStatus.COMPLETED, "response_hash": "d" * 64}
    )
    file_backend.finalize(pending.content_id, final, "hmac-final")

    got_final, hmac = file_backend.get(pending.content_id) or (None, None)
    assert got_final is not None and got_final.status == RecordStatus.COMPLETED
    assert hmac == "hmac-final"
    assert got_final.response_hash == "d" * 64

    pending_report = file_backend.query(QueryFilters(status=RecordStatus.PENDING))
    assert pending_report.total_count == 0


def test_finalize_inserts_when_pending_row_missing(
    file_backend: SQLiteBackend,
) -> None:
    record = _make_record()
    file_backend.finalize(record.content_id, record, "hmac")
    fetched, hmac = file_backend.get(record.content_id) or (None, None)
    assert fetched == record and hmac == "hmac"


def test_finalize_rejects_content_id_mismatch(file_backend: SQLiteBackend) -> None:
    record = _make_record()
    with pytest.raises(ValueError, match="does not match"):
        file_backend.finalize("other-content-id", record)


# --- Bulk and buffered writes -------------------------------------------------


def test_write_many_bulk_insert(file_backend: SQLiteBackend) -> None:
    pairs = [(_make_record(), f"hmac-{i}") for i in range(50)]
    file_backend.write_many(pairs)
    report = file_backend.query(QueryFilters(limit=1000))
    assert report.total_count == 50
    assert all(r.content_id for r in report.records)


def test_write_many_empty_is_noop(file_backend: SQLiteBackend) -> None:
    file_backend.write_many([])
    assert file_backend.query(QueryFilters()).total_count == 0


def test_buffered_writer_autoflush_batches(file_backend: SQLiteBackend) -> None:
    with patch.object(file_backend, "write_many", wraps=file_backend.write_many) as spy:
        writer = BufferedWriter(file_backend, max_buffer_size=5)
        for i in range(12):
            writer.add(_make_record(), f"bh-{i}")
        assert spy.call_count == 2, "flush at 5 and 10"
        writer.flush()
    assert spy.call_count == 3
    assert [len(call.args[0]) for call in spy.call_args_list] == [5, 5, 2]
    assert file_backend.query(QueryFilters(limit=1000)).total_count == 12


def test_buffered_writer_context_manager_flushes(file_backend: SQLiteBackend) -> None:
    with BufferedWriter(file_backend, max_buffer_size=100) as writer:
        for _ in range(3):
            writer.add(_make_record(), None)
    # No explicit flush — exit must have flushed the 3 buffered records.
    assert file_backend.query(QueryFilters(limit=1000)).total_count == 3


def test_buffered_writer_rejects_bad_size(file_backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="max_buffer_size"):
        BufferedWriter(file_backend, max_buffer_size=0)


def test_buffered_writer_close_flushes_but_keeps_backend_open(
    file_backend: SQLiteBackend,
) -> None:
    writer = BufferedWriter(file_backend, max_buffer_size=10)
    writer.add(_make_record(), None)
    writer.close()
    assert file_backend.query(QueryFilters()).total_count == 1
    # Backend stays usable — the writer does not own the backend's lifecycle.
    file_backend.write(_make_record(), None)
    assert file_backend.query(QueryFilters()).total_count == 2


# --- Retention ----------------------------------------------------------------


def test_purge_deletes_only_expired_records(file_backend: SQLiteBackend) -> None:
    now = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    old_ts = now - timedelta(days=10)
    new_ts = now - timedelta(days=1)
    for _ in range(3):
        file_backend.write(_make_record(timestamp=old_ts), None)
    for _ in range(2):
        file_backend.write(_make_record(timestamp=new_ts), None)

    purged = file_backend.purge(5, now=now)
    assert purged == 3
    report = file_backend.query(QueryFilters(limit=1000))
    assert report.total_count == 2
    assert all(
        r.timestamp.replace(tzinfo=timezone.utc) >= now - timedelta(days=5)
        for r in report.records
    )


def test_purge_zero_retention_deletes_everything(
    file_backend: SQLiteBackend,
) -> None:
    now = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    file_backend.write(_make_record(timestamp=now - timedelta(seconds=1)), None)
    assert file_backend.purge(0, now=now) == 1
    assert file_backend.query(QueryFilters()).total_count == 0


def test_purge_rejects_negative_retention(file_backend: SQLiteBackend) -> None:
    with pytest.raises(ValueError, match="retention_days"):
        file_backend.purge(-1)


def test_purge_accepts_naive_now_as_utc(file_backend: SQLiteBackend) -> None:
    aware_now = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
    file_backend.write(_make_record(timestamp=aware_now - timedelta(days=2)), None)
    assert file_backend.purge(1, now=aware_now.replace(tzinfo=None)) == 1


# --- Lifecycle ----------------------------------------------------------------


def test_close_disposes_engine_pool(file_backend: SQLiteBackend) -> None:
    with patch.object(file_backend._engine, "dispose") as spy:
        file_backend.close()
        spy.assert_called_once()


def test_close_is_idempotent(file_backend: SQLiteBackend) -> None:
    file_backend.close()
    file_backend.close()  # must not raise


def test_sync_context_manager(file_backend: SQLiteBackend) -> None:
    with file_backend as backend:
        backend.write(_make_record(), None)
        report = backend.query(QueryFilters())
        assert report.total_count == 1


def test_dispose_alias_matches_close(file_backend: SQLiteBackend) -> None:
    with patch.object(file_backend, "close") as spy:
        file_backend.dispose()
        spy.assert_called_once()


# --- Async parity --------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_deterministic_order_and_keyset(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'async.db'}")
    await backend.create_tables()
    try:
        base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
        for i in range(25):
            await backend.write(
                _make_record(timestamp=base_ts + timedelta(minutes=i % 4)), None
            )

        seen: list[str] = []
        cursor: str | None = None
        while True:
            filters = (
                QueryFilters(cursor=cursor, limit=10)
                if cursor
                else QueryFilters(limit=10)
            )
            report = await backend.query(filters)
            seen.extend(r.content_id for r in report.records)
            next_cursor = report.next_cursor
            if next_cursor is None:
                break
            cursor = next_cursor
        assert len(seen) == len(set(seen)) == 25

        first = [r.content_id for r in (await backend.query(QueryFilters())).records]
        second = [r.content_id for r in (await backend.query(QueryFilters())).records]
        assert first == second
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_async_write_ahead_pending_then_finalize(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'wa.db'}")
    await backend.create_tables()
    try:
        pending = _make_record(status=RecordStatus.PENDING)
        await backend.write(pending, None)
        got = await backend.get(pending.content_id)
        assert got is not None and got[0].status == RecordStatus.PENDING

        final = pending.model_copy(update={"status": RecordStatus.COMPLETED})
        await backend.finalize(pending.content_id, final, "hmac-a")
        got_final, hmac = await backend.get(pending.content_id) or (None, None)
        assert got_final is not None and got_final.status == RecordStatus.COMPLETED
        assert hmac == "hmac-a"
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_async_write_many_and_buffered_writer(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'bulk.db'}")
    await backend.create_tables()
    try:
        pairs = [(_make_record(), None) for _ in range(20)]
        await backend.write_many(pairs)
        assert (await backend.query(QueryFilters(limit=1000))).total_count == 20

        async with AsyncBufferedWriter(backend, max_buffer_size=7) as writer:
            for _ in range(16):
                await writer.add(_make_record(), None)
        # 16 adds => auto-flush at 7 and 14; exit flushes the last 2.
        assert (await backend.query(QueryFilters(limit=1000))).total_count == 36
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_async_purge(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'purge.db'}")
    await backend.create_tables()
    try:
        now = datetime(2026, 9, 17, 12, 0, tzinfo=timezone.utc)
        await backend.write(_make_record(timestamp=now - timedelta(days=30)), None)
        await backend.write(_make_record(timestamp=now), None)
        assert await backend.purge(7, now=now) == 1
        assert (await backend.query(QueryFilters())).total_count == 1
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_async_close_and_context_manager(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'life.db'}")
    async with backend as b:
        await b.create_tables()
        await b.write(_make_record(), None)
        assert (await b.query(QueryFilters())).total_count == 1
    await backend.close()  # idempotent


# --- Migration 0002 -------------------------------------------------------------


def _alembic_config(url: str) -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def test_migration_0002_up_down_sqlite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # env.py prefers AISTAMP_DATABASE_URL over the ini url — neutralize it.
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    db_path = tmp_path / "mig.db"
    cfg = _alembic_config(f"sqlite:///{db_path}")

    command.upgrade(cfg, "head")
    raw = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in raw.execute("PRAGMA table_info(provenance_records)")
        }
        indexes = {
            row[0]
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        raw.close()

    assert {
        "key_id",
        "sig_algo",
        "record_version",
        "prev_hash",
        "scope_sequence",
        "error_type",
        "error_message",
    } <= columns
    for index_name in (
        "ix_provenance_records_app_id_timestamp",
        "ix_provenance_records_user_id_timestamp",
        "ix_provenance_records_feature_id",
    ):
        assert index_name in indexes

    command.downgrade(cfg, "0001")
    raw = sqlite3.connect(db_path)
    try:
        columns = {
            row[1] for row in raw.execute("PRAGMA table_info(provenance_records)")
        }
        indexes = {
            row[0]
            for row in raw.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    finally:
        raw.close()
    assert not columns & {
        "key_id",
        "sig_algo",
        "record_version",
        "prev_hash",
        "scope_sequence",
        "error_type",
        "error_message",
    }
    assert "ix_provenance_records_app_id_timestamp" not in indexes

    command.upgrade(cfg, "head")  # re-upgrade after downgrade must work


def test_migration_matches_orm_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh create_tables() and a fresh migration must converge on the
    same column set."""
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    db_path = tmp_path / "converge.db"
    cfg = _alembic_config(f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")

    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'createall.db'}")
    backend.create_tables()

    def _columns(path: Path) -> set[str]:
        raw = sqlite3.connect(path)
        try:
            return {
                row[1] for row in raw.execute("PRAGMA table_info(provenance_records)")
            }
        finally:
            raw.close()

    assert _columns(db_path) == _columns(tmp_path / "createall.db")
    backend.close()


# --- Reviewer regressions: offset paging, hmac preservation, re-buffer ------


def test_offset_middle_page_matches_legacy_behavior(
    file_backend: SQLiteBackend,
) -> None:
    """0.1.x offset paging must keep working: offset=2 skips the first two
    rows and returns rows 3-4, never page 1 forever."""
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    contents: list[str] = []
    for i in range(5):
        record = _make_record(timestamp=base_ts + timedelta(seconds=i))
        file_backend.write(record, None)
        contents.append(record.content_id)

    middle = file_backend.query(QueryFilters(limit=2, offset=2))
    assert [r.content_id for r in middle.records] == contents[2:4]
    last = file_backend.query(QueryFilters(limit=2, offset=4))
    assert [r.content_id for r in last.records] == contents[4:]


def test_offset_composes_with_filters(file_backend: SQLiteBackend) -> None:
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    for i in range(4):
        file_backend.write(
            _make_record(timestamp=base_ts + timedelta(seconds=i), user_id="bulk"),
            None,
        )
        file_backend.write(
            _make_record(timestamp=base_ts + timedelta(seconds=i), user_id="other"),
            None,
        )

    all_bulk = [
        r.content_id
        for r in file_backend.query(QueryFilters(user_id="bulk", limit=10)).records
    ]
    page = file_backend.query(QueryFilters(user_id="bulk", limit=2, offset=1))
    assert [r.content_id for r in page.records] == all_bulk[1:3]


def test_offset_cannot_combine_with_keyset() -> None:
    ts = datetime(2026, 9, 17, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        QueryFilters(cursor="2026-09-17T00:00:00|1", offset=5)
    with pytest.raises(ValidationError):
        QueryFilters(after_timestamp=ts, after_id=1, offset=5)


def test_finalize_none_preserves_existing_hmac(
    file_backend: SQLiteBackend,
) -> None:
    record = _make_record()
    file_backend.write(record, "original-hmac")
    final = record.model_copy(update={"status": RecordStatus.COMPLETED})
    file_backend.finalize(record.content_id, final, None)
    fetched, hmac = file_backend.get(record.content_id) or (None, None)
    assert fetched is not None
    assert hmac == "original-hmac", "finalize(None) must not erase a signature"


class _FlakyBackend:
    """Wraps a backend; write_many fails N times before succeeding."""

    def __init__(self, backend: SQLiteBackend, failures: int) -> None:
        self._backend = backend
        self._failures = failures

    def write_many(self, items: Sequence[tuple[ProvenanceRecord, str | None]]) -> None:
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("simulated transient write failure")
        self._backend.write_many(items)


class _FlakyAsyncBackend:
    """Async twin of _FlakyBackend."""

    def __init__(self, backend: AsyncSQLiteBackend, failures: int) -> None:
        self._backend = backend
        self._failures = failures

    async def write_many(
        self, items: Sequence[tuple[ProvenanceRecord, str | None]]
    ) -> None:
        if self._failures > 0:
            self._failures -= 1
            raise RuntimeError("simulated transient write failure")
        await self._backend.write_many(items)


def test_buffered_writer_rebuffers_failed_batch(
    file_backend: SQLiteBackend,
) -> None:
    flaky = _FlakyBackend(file_backend, failures=1)
    writer = BufferedWriter(flaky, max_buffer_size=2)
    writer.add(_make_record(), None)
    # The caller must see the failure — but the batch must survive it.
    with pytest.raises(RuntimeError, match="simulated transient write failure"):
        writer.add(_make_record(), None)  # autoflush fires and fails
    assert len(writer) == 2, "failed batch must be re-buffered, not dropped"
    writer.flush()  # retry succeeds now that the backend healed
    assert file_backend.query(QueryFilters(limit=1000)).total_count == 2


@pytest.mark.asyncio
async def test_async_buffered_writer_rebuffers_failed_batch(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'flaky.db'}")
    await backend.create_tables()
    try:
        writer = AsyncBufferedWriter(
            _FlakyAsyncBackend(backend, failures=1), max_buffer_size=2
        )
        await writer.add(_make_record(), None)
        # The caller must see the failure — but the batch must survive it.
        with pytest.raises(RuntimeError, match="simulated transient write failure"):
            await writer.add(_make_record(), None)  # autoflush fires and fails
        assert len(writer) == 2, "failed batch must be re-buffered, not dropped"
        await writer.flush()  # retry succeeds now that the backend healed
        assert (await backend.query(QueryFilters(limit=1000))).total_count == 2
    finally:
        await backend.close()


# --- Security audit P1-3: PENDING evidence must not present as final --------


def test_query_excludes_pending_by_default(file_backend: SQLiteBackend) -> None:
    file_backend.write(_make_record(), None)  # COMPLETED
    file_backend.write(_make_record(status=RecordStatus.PENDING), None)

    report = file_backend.query(QueryFilters())

    assert [r.status for r in report.records] == [RecordStatus.COMPLETED]
    assert report.total_count == 1
    assert report.filters_applied["include_pending"] is False


def test_query_include_pending_opt_in(file_backend: SQLiteBackend) -> None:
    file_backend.write(_make_record(), None)
    file_backend.write(_make_record(status=RecordStatus.PENDING), None)

    report = file_backend.query(QueryFilters(include_pending=True))

    assert report.total_count == 2
    assert {r.status for r in report.records} == {
        RecordStatus.COMPLETED,
        RecordStatus.PENDING,
    }


def test_query_explicit_pending_status_returns_pending(
    file_backend: SQLiteBackend,
) -> None:
    """Filtering status=PENDING is explicit intent — the opt-in flag is for
    report/export paths that must not see incomplete evidence by default."""
    file_backend.write(_make_record(status=RecordStatus.PENDING), None)

    report = file_backend.query(QueryFilters(status=RecordStatus.PENDING))

    assert report.total_count == 1
    assert report.records[0].status is RecordStatus.PENDING


def test_export_csv_excludes_pending_by_default(
    file_backend: SQLiteBackend,
) -> None:
    """The auditor's PoV: to_csv used to emit PENDING rows verbatim."""
    file_backend.write(_make_record(), None)
    file_backend.write(_make_record(status=RecordStatus.PENDING), None)
    exporter = AuditExporter(file_backend)

    default_csv = exporter.to_csv(exporter.query(QueryFilters()))
    opt_in_csv = exporter.to_csv(exporter.query(QueryFilters(include_pending=True)))

    assert "PENDING" not in default_csv
    assert "PENDING" in opt_in_csv  # surfaced loudly, with its status column


@pytest.mark.asyncio
async def test_async_query_excludes_pending_by_default(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'pending.db'}")
    await backend.create_tables()
    try:
        await backend.write(_make_record(), None)
        await backend.write(_make_record(status=RecordStatus.PENDING), None)

        report = await backend.query(QueryFilters())
        assert report.total_count == 1
        assert report.records[0].status is RecordStatus.COMPLETED

        opt_in = await backend.query(QueryFilters(include_pending=True))
        assert opt_in.total_count == 2
    finally:
        await backend.close()


# --- Security audit P1-5: chain-linked deletion must be detectable ----------


def test_purge_writes_anchor_and_preserves_survivor(
    file_backend: SQLiteBackend,
) -> None:
    base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
    old_1 = _make_record(timestamp=base_ts)
    old_2 = _make_record(timestamp=base_ts + timedelta(minutes=1), prev_hash="c" * 64)
    survivor = _make_record(
        timestamp=base_ts + timedelta(minutes=2), prev_hash="d" * 64
    )
    for record in (old_1, old_2, survivor):
        file_backend.write(record, None)

    # Cutoff lands between old_2 and survivor: now - 1 day == base_ts + 90 s.
    now = base_ts + timedelta(days=1, seconds=90)
    deleted = file_backend.purge(1, now=now)

    assert deleted == 2
    fetched, _ = file_backend.get(survivor.content_id) or (None, None)
    assert fetched is not None and fetched.prev_hash == "d" * 64

    anchors = file_backend.list_purge_anchors()
    assert len(anchors) == 1
    anchor = anchors[0]
    assert anchor.purged_count == 2
    expected_cutoff = (base_ts + timedelta(seconds=90)).replace(tzinfo=None)
    assert anchor.purged_before.replace(tzinfo=None) == expected_cutoff
    assert anchor.deleted_prev_hashes == [None, "c" * 64]  # chain order
    assert anchor.anchor_created_at.replace(tzinfo=None) == now.replace(tzinfo=None)


def test_purge_noop_writes_no_anchor(file_backend: SQLiteBackend) -> None:
    file_backend.write(_make_record(), None)

    deleted = file_backend.purge(30, now=datetime(2026, 9, 17, tzinfo=timezone.utc))

    assert deleted == 0
    assert file_backend.list_purge_anchors() == []


def test_purge_anchor_journal_persists_across_reopen(tmp_path: Path) -> None:
    url = f"sqlite:///{tmp_path / 'journal.db'}"
    backend = SQLiteBackend(url)
    backend.create_tables()
    backend.write(
        _make_record(timestamp=datetime(2026, 9, 15, tzinfo=timezone.utc)), None
    )
    assert backend.purge(1, now=datetime(2026, 9, 18, tzinfo=timezone.utc)) == 1
    backend.close()

    reopened = SQLiteBackend(url)
    try:
        anchors = reopened.list_purge_anchors()
        assert len(anchors) == 1
        assert anchors[0].purged_count == 1
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_async_purge_writes_anchor(tmp_path: Path) -> None:
    backend = AsyncSQLiteBackend(f"sqlite+aiosqlite:///{tmp_path / 'anchor.db'}")
    await backend.create_tables()
    try:
        base_ts = datetime(2026, 9, 17, 0, 0, tzinfo=timezone.utc)
        await backend.write(_make_record(timestamp=base_ts, prev_hash="e" * 64), None)

        deleted = await backend.purge(1, now=base_ts + timedelta(days=1, seconds=1))

        assert deleted == 1
        anchors = await backend.list_purge_anchors()
        assert len(anchors) == 1
        assert anchors[0].purged_count == 1
        assert anchors[0].deleted_prev_hashes == ["e" * 64]
    finally:
        await backend.close()


def test_migration_creates_purge_anchor_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Migration 0002 and create_tables() must both produce the journal."""
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    db_path = tmp_path / "anchor-migrate.db"
    cfg = _alembic_config(f"sqlite:///{db_path}")
    command.upgrade(cfg, "head")

    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'anchor-createall.db'}")
    backend.create_tables()
    backend.close()

    def _journal_columns(path: Path) -> set[str]:
        raw = sqlite3.connect(path)
        try:
            return {row[1] for row in raw.execute("PRAGMA table_info(purge_anchors)")}
        finally:
            raw.close()

    assert _journal_columns(db_path) == _journal_columns(
        tmp_path / "anchor-createall.db"
    )
    assert {
        "id",
        "purged_before",
        "purged_count",
        "deleted_prev_hashes",
        "anchor_created_at",
        "signature",
    } <= _journal_columns(db_path)
