from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio

from aistamp.models import ProvenanceRecord, QueryFilters, RecordStatus
from aistamp.store.async_backend import AsyncPostgreSQLBackend
from aistamp.store.backend import PostgreSQLBackend

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
