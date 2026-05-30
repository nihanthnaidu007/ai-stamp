from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
import pytest_asyncio

from aistamp.models import (
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.store.async_backend import AsyncSQLiteBackend

pytestmark = pytest.mark.asyncio


def _make_record(**overrides: Any) -> ProvenanceRecord:
    base = dict(
        content_id=str(uuid.uuid4()),
        app_id="test_app",
        feature_id="test_feature",
        user_id="test_user",
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


@pytest_asyncio.fixture
async def async_sqlite_backend():
    backend = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    await backend.create_tables()
    yield backend
    await backend._engine.dispose()


async def test_async_write_succeeds(async_sqlite_backend: AsyncSQLiteBackend) -> None:
    # async write() must persist a record without raising.
    await async_sqlite_backend.write(_make_record(), "hmac")


async def test_async_get_returns_correct_record(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async get() must return the same record that was written.
    record = _make_record()
    await async_sqlite_backend.write(record, "hmac")
    result = await async_sqlite_backend.get(record.content_id)
    assert result is not None
    fetched, _ = result
    assert fetched.content_id == record.content_id
    assert fetched.app_id == record.app_id
    assert fetched.model == record.model


async def test_async_get_returns_none_for_unknown_id(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async get() must return None for a content_id never written.
    assert await async_sqlite_backend.get("missing") is None


async def test_async_get_returns_tuple_with_hmac(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async get() must return (ProvenanceRecord, hmac_string) tuple.
    record = _make_record()
    await async_sqlite_backend.write(record, "test_async_hmac")
    result = await async_sqlite_backend.get(record.content_id)
    assert result is not None
    _, hmac = result
    assert hmac == "test_async_hmac"


async def test_async_get_returns_none_hmac_when_not_stored(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # If written with hmac=None, async get() must return None for hmac.
    record = _make_record()
    await async_sqlite_backend.write(record, None)
    result = await async_sqlite_backend.get(record.content_id)
    assert result is not None
    _, hmac = result
    assert hmac is None


async def test_async_get_deserializes_pii_result(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async get() must deserialize pii_result back to PIIResult, not a raw dict.
    pii = PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=PIISeverity.MEDIUM,
        match_count=1,
    )
    record = _make_record(pii_result=pii)
    await async_sqlite_backend.write(record, "h")
    result = await async_sqlite_backend.get(record.content_id)
    assert result is not None
    fetched, _ = result
    assert isinstance(fetched.pii_result, PIIResult)


async def test_async_get_deserializes_policy_decision(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async get() must deserialize policy_decision back to PolicyDecision.
    decision = PolicyDecision(action=PolicyAction.WARN, rule_name="r", reason="r")
    record = _make_record(policy_decision=decision)
    await async_sqlite_backend.write(record, "h")
    result = await async_sqlite_backend.get(record.content_id)
    assert result is not None
    fetched, _ = result
    assert isinstance(fetched.policy_decision, PolicyDecision)


async def test_async_query_returns_matching_records(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async query() with user_id filter must return only records for that user.
    await async_sqlite_backend.write(_make_record(user_id="alice"), "h")
    await async_sqlite_backend.write(_make_record(user_id="bob"), "h")
    report = await async_sqlite_backend.query(QueryFilters(user_id="alice"))
    assert len(report.records) == 1
    assert report.records[0].user_id == "alice"


async def test_async_query_returns_empty_for_no_match(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async query() with filter matching no records must return empty AuditReport.
    report = await async_sqlite_backend.query(QueryFilters(user_id="nobody"))
    assert report.records == []
    assert report.total_count == 0


async def test_async_query_respects_limit(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async query() must respect limit in QueryFilters.
    for _ in range(5):
        await async_sqlite_backend.write(_make_record(user_id="multi"), "h")
    report = await async_sqlite_backend.query(QueryFilters(user_id="multi", limit=2))
    assert len(report.records) == 2
    assert report.total_count == 5


async def test_async_query_filters_by_status(
    async_sqlite_backend: AsyncSQLiteBackend,
) -> None:
    # async query() with status=COMPLETED must return only COMPLETED records.
    await async_sqlite_backend.write(_make_record(status=RecordStatus.COMPLETED), "h")
    await async_sqlite_backend.write(_make_record(status=RecordStatus.ERROR), "h")
    report = await async_sqlite_backend.query(
        QueryFilters(status=RecordStatus.COMPLETED)
    )
    assert len(report.records) == 1
    assert report.records[0].status == RecordStatus.COMPLETED
