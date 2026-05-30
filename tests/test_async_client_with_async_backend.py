from __future__ import annotations

import pytest
import pytest_asyncio

from aistamp import Config
from aistamp.client import AsyncProvenanceClient
from aistamp.models import QueryFilters
from aistamp.store import SQLiteBackend
from aistamp.store.async_backend import AsyncSQLiteBackend

pytestmark = pytest.mark.asyncio


_SECRET = "async-client-with-async-backend-32chars!"


@pytest_asyncio.fixture
async def async_backend():
    b = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    await b.create_tables()
    yield b
    await b._engine.dispose()


def _config() -> Config:
    return Config(
        secret_key=_SECRET,
        database_url="sqlite+aiosqlite:///:memory:",
    )


async def test_async_client_uses_async_backend(
    async_backend: AsyncSQLiteBackend,
) -> None:
    # AsyncProvenanceClient initialized with AsyncSQLiteBackend must write to it.
    async def llm(prompt: str) -> str:
        return "ok"

    client = AsyncProvenanceClient(
        llm,
        config=_config(),
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=async_backend,
    )
    await client.chat("hi")
    report = await async_backend.query(QueryFilters(user_id="u"))
    assert report.total_count == 1


async def test_async_client_async_backend_record_has_correct_fields(
    async_backend: AsyncSQLiteBackend,
) -> None:
    # The stored record must have correct app_id, user_id, model, status.
    async def llm(prompt: str) -> str:
        return "ok"

    client = AsyncProvenanceClient(
        llm,
        config=_config(),
        app_id="my_app",
        feature_id="f",
        user_id="my_user",
        backend=async_backend,
    )
    await client.chat("hello", model="explicit-model")
    report = await async_backend.query(QueryFilters(user_id="my_user"))
    record = report.records[0]
    assert record.app_id == "my_app"
    assert record.user_id == "my_user"
    assert record.model == "explicit-model"
    assert record.status.value == "COMPLETED"


async def test_async_client_falls_back_to_sync_backend() -> None:
    # AsyncProvenanceClient with a sync SQLiteBackend must still work.
    sync_backend = SQLiteBackend("sqlite:///:memory:")
    sync_backend.create_tables()

    async def llm(prompt: str) -> str:
        return "ok"

    client = AsyncProvenanceClient(
        llm,
        config=_config(),
        app_id="a",
        feature_id="f",
        user_id="sync_u",
        backend=sync_backend,
    )
    await client.chat("hi")
    report = sync_backend.query(QueryFilters(user_id="sync_u"))
    assert report.total_count == 1


async def test_async_client_async_backend_pii_detected(
    async_backend: AsyncSQLiteBackend,
) -> None:
    # AsyncProvenanceClient with async_backend must detect PII in response.
    async def llm(prompt: str) -> str:
        return "Email me at alice@example.com please"

    client = AsyncProvenanceClient(
        llm,
        config=_config(),
        app_id="a",
        feature_id="f",
        user_id="pii_u",
        backend=async_backend,
    )
    await client.chat("how do I reach you")
    report = await async_backend.query(QueryFilters(user_id="pii_u"))
    pii = report.records[0].pii_result
    assert pii is not None
    assert pii.match_count > 0
