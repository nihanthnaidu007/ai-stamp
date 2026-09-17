"""v0.2 async-lifecycle compatibility checks.

The async client must mirror the sync client (chat -> str, stamp /
chat_detailed -> StampResult), close its backend resources on ``aclose()``
idempotently, and apply the streaming-redaction hotfix on the async stream
path too.
"""

from __future__ import annotations

import json

import _fakes
import pytest
import pytest_asyncio

from aistamp.client import AsyncProvenanceClient, StampResult
from aistamp.client._pipeline import redact_prompt
from aistamp.config import Config
from aistamp.fingerprint import hash_content
from aistamp.models import QueryFilters, RecordStatus
from aistamp.store.async_backend import AsyncSQLiteBackend

pytestmark = pytest.mark.asyncio

_SECRET = "compat-kit-secret-key-0-2-32-chars!!"
_MODEL = "gpt-4o-mini"
_SSN = "123-45-6789"
_PII_PROMPT = f"Employee SSN is {_SSN}. Summarize benefits."


def _config(**overrides: object) -> Config:
    params: dict[str, object] = {
        "secret_key": _SECRET,
        "database_url": "sqlite+aiosqlite:///:memory:",
    }
    params.update(overrides)
    return Config(**params)  # type: ignore[arg-type]


@pytest_asyncio.fixture
async def async_backend() -> AsyncSQLiteBackend:
    b = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    await b.create_tables()
    yield b
    await b._engine.dispose()


async def _llm(prompt: str) -> str:
    return f"echo: {prompt}"


def _client(backend: AsyncSQLiteBackend) -> AsyncProvenanceClient:
    return AsyncProvenanceClient(
        _llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )


# V2-09 ----------------------------------------------------------------------
async def test_v2_09_async_chat_stamp_chat_detailed_parity(
    async_backend: AsyncSQLiteBackend,
) -> None:
    client = _client(async_backend)
    text = await client.chat("hello", _MODEL)
    stamped = await client.stamp("hello", _MODEL)
    detailed = await client.chat_detailed("hello", _MODEL)
    assert isinstance(text, str)
    assert isinstance(stamped, StampResult)
    assert isinstance(detailed, StampResult)
    assert stamped.text == detailed.text == text
    assert stamped.record.status == RecordStatus.COMPLETED
    report = await async_backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 3


# V2-10 ----------------------------------------------------------------------
async def test_v2_10_async_streaming_redaction_parity(
    monkeypatch: pytest.MonkeyPatch, async_backend: AsyncSQLiteBackend
) -> None:
    """Hotfix #12 covers the async stream path too."""
    fake = _fakes.FakeAsyncOpenAI(
        stream_chunks=[
            _fakes.chunk("benefits: "),
            _fakes.chunk("dental"),
            _fakes.chunk("", usage=_fakes.fake_response("ignored").usage),
        ]
    )
    _fakes.install_fake_openai(monkeypatch, _fakes.FakeOpenAI, _fakes.FakeAsyncOpenAI)
    client = AsyncProvenanceClient(
        fake,
        config=_config(redact_before_send=True),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=async_backend,
    )
    stream = await client.stamp_stream(_PII_PROMPT, _MODEL)
    chunks = [c async for c in stream]
    result = stream.result
    payload = json.dumps(fake.calls, default=str)
    assert _SSN not in payload, "raw PII reached the provider on the async stream path"
    redacted = redact_prompt(_PII_PROMPT)
    assert result.record.prompt_hash == hash_content(redacted)
    assert result.record.prompt_hash != hash_content(_PII_PROMPT)
    assert result.text == "".join(chunks)


# V2-11 ----------------------------------------------------------------------
async def test_v2_11_async_lifecycle_create_tables_and_aclose_idempotent() -> None:
    backend = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    client = AsyncProvenanceClient(
        _llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    await client.create_tables()
    text = await client.chat("hello", _MODEL)
    assert isinstance(text, str)
    await client.aclose()
    await client.aclose()  # documented idempotent
