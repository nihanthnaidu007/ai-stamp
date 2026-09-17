"""v0.2 client-surface compatibility checks (sync client).

These are CI-guarded invariants for consumers moving from 0.1.x to 0.2:
``chat()`` keeps returning ``str`` while ``stamp()`` / ``chat_detailed()``
return the rich ``StampResult``, the ``AIStampError`` taxonomy is catchable
through one base class, and the streaming-redaction hotfix holds on every
streaming path.
"""

from __future__ import annotations

import json

import _fakes
import pytest

from aistamp.client import ProvenanceClient, StampResult
from aistamp.client._pipeline import redact_prompt
from aistamp.config import Config
from aistamp.errors import (
    AIStampError,
    ConfigError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    StampError,
)
from aistamp.fingerprint import hash_content
from aistamp.models import RecordStatus
from aistamp.store import SQLiteBackend

_SECRET = "compat-kit-secret-key-0-2-32-chars!!"
_MODEL = "gpt-4o-mini"
_SSN = "123-45-6789"
_PII_PROMPT = f"Employee SSN is {_SSN}. Summarize benefits."


def _config(**overrides: object) -> Config:
    params: dict[str, object] = {
        "secret_key": _SECRET,
        "database_url": "sqlite:///:memory:",
    }
    params.update(overrides)
    return Config(**params)  # type: ignore[arg-type]


@pytest.fixture
def backend() -> SQLiteBackend:
    b = SQLiteBackend("sqlite:///:memory:")
    b.create_tables()
    return b


def _client(
    backend: SQLiteBackend, llm: object, config: Config | None = None
) -> ProvenanceClient:
    return ProvenanceClient(
        llm,
        config=config or _config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )


def _callable_llm(prompt: str) -> str:
    return f"echo: {prompt}"


# V2-01 ----------------------------------------------------------------------
def test_v2_01_chat_returns_str_while_stamp_returns_stamp_result(
    backend: SQLiteBackend,
) -> None:
    """0.1 contract kept: chat() is str; the 0.2 rich path is StampResult."""
    client = _client(backend, _callable_llm)
    text = client.chat("hello", _MODEL)
    result = client.stamp("hello", _MODEL)
    assert isinstance(text, str)
    assert isinstance(result, StampResult)
    assert result.text == text


# V2-02 ----------------------------------------------------------------------
def test_v2_02_chat_detailed_returns_full_stamp_result(backend: SQLiteBackend) -> None:
    client = _client(backend, _callable_llm)
    result = client.chat_detailed(
        "hello",
        _MODEL,
        metadata={"case": "C-1"},
        conversation_id="conv-1",
        request_id="req-1",
    )
    assert isinstance(result, StampResult)
    assert result.record.status == RecordStatus.COMPLETED
    assert result.content_id == result.record.content_id
    assert result.conversation_id == "conv-1"
    assert result.request_id == "req-1"
    assert result.metadata == {"case": "C-1"}
    stored = backend.get(result.content_id)
    assert stored is not None


# V2-03 ----------------------------------------------------------------------
def test_v2_03_stamp_result_record_binds_content_hashes(backend: SQLiteBackend) -> None:
    client = _client(backend, _callable_llm)
    result = client.stamp("bind-me", _MODEL)
    record = result.record
    assert record.prompt_hash == hash_content("bind-me")
    assert record.response_hash == hash_content(result.text)
    stored = backend.get(result.content_id)
    assert stored is not None


# V2-04 ----------------------------------------------------------------------
def test_v2_04_error_taxonomy_single_base_class() -> None:
    """Every public error derives from AIStampError; legacy paths hold."""
    error_names = (
        "StampError",
        "ProviderError",
        "ProviderTimeoutError",
        "ProviderRateLimitError",
        "ProviderAuthError",
        "ProviderResponseError",
        "ConfigError",
    )
    for name in error_names:
        assert issubclass(getattr(_errors_module(), name), AIStampError), name
    # Legacy 0.1 import paths keep working and point at the same class.
    from aistamp.client import StampError as legacy_client
    from aistamp.client.sync import StampError as legacy_sync

    assert legacy_client is _errors_module().StampError
    assert legacy_sync is _errors_module().StampError
    # 0.1 invariant kept: policy blocks are not StampErrors.
    from aistamp.policy import PolicyViolationError

    assert not issubclass(PolicyViolationError, StampError)
    with pytest.raises(AIStampError):
        raise StampError("caught via base", content_id="cid")


# V2-05 ----------------------------------------------------------------------
def test_v2_05_error_attributes_content_id_status_retry_after() -> None:
    err = ProviderRateLimitError(
        "rate limited", content_id="cid-1", status_code=429, retry_after=30.0
    )
    assert err.content_id == "cid-1"
    assert err.status_code == 429
    assert err.retry_after == 30.0
    assert isinstance(err, ProviderError)
    assert isinstance(err, AIStampError)
    for cls in (ProviderTimeoutError, ProviderAuthError, ProviderResponseError):
        assert issubclass(cls, ProviderError)
    # ConfigError stays a ValueError so old `except ValueError` handlers work.
    assert issubclass(ConfigError, ValueError)
    assert issubclass(ConfigError, AIStampError)


# V2-06 ----------------------------------------------------------------------
def test_v2_06_streaming_redaction_provider_never_sees_raw_pii(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    """Hotfix #12, streaming path: the provider receives only redacted text."""
    fake = _fakes.FakeOpenAI(
        stream_chunks=[_fakes.chunk("benefits: "), _fakes.chunk("dental")]
    )
    _fakes.install_fake_openai(monkeypatch, _fakes.FakeOpenAI, _fakes.FakeAsyncOpenAI)
    client = _client(backend, fake, _config(redact_before_send=True))
    stream = client.stamp_stream(_PII_PROMPT, _MODEL)
    list(stream)
    result = stream.result
    payload = json.dumps(fake.calls, default=str)
    assert _SSN not in payload, "raw PII reached the provider on a streaming path"
    redacted = redact_prompt(_PII_PROMPT)
    assert redacted != _PII_PROMPT
    assert result.record.prompt_hash == hash_content(redacted)
    assert result.record.prompt_hash != hash_content(_PII_PROMPT)


# V2-07 ----------------------------------------------------------------------
def test_v2_07_streaming_redaction_opt_out_sends_raw_prompt(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    """The redact_before_send=False opt-out is the documented off-switch."""
    fake = _fakes.FakeOpenAI(stream_chunks=[_fakes.chunk("ok")])
    _fakes.install_fake_openai(monkeypatch, _fakes.FakeOpenAI, _fakes.FakeAsyncOpenAI)
    client = _client(backend, fake, _config(redact_before_send=False))
    stream = client.stamp_stream(_PII_PROMPT, _MODEL)
    list(stream)
    assert _SSN in json.dumps(fake.calls, default=str)
    assert stream.result.record.prompt_hash == hash_content(_PII_PROMPT)


# V2-08 ----------------------------------------------------------------------
def test_v2_08_stamp_stream_chunk_passthrough_and_result_lifecycle(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _fakes.FakeOpenAI(
        stream_chunks=[
            _fakes.chunk("hello "),
            _fakes.chunk("world"),
            _fakes.chunk("!", usage=_fakes.fake_response("ignored").usage),
        ]
    )
    _fakes.install_fake_openai(monkeypatch, _fakes.FakeOpenAI, _fakes.FakeAsyncOpenAI)
    client = _client(backend, fake)
    stream = client.stamp_stream("hi", _MODEL)
    with pytest.raises(StampError):
        _ = stream.result  # not available before exhaustion
    chunks = list(stream)
    assert chunks == ["hello ", "world", "!"]
    result = stream.result
    assert result.text == "hello world!"
    assert result.usage is not None
    assert result.usage.response_tokens == 5
    assert result.content_id
    assert backend.get(result.content_id) is not None


def _errors_module() -> object:
    import aistamp.errors as errors_module

    return errors_module
