"""v0.2 client-layer tests: rich results, resilience, streaming, hardening.

All provider SDKs are offline fakes installed via ``sys.modules`` injection.
Retry sleeping is replaced at the ``aistamp.client._retry`` module seams, so
no test ever waits on backoff delays.
"""

from __future__ import annotations

import sys
import threading
import types
import uuid
from collections.abc import AsyncIterator, Iterator
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError

import pytest
from pydantic import ValidationError

import aistamp.client._pipeline as pipeline_module
import aistamp.pii
from aistamp.client import (
    AsyncProvenanceClient,
    ProvenanceClient,
    StampResult,
    TokenUsage,
)
from aistamp.client import http as http_module
from aistamp.client._pipeline import (
    CaptureContext,
    apply_pre_call_policy,
    classify_provider_error,
    report_persist_failure,
    report_persist_failure_async,
    try_persist_async,
    try_persist_sync,
)
from aistamp.client._retry import compute_delay
from aistamp.client.async_ import _build_default_async_backend
from aistamp.client.http import GenericHTTPClient, wrap_http_error
from aistamp.config import Config, interpolate_env_vars
from aistamp.errors import (
    AIStampError,
    ConfigError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    StampError,
)
from aistamp.fingerprint import hash_content
from aistamp.models import (
    PIIResult,
    PolicyAction,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.policy.rules import RuleConditions, RuleConfig
from aistamp.store import SQLiteBackend
from aistamp.store.async_backend import AsyncPostgreSQLBackend

_SECRET = "v2-test-secret-key-minimum-32-chars!"


# ---------------------------------------------------------------------------
# Fake provider SDKs
# ---------------------------------------------------------------------------


class _FakeRateLimitError(Exception):
    status_code = 429


class _FakeServerError(Exception):
    status_code = 503


class _FakeAuthError(Exception):
    status_code = 401


class _FakeOpenAIResponse:
    def __init__(self, text: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=text))]
        self.usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)


class _FakeAnthropicResponse:
    def __init__(self, text: str) -> None:
        self.content = [SimpleNamespace(text=text)]
        self.usage = SimpleNamespace(input_tokens=9, output_tokens=4)


def _chunk(delta: str, usage: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))],
        usage=usage,
    )


class _FakeOpenAI:
    """Offline stand-in for ``openai.OpenAI`` (create + streaming)."""

    def __init__(
        self,
        *,
        response: Any = None,
        errors: list[BaseException] | None = None,
        stream_chunks: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._errors = list(errors or [])
        self._stream_chunks = stream_chunks
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        if self._stream_chunks is not None and kwargs.get("stream"):
            return iter(self._stream_chunks)
        if self._response is None:
            raise AssertionError("fake OpenAI configured without a response")
        return self._response


class _FakeAsyncOpenAI:
    """Offline stand-in for ``openai.AsyncOpenAI``."""

    def __init__(
        self,
        *,
        response: Any = None,
        errors: list[BaseException] | None = None,
        stream_chunks: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._errors = list(errors or [])
        self._stream_chunks = stream_chunks
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        if self._stream_chunks is not None and kwargs.get("stream"):
            return self._agen()
        if self._response is None:
            raise AssertionError("fake AsyncOpenAI configured without a response")
        return self._response

    async def _agen(self) -> AsyncIterator[Any]:
        for chunk in self._stream_chunks or []:
            yield chunk


class _DualIterable:
    """Sync- and async-iterable string sequence.

    Mirrors the real Anthropic SDK, where ``text_stream`` is sync-iterable
    on ``MessageStream`` and async-iterable on ``AsyncMessageStream``.
    """

    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def __iter__(self) -> Iterator[str]:
        return iter(self._texts)

    def __aiter__(self) -> AsyncIterator[str]:
        async def _gen() -> AsyncIterator[str]:
            for text in self._texts:
                yield text

        return _gen()


class _AwaitableFinalMessage:
    """Final-message stand-in that works awaited (async SDK) and plain (sync)."""

    def __init__(self, usage: Any) -> None:
        self.usage = usage

    def __await__(self) -> Any:
        async def _coro() -> _AwaitableFinalMessage:
            return self

        return _coro().__await__()


class _FakeAnthropicStream:
    def __init__(self, texts: list[str], final_usage: Any) -> None:
        self._texts = texts
        self._final_usage = final_usage

    def __enter__(self) -> _FakeAnthropicStream:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    async def __aenter__(self) -> _FakeAnthropicStream:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    @property
    def text_stream(self) -> _DualIterable:
        return _DualIterable(self._texts)

    def get_final_message(self) -> _AwaitableFinalMessage:
        return _AwaitableFinalMessage(self._final_usage)


class _FakeAnthropic:
    """Offline stand-in for ``anthropic.Anthropic``."""

    def __init__(
        self,
        *,
        response: Any = None,
        errors: list[BaseException] | None = None,
        stream_texts: list[str] | None = None,
        final_usage: Any = None,
    ) -> None:
        self._response = response
        self._errors = list(errors or [])
        self._stream = _FakeAnthropicStream(stream_texts or [], final_usage)
        self.calls: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create, stream=self._open_stream)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        if self._response is None:
            raise AssertionError("fake Anthropic configured without a response")
        return self._response

    def _open_stream(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.calls.append(kwargs)
        return self._stream


class _FakeAsyncAnthropic(_FakeAnthropic):
    """Offline stand-in for ``anthropic.AsyncAnthropic`` (awaited create)."""

    async def _create(self, **kwargs: Any) -> Any:  # type: ignore[override]
        self.calls.append(kwargs)
        if self._errors:
            raise self._errors.pop(0)
        if self._response is None:
            raise AssertionError("fake AsyncAnthropic configured without a response")
        return self._response

    def _open_stream(self, **kwargs: Any) -> _FakeAnthropicStream:
        self.calls.append(kwargs)
        return self._stream


def _install_sdk(monkeypatch: pytest.MonkeyPatch, name: str, **attrs: Any) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _config(**overrides: Any) -> Config:
    params: dict[str, Any] = {
        "secret_key": _SECRET,
        "database_url": "sqlite:///:memory:",
    }
    params.update(overrides)
    return Config(**params)


@pytest.fixture
def backend() -> SQLiteBackend:
    b = SQLiteBackend("sqlite:///:memory:")
    b.create_tables()
    return b


def _client(
    llm: Any, backend: SQLiteBackend, config: Config | None = None, **kw: Any
) -> ProvenanceClient:
    return ProvenanceClient(
        llm,
        config=config or _config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
        **kw,
    )


@pytest.fixture
def sleep_log(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    log: list[float] = []

    def _sleep(seconds: float) -> None:
        log.append(seconds)

    monkeypatch.setattr("aistamp.client._retry._sleep", _sleep)
    return log


@pytest.fixture
def asleep_log(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    log: list[float] = []

    async def _asleep(seconds: float) -> None:
        log.append(seconds)

    monkeypatch.setattr("aistamp.client._retry._asleep", _asleep)
    return log


# ---------------------------------------------------------------------------
# Rich results
# ---------------------------------------------------------------------------


def test_stamp_returns_rich_result(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "rich text", backend)
    result = client.stamp("hello")
    assert isinstance(result, StampResult)
    assert result.text == "rich text"
    assert result.content_id
    assert result.record.content_id == result.content_id
    assert result.record.status.value == "COMPLETED"
    assert result.record.key_id == "default"
    assert result.usage is None
    assert result.decision is None


def test_chat_still_returns_plain_string(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "plain", backend)
    assert client.chat("hi") == "plain"


def test_chat_detailed_matches_stamp(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "same", backend)
    first = client.stamp("q")
    second = client.chat_detailed("q")
    assert second.text == first.text == "same"
    assert second.record.user_id == first.record.user_id


def test_usage_captured_from_openai_response(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(response=_FakeOpenAIResponse("ok"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    result = _client(fake, backend).stamp("q", model="gpt-4o")
    assert result.usage == TokenUsage(prompt_tokens=11, response_tokens=7)
    assert result.record.prompt_tokens == 11
    assert result.record.response_tokens == 7


def test_usage_captured_from_anthropic_response(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAnthropic(response=_FakeAnthropicResponse("ok"))
    _install_sdk(
        monkeypatch,
        "anthropic",
        Anthropic=_FakeAnthropic,
        AsyncAnthropic=_FakeAsyncAnthropic,
    )
    result = _client(fake, backend).stamp("q", model="claude-3")
    assert result.usage == TokenUsage(prompt_tokens=9, response_tokens=4)


def test_per_call_identity_overrides(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "x", backend)
    result = client.stamp(
        "q", app_id="other-app", feature_id="other-feat", user_id="other-user"
    )
    assert result.record.app_id == "other-app"
    assert result.record.feature_id == "other-feat"
    assert result.record.user_id == "other-user"
    # Overrides are per-call; constructor values apply again afterwards.
    next_result = client.stamp("q2")
    assert next_result.record.app_id == "app"
    assert next_result.record.user_id == "user"


def test_correlation_fields_echoed(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "x", backend)
    result = client.stamp(
        "q", metadata={"session": "s1"}, conversation_id="conv-1", request_id="req-9"
    )
    assert result.metadata == {"session": "s1"}
    assert result.conversation_id == "conv-1"
    assert result.request_id == "req-9"


# ---------------------------------------------------------------------------
# Provider kwargs passthrough and max_tokens
# ---------------------------------------------------------------------------


def test_kwargs_forwarded_and_max_tokens_configurable(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(response=_FakeOpenAIResponse("ok"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_tokens=250)
    client.stamp("q", model="gpt-4o", temperature=0.3, system="be brief")
    create_kwargs = fake.calls[0]
    assert create_kwargs["model"] == "gpt-4o"
    assert create_kwargs["messages"] == [{"role": "user", "content": "q"}]
    assert create_kwargs["temperature"] == 0.3
    assert create_kwargs["system"] == "be brief"
    # Client-level default applies; proves the 0.1 hardcoded 1024 is gone.
    assert create_kwargs["max_tokens"] == 250
    # Per-call max_tokens wins over the client default.
    client.stamp("q2", model="gpt-4o", max_tokens=99)
    assert fake.calls[1]["max_tokens"] == 99


def test_max_tokens_none_omits_provider_param(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(response=_FakeOpenAIResponse("ok"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_tokens=None)
    client.stamp("q", model="gpt-4o")
    assert "max_tokens" not in fake.calls[0]


def test_messages_kwarg_replaces_default_payload(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(response=_FakeOpenAIResponse("ok"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    result = _client(fake, backend).stamp("q", model="gpt-4o", messages=messages)
    assert fake.calls[0]["messages"] == messages
    # The prompt remains what is hashed and stamped.
    assert result.record.prompt_hash == hash_content("q")


# ---------------------------------------------------------------------------
# Resilience: retries, timeouts, taxonomy
# ---------------------------------------------------------------------------


def test_rate_limit_retried_then_success(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, sleep_log: list[float]
) -> None:
    fake = _FakeOpenAI(
        response=_FakeOpenAIResponse("ok"),
        errors=[_FakeRateLimitError("slow down"), _FakeRateLimitError("again")],
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_retries=3)
    result = client.stamp("q")
    assert result.text == "ok"
    assert len(fake.calls) == 3  # initial + two retries
    assert len(sleep_log) == 2
    assert all(delay >= 0 for delay in sleep_log)


def test_rate_limit_exhausted_raises_with_content_id(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, sleep_log: list[float]
) -> None:
    fake = _FakeOpenAI(
        response=_FakeOpenAIResponse("ok"),
        errors=[_FakeRateLimitError("busy")] * 10,
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_retries=1)
    with pytest.raises(ProviderRateLimitError) as excinfo:
        client.stamp("q")
    assert len(fake.calls) == 2  # initial + one retry
    assert len(sleep_log) == 1
    assert excinfo.value.content_id  # evidence for operators


def test_timeout_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, sleep_log: list[float]
) -> None:
    fake = _FakeOpenAI(
        response=_FakeOpenAIResponse("late but ok"),
        errors=[TimeoutError("read timed out"), TimeoutError("again")],
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_retries=3)
    result = client.stamp("q")
    assert result.text == "late but ok"
    assert len(fake.calls) == 3


def test_server_errors_are_retried(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, sleep_log: list[float]
) -> None:
    fake = _FakeOpenAI(
        response=_FakeOpenAIResponse("ok"),
        errors=[_FakeServerError("maintenance"), _FakeServerError("still down")],
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_retries=3)
    result = client.stamp("q")
    assert result.text == "ok"
    assert len(fake.calls) == 3


def test_auth_error_is_not_retried(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, sleep_log: list[float]
) -> None:
    fake = _FakeOpenAI(
        response=_FakeOpenAIResponse("ok"), errors=[_FakeAuthError("bad key")]
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend, max_retries=3)
    # A foreign 401 is classified into the taxonomy, not retried.
    with pytest.raises(ProviderAuthError) as excinfo:
        client.stamp("q")
    assert len(fake.calls) == 1  # no retry on 4xx auth failures
    assert sleep_log == []
    assert excinfo.value.content_id


def test_error_taxonomy_under_ai_stamp_error() -> None:
    for cls in (
        StampError,
        ProviderTimeoutError,
        ProviderRateLimitError,
        ProviderAuthError,
        ProviderResponseError,
        ConfigError,
    ):
        assert issubclass(cls, AIStampError)


def test_backoff_delay_bounds() -> None:
    # A ceiling-returning sampler keeps the jitter inside the computed window.
    rand = lambda _low, high: high  # noqa: E731
    assert compute_delay(0, base_delay=0.5, max_delay=8.0, rand=rand) == 0.5
    assert compute_delay(3, base_delay=0.5, max_delay=8.0, rand=rand) == 4.0
    # Capped at max_delay.
    assert compute_delay(10, base_delay=0.5, max_delay=8.0, rand=rand) == 8.0


def test_http_client_wraps_rate_limit(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    from urllib.error import HTTPError

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        raise HTTPError(request.full_url, 429, "Too Many Requests", None, None)

    monkeypatch.setattr("aistamp.client.http.urlopen", fake_urlopen)
    http_client = GenericHTTPClient(endpoint="http://llm.test/v1", timeout_seconds=1.0)
    client = _client(http_client, backend, max_retries=1)
    with pytest.raises(ProviderRateLimitError):
        client.stamp("q")


def test_http_client_wraps_connection_error(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    from urllib.error import URLError

    def fake_urlopen(request: Any, timeout: float | None = None) -> Any:
        raise URLError("connection refused")

    monkeypatch.setattr("aistamp.client.http.urlopen", fake_urlopen)
    http_client = GenericHTTPClient(endpoint="http://llm.test/v1", timeout_seconds=1.0)
    client = _client(http_client, backend, max_retries=0)
    with pytest.raises(ProviderResponseError):
        client.stamp("q")


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_stream_stamps_concatenation_openai(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(
        stream_chunks=[
            _chunk("Hel"),
            _chunk("lo"),
            _chunk("", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3)),
            _chunk(" world"),
        ]
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend)
    stream = client.stamp_stream("q")
    parts = list(stream)
    assert parts == ["Hel", "lo", " world"]
    result = stream.result
    assert result.text == "Hello world"
    assert result.usage == TokenUsage(prompt_tokens=5, response_tokens=3)
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].content_id == result.content_id


def test_stream_result_unavailable_before_exhaustion(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeOpenAI(stream_chunks=[_chunk("a"), _chunk("b")])
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(fake, backend)
    stream = client.stamp_stream("q")
    with pytest.raises(StampError, match="fully consumed"):
        stream.result  # noqa: B018 — property raises by design
    assert list(stream) == ["a", "b"]
    assert stream.result.text == "ab"


def test_stamp_stream_rejects_non_streaming_client(backend: SQLiteBackend) -> None:
    client = _client(lambda p: "x", backend)
    with pytest.raises(StampError, match="streaming"):
        client.stamp_stream("q")


def test_stream_stamps_concatenation_anthropic(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAnthropic(
        stream_texts=["bon", "jour"],
        final_usage=SimpleNamespace(input_tokens=9, output_tokens=4),
    )
    _install_sdk(
        monkeypatch,
        "anthropic",
        Anthropic=_FakeAnthropic,
        AsyncAnthropic=_FakeAsyncAnthropic,
    )
    client = _client(fake, backend)
    stream = client.stamp_stream("q")
    parts = list(stream)
    assert parts == ["bon", "jour"]
    assert stream.result.text == "bonjour"
    assert stream.result.usage == TokenUsage(prompt_tokens=9, response_tokens=4)


# ---------------------------------------------------------------------------
# Persist-failure hook
# ---------------------------------------------------------------------------


def test_on_persist_error_callback_invoked(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    def _boom(record: Any, hmac: str) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(backend, "write", _boom)
    seen: list[tuple[Any, BaseException]] = []

    def on_error(record: Any, exc: BaseException) -> None:
        seen.append((record, exc))

    client = _client(lambda p: "ok", backend, on_persist_error=on_error)
    result = client.stamp("q")  # must not raise
    assert result.text == "ok"
    assert len(seen) == 1
    record, exc = seen[0]
    assert record.content_id == result.content_id
    assert isinstance(exc, RuntimeError)


def test_persist_failure_is_silent_without_callback(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    def _boom(record: Any, hmac: str) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(backend, "write", _boom)
    client = _client(lambda p: "ok", backend)
    result = client.stamp("q")
    assert result.text == "ok"


# ---------------------------------------------------------------------------
# Redaction before send
# ---------------------------------------------------------------------------


def test_redact_before_send_redacts_prompt(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    monkeypatch.setattr(
        aistamp.pii,
        "redact_text",
        lambda text, matches=None, placeholder="[REDACTED]": placeholder,
        raising=False,
    )
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "ok"

    client = _client(llm, backend, _config(redact_before_send=True))
    result = client.stamp("my email is a@b.com")
    assert seen == ["[REDACTED]"]  # provider never saw the raw prompt
    assert result.record.prompt_hash == hash_content("[REDACTED]")


def test_redact_before_send_without_redact_text_refuses(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    monkeypatch.delattr(aistamp.pii, "redact_text", raising=False)
    client = _client(lambda p: "ok", backend, _config(redact_before_send=True))
    with pytest.raises(AIStampError, match="redact_text"):
        client.stamp("secret prompt")


# ---------------------------------------------------------------------------
# Config hardening
# ---------------------------------------------------------------------------


def test_yaml_env_interpolation(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('secret_key: "${TEST_SK}"\ndatabase_url: "sqlite:///:memory:"\n')
    monkeypatch.setenv("TEST_SK", "k" * 32)
    config = Config.from_yaml(path)
    assert config.secret_key_value == "k" * 32


def test_yaml_missing_env_var_raises_config_error(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        'secret_key: "${TEST_SK_ABSENT}"\ndatabase_url: "sqlite:///:memory:"\n'
    )
    monkeypatch.delenv("TEST_SK_ABSENT", raising=False)
    with pytest.raises(ConfigError, match="TEST_SK_ABSENT"):
        Config.from_yaml(path)


def test_missing_yaml_file_raises_config_error(tmp_path: Any) -> None:
    with pytest.raises(ConfigError, match="not found"):
        Config.from_yaml(tmp_path / "nope.yaml")


def test_database_url_scheme_rejected() -> None:
    # Field-level validation keeps the 0.1 ValidationError contract; the
    # message names the offending scheme and the supported set.
    with pytest.raises(ValidationError, match="Unsupported database_url scheme 'ftp'"):
        Config(secret_key="a" * 32, database_url="ftp://nope")


def test_config_key_id_default() -> None:
    assert _config().key_id == "default"


def test_config_key_id_is_the_rotation_declaration_point(
    backend: SQLiteBackend,
) -> None:
    # Key-rotation callers declare the active key id once on Config; every
    # record produced under that config is signed and persisted with it.
    config = _config(key_id="key-2026-01")
    client = _client(lambda p: "ok", backend, config)
    result = client.stamp("q")
    assert result.record.key_id == "key-2026-01"
    report = backend.query(QueryFilters(user_id="user"))
    assert report.records[0].key_id == "key-2026-01"


def test_secret_key_never_leaks_via_repr_str_or_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    secret = "super-secret-value-0123456789abcdef"
    config = Config(secret_key=secret, database_url="sqlite:///:memory:")
    assert secret not in repr(config)
    assert secret not in str(config)
    assert secret not in repr(config.secret_key)
    assert secret not in str(config.secret_key)
    # The plaintext is reachable only through the explicit secret_key_value
    # accessor — never through incidental stringification.
    assert config.secret_key_value == secret
    leaky_logger = logging.getLogger("aistamp.test.leak-check")
    with caplog.at_level(logging.DEBUG, logger="aistamp.test.leak-check"):
        leaky_logger.warning("loaded config: %s", config)
        leaky_logger.warning("secret field: %s", config.secret_key)
    assert secret not in caplog.text


def test_redact_before_send_off_by_default_sends_raw_prompt(
    backend: SQLiteBackend,
) -> None:
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "ok"

    client = _client(llm, backend)  # default Config: redact_before_send=False
    result = client.stamp("my email is a@b.com")
    assert seen == ["my email is a@b.com"]  # raw prompt reaches the provider
    assert result.record.prompt_hash == hash_content("my email is a@b.com")


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_stamp_rich_result_with_fake_async_openai(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAsyncOpenAI(response=_FakeOpenAIResponse("async ok"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    result = await client.stamp("q", model="gpt-4o")
    assert result.text == "async ok"
    assert result.usage == TokenUsage(prompt_tokens=11, response_tokens=7)
    assert result.record.status.value == "COMPLETED"
    assert fake.calls[0]["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_async_stamp_uses_aiosqlite_default_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    fake = _FakeAsyncOpenAI(response=_FakeOpenAIResponse("persisted"))
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    db_path = tmp_path / "v2async.db"
    config = _config(database_url=f"sqlite:///{db_path}")
    client = AsyncProvenanceClient(
        fake, config=config, app_id="app", feature_id="feat", user_id="user"
    )
    result = await client.stamp("q")
    await client.aclose()
    # Parity: the default async backend persisted a complete record.
    sync_reader = SQLiteBackend(f"sqlite:///{db_path}")
    report = sync_reader.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].content_id == result.content_id


@pytest.mark.asyncio
async def test_async_retries_rate_limit(
    monkeypatch: pytest.MonkeyPatch,
    backend: SQLiteBackend,
    asleep_log: list[float],
) -> None:
    fake = _FakeAsyncOpenAI(
        response=_FakeOpenAIResponse("ok"),
        errors=[_FakeRateLimitError("busy"), _FakeRateLimitError("still busy")],
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
        max_retries=3,
    )
    result = await client.stamp("q")
    assert result.text == "ok"
    assert len(fake.calls) == 3
    assert len(asleep_log) == 2


@pytest.mark.asyncio
async def test_async_error_carries_content_id(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend, asleep_log: list[float]
) -> None:
    fake = _FakeAsyncOpenAI(
        response=_FakeOpenAIResponse("ok"),
        errors=[_FakeRateLimitError("busy")] * 10,
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
        max_retries=1,
    )
    with pytest.raises(ProviderRateLimitError) as excinfo:
        await client.stamp("q")
    assert excinfo.value.content_id


@pytest.mark.asyncio
async def test_async_on_persist_error_awaits_callback(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    def _boom(record: Any, hmac: str) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(backend, "write", _boom)
    seen: list[tuple[Any, BaseException]] = []

    async def on_error(record: Any, exc: BaseException) -> None:
        seen.append((record, exc))

    client = AsyncProvenanceClient(
        lambda p: "ok",
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
        on_persist_error=on_error,
    )
    result = await client.stamp("q")
    assert result.text == "ok"
    assert len(seen) == 1
    assert seen[0][0].content_id == result.content_id


@pytest.mark.asyncio
async def test_async_redact_before_send_redacts_prompt(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    monkeypatch.setattr(
        aistamp.pii,
        "redact_text",
        lambda text, matches=None, placeholder="[REDACTED]": placeholder,
        raising=False,
    )
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "ok"

    client = AsyncProvenanceClient(
        llm,
        config=_config(redact_before_send=True),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    result = await client.stamp("my email is a@b.com")
    assert seen == ["[REDACTED]"]
    assert result.record.prompt_hash == hash_content("[REDACTED]")


@pytest.mark.asyncio
async def test_async_create_tables_parity(tmp_path: Any) -> None:
    db_path = tmp_path / "parity.db"
    config = _config(database_url=f"sqlite:///{db_path}")
    client = AsyncProvenanceClient(
        lambda p: "x", config=config, app_id="app", feature_id="feat", user_id="user"
    )
    await client.create_tables()
    result = await client.stamp("q")
    await client.aclose()
    sync_reader = SQLiteBackend(f"sqlite:///{db_path}")
    report = sync_reader.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].content_id == result.content_id


@pytest.mark.asyncio
async def test_async_aclose_is_idempotent(tmp_path: Any) -> None:
    config = _config(database_url=f"sqlite:///{tmp_path / 'aclose.db'}")
    client = AsyncProvenanceClient(
        lambda p: "x", config=config, app_id="app", feature_id="feat", user_id="user"
    )
    await client.aclose()
    await client.aclose()  # second dispose is a no-op, not an error


@pytest.mark.asyncio
async def test_async_stream_stamps_concatenation(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAsyncOpenAI(
        stream_chunks=[
            _chunk("one "),
            _chunk("two", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=3)),
        ]
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    stream = await client.stamp_stream("q")
    parts = [chunk async for chunk in stream]
    assert parts == ["one ", "two"]
    result = stream.result
    assert result.text == "one two"
    assert result.usage == TokenUsage(prompt_tokens=5, response_tokens=3)
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1


# ---------------------------------------------------------------------------
# Coverage: config hardening edge paths
# ---------------------------------------------------------------------------


def test_config_env_interpolation_missing_var_raises(tmp_path: Any) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(
        f"secret_key: {_SECRET}\n"
        "database_url: sqlite:///./x.db\n"
        "log_level: ${AISTAMP_TEST_MISSING_ENV}\n"
    )
    with pytest.raises(ConfigError, match="AISTAMP_TEST_MISSING_ENV"):
        Config.from_yaml(path)


def test_interpolate_env_vars_lists_and_dicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AISTAMP_TEST_V", "v")
    out = interpolate_env_vars(
        ["${AISTAMP_TEST_V}", {"k": "${AISTAMP_TEST_V}", "n": 1}]
    )
    assert out == ["v", {"k": "v", "n": 1}]


def test_config_rejects_database_url_without_scheme() -> None:
    with pytest.raises(ValidationError, match="scheme"):
        Config(secret_key=_SECRET, database_url="aistamp.db")


def test_config_from_env_optional_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AISTAMP_SECRET_KEY", _SECRET)
    monkeypatch.setenv("AISTAMP_DATABASE_URL", "sqlite:///./env.db")
    monkeypatch.setenv("AISTAMP_LOG_LEVEL", "debug")
    monkeypatch.setenv("AISTAMP_KEY_ID", "k-1")
    monkeypatch.setenv("AISTAMP_REDACT_BEFORE_SEND", "true")
    cfg = Config.from_env()
    assert cfg.database_url == "sqlite:///./env.db"
    assert cfg.log_level == "DEBUG"
    assert cfg.key_id == "k-1"
    assert cfg.redact_before_send is True


def test_config_from_yaml_rejects_non_mapping(tmp_path: Any) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("- one\n- two\n")
    with pytest.raises(ConfigError, match="YAML mapping"):
        Config.from_yaml(path)


def test_config_from_yaml_requires_secret_key(tmp_path: Any) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text("database_url: sqlite:///./x.db\n")
    with pytest.raises(ConfigError, match="secret_key"):
        Config.from_yaml(path)


def test_config_load_dispatch(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cfg.yaml"
    path.write_text(f"secret_key: {_SECRET}\n")
    assert Config.load(path).secret_key_value == _SECRET
    monkeypatch.setenv("AISTAMP_SECRET_KEY", _SECRET)
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    assert Config.load(None).secret_key_value == _SECRET


def test_config_yaml_roundtrip_with_interpolation(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AISTAMP_TEST_KID", "rotated")
    path = tmp_path / "cfg.yaml"
    path.write_text(f"secret_key: {_SECRET}\nkey_id: ${{AISTAMP_TEST_KID}}\n")
    assert Config.from_yaml(path).key_id == "rotated"


# ---------------------------------------------------------------------------
# Coverage: HTTP transport error taxonomy
# ---------------------------------------------------------------------------


def test_wrap_http_error_taxonomy() -> None:
    assert isinstance(wrap_http_error(TimeoutError("t")), ProviderTimeoutError)
    auth = wrap_http_error(HTTPError("http://x", 401, "no", {}, None))
    assert isinstance(auth, ProviderAuthError)
    assert auth.status_code == 401
    resp = wrap_http_error(HTTPError("http://x", 503, "down", {}, None))
    assert isinstance(resp, ProviderResponseError)
    assert resp.status_code == 503
    assert isinstance(
        wrap_http_error(URLError(TimeoutError("socket"))), ProviderTimeoutError
    )
    conn = wrap_http_error(URLError(OSError("refused")))
    assert isinstance(conn, ProviderResponseError)


def test_parse_retry_after_edge_cases() -> None:
    assert http_module._parse_retry_after(None) is None
    assert http_module._parse_retry_after({"Retry-After": "soon"}) is None


def test_generic_http_client_timeout_wraps_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = GenericHTTPClient("http://provider.invalid/v1/chat")

    def _raise(*args: Any, **kwargs: Any) -> None:
        raise TimeoutError("socket timed out")

    monkeypatch.setattr(http_module, "urlopen", _raise)
    with pytest.raises(ProviderTimeoutError):
        client.complete("p", "m")


# ---------------------------------------------------------------------------
# Coverage: constructor validation and stream failure paths
# ---------------------------------------------------------------------------


def test_sync_constructor_validation(backend: SQLiteBackend) -> None:
    def _mk(**kw: Any) -> ProvenanceClient:
        return _client(lambda p: "x", backend, **kw)

    with pytest.raises(ValueError, match="max_retries"):
        _mk(max_retries=-1)
    with pytest.raises(ValueError, match="delays"):
        _mk(retry_base_delay=0)
    with pytest.raises(ValueError, match="request_timeout"):
        _mk(request_timeout=0)
    with pytest.raises(ValueError, match="max_tokens"):
        _mk(max_tokens=0)

    async def _cb(record: Any, error: Any) -> None:
        return None

    with pytest.raises(TypeError, match="synchronous callable"):
        _mk(on_persist_error=_cb)


def test_async_constructor_validation(backend: SQLiteBackend) -> None:
    def _mk(**kw: Any) -> AsyncProvenanceClient:
        return AsyncProvenanceClient(
            lambda p: "x",
            config=_config(),
            app_id="app",
            feature_id="feat",
            user_id="user",
            backend=backend,
            **kw,
        )

    with pytest.raises(ValueError, match="max_retries"):
        _mk(max_retries=-1)
    with pytest.raises(ValueError, match="delays"):
        _mk(retry_base_delay=0)
    with pytest.raises(ValueError, match="request_timeout"):
        _mk(request_timeout=0)
    with pytest.raises(ValueError, match="max_tokens"):
        _mk(max_tokens=0)
    with pytest.raises(TypeError, match="callable"):
        _mk(on_persist_error="nope")


class _BrokenStreamOpenAI(_FakeOpenAI):
    """Streams one chunk then explodes mid-iteration."""

    def _create(self, **kwargs: Any) -> Iterator[Any]:
        self.calls.append(kwargs)

        def _gen() -> Iterator[Any]:
            yield _chunk("partial ")
            raise RuntimeError("mid-stream explosion")

        return _gen()


class _BrokenAsyncStreamOpenAI(_FakeAsyncOpenAI):
    """Async twin: streams one chunk then explodes mid-iteration."""

    async def _create(self, **kwargs: Any) -> AsyncIterator[Any]:
        self.calls.append(kwargs)

        async def _gen() -> AsyncIterator[Any]:
            yield _chunk("partial ")
            raise RuntimeError("async mid-stream explosion")

        return _gen()


def test_sync_stream_mid_stream_error_persists_record(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    _install_sdk(
        monkeypatch, "openai", OpenAI=_BrokenStreamOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = _client(_BrokenStreamOpenAI(), backend)
    stream = client.stamp_stream("q")
    with pytest.raises(StampError, match="mid-stream explosion"):
        list(stream)
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].error_type == "RuntimeError"
    assert report.records[0].error_message == "mid-stream explosion"


@pytest.mark.asyncio
async def test_async_stream_mid_stream_error_persists_record(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    _install_sdk(
        monkeypatch,
        "openai",
        OpenAI=_FakeOpenAI,
        AsyncOpenAI=_BrokenAsyncStreamOpenAI,
    )
    client = AsyncProvenanceClient(
        _BrokenAsyncStreamOpenAI(),
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    stream = await client.stamp_stream("q")
    with pytest.raises(StampError, match="async mid-stream explosion"):
        [chunk async for chunk in stream]
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].error_type == "RuntimeError"


def test_sync_stream_dispatch_fallthrough_rejects(backend: SQLiteBackend) -> None:
    client = _client(GenericHTTPClient("http://provider.invalid"), backend)
    with pytest.raises(StampError, match="streaming is not supported"):
        list(client._stream_dispatch("q", "m", {}, []))


@pytest.mark.asyncio
async def test_async_stream_result_unavailable_before_exhaustion(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAsyncOpenAI(stream_chunks=[_chunk("a")])
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    stream = await client.stamp_stream("q")
    with pytest.raises(StampError, match="fully consumed"):
        stream.result  # noqa: B018 — the property raising IS the assertion


@pytest.mark.asyncio
async def test_async_stream_stamps_concatenation_anthropic(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAsyncAnthropic(
        stream_texts=["alpha ", "beta"],
        final_usage=SimpleNamespace(input_tokens=7, output_tokens=9),
    )
    _install_sdk(
        monkeypatch,
        "anthropic",
        Anthropic=_FakeAnthropic,
        AsyncAnthropic=_FakeAsyncAnthropic,
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    stream = await client.stamp_stream("q")
    parts = [chunk async for chunk in stream]
    assert parts == ["alpha ", "beta"]
    result = stream.result
    assert result.text == "alpha beta"
    assert result.usage == TokenUsage(prompt_tokens=7, response_tokens=9)


# ---------------------------------------------------------------------------
# Coverage: async dispatch error paths and policy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_stamp_wraps_generic_provider_exception(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    fake = _FakeAsyncOpenAI(errors=[RuntimeError("boom")])
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    with pytest.raises(StampError, match="boom") as exc_info:
        await client.stamp("q")
    assert exc_info.value.content_id is not None
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].error_type == "RuntimeError"


@pytest.mark.asyncio
async def test_async_stamp_propagates_ai_stamp_error(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    fake = _FakeAsyncOpenAI(errors=[ProviderAuthError("denied")])
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    with pytest.raises(ProviderAuthError) as exc_info:
        await client.stamp("q")
    assert exc_info.value.content_id is not None


@pytest.mark.asyncio
async def test_async_policy_block_persists_error_record(
    backend: SQLiteBackend,
) -> None:
    engine = PolicyEngine(
        rules=[RuleConfig("block_all", RuleConditions(), PolicyAction.BLOCK)],
        model_tiers={},
    )
    client = AsyncProvenanceClient(
        lambda p: "x",
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
        engine=engine,
    )
    with pytest.raises(PolicyViolationError):
        await client.stamp("q")
    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1  # blocked call still persists evidence


@pytest.mark.asyncio
async def test_async_callable_non_str_rejected(backend: SQLiteBackend) -> None:
    client = AsyncProvenanceClient(
        lambda p: 123,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    with pytest.raises(StampError, match="must return str"):
        await client.stamp("q")


@pytest.mark.asyncio
async def test_async_create_tables_with_sync_backend(backend: SQLiteBackend) -> None:
    client = AsyncProvenanceClient(
        lambda p: "x",
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    await client.create_tables()  # sync backend DDL stays on the loop


def test_build_default_async_backend_postgres() -> None:
    pytest.importorskip("asyncpg")
    backend = _build_default_async_backend("postgresql+asyncpg://u:p@localhost/db")
    assert isinstance(backend, AsyncPostgreSQLBackend)
    with pytest.raises(ConfigError, match="database_url"):
        _build_default_async_backend("mysql://u:p@localhost/db")


# ---------------------------------------------------------------------------
# Coverage: pipeline helpers (classification, policy WARN, persist hooks)
# ---------------------------------------------------------------------------


def _ctx(**overrides: Any) -> CaptureContext:
    fields: dict[str, Any] = dict(
        content_id=str(uuid.uuid4()),
        app_id="app",
        feature_id="feat",
        user_id="user",
        model="m",
        prompt="p",
        prompt_hash=hash_content("p"),
        start_time=0,
        timestamp=datetime.now(timezone.utc),
    )
    fields.update(overrides)
    return CaptureContext(**fields)


def _pii_result() -> PIIResult:
    return PIIResult(
        prompt_matches=[], response_matches=[], highest_severity=None, match_count=0
    )


def test_classify_provider_error_matrix() -> None:
    class _Status(Exception):
        def __init__(self, status: int) -> None:
            self.status_code = status

    rate = classify_provider_error(_Status(429), content_id="c")
    assert isinstance(rate, ProviderRateLimitError)
    assert rate.content_id == "c"
    assert isinstance(
        classify_provider_error(_Status(500), content_id="c"), ProviderResponseError
    )
    assert isinstance(
        classify_provider_error(TimeoutError("t"), content_id="c"), ProviderTimeoutError
    )
    generic = classify_provider_error(RuntimeError("x"), content_id="c")
    assert isinstance(generic, StampError)
    assert generic.content_id == "c"
    already = ProviderAuthError("a")
    out = classify_provider_error(already, content_id="c")
    assert out is already
    assert already.content_id == "c"


def test_apply_pre_call_policy_warn_branch() -> None:
    ctx = _ctx()
    engine = PolicyEngine(
        rules=[RuleConfig("warn_all", RuleConditions(), PolicyAction.WARN)],
        model_tiers={},
    )
    apply_pre_call_policy(ctx, _pii_result(), engine)
    assert ctx.policy_decision is not None
    assert ctx.policy_decision.action == PolicyAction.WARN


def test_redact_prompt_missing_module_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "aistamp.pii", None)
    with pytest.raises(AIStampError, match="redact_before_send"):
        pipeline_module.redact_prompt("hello")


def test_report_persist_failure_callback_invoked_and_swallowed(
    backend: SQLiteBackend,
) -> None:
    record = pipeline_module.build_record(_ctx())
    calls: list[Any] = []
    report_persist_failure(
        record, RuntimeError("persist down"), lambda r, e: calls.append((r, e))
    )
    assert len(calls) == 1

    def _bad(record: Any, error: Any) -> None:
        raise RuntimeError("callback blew up")

    report_persist_failure(record, RuntimeError("persist down"), _bad)  # must not raise


@pytest.mark.asyncio
async def test_report_persist_failure_async_awaits_and_swallows(
    backend: SQLiteBackend,
) -> None:
    record = pipeline_module.build_record(_ctx())
    calls: list[str] = []

    async def _cb(r: Any, e: Any) -> None:
        calls.append(r.content_id)

    await report_persist_failure_async(record, RuntimeError("x"), _cb)
    assert calls == [record.content_id]

    async def _bad(r: Any, e: Any) -> None:
        raise RuntimeError("cb")

    await report_persist_failure_async(record, RuntimeError("x"), _bad)  # swallowed


def test_try_persist_sync_reports_failure(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()

    def _fail(record: Any, config: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(pipeline_module, "persist_record", _fail)
    calls: list[Any] = []
    try_persist_sync(ctx, backend, _config(), lambda r, e: calls.append(r))
    assert len(calls) == 1


def test_try_persist_sync_build_failure_is_swallowed(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(record: Any) -> ProvenanceRecord:
        raise RuntimeError("no record")

    monkeypatch.setattr(pipeline_module, "build_record", _boom)
    try_persist_sync(_ctx(), backend, _config(), None)  # must not raise


@pytest.mark.asyncio
async def test_try_persist_async_reports_failure(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = _ctx()

    async def _fail(record: Any, config: Any) -> None:
        raise RuntimeError("db down")

    monkeypatch.setattr(pipeline_module, "persist_record_async", _fail)
    calls: list[Any] = []
    await try_persist_async(ctx, backend, _config(), lambda r, e: calls.append(r))
    assert len(calls) == 1


@pytest.mark.asyncio
async def test_try_persist_async_build_failure_is_swallowed(
    backend: SQLiteBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(record: Any) -> ProvenanceRecord:
        raise RuntimeError("no record")

    monkeypatch.setattr(pipeline_module, "build_record", _boom)
    await try_persist_async(_ctx(), backend, _config(), None)  # must not raise


# ---------------------------------------------------------------------------
# P0 hotfix: streaming redaction parity, abandoned-stream evidence,
# off-loop sync callables
# ---------------------------------------------------------------------------


def test_sync_stream_redacts_before_send(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    monkeypatch.setattr(
        aistamp.pii,
        "redact_text",
        lambda text, matches=None, placeholder="[REDACTED]": placeholder,
        raising=False,
    )
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    fake = _FakeOpenAI(stream_chunks=[_chunk("ok-1"), _chunk("-2")])
    client = _client(fake, backend, _config(redact_before_send=True))

    stream = client.stamp_stream("my email is a@b.com")
    assert "".join(stream) == "ok-1-2"

    # The provider must never see the raw prompt, and the persisted
    # evidence must cover the redacted text.
    assert fake.calls[0]["messages"][0]["content"] == "[REDACTED]"
    assert stream.result.record.prompt_hash == hash_content("[REDACTED]")


@pytest.mark.asyncio
async def test_async_stream_redacts_before_send(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    monkeypatch.setattr(
        aistamp.pii,
        "redact_text",
        lambda text, matches=None, placeholder="[REDACTED]": placeholder,
        raising=False,
    )
    fake = _FakeAsyncOpenAI(stream_chunks=[_chunk("ok-1"), _chunk("-2")])
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(redact_before_send=True),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )

    chunks: list[str] = []
    stream = await client.stamp_stream("my email is a@b.com")
    async for chunk in stream:
        chunks.append(chunk)
    assert "".join(chunks) == "ok-1-2"

    assert fake.calls[0]["messages"][0]["content"] == "[REDACTED]"
    report = backend.query(QueryFilters(user_id="user"))
    assert report.records[0].prompt_hash == hash_content("[REDACTED]")


def test_sync_stream_abandoned_persists_error_record(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    fake = _FakeOpenAI(stream_chunks=[_chunk("a"), _chunk("b"), _chunk("c")])
    client = _client(fake, backend)

    stream = client.stamp_stream("q")
    it = stream._chunks
    assert next(it) == "a"
    it.close()  # abandon mid-stream: no result is ever stamped

    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].status == RecordStatus.ERROR
    assert report.records[0].error_type == "StampError"
    assert report.records[0].error_message == "stream abandoned before completion"


@pytest.mark.asyncio
async def test_async_stream_abandoned_persists_error_record(
    monkeypatch: pytest.MonkeyPatch, backend: SQLiteBackend
) -> None:
    fake = _FakeAsyncOpenAI(stream_chunks=[_chunk("a"), _chunk("b"), _chunk("c")])
    _install_sdk(
        monkeypatch, "openai", OpenAI=_FakeOpenAI, AsyncOpenAI=_FakeAsyncOpenAI
    )
    client = AsyncProvenanceClient(
        fake,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )

    stream = await client.stamp_stream("q")
    agen = stream._chunks
    assert await agen.__anext__() == "a"
    await agen.aclose()  # abandon mid-stream

    report = backend.query(QueryFilters(user_id="user"))
    assert report.total_count == 1
    assert report.records[0].status == RecordStatus.ERROR
    assert report.records[0].error_type == "StampError"
    assert report.records[0].error_message == "stream abandoned before completion"


@pytest.mark.asyncio
async def test_async_sync_callable_runs_off_loop(backend: SQLiteBackend) -> None:
    loop_thread = threading.get_ident()
    call_threads: list[int] = []

    def blocking_llm(prompt: str) -> str:
        call_threads.append(threading.get_ident())
        return "ok"

    client = AsyncProvenanceClient(
        blocking_llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    result = await client.stamp("q")
    assert result.text == "ok"
    assert call_threads and call_threads[0] != loop_thread


@pytest.mark.asyncio
async def test_async_awaitable_callable_still_works(backend: SQLiteBackend) -> None:
    async def async_llm(prompt: str) -> str:
        return f"hello {prompt}"

    client = AsyncProvenanceClient(
        async_llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    result = await client.stamp("q")
    assert result.text == "hello q"
