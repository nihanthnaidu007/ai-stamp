"""Adapter tests for the OpenAI sync/async code paths.

The real ``openai`` package is not a test dependency. Both adapters import it
lazily inside the call path and dispatch via ``isinstance``, so injecting a
fake ``openai`` module through ``sys.modules`` exercises the exact dispatch
the production code performs against the real SDK.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from aistamp.client import (
    AsyncProvenanceClient,
    ProvenanceClient,
    StampError,
)
from aistamp.config import Config
from aistamp.fingerprint import hash_content
from aistamp.models import QueryFilters, RecordStatus
from aistamp.store import SQLiteBackend

SECRET_KEY = "test-secret-key-for-aistamp-unit-tests-32chars"


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(
        self, prompt_tokens: int | None, completion_tokens: int | None
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _FakeCompletionResponse:
    def __init__(self, content: str | None, usage: _FakeUsage | None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = usage


class _FakeChat:
    def __init__(self, completions: Any) -> None:
        self.completions = completions


Behavior = Callable[[str, list[dict[str, str]]], _FakeCompletionResponse]


def _echo_behavior(
    model: str, messages: list[dict[str, str]]
) -> _FakeCompletionResponse:
    return _FakeCompletionResponse(
        content=f"Reply to: {messages[0]['content']}",
        usage=_FakeUsage(prompt_tokens=12, completion_tokens=34),
    )


@dataclass(frozen=True)
class _FakeSdk:
    """Injected fake ``openai`` module plus the client classes it defines."""

    module: types.ModuleType
    sync_cls: type
    async_cls: type


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    behavior: Behavior,
) -> _FakeSdk:
    module = types.ModuleType("openai")

    class _SyncCompletions:
        def create(
            self, *, model: str, messages: list[dict[str, str]]
        ) -> _FakeCompletionResponse:
            return behavior(model, messages)

    class _AsyncCompletions:
        async def create(
            self, *, model: str, messages: list[dict[str, str]]
        ) -> _FakeCompletionResponse:
            return behavior(model, messages)

    class _SyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _FakeChat(_SyncCompletions())

    class _AsyncOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            self.chat = _FakeChat(_AsyncCompletions())

    # ModuleType carries no statically declared attributes; the adapter
    # resolves these names dynamically at import time.
    sdk_module: Any = module
    sdk_module.OpenAI = _SyncOpenAI
    sdk_module.AsyncOpenAI = _AsyncOpenAI
    monkeypatch.setitem(sys.modules, "openai", module)
    return _FakeSdk(module=module, sync_cls=_SyncOpenAI, async_cls=_AsyncOpenAI)


def _make_config() -> Config:
    return Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
    )


def _make_backend(config: Config) -> SQLiteBackend:
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()
    return backend


async def _chat(
    use_async: bool,
    sdk: _FakeSdk,
    config: Config,
    backend: SQLiteBackend,
    prompt: str,
    model: str | None = None,
) -> str:
    """Dispatch one chat call through the sync or async adapter."""
    if use_async:
        client = AsyncProvenanceClient(
            sdk.async_cls(),
            config=config,
            app_id="test_app",
            feature_id="test_feature",
            user_id="test_user",
            backend=backend,
        )
        return await client.chat(prompt, model=model)
    sync_client = ProvenanceClient(
        sdk.sync_cls(),
        config=config,
        app_id="test_app",
        feature_id="test_feature",
        user_id="test_user",
        backend=backend,
    )
    return sync_client.chat(prompt, model=model)


def _rejection_client(
    use_async: bool, config: Config, backend: SQLiteBackend
) -> ProvenanceClient | AsyncProvenanceClient:
    """Build a client around an object no adapter branch accepts."""
    if use_async:
        return AsyncProvenanceClient(
            object(),
            config=config,
            app_id="a",
            feature_id="f",
            user_id="u",
            backend=backend,
        )
    return ProvenanceClient(
        object(),
        config=config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )


# --- End-to-end stamping through the OpenAI adapter -------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_openai_adapter_stamps_record(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    sdk = _install_fake_openai(monkeypatch, _echo_behavior)
    config = _make_config()
    backend = _make_backend(config)

    response = await _chat(
        use_async, sdk, config, backend, "hello world", model="gpt-4o-mini"
    )

    assert response == "Reply to: hello world"
    report = backend.query(QueryFilters())
    assert report.total_count == 1
    record = report.records[0]
    assert record.status == RecordStatus.COMPLETED
    assert record.model == "gpt-4o-mini"
    assert record.prompt_hash == hash_content("hello world")
    assert record.response_hash == hash_content(response)
    assert record.prompt_tokens == 12
    assert record.response_tokens == 34
    assert record.pii_result is not None and record.pii_result.match_count == 0
    # No engine configured → no policy evaluation, no decision recorded.
    assert record.policy_decision is None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_openai_adapter_resolves_default_model(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    captured_models: list[str] = []

    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeCompletionResponse:
        captured_models.append(model)
        return _FakeCompletionResponse("ok", _FakeUsage(1, 1))

    sdk = _install_fake_openai(monkeypatch, behavior)
    config = _make_config()
    backend = _make_backend(config)

    await _chat(use_async, sdk, config, backend, "hi")

    # The adapter must infer the OpenAI default model when none is given.
    assert captured_models == ["gpt-4o"]
    report = backend.query(QueryFilters())
    assert report.records[0].model == "gpt-4o"


# --- Response-shape sweep (extraction robustness) ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("content", "usage", "expected_tokens"),
    [
        ("text!", _FakeUsage(5, 6), (5, 6)),
        ("text!", None, (None, None)),
        (None, _FakeUsage(5, 6), (5, 6)),
    ],
    ids=["full-usage", "no-usage", "null-content"],
)
async def test_openai_adapter_response_shape_sweep(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    content: str | None,
    usage: _FakeUsage | None,
    expected_tokens: tuple[int | None, int | None],
) -> None:
    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeCompletionResponse:
        return _FakeCompletionResponse(content=content, usage=usage)

    sdk = _install_fake_openai(monkeypatch, behavior)
    config = _make_config()
    backend = _make_backend(config)

    response = await _chat(use_async, sdk, config, backend, "p", model="gpt-4o")

    expected_text = content if content is not None else ""
    assert response == expected_text
    record = backend.query(QueryFilters()).records[0]
    assert record.prompt_tokens == expected_tokens[0]
    assert record.response_tokens == expected_tokens[1]


# --- Error handling ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_openai_adapter_sdk_error_wrapped_as_stamp_error(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeCompletionResponse:
        raise RuntimeError("boom")

    sdk = _install_fake_openai(monkeypatch, behavior)
    config = _make_config()
    backend = _make_backend(config)

    with pytest.raises(StampError, match="LLM call failed: RuntimeError: boom"):
        await _chat(use_async, sdk, config, backend, "hi")

    # The failed call must still land in the audit trail as an ERROR record.
    report = backend.query(QueryFilters())
    assert report.total_count == 1
    record = report.records[0]
    assert record.status == RecordStatus.ERROR
    assert record.response_hash is None


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_adapter_rejects_unsupported_client_type(
    use_async: bool,
) -> None:
    # No fake module needed: object() matches no adapter branch.
    config = _make_config()
    backend = _make_backend(config)
    client = _rejection_client(use_async, config, backend)

    with pytest.raises(StampError, match="Unsupported LLM client type: object"):
        if isinstance(client, AsyncProvenanceClient):
            await client.chat("hi")
        else:
            client.chat("hi")
