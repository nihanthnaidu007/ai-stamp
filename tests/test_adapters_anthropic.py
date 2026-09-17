"""Adapter tests for the Anthropic sync/async code paths.

Mirrors tests/test_adapters_openai.py: the real ``anthropic`` package is not
a test dependency, so a fake module injected through ``sys.modules`` drives
the exact isinstance-dispatch the production code performs.
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

DEFAULT_ANTHROPIC_MODEL = "claude-3-5-sonnet-20241022"


class _FakeContentBlock:
    def __init__(self, text: str | None) -> None:
        self.text = text


class _FakeUsage:
    def __init__(self, input_tokens: int | None, output_tokens: int | None) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeMessageResponse:
    def __init__(
        self, content: list[_FakeContentBlock], usage: _FakeUsage | None
    ) -> None:
        self.content = content
        self.usage = usage


Behavior = Callable[[str, list[dict[str, str]]], _FakeMessageResponse]


def _echo_behavior(model: str, messages: list[dict[str, str]]) -> _FakeMessageResponse:
    return _FakeMessageResponse(
        content=[_FakeContentBlock(f"Reply to: {messages[0]['content']}")],
        usage=_FakeUsage(input_tokens=21, output_tokens=43),
    )


class _FakeSyncMessages:
    def __init__(self, behavior: Behavior) -> None:
        self._behavior = behavior

    def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> _FakeMessageResponse:
        return self._behavior(model, messages)


class _FakeAsyncMessages:
    def __init__(self, behavior: Behavior) -> None:
        self._behavior = behavior

    async def create(
        self,
        *,
        model: str,
        max_tokens: int,
        messages: list[dict[str, str]],
    ) -> _FakeMessageResponse:
        return self._behavior(model, messages)


@dataclass(frozen=True)
class _FakeSdk:
    """Injected fake ``anthropic`` module plus the client classes it defines."""

    module: types.ModuleType
    sync_cls: type
    async_cls: type


def _install_fake_anthropic(
    monkeypatch: pytest.MonkeyPatch,
    behavior: Behavior,
) -> _FakeSdk:
    module = types.ModuleType("anthropic")

    class _SyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _FakeSyncMessages(behavior)

    class _AsyncAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            self.messages = _FakeAsyncMessages(behavior)

    # ModuleType carries no statically declared attributes; the adapter
    # resolves these names dynamically at import time.
    sdk_module: Any = module
    sdk_module.Anthropic = _SyncAnthropic
    sdk_module.AsyncAnthropic = _AsyncAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return _FakeSdk(module=module, sync_cls=_SyncAnthropic, async_cls=_AsyncAnthropic)


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


# --- End-to-end stamping through the Anthropic adapter ----------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_anthropic_adapter_stamps_record(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    sdk = _install_fake_anthropic(monkeypatch, _echo_behavior)
    config = _make_config()
    backend = _make_backend(config)

    response = await _chat(
        use_async, sdk, config, backend, "hello world", model="claude-3-5-haiku"
    )

    assert response == "Reply to: hello world"
    report = backend.query(QueryFilters())
    assert report.total_count == 1
    record = report.records[0]
    assert record.status == RecordStatus.COMPLETED
    assert record.model == "claude-3-5-haiku"
    assert record.prompt_hash == hash_content("hello world")
    assert record.response_hash == hash_content(response)
    assert record.prompt_tokens == 21
    assert record.response_tokens == 43
    assert record.pii_result is not None and record.pii_result.match_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_anthropic_adapter_resolves_default_model(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    captured_models: list[str] = []

    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeMessageResponse:
        captured_models.append(model)
        return _FakeMessageResponse(
            content=[_FakeContentBlock("ok")], usage=_FakeUsage(1, 1)
        )

    sdk = _install_fake_anthropic(monkeypatch, behavior)
    config = _make_config()
    backend = _make_backend(config)

    await _chat(use_async, sdk, config, backend, "hi")

    # The adapter must infer the Anthropic default model when none is given.
    assert captured_models == [DEFAULT_ANTHROPIC_MODEL]
    report = backend.query(QueryFilters())
    assert report.records[0].model == DEFAULT_ANTHROPIC_MODEL


# --- Response-shape sweep (extraction robustness) ---------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize(
    ("blocks", "usage", "expected_tokens"),
    [
        ([_FakeContentBlock("text!")], _FakeUsage(7, 8), (7, 8)),
        ([_FakeContentBlock("text!")], None, (None, None)),
        ([], _FakeUsage(7, 8), (7, 8)),
    ],
    ids=["full-usage", "no-usage", "empty-content"],
)
async def test_anthropic_adapter_response_shape_sweep(
    monkeypatch: pytest.MonkeyPatch,
    use_async: bool,
    blocks: list[_FakeContentBlock],
    usage: _FakeUsage | None,
    expected_tokens: tuple[int | None, int | None],
) -> None:
    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeMessageResponse:
        return _FakeMessageResponse(content=blocks, usage=usage)

    sdk = _install_fake_anthropic(monkeypatch, behavior)
    config = _make_config()
    backend = _make_backend(config)

    response = await _chat(
        use_async, sdk, config, backend, "p", model=DEFAULT_ANTHROPIC_MODEL
    )

    expected_text = blocks[0].text if blocks else ""
    assert response == expected_text
    record = backend.query(QueryFilters()).records[0]
    assert record.prompt_tokens == expected_tokens[0]
    assert record.response_tokens == expected_tokens[1]


# --- Error handling ----------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True], ids=["sync", "async"])
async def test_anthropic_adapter_sdk_error_wrapped_as_stamp_error(
    monkeypatch: pytest.MonkeyPatch, use_async: bool
) -> None:
    def behavior(model: str, messages: list[dict[str, str]]) -> _FakeMessageResponse:
        raise RuntimeError("boom")

    sdk = _install_fake_anthropic(monkeypatch, behavior)
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
