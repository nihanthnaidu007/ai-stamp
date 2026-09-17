"""Offline provider fakes for the v0.2 compatibility kit.

Mirrors the sys.modules-injection pattern of ``tests/test_client_v2.py``: the
streaming adapters isinstance-check against ``openai.OpenAI`` /
``openai.AsyncOpenAI``, so the kit installs a fake ``openai`` module holding
these classes. Every fake records the exact kwargs it was called with so the
redaction checks can assert on what the provider really received.
"""

from __future__ import annotations

import sys
import types
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest


def chunk(delta: str, usage: Any = None) -> SimpleNamespace:
    """One OpenAI-style streaming chunk."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=delta))],
        usage=usage,
    )


class FakeOpenAI:
    """Offline stand-in for ``openai.OpenAI`` (non-streaming + streaming)."""

    def __init__(
        self,
        *,
        response: Any = None,
        stream_chunks: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._stream_chunks = stream_chunks or []
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self._stream_chunks)
        if self._response is None:
            raise AssertionError("fake OpenAI configured without a response")
        return self._response


class FakeAsyncOpenAI:
    """Offline stand-in for ``openai.AsyncOpenAI``."""

    def __init__(
        self,
        *,
        response: Any = None,
        stream_chunks: list[Any] | None = None,
    ) -> None:
        self._response = response
        self._stream_chunks = stream_chunks or []
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return self._agen()
        if self._response is None:
            raise AssertionError("fake AsyncOpenAI configured without a response")
        return self._response

    async def _agen(self) -> AsyncIterator[Any]:
        for c in self._stream_chunks:
            yield c


def fake_response(
    text: str, *, prompt_tokens: int = 3, completion_tokens: int = 5
) -> Any:
    """OpenAI-style non-streaming response with usage."""
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=usage,
    )


def install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    sync_cls: type,
    async_cls: type,
) -> None:
    """Point the streaming adapters' ``import openai`` at the fakes."""
    module = types.ModuleType("openai")
    module.OpenAI = sync_cls  # type: ignore[attr-defined]
    module.AsyncOpenAI = async_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
