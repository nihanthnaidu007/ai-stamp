from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import AsyncIterator
from typing import Any

from aistamp.client._pipeline import (
    CaptureContext,
    PersistErrorCallback,
    attach_content_id,
    build_pre_call_context,
    build_record,
    classify_provider_error,
    finalize_context_async,
    persist_record_async,
    redact_prompt,
    report_persist_failure_async,
    run_pre_call_policy_async,
    try_persist_async,
    validate_prompt,
)
from aistamp.client._providers import (
    _import_anthropic,
    _import_openai,
    build_create_kwargs,
    dispatch_sync,
    parse_anthropic_response,
    parse_openai_response,
    resolve_model,
    validate_llm_client,
    with_http_timeout,
)
from aistamp.client._retry import acall_with_retries
from aistamp.client.http import GenericHTTPClient
from aistamp.client.results import AsyncStreamStamp, StampResult, TokenUsage
from aistamp.config import Config
from aistamp.errors import AIStampError, ConfigError, StampError
from aistamp.fingerprint import hash_content
from aistamp.models import RecordStatus
from aistamp.pii.patterns import PatternConfig
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.store.async_backend import (
    AsyncSQLiteBackend,
    AsyncStoreBackend,
)
from aistamp.store.backend import SQLiteBackend, StoreBackend

logger = logging.getLogger("aistamp.client")

__all__ = ["AsyncProvenanceClient"]


def _build_default_async_backend(database_url: str) -> AsyncStoreBackend:
    """Default backend for the async client: aiosqlite (or asyncpg) engine.

    ``sqlite:///`` URLs are upgraded to the aiosqlite driver so writes run
    natively on the event loop.
    """
    url = database_url
    if url.startswith("sqlite:///"):
        url = "sqlite+aiosqlite:///" + url[len("sqlite:///") :]
    if url.startswith("sqlite+aiosqlite"):
        return AsyncSQLiteBackend(url)
    if url.startswith("postgresql+asyncpg"):
        from aistamp.store.async_backend import AsyncPostgreSQLBackend

        return AsyncPostgreSQLBackend(url)
    raise ConfigError(
        "AsyncProvenanceClient's default backend requires a sqlite:///"
        f" or postgresql+asyncpg:// database_url (got {database_url!r});"
        " pass an explicit async backend for other schemes."
    )


class AsyncProvenanceClient:
    """
    Wraps an async LLM client and stamps every call with a provenance record.

    Supports ``openai.AsyncOpenAI``, ``anthropic.AsyncAnthropic``,
    ``GenericHTTPClient``, and callables. Sync client types (sync SDK
    instances, callables, the HTTP fallback) are dispatched via
    ``asyncio.to_thread`` so the event loop is never blocked; sync backends
    passed explicitly are written off-loop the same way.

    Defaults to an aiosqlite backend; PII scans and redaction run in worker
    threads; ``create_tables()`` is available as a coroutine for backends
    that cannot run DDL from a sync constructor.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        config: Config,
        app_id: str,
        feature_id: str,
        user_id: str,
        engine: PolicyEngine | None = None,
        backend: AsyncStoreBackend | StoreBackend | None = None,
        extra_patterns: list[PatternConfig] | None = None,
        use_spacy: bool = False,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        on_persist_error: PersistErrorCallback | None = None,
    ) -> None:
        validate_llm_client(llm_client, is_async=True)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_delay <= 0 or retry_max_delay <= 0:
            raise ValueError("retry delays must be positive")
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if on_persist_error is not None and not callable(on_persist_error):
            raise TypeError("on_persist_error must be callable")

        self._llm_client = llm_client
        self._config = config
        self._app_id = app_id
        self._feature_id = feature_id
        self._user_id = user_id
        self._engine = engine
        self._extra_patterns = extra_patterns
        self._use_spacy = use_spacy
        self._max_tokens = max_tokens
        self._request_timeout = request_timeout
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._on_persist_error = on_persist_error

        if backend is None:
            self._backend: AsyncStoreBackend | StoreBackend = (
                _build_default_async_backend(config.database_url)
            )
            # create_tables parity with the sync client: SQLite DDL is cheap,
            # and the sync driver can run it from this sync constructor on the
            # same file the async engine uses. Non-sqlite default backends
            # create tables via `await client.create_tables()` or migrations.
            if isinstance(self._backend, AsyncSQLiteBackend):
                ddl_url = config.database_url
                if ddl_url.startswith("sqlite+aiosqlite:///"):
                    ddl_url = "sqlite:///" + ddl_url[len("sqlite+aiosqlite:///") :]
                SQLiteBackend(ddl_url).create_tables()
        else:
            self._backend = backend

        self._http_client = (
            with_http_timeout(llm_client, request_timeout)
            if isinstance(llm_client, GenericHTTPClient)
            else llm_client
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self, prompt: str, model: str | None = None, **provider_kwargs: Any
    ) -> str:
        """0.1.x-compatible surface: returns only the response text."""
        return (await self.stamp(prompt, model, **provider_kwargs)).text

    async def stamp(
        self,
        prompt: str,
        model: str | None = None,
        *,
        app_id: str | None = None,
        feature_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, str] | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        **provider_kwargs: Any,
    ) -> StampResult:
        """Run one stamped call and return text, content_id, record and usage.

        Per-call ``app_id`` / ``feature_id`` / ``user_id`` override the
        constructor values for this call only.
        """
        return await self._execute(
            prompt,
            model,
            app_id=app_id,
            feature_id=feature_id,
            user_id=user_id,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
            provider_kwargs=provider_kwargs,
        )

    async def chat_detailed(
        self,
        prompt: str,
        model: str | None = None,
        *,
        app_id: str | None = None,
        feature_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, str] | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        **provider_kwargs: Any,
    ) -> StampResult:
        """Alias for :meth:`stamp`."""
        return await self.stamp(
            prompt,
            model,
            app_id=app_id,
            feature_id=feature_id,
            user_id=user_id,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
            **provider_kwargs,
        )

    async def stamp_stream(
        self,
        prompt: str,
        model: str | None = None,
        *,
        app_id: str | None = None,
        feature_id: str | None = None,
        user_id: str | None = None,
        metadata: dict[str, str] | None = None,
        conversation_id: str | None = None,
        request_id: str | None = None,
        **provider_kwargs: Any,
    ) -> AsyncStreamStamp:
        """Stream provider chunks and stamp the final concatenation.

        Pre-call policy runs eagerly (a BLOCK raises here, before any chunk).
        ``async for`` over the returned :class:`AsyncStreamStamp` yields
        provider chunks; after exhaustion, ``stream.result`` holds the final
        :class:`StampResult`. Streaming is not retried — a partially consumed
        stream cannot be replayed safely.
        """
        validate_prompt(prompt)
        if not self._supports_streaming():
            raise StampError(
                "streaming is not supported for this LLM client type;"
                " use stamp() instead."
            )

        resolved_model = resolve_model(self._llm_client, model)
        ctx = self._build_context(
            prompt,
            resolved_model,
            app_id=app_id,
            feature_id=feature_id,
            user_id=user_id,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        await self._run_pre_call_policy_async(ctx)

        result_cell: list[StampResult] = []

        async def generate() -> AsyncIterator[str]:
            usage_cell: list[tuple[int | None, int | None]] = []
            chunks: list[str] = []
            try:
                async for chunk_text in self._stream_dispatch(
                    ctx.prompt, resolved_model, provider_kwargs, usage_cell
                ):
                    chunks.append(chunk_text)
                    yield chunk_text
            except Exception as exc:
                ctx.status = RecordStatus.ERROR
                ctx.error = exc
                attach_content_id(exc, ctx.content_id)
                await try_persist_async(
                    ctx, self._backend, self._config, self._on_persist_error
                )
                if isinstance(exc, AIStampError):
                    raise
                raise StampError(
                    f"LLM call failed: {type(exc).__name__}: {exc}",
                    content_id=ctx.content_id,
                ) from exc

            prompt_tokens, response_tokens = (
                usage_cell[-1] if usage_cell else (None, None)
            )
            result_cell.append(
                await self._finalize_and_persist_async(
                    ctx,
                    "".join(chunks),
                    prompt_tokens,
                    response_tokens,
                    metadata=metadata,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
            )

        return AsyncStreamStamp(generate(), result_cell)

    async def create_tables(self) -> None:
        """Create the schema on the configured backend (awaitable)."""
        if isinstance(self._backend, AsyncStoreBackend):
            await self._backend.create_tables()
        else:
            # Sync backends stay on the loop: thread-affine connections
            # (SQLite in-memory) make to_thread unsound for arbitrary sync
            # backends, and DDL is a one-shot call.
            self._backend.create_tables()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    async def _execute(
        self,
        prompt: str,
        model: str | None,
        *,
        app_id: str | None,
        feature_id: str | None,
        user_id: str | None,
        metadata: dict[str, str] | None,
        conversation_id: str | None,
        request_id: str | None,
        provider_kwargs: dict[str, Any],
    ) -> StampResult:
        validate_prompt(prompt)

        resolved_model = resolve_model(self._llm_client, model)
        ctx = self._build_context(
            prompt,
            resolved_model,
            app_id=app_id,
            feature_id=feature_id,
            user_id=user_id,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
        )
        await self._run_pre_call_policy_async(ctx)

        try:
            if self._config.redact_before_send:
                dispatch_prompt = await asyncio.to_thread(redact_prompt, ctx.prompt)
                ctx.prompt = dispatch_prompt
                ctx.prompt_hash = hash_content(dispatch_prompt)
            else:
                dispatch_prompt = ctx.prompt
            (
                response_text,
                prompt_tokens,
                response_tokens,
            ) = await self._dispatch_with_retries(
                dispatch_prompt, resolved_model, provider_kwargs
            )
        except AIStampError as exc:
            attach_content_id(exc, ctx.content_id)
            ctx.status = RecordStatus.ERROR
            ctx.error = exc
            await try_persist_async(
                ctx, self._backend, self._config, self._on_persist_error
            )
            raise
        except Exception as exc:
            ctx.status = RecordStatus.ERROR
            ctx.error = exc
            await try_persist_async(
                ctx, self._backend, self._config, self._on_persist_error
            )
            raise classify_provider_error(exc, ctx.content_id) from exc

        return await self._finalize_and_persist_async(
            ctx,
            response_text,
            prompt_tokens,
            response_tokens,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
        )

    def _build_context(
        self,
        prompt: str,
        resolved_model: str,
        *,
        app_id: str | None,
        feature_id: str | None,
        user_id: str | None,
        metadata: dict[str, str] | None,
        conversation_id: str | None,
        request_id: str | None,
    ) -> CaptureContext:
        return build_pre_call_context(
            prompt=prompt,
            model=resolved_model,
            app_id=app_id if app_id is not None else self._app_id,
            feature_id=feature_id if feature_id is not None else self._feature_id,
            user_id=user_id if user_id is not None else self._user_id,
            key_id=self._config.key_id,
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
        )

    async def _run_pre_call_policy_async(self, ctx: CaptureContext) -> None:
        try:
            await run_pre_call_policy_async(
                ctx, self._engine, self._extra_patterns, self._use_spacy
            )
        except PolicyViolationError:
            ctx.status = RecordStatus.BLOCKED
            await try_persist_async(
                ctx, self._backend, self._config, self._on_persist_error
            )
            raise

    async def _finalize_and_persist_async(
        self,
        ctx: CaptureContext,
        response_text: str,
        prompt_tokens: int | None,
        response_tokens: int | None,
        *,
        metadata: dict[str, str] | None,
        conversation_id: str | None,
        request_id: str | None,
    ) -> StampResult:
        try:
            await finalize_context_async(
                ctx,
                response_text,
                prompt_tokens,
                response_tokens,
                self._engine,
                self._extra_patterns,
                self._use_spacy,
            )
        except PolicyViolationError:
            await try_persist_async(
                ctx, self._backend, self._config, self._on_persist_error
            )
            raise

        record = build_record(ctx)
        try:
            await persist_record_async(record, self._backend, self._config)
        except Exception as exc:
            await report_persist_failure_async(record, exc, self._on_persist_error)

        return StampResult(
            text=response_text,
            content_id=ctx.content_id,
            record=record,
            decision=record.policy_decision,
            usage=(
                TokenUsage(prompt_tokens=prompt_tokens, response_tokens=response_tokens)
                if prompt_tokens is not None or response_tokens is not None
                else None
            ),
            metadata=metadata,
            conversation_id=conversation_id,
            request_id=request_id,
        )

    async def _dispatch_with_retries(
        self, prompt: str, model: str, provider_kwargs: dict[str, Any]
    ) -> tuple[str, int | None, int | None]:
        return await acall_with_retries(
            lambda: self._dispatch(prompt, model, provider_kwargs),
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            max_delay=self._retry_max_delay,
        )

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self, prompt: str, model: str, provider_kwargs: dict[str, Any]
    ) -> tuple[str, int | None, int | None]:
        client = self._http_client
        openai = _import_openai()
        if openai is not None and isinstance(client, openai.AsyncOpenAI):
            create_kwargs = build_create_kwargs(
                prompt,
                model,
                provider_kwargs,
                max_tokens=self._max_tokens,
                request_timeout=self._request_timeout,
            )
            resp = await client.chat.completions.create(**create_kwargs)
            return parse_openai_response(resp)

        anthropic = _import_anthropic()
        if anthropic is not None and isinstance(client, anthropic.AsyncAnthropic):
            create_kwargs = build_create_kwargs(
                prompt,
                model,
                provider_kwargs,
                max_tokens=self._max_tokens,
                request_timeout=self._request_timeout,
            )
            resp = await client.messages.create(**create_kwargs)
            return parse_anthropic_response(resp)

        # Async/awaitable callables run on the loop (0.1 contract: (str) ->
        # str | Awaitable[str]); sync callables also resolve here, since a
        # result's awaitability is only knowable after calling.
        if callable(client):
            result = client(prompt)
            text = await result if inspect.isawaitable(result) else result
            if not isinstance(text, str):
                raise StampError(
                    f"Generic callable must return str or Awaitable[str],"
                    f" got {type(text).__name__}"
                )
            return text, None, None

        # Remaining sync client types (HTTP fallback, sync SDK instances)
        # run off the event loop.
        return await asyncio.to_thread(
            dispatch_sync,
            client,
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )

    def _supports_streaming(self) -> bool:
        client = self._llm_client
        openai = _import_openai()
        if openai is not None and isinstance(client, openai.AsyncOpenAI):
            return True
        anthropic = _import_anthropic()
        if anthropic is not None and isinstance(client, anthropic.AsyncAnthropic):
            return True
        return False

    async def _stream_dispatch(
        self,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> AsyncIterator[str]:
        """Yield provider stream chunks; append final usage to *usage_cell*."""
        client = self._llm_client
        openai = _import_openai()
        if openai is not None and isinstance(client, openai.AsyncOpenAI):
            async for chunk in self._stream_openai_async(
                client, prompt, model, provider_kwargs, usage_cell
            ):
                yield chunk
            return
        anthropic = _import_anthropic()
        if anthropic is not None and isinstance(client, anthropic.AsyncAnthropic):
            async for text in self._stream_anthropic_async(
                client, prompt, model, provider_kwargs, usage_cell
            ):
                yield text
            return
        raise StampError(
            "streaming is not supported for this LLM client type; use stamp() instead."
        )

    async def _stream_openai_async(
        self,
        client: Any,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> AsyncIterator[str]:
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )
        stream = await client.chat.completions.create(**create_kwargs, stream=True)
        async for chunk in stream:
            # Usage can arrive on a final chunk with empty choices (OpenAI's
            # stream_options include_usage), so capture it before the delta check.
            usage = getattr(chunk, "usage", None)
            if usage is not None:
                usage_cell.append(
                    (
                        getattr(usage, "prompt_tokens", None),
                        getattr(usage, "completion_tokens", None),
                    )
                )
            choices = getattr(chunk, "choices", None)
            delta = choices[0].delta.content if choices else None
            if delta:
                yield delta

    async def _stream_anthropic_async(
        self,
        client: Any,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> AsyncIterator[str]:
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )
        async with client.messages.stream(**create_kwargs) as stream:
            async for text in stream.text_stream:
                if text:
                    yield text
            final = await stream.get_final_message()
        usage = getattr(final, "usage", None)
        if usage is not None:
            usage_cell.append(
                (
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )
            )
