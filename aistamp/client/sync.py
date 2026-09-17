from __future__ import annotations

import inspect
import logging
from collections.abc import Iterator
from typing import Any

from aistamp.client._pipeline import (
    CaptureContext,
    PersistErrorCallback,
    attach_content_id,
    build_pre_call_context,
    build_record,
    classify_provider_error,
    finalize_context,
    persist_record,
    redact_prompt,
    report_persist_failure,
    run_pre_call_policy,
    try_persist_sync,
    validate_prompt,
)
from aistamp.client._providers import (
    _import_anthropic,
    _import_openai,
    build_create_kwargs,
    dispatch_sync,
    resolve_model,
    validate_llm_client,
    with_http_timeout,
)
from aistamp.client._retry import call_with_retries
from aistamp.client.http import GenericHTTPClient
from aistamp.client.results import StampResult, StreamStamp, TokenUsage
from aistamp.config import Config
from aistamp.errors import AIStampError, StampError
from aistamp.fingerprint import hash_content
from aistamp.models import RecordStatus
from aistamp.pii.patterns import PatternConfig
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.store.backend import SQLiteBackend, StoreBackend

logger = logging.getLogger("aistamp.client")

__all__ = [
    "ProvenanceClient",
    "StampError",  # re-exported for 0.1.x import-path compatibility
]

# TODO(phase-v2): add batch processing


class ProvenanceClient:
    """
    Wraps a synchronous LLM client and stamps every call with a provenance record.

    Supports OpenAI, Anthropic, GenericHTTPClient, and callables.

    v0.2 additions: ``stamp()`` / ``chat_detailed()`` return a
    :class:`~aistamp.client.StampResult` (text, content_id, record, decision,
    usage); ``stamp_stream()`` captures provider streams and stamps the final
    concatenation; provider calls are retried with exponential backoff and
    jitter on 429/5xx/timeouts; ``max_tokens`` and ``request_timeout`` are
    configurable; extra ``**kwargs`` are forwarded to the provider SDK.
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
        backend: StoreBackend | None = None,
        extra_patterns: list[PatternConfig] | None = None,
        use_spacy: bool = False,
        max_tokens: int | None = None,
        request_timeout: float | None = None,
        max_retries: int = 2,
        retry_base_delay: float = 0.5,
        retry_max_delay: float = 8.0,
        on_persist_error: PersistErrorCallback | None = None,
    ) -> None:
        validate_llm_client(llm_client, is_async=False)
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if retry_base_delay <= 0 or retry_max_delay <= 0:
            raise ValueError("retry delays must be positive")
        if request_timeout is not None and request_timeout <= 0:
            raise ValueError("request_timeout must be positive")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if on_persist_error is not None and inspect.iscoroutinefunction(
            on_persist_error
        ):
            raise TypeError(
                "on_persist_error must be a synchronous callable for"
                " ProvenanceClient; async callbacks require AsyncProvenanceClient."
            )

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
            self._backend: StoreBackend = SQLiteBackend(config.database_url)
            self._backend.create_tables()
        else:
            self._backend = backend

        # The HTTP fallback carries its own timeout; the client-level knob
        # overrides it when set. dataclasses.replace works on the frozen
        # dataclass and leaves the caller's instance untouched.
        self._http_client = (
            with_http_timeout(llm_client, request_timeout)
            if isinstance(llm_client, GenericHTTPClient)
            else llm_client
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self, prompt: str, model: str | None = None, **provider_kwargs: Any
    ) -> str:
        """0.1.x-compatible surface: returns only the response text."""
        return self._execute(
            prompt,
            model,
            app_id=None,
            feature_id=None,
            user_id=None,
            metadata=None,
            conversation_id=None,
            request_id=None,
            provider_kwargs=provider_kwargs,
        ).text

    def stamp(
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
        return self._execute(
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

    def chat_detailed(
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
        return self.stamp(
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

    def stamp_stream(
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
    ) -> StreamStamp:
        """Stream provider chunks and stamp the final concatenation.

        Pre-call policy runs eagerly (a BLOCK raises here, before any chunk).
        Iterating the returned :class:`StreamStamp` yields provider chunks;
        after exhaustion, ``stream.result`` holds the final
        :class:`StampResult` stamped over the concatenation of all chunks.

        Streaming is not retried — retrying a partially consumed stream would
        duplicate output.
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
        self._run_pre_call_policy(ctx)

        result_cell: list[StampResult] = []

        def generate() -> Iterator[str]:
            usage_cell: list[tuple[int | None, int | None]] = []
            chunks: list[str] = []
            try:
                for chunk_text in self._stream_dispatch(
                    ctx.prompt, resolved_model, provider_kwargs, usage_cell
                ):
                    chunks.append(chunk_text)
                    yield chunk_text
            except Exception as exc:
                ctx.status = RecordStatus.ERROR
                ctx.error = exc
                attach_content_id(exc, ctx.content_id)
                try_persist_sync(
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
                self._finalize_and_persist(
                    ctx,
                    "".join(chunks),
                    prompt_tokens,
                    response_tokens,
                    metadata=metadata,
                    conversation_id=conversation_id,
                    request_id=request_id,
                )
            )

        return StreamStamp(generate(), result_cell)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def _execute(
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
        self._run_pre_call_policy(ctx)

        try:
            if self._config.redact_before_send:
                dispatch_prompt = redact_prompt(ctx.prompt)
                ctx.prompt = dispatch_prompt
                ctx.prompt_hash = hash_content(dispatch_prompt)
            else:
                dispatch_prompt = ctx.prompt
            response_text, prompt_tokens, response_tokens = self._dispatch_with_retries(
                dispatch_prompt, resolved_model, provider_kwargs
            )
        except AIStampError as exc:
            attach_content_id(exc, ctx.content_id)
            ctx.status = RecordStatus.ERROR
            ctx.error = exc
            try_persist_sync(ctx, self._backend, self._config, self._on_persist_error)
            raise
        except Exception as exc:
            ctx.status = RecordStatus.ERROR
            ctx.error = exc
            try_persist_sync(ctx, self._backend, self._config, self._on_persist_error)
            raise classify_provider_error(exc, ctx.content_id) from exc

        return self._finalize_and_persist(
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

    def _run_pre_call_policy(self, ctx: CaptureContext) -> None:
        try:
            run_pre_call_policy(
                ctx, self._engine, self._extra_patterns, self._use_spacy
            )
        except PolicyViolationError:
            ctx.status = RecordStatus.BLOCKED
            try_persist_sync(ctx, self._backend, self._config, self._on_persist_error)
            raise

    def _finalize_and_persist(
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
            finalize_context(
                ctx,
                response_text,
                prompt_tokens,
                response_tokens,
                self._engine,
                self._extra_patterns,
                self._use_spacy,
            )
        except PolicyViolationError:
            try_persist_sync(ctx, self._backend, self._config, self._on_persist_error)
            raise

        record = build_record(ctx)
        try:
            persist_record(record, self._backend, self._config)
        except Exception as exc:
            report_persist_failure(record, exc, self._on_persist_error)

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

    def _dispatch_with_retries(
        self, prompt: str, model: str, provider_kwargs: dict[str, Any]
    ) -> tuple[str, int | None, int | None]:
        return call_with_retries(
            lambda: self._dispatch(prompt, model, provider_kwargs),
            max_retries=self._max_retries,
            base_delay=self._retry_base_delay,
            max_delay=self._retry_max_delay,
        )

    # ------------------------------------------------------------------
    # Provider dispatch
    # ------------------------------------------------------------------

    def _dispatch(
        self, prompt: str, model: str, provider_kwargs: dict[str, Any]
    ) -> tuple[str, int | None, int | None]:
        return dispatch_sync(
            self._http_client,
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )

    def _supports_streaming(self) -> bool:
        client = self._llm_client
        openai = _import_openai()
        if openai is not None and isinstance(client, openai.OpenAI):
            return True
        anthropic = _import_anthropic()
        if anthropic is not None and isinstance(client, anthropic.Anthropic):
            return True
        return False

    def _stream_dispatch(
        self,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> Iterator[str]:
        """Yield provider stream chunks; append final usage to *usage_cell*."""
        client = self._llm_client
        openai = _import_openai()
        if openai is not None and isinstance(client, openai.OpenAI):
            yield from self._stream_openai(
                client, prompt, model, provider_kwargs, usage_cell
            )
            return
        anthropic = _import_anthropic()
        if anthropic is not None and isinstance(client, anthropic.Anthropic):
            yield from self._stream_anthropic(
                client, prompt, model, provider_kwargs, usage_cell
            )
            return
        raise StampError(
            "streaming is not supported for this LLM client type; use stamp() instead."
        )

    def _stream_openai(
        self,
        client: Any,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> Iterator[str]:
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )
        stream = client.chat.completions.create(**create_kwargs, stream=True)
        for chunk in stream:
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

    def _stream_anthropic(
        self,
        client: Any,
        prompt: str,
        model: str,
        provider_kwargs: dict[str, Any],
        usage_cell: list[tuple[int | None, int | None]],
    ) -> Iterator[str]:
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=self._max_tokens,
            request_timeout=self._request_timeout,
        )
        with client.messages.stream(**create_kwargs) as stream:
            for text in stream.text_stream:
                if text:
                    yield text
            final = stream.get_final_message()
        usage = getattr(final, "usage", None)
        if usage is not None:
            usage_cell.append(
                (
                    getattr(usage, "input_tokens", None),
                    getattr(usage, "output_tokens", None),
                )
            )
