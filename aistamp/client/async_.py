from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

from aistamp.client._pipeline import (
    CaptureContext,
    build_pre_call_context,
    build_record,
    finalize_context,
    persist_record,
    run_pre_call_policy,
)
from aistamp.client.http import GenericHTTPClient
from aistamp.client.sync import StampError
from aistamp.config import Config
from aistamp.fingerprint import sign_record
from aistamp.models import ProvenanceRecord, RecordStatus
from aistamp.pii.patterns import PatternConfig
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.store.async_backend import AsyncStoreBackend
from aistamp.store.backend import SQLiteBackend, StoreBackend

logger = logging.getLogger("aistamp.client")


class AsyncProvenanceClient:
    """
    Wraps an asynchronous LLM client and stamps every call with a provenance record.

    Supports openai.AsyncOpenAI, anthropic.AsyncAnthropic, GenericHTTPClient,
    and any callable matching (str) -> str or (str) -> Awaitable[str].
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
        backend: StoreBackend | AsyncStoreBackend | None = None,
        extra_patterns: list[PatternConfig] | None = None,
        use_spacy: bool = False,
    ) -> None:
        self._llm_client = llm_client
        self._config = config
        self._app_id = app_id
        self._feature_id = feature_id
        self._user_id = user_id
        self._engine = engine
        self._extra_patterns = extra_patterns
        self._use_spacy = use_spacy

        self._backend: StoreBackend | AsyncStoreBackend
        if backend is None:
            sync_backend = SQLiteBackend(config.database_url)
            sync_backend.create_tables()
            self._backend = sync_backend
        else:
            # For async backends create_tables() is the caller's responsibility.
            self._backend = backend

    async def chat(self, prompt: str, model: str | None = None) -> str:
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
        if not prompt.strip():
            raise ValueError("prompt must be a non-empty, non-whitespace string")

        resolved_model = self._resolve_model(model)
        ctx = build_pre_call_context(
            prompt=prompt,
            model=resolved_model,
            app_id=self._app_id,
            feature_id=self._feature_id,
            user_id=self._user_id,
        )

        try:
            run_pre_call_policy(
                ctx, self._engine, self._extra_patterns, self._use_spacy
            )
        except PolicyViolationError:
            ctx.status = RecordStatus.BLOCKED
            await self._try_persist(ctx)
            raise

        try:
            response_text, prompt_tokens, response_tokens = await self._call_llm_async(
                prompt, resolved_model
            )
        except StampError:
            ctx.status = RecordStatus.ERROR
            await self._try_persist(ctx)
            raise
        except Exception as e:
            ctx.status = RecordStatus.ERROR
            ctx.error = e
            await self._try_persist(ctx)
            raise StampError(f"LLM call failed: {type(e).__name__}: {e}") from e

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
            await self._try_persist(ctx)
            raise

        record = build_record(ctx)
        await self._async_persist(record)
        return response_text

    async def _async_persist(self, record: ProvenanceRecord) -> str:
        hmac = sign_record(record, self._config.secret_key)
        if isinstance(self._backend, AsyncStoreBackend):
            await self._backend.write(record, hmac)
        else:
            persist_record(record, self._backend, self._config)
        return record.content_id

    async def _try_persist(self, ctx: CaptureContext) -> None:
        try:
            record = build_record(ctx)
            await self._async_persist(record)
        except Exception as e:
            logger.warning(
                "Failed to persist partial record content_id=%s: %s",
                ctx.content_id,
                e,
            )

    async def _call_llm_async(
        self,
        prompt: str,
        model: str,
    ) -> tuple[str, int | None, int | None]:
        if isinstance(self._llm_client, GenericHTTPClient):
            return await asyncio.to_thread(self._llm_client.complete, prompt, model)

        try:
            import openai

            if isinstance(self._llm_client, openai.AsyncOpenAI):
                resp = await self._llm_client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.choices[0].message.content or ""
                pt = resp.usage.prompt_tokens if resp.usage else None
                rt = resp.usage.completion_tokens if resp.usage else None
                return text, pt, rt
        except ImportError:
            pass

        try:
            import anthropic

            if isinstance(self._llm_client, anthropic.AsyncAnthropic):
                resp = await self._llm_client.messages.create(
                    model=model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text if resp.content else ""
                pt = resp.usage.input_tokens if resp.usage else None
                rt = resp.usage.output_tokens if resp.usage else None
                return text, pt, rt
        except ImportError:
            pass

        if callable(self._llm_client):
            result = self._llm_client(prompt)
            if inspect.isawaitable(result):
                text = await result
            else:
                text = result
            if not isinstance(text, str):
                raise StampError(
                    f"Generic callable must return str or Awaitable[str],"
                    f" got {type(text).__name__}"
                )
            return text, None, None

        raise StampError(
            f"Unsupported LLM client type: {type(self._llm_client).__name__}. "
            "Expected openai.AsyncOpenAI, anthropic.AsyncAnthropic, "
            "GenericHTTPClient, or a callable (str) -> str | Awaitable[str]."
        )

    def _resolve_model(self, model: str | None) -> str:
        if model:
            return model

        try:
            import openai

            if isinstance(self._llm_client, openai.AsyncOpenAI):
                return "gpt-4o"
        except ImportError:
            pass

        try:
            import anthropic

            if isinstance(self._llm_client, anthropic.AsyncAnthropic):
                return "claude-3-5-sonnet-20241022"
        except ImportError:
            pass

        return "unknown"
