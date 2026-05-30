from __future__ import annotations

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
from aistamp.config import Config
from aistamp.models import RecordStatus
from aistamp.pii.patterns import PatternConfig
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.store.backend import SQLiteBackend, StoreBackend

logger = logging.getLogger("aistamp.client")


# TODO(phase-v2): add streaming support
# TODO(phase-v2): add batch processing
# TODO(phase-v2): multi-turn conversation support
# TODO(phase-v2): add retry with exponential backoff


class StampError(Exception):
    """
    Raised when ai-stamp encounters an internal error during the stamping pipeline.
    Wraps exceptions from unsupported client types or unexpected failures.
    """


class ProvenanceClient:
    """
    Wraps a synchronous LLM client and stamps every call with a provenance record.

    Supports OpenAI, Anthropic, GenericHTTPClient, and callables.
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
    ) -> None:
        self._llm_client = llm_client
        self._config = config
        self._app_id = app_id
        self._feature_id = feature_id
        self._user_id = user_id
        self._engine = engine
        self._extra_patterns = extra_patterns
        self._use_spacy = use_spacy

        if backend is None:
            self._backend: StoreBackend = SQLiteBackend(config.database_url)
            self._backend.create_tables()
        else:
            self._backend = backend

    def chat(self, prompt: str, model: str | None = None) -> str:
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
            self._try_persist(ctx)
            raise

        try:
            response_text, prompt_tokens, response_tokens = self._call_llm(
                prompt, resolved_model
            )
        except StampError:
            ctx.status = RecordStatus.ERROR
            self._try_persist(ctx)
            raise
        except Exception as e:
            ctx.status = RecordStatus.ERROR
            ctx.error = e
            self._try_persist(ctx)
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
            self._try_persist(ctx)
            raise

        record = build_record(ctx)
        persist_record(record, self._backend, self._config)
        return response_text

    def _try_persist(self, ctx: CaptureContext) -> None:
        try:
            record = build_record(ctx)
            persist_record(record, self._backend, self._config)
        except Exception as e:
            logger.warning(
                "Failed to persist partial record content_id=%s: %s",
                ctx.content_id,
                e,
            )

    def _call_llm(
        self,
        prompt: str,
        model: str,
    ) -> tuple[str, int | None, int | None]:
        if isinstance(self._llm_client, GenericHTTPClient):
            return self._llm_client.complete(prompt, model)

        try:
            import openai

            if isinstance(self._llm_client, openai.OpenAI):
                resp = self._llm_client.chat.completions.create(
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

            if isinstance(self._llm_client, anthropic.Anthropic):
                resp = self._llm_client.messages.create(
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
            text = self._llm_client(prompt)
            if not isinstance(text, str):
                raise StampError(
                    f"Generic callable must return str, got {type(text).__name__}"
                )
            return text, None, None

        raise StampError(
            f"Unsupported LLM client type: {type(self._llm_client).__name__}. "
            "Expected openai.OpenAI, anthropic.Anthropic, GenericHTTPClient, "
            "or a callable (str) -> str."
        )

    def _resolve_model(self, model: str | None) -> str:
        if model:
            return model

        try:
            import openai

            if isinstance(self._llm_client, openai.OpenAI):
                return "gpt-4o"
        except ImportError:
            pass

        try:
            import anthropic

            if isinstance(self._llm_client, anthropic.Anthropic):
                return "claude-3-5-sonnet-20241022"
        except ImportError:
            pass

        return "unknown"
