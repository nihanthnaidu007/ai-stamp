from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import cast

from aistamp.client._retry import _TIMEOUT_EXCEPTION_NAMES
from aistamp.config import Config
from aistamp.errors import (
    AIStampError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
    StampError,
)
from aistamp.fingerprint import generate_content_id, hash_content, sign_record
from aistamp.models import (
    PIIResult,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.pii import scan_prompt_and_response
from aistamp.pii.patterns import PatternConfig
from aistamp.policy.engine import PolicyEngine, PolicyViolationError
from aistamp.store.async_backend import AsyncStoreBackend
from aistamp.store.backend import StoreBackend

logger = logging.getLogger("aistamp.client")

# Callbacks invoked when a provenance record could not be persisted. Sync
# clients require the sync variant; async clients accept either and await
# awaitable results.
PersistErrorCallback = Callable[[ProvenanceRecord, BaseException], None]
AsyncPersistErrorCallback = Callable[[ProvenanceRecord, BaseException], Awaitable[None]]

# Upper bound for the persisted error_message (the store/schema.py column
# is Text, unbounded) so a pathological provider message cannot produce an
# unbounded row and cannot fail the very write that is supposed to record it.
_MAX_ERROR_MESSAGE_LENGTH = 2000


@dataclass
class CaptureContext:
    content_id: str
    app_id: str
    feature_id: str
    user_id: str
    model: str
    prompt: str
    prompt_hash: str
    start_time: int
    timestamp: datetime
    response: str | None = None
    response_hash: str | None = None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    latency_ms: float | None = None
    pii_result: PIIResult | None = None
    policy_decision: PolicyDecision | None = None
    status: RecordStatus = RecordStatus.COMPLETED
    error: Exception | None = None
    key_id: str = "default"
    metadata: dict[str, str] | None = None
    conversation_id: str | None = None
    request_id: str | None = None


def validate_prompt(prompt: str) -> None:
    """Shared prompt validation for sync and async clients."""
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    if not prompt.strip():
        raise ValueError("prompt must be a non-empty, non-whitespace string")


def build_pre_call_context(
    prompt: str,
    model: str,
    app_id: str,
    feature_id: str,
    user_id: str,
    *,
    key_id: str = "default",
    metadata: dict[str, str] | None = None,
    conversation_id: str | None = None,
    request_id: str | None = None,
) -> CaptureContext:
    return CaptureContext(
        content_id=generate_content_id(),
        app_id=app_id,
        feature_id=feature_id,
        user_id=user_id,
        model=model,
        prompt=prompt,
        prompt_hash=hash_content(prompt),
        start_time=time.perf_counter_ns(),
        timestamp=datetime.now(timezone.utc),
        key_id=key_id,
        metadata=metadata,
        conversation_id=conversation_id,
        request_id=request_id,
    )


def redact_prompt(prompt: str) -> str:
    """Scrub PII from a prompt before it is dispatched to the provider.

    ``aistamp.pii.redact_text`` is resolved lazily at call time: the PII track
    owns it and may not exist on this branch yet, and resolving inside the
    function makes ``aistamp.pii.redact_text`` a monkeypatchable seam for
    tests and in-flight branches.

    Raises:
        AIStampError: if ``redact_text`` is unavailable. Refusing to send an
            unredacted prompt is deliberate — silently downgrading a
            redaction request would leak exactly the data it exists to
            protect.
    """
    try:
        import aistamp.pii as pii_module
    except ImportError as exc:
        raise AIStampError(
            "redact_before_send is enabled, but aistamp.pii.redact_text is not"
            " available. Refusing to send an unredacted prompt to the provider;"
            " upgrade ai-stamp or disable redact_before_send."
        ) from exc
    # getattr keeps this correct whether or not the PII track has landed
    # redact_text yet, and resolves at call time so monkeypatching
    # aistamp.pii.redact_text works as a test seam.
    redact = cast(
        "Callable[[str], str] | None", getattr(pii_module, "redact_text", None)
    )
    if redact is None:
        raise AIStampError(
            "redact_before_send is enabled, but aistamp.pii.redact_text is not"
            " available. Refusing to send an unredacted prompt to the provider;"
            " upgrade ai-stamp or disable redact_before_send."
        )
    return redact(prompt)


def _record_from_context(ctx: CaptureContext) -> ProvenanceRecord:
    error = ctx.error
    error_message: str | None = None
    if error is not None:
        error_message = str(error)[:_MAX_ERROR_MESSAGE_LENGTH]
    return ProvenanceRecord(
        content_id=ctx.content_id,
        app_id=ctx.app_id,
        feature_id=ctx.feature_id,
        user_id=ctx.user_id,
        model=ctx.model,
        prompt_hash=ctx.prompt_hash,
        response_hash=ctx.response_hash,
        prompt_tokens=ctx.prompt_tokens,
        response_tokens=ctx.response_tokens,
        latency_ms=ctx.latency_ms,
        timestamp=ctx.timestamp,
        status=ctx.status,
        pii_result=ctx.pii_result,
        policy_decision=ctx.policy_decision,
        key_id=ctx.key_id,
        error_type=(type(error).__name__ if error is not None else None),
        error_message=error_message,
    )


# ----------------------------------------------------------------------
# Pre-call phase (PII scan is isolated so async clients can offload it)
# ----------------------------------------------------------------------


def scan_pre_call(
    ctx: CaptureContext,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> PIIResult:
    return scan_prompt_and_response(ctx.prompt, "", extra_patterns, use_spacy)


def apply_pre_call_policy(
    ctx: CaptureContext,
    pii_scan: PIIResult,
    engine: PolicyEngine | None,
) -> None:
    ctx.pii_result = pii_scan

    if engine is None:
        return

    record = _record_from_context(ctx)
    try:
        decision = engine.evaluate(record)
    except PolicyViolationError as e:
        # Attach the BLOCK decision before re-raising so the persisted record
        # carries the rule that triggered the block, not just status=BLOCKED.
        ctx.policy_decision = e.decision
        raise

    ctx.policy_decision = decision
    if decision.action == PolicyAction.WARN:
        logger.warning(
            "Pre-call policy WARN content_id=%s rule=%s",
            ctx.content_id,
            decision.rule_name,
        )


def run_pre_call_policy(
    ctx: CaptureContext,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    apply_pre_call_policy(ctx, scan_pre_call(ctx, extra_patterns, use_spacy), engine)


async def run_pre_call_policy_async(
    ctx: CaptureContext,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    """Async twin: runs the (CPU-bound) PII scan off the event loop."""
    pii_scan = await asyncio.to_thread(scan_pre_call, ctx, extra_patterns, use_spacy)
    apply_pre_call_policy(ctx, pii_scan, engine)


# ----------------------------------------------------------------------
# Post-call phase (same split)
# ----------------------------------------------------------------------


def scan_finalize(
    ctx: CaptureContext,
    response_text: str,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> PIIResult:
    return scan_prompt_and_response(
        ctx.prompt, response_text, extra_patterns, use_spacy
    )


def apply_finalize_context(
    ctx: CaptureContext,
    response_text: str,
    prompt_tokens: int | None,
    response_tokens: int | None,
    pii_scan: PIIResult,
    engine: PolicyEngine | None,
) -> None:
    ctx.response = response_text
    ctx.response_hash = hash_content(response_text)
    elapsed_ms = (time.perf_counter_ns() - ctx.start_time) / 1_000_000
    ctx.latency_ms = max(elapsed_ms, 0.000001)
    ctx.prompt_tokens = prompt_tokens
    ctx.response_tokens = response_tokens
    ctx.pii_result = pii_scan

    if engine is None:
        return

    record = _record_from_context(ctx)
    try:
        decision = engine.evaluate(record)
    except PolicyViolationError as e:
        # Attach the BLOCK decision and set status before re-raising so the
        # persisted record carries both the BLOCKED status and the matched rule.
        ctx.status = RecordStatus.BLOCKED
        ctx.policy_decision = e.decision
        raise

    ctx.policy_decision = decision
    if decision.action == PolicyAction.WARN:
        logger.warning(
            "Post-call policy WARN content_id=%s rule=%s",
            ctx.content_id,
            decision.rule_name,
        )


def finalize_context(
    ctx: CaptureContext,
    response_text: str,
    prompt_tokens: int | None,
    response_tokens: int | None,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    pii_scan = scan_finalize(ctx, response_text, extra_patterns, use_spacy)
    apply_finalize_context(
        ctx, response_text, prompt_tokens, response_tokens, pii_scan, engine
    )


async def finalize_context_async(
    ctx: CaptureContext,
    response_text: str,
    prompt_tokens: int | None,
    response_tokens: int | None,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    """Async twin: runs the (CPU-bound) PII scan off the event loop."""
    pii_scan = await asyncio.to_thread(
        scan_finalize, ctx, response_text, extra_patterns, use_spacy
    )
    apply_finalize_context(
        ctx, response_text, prompt_tokens, response_tokens, pii_scan, engine
    )


# ----------------------------------------------------------------------
# Record build + persist
# ----------------------------------------------------------------------


def build_record(ctx: CaptureContext) -> ProvenanceRecord:
    return _record_from_context(ctx)


def persist_record(
    record: ProvenanceRecord,
    backend: StoreBackend,
    config: Config,
) -> str:
    hmac = sign_record(record, config.secret_key_value)
    backend.write(record, hmac)
    logger.debug(
        "Stamped content_id=%s model=%s status=%s",
        record.content_id,
        record.model,
        record.status.value,
    )
    return record.content_id


async def persist_record_async(
    record: ProvenanceRecord,
    backend: AsyncStoreBackend | StoreBackend,
    config: Config,
) -> str:
    hmac = sign_record(record, config.secret_key_value)
    if isinstance(backend, AsyncStoreBackend):
        await backend.write(record, hmac)
    else:
        # Sync backends are called directly: they are the documented 0.1 sync
        # API, and offloading to a worker thread would break thread-affine
        # connections (e.g. SQLite in-memory databases live per connection).
        backend.write(record, hmac)
    logger.debug(
        "Stamped content_id=%s model=%s status=%s",
        record.content_id,
        record.model,
        record.status.value,
    )
    return record.content_id


def attach_content_id(error: BaseException, content_id: str) -> None:
    """Expose the capture context's content_id on a raised library error."""
    if isinstance(error, AIStampError) and error.content_id is None:
        error.content_id = content_id


def classify_provider_error(
    exc: BaseException, content_id: str | None = None
) -> AIStampError:
    """Map a foreign provider exception onto the ai-stamp error taxonomy.

    SDK exceptions are recognized by ``status_code`` attribute, ``TimeoutError``
    inheritance, or a known timeout-exception name (the same heuristics the
    retry classifier uses). Library errors pass through with content_id set.
    """
    if isinstance(exc, AIStampError):
        if exc.content_id is None and content_id is not None:
            exc.content_id = content_id
        return exc
    message = f"{type(exc).__name__}: {exc}"
    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        if status_code == 429:
            return ProviderRateLimitError(
                message, content_id=content_id, status_code=status_code
            )
        if status_code in (401, 403):
            return ProviderAuthError(
                message, content_id=content_id, status_code=status_code
            )
        if status_code >= 500:
            return ProviderResponseError(
                message, content_id=content_id, status_code=status_code
            )
    if isinstance(exc, TimeoutError) or type(exc).__name__ in _TIMEOUT_EXCEPTION_NAMES:
        return ProviderTimeoutError(message, content_id=content_id)
    return StampError(f"LLM call failed: {message}", content_id=content_id)


def report_persist_failure(
    record: ProvenanceRecord,
    error: BaseException,
    on_persist_error: PersistErrorCallback | None,
) -> None:
    """Log a persist failure and surface it through the callback hook.

    Never raises: the primary outcome (LLM response or error) must not be
    masked by an audit-side failure or a faulty callback. A raising callback
    is logged via ``logger.exception`` so it is observed, not swallowed.
    """
    logger.warning(
        "Failed to persist provenance record content_id=%s: %s",
        record.content_id,
        error,
    )
    if on_persist_error is None:
        return
    try:
        on_persist_error(record, error)
    except Exception:
        logger.exception(
            "on_persist_error callback raised for content_id=%s",
            record.content_id,
        )


async def report_persist_failure_async(
    record: ProvenanceRecord,
    error: BaseException,
    on_persist_error: PersistErrorCallback | AsyncPersistErrorCallback | None,
) -> None:
    """Async twin of :func:`report_persist_failure`; awaits async callbacks."""
    logger.warning(
        "Failed to persist provenance record content_id=%s: %s",
        record.content_id,
        error,
    )
    if on_persist_error is None:
        return
    try:
        outcome = on_persist_error(record, error)
        if inspect.isawaitable(outcome):
            await outcome
    except Exception:
        logger.exception(
            "on_persist_error callback raised for content_id=%s",
            record.content_id,
        )


def try_persist_sync(
    ctx: CaptureContext,
    backend: StoreBackend,
    config: Config,
    on_persist_error: PersistErrorCallback | None,
) -> None:
    """Best-effort persist of a partial record after a failure. Never raises."""
    record: ProvenanceRecord | None = None
    try:
        record = build_record(ctx)
        persist_record(record, backend, config)
    except Exception as e:
        if record is None:
            # build_record is pure model construction; a failure here leaves
            # nothing to persist. Log and return — the primary error the
            # caller is already handling still propagates.
            logger.warning(
                "Failed to build partial provenance record content_id=%s: %s",
                ctx.content_id,
                e,
            )
            return
        report_persist_failure(record, e, on_persist_error)


async def try_persist_async(
    ctx: CaptureContext,
    backend: AsyncStoreBackend | StoreBackend,
    config: Config,
    on_persist_error: PersistErrorCallback | AsyncPersistErrorCallback | None,
) -> None:
    """Async twin of :func:`try_persist_sync`. Never raises."""
    record: ProvenanceRecord | None = None
    try:
        record = build_record(ctx)
        await persist_record_async(record, backend, config)
    except Exception as e:
        if record is None:
            logger.warning(
                "Failed to build partial provenance record content_id=%s: %s",
                ctx.content_id,
                e,
            )
            return
        await report_persist_failure_async(record, e, on_persist_error)
