from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from aistamp.config import Config
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
from aistamp.store.backend import StoreBackend

logger = logging.getLogger("aistamp.client")


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


def build_pre_call_context(
    prompt: str,
    model: str,
    app_id: str,
    feature_id: str,
    user_id: str,
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
    )


def _record_from_context(ctx: CaptureContext) -> ProvenanceRecord:
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
    )


def run_pre_call_policy(
    ctx: CaptureContext,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    pii_scan = scan_prompt_and_response(ctx.prompt, "", extra_patterns, use_spacy)
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


def finalize_context(
    ctx: CaptureContext,
    response_text: str,
    prompt_tokens: int | None,
    response_tokens: int | None,
    engine: PolicyEngine | None,
    extra_patterns: list[PatternConfig] | None,
    use_spacy: bool,
) -> None:
    ctx.response = response_text
    ctx.response_hash = hash_content(response_text)
    elapsed_ms = (time.perf_counter_ns() - ctx.start_time) / 1_000_000
    ctx.latency_ms = max(elapsed_ms, 0.000001)
    ctx.prompt_tokens = prompt_tokens
    ctx.response_tokens = response_tokens

    ctx.pii_result = scan_prompt_and_response(
        ctx.prompt, response_text, extra_patterns, use_spacy
    )

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


def build_record(ctx: CaptureContext) -> ProvenanceRecord:
    return _record_from_context(ctx)


def persist_record(
    record: ProvenanceRecord,
    backend: StoreBackend,
    config: Config,
) -> str:
    hmac = sign_record(record, config.secret_key)
    backend.write(record, hmac)
    logger.debug(
        "Stamped content_id=%s model=%s status=%s",
        record.content_id,
        record.model,
        record.status.value,
    )
    return record.content_id
