"""Retry classification and exponential backoff with full jitter.

Retries apply to transient provider failures: timeouts, HTTP 429, and HTTP
5xx. The dispatcher never retries policy blocks, auth failures, or other
client errors.

``_sleep`` / ``_asleep`` are module-level so tests can substitute no-op
sleep functions without patching ``time``/``asyncio`` globally.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

from aistamp.errors import (
    AIStampError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)

logger = logging.getLogger("aistamp.client")

# Provider SDKs whose timeout exceptions neither subclass TimeoutError nor
# carry a ``status_code`` attribute are recognised by class name (e.g.
# openai.APITimeoutError, httpx.ReadTimeout). Name matching is a deliberate
# heuristic: the SDK classes are not importable in offline test environments.
_TIMEOUT_EXCEPTION_NAMES = frozenset(
    {
        "APITimeoutError",
        "ReadTimeout",
        "ConnectTimeout",
        "WriteTimeout",
        "PoolTimeout",
    }
)

_sleep = time.sleep
_asleep = asyncio.sleep

T = TypeVar("T")


def is_retryable(exc: BaseException) -> bool:
    """Return True when *exc* is a transient failure worth retrying.

    Retryable: timeouts, HTTP 429, and HTTP 5xx — either raised as ai-stamp
    library errors or detected on foreign SDK exceptions via a ``status_code``
    attribute, ``TimeoutError`` inheritance, or a known timeout-exception name.
    """
    if isinstance(exc, (ProviderTimeoutError, ProviderRateLimitError)):
        return True
    if isinstance(exc, AIStampError):
        if isinstance(exc, ProviderResponseError):
            return exc.status_code is not None and exc.status_code >= 500
        return False

    status_code = getattr(exc, "status_code", None)
    if isinstance(status_code, int):
        return status_code == 429 or status_code >= 500

    if isinstance(exc, TimeoutError):
        return True

    return type(exc).__name__ in _TIMEOUT_EXCEPTION_NAMES


def compute_delay(
    attempt: int,
    *,
    base_delay: float,
    max_delay: float,
    rand: Callable[[float, float], float] = random.uniform,
) -> float:
    """Full-jitter backoff: uniform in ``[0, base_delay * 2**attempt]``, capped.

    ``attempt`` is 0-based over retries (the first retry uses ``attempt=0``).
    """
    ceiling = min(max_delay, base_delay * (2**attempt))
    return max(0.0, rand(0.0, ceiling))


def call_with_retries(
    fn: Callable[[], T],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """Run *fn* with exponential backoff + full jitter on transient failures.

    ``max_retries`` counts retries after the initial attempt, so a call may
    execute up to ``max_retries + 1`` times. Non-retryable and final failures
    propagate unchanged.
    """
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = compute_delay(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "Transient provider error (%s: %s); retrying in %.3fs (attempt %d/%d).",
                type(exc).__name__,
                exc,
                delay,
                attempt + 1,
                max_retries,
            )
            _sleep(delay)
    raise AssertionError("retry loop exited without returning")  # pragma: no cover


async def acall_with_retries(
    fn: Callable[[], Awaitable[T]],
    *,
    max_retries: int,
    base_delay: float,
    max_delay: float,
) -> T:
    """Async twin of :func:`call_with_retries` (``await``-friendly sleeping)."""
    attempts = max_retries + 1
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as exc:
            if not is_retryable(exc) or attempt == attempts - 1:
                raise
            delay = compute_delay(attempt, base_delay=base_delay, max_delay=max_delay)
            logger.warning(
                "Transient provider error (%s: %s); retrying in %.3fs (attempt %d/%d).",
                type(exc).__name__,
                exc,
                delay,
                attempt + 1,
                max_retries,
            )
            await _asleep(delay)
    raise AssertionError("retry loop exited without returning")  # pragma: no cover
