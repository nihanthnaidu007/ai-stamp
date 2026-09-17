"""Rich result types for the v0.2 stamping API.

``StampResult`` answers the question 0.1 could not: what did the wrapper just
stamp? ``StreamStamp`` / ``AsyncStreamStamp`` expose streaming chunks while
stamping the final concatenation once the stream is exhausted.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass

from aistamp.errors import StampError
from aistamp.models import PolicyDecision, ProvenanceRecord

__all__ = [
    "TokenUsage",
    "StampResult",
    "StreamStamp",
    "AsyncStreamStamp",
    "PersistErrorCallback",
    "AsyncPersistErrorCallback",
]


@dataclass(frozen=True)
class TokenUsage:
    """Token counts reported by the provider, when available."""

    prompt_tokens: int | None
    response_tokens: int | None


@dataclass(frozen=True)
class StampResult:
    """Everything one stamped LLM call produced.

    Attributes:
        text: the full response text (what ``chat()`` would have returned).
        content_id: identifier of the persisted provenance record.
        record: the full ``ProvenanceRecord`` (hashes, PII, policy, status).
        decision: the post-call policy decision, if a policy engine ran.
        usage: token usage reported by the provider, when available.
        metadata / conversation_id / request_id: per-call correlation values
            echoed back so callers can link their own identifiers to
            ``content_id``.
    """

    text: str
    content_id: str
    record: ProvenanceRecord
    decision: PolicyDecision | None
    usage: TokenUsage | None
    metadata: dict[str, str] | None = None
    conversation_id: str | None = None
    request_id: str | None = None


def _missing_result() -> StampResult:
    raise StampError(
        "The stream result is only available after the stream has been fully consumed."
    )


class StreamStamp:
    """Iterable over streamed text chunks.

    Iterate to receive chunks exactly as the provider produced them; once the
    stream is exhausted, :attr:`result` holds the final :class:`StampResult`
    stamped over the concatenation of all chunks.
    """

    def __init__(
        self,
        chunks: Iterator[str],
        result_cell: list[StampResult],
    ) -> None:
        self._chunks = chunks
        # One-element cell the producer's generator fills just before it ends.
        self._result_cell = result_cell

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        return next(self._chunks)

    @property
    def result(self) -> StampResult:
        if not self._result_cell:
            return _missing_result()
        return self._result_cell[0]


class AsyncStreamStamp:
    """Async twin of :class:`StreamStamp` (``async for`` + ``.result``)."""

    def __init__(
        self,
        chunks: AsyncIterator[str],
        result_cell: list[StampResult],
    ) -> None:
        self._chunks = chunks
        self._result_cell = result_cell

    def __aiter__(self) -> AsyncIterator[str]:
        return self

    async def __anext__(self) -> str:
        return await self._chunks.__anext__()

    @property
    def result(self) -> StampResult:
        if not self._result_cell:
            return _missing_result()
        return self._result_cell[0]


# Callback invoked (never raised through) when a provenance record could not
# be persisted. Sync clients require the sync variant; async clients accept
# either, awaiting the result when it is awaitable.
PersistErrorCallback = Callable[[ProvenanceRecord, BaseException], None]
AsyncPersistErrorCallback = Callable[[ProvenanceRecord, BaseException], Awaitable[None]]
