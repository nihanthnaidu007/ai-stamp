from __future__ import annotations

import asyncio
import threading
from types import TracebackType

from aistamp.models import ProvenanceRecord
from aistamp.store.async_backend import AsyncStoreBackend
from aistamp.store.backend import StoreBackend

WritePair = tuple[ProvenanceRecord, str | None]


class BufferedWriter:
    """Buffers provenance writes and flushes them as transaction batches.

    For high-throughput stamping: ``add()`` is cheap and a flush (manual,
    automatic when the buffer reaches ``max_buffer_size``, or on close)
    persists the accumulated batch in a single transaction.

    close() flushes the buffer but does NOT dispose the wrapped backend —
    the caller owns the backend's lifecycle.
    """

    def __init__(self, backend: StoreBackend, max_buffer_size: int = 100) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1")
        self._backend = backend
        self._max_buffer_size = max_buffer_size
        self._buffer: list[WritePair] = []
        self._lock = threading.Lock()

    def add(self, record: ProvenanceRecord, hmac_signature: str | None = None) -> None:
        """Buffer one record; flush automatically when the buffer is full."""
        pending: list[WritePair] = []
        with self._lock:
            self._buffer.append((record, hmac_signature))
            if len(self._buffer) >= self._max_buffer_size:
                pending = self._buffer
                self._buffer = []
        if pending:
            self._backend.write_many(pending)

    def flush(self) -> None:
        """Persist all buffered records now. Safe to call on an empty buffer."""
        with self._lock:
            pending = self._buffer
            self._buffer = []
        if pending:
            self._backend.write_many(pending)

    def close(self) -> None:
        """Flush any remaining buffered records. Does not close the backend."""
        self.flush()

    def __enter__(self) -> BufferedWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


class AsyncBufferedWriter:
    """Async counterpart of :class:`BufferedWriter` for async backends."""

    def __init__(self, backend: AsyncStoreBackend, max_buffer_size: int = 100) -> None:
        if max_buffer_size < 1:
            raise ValueError("max_buffer_size must be >= 1")
        self._backend = backend
        self._max_buffer_size = max_buffer_size
        self._buffer: list[WritePair] = []
        self._lock = asyncio.Lock()

    async def add(
        self, record: ProvenanceRecord, hmac_signature: str | None = None
    ) -> None:
        """Buffer one record; flush automatically when the buffer is full."""
        pending: list[WritePair] = []
        async with self._lock:
            self._buffer.append((record, hmac_signature))
            if len(self._buffer) >= self._max_buffer_size:
                pending = self._buffer
                self._buffer = []
        if pending:
            await self._backend.write_many(pending)

    async def flush(self) -> None:
        """Persist all buffered records now. Safe to call on an empty buffer."""
        async with self._lock:
            pending = self._buffer
            self._buffer = []
        if pending:
            await self._backend.write_many(pending)

    async def close(self) -> None:
        """Flush any remaining buffered records. Does not close the backend."""
        await self.flush()

    async def __aenter__(self) -> AsyncBufferedWriter:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
