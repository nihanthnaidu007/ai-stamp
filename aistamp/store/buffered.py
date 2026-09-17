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

    def _write_batch(self, pending: list[WritePair]) -> None:
        """Persist one batch; re-buffer it if the backend fails.

        A transient write_many error must not silently drop audit records:
        the batch returns to the front of the buffer so the next flush
        retries it, preserving order against concurrent adds.
        """
        try:
            self._backend.write_many(pending)
        except Exception:
            with self._lock:
                self._buffer = pending + self._buffer
            raise

    def add(self, record: ProvenanceRecord, hmac_signature: str | None = None) -> None:
        """Buffer one record; flush automatically when the buffer is full."""
        pending: list[WritePair] = []
        with self._lock:
            self._buffer.append((record, hmac_signature))
            if len(self._buffer) >= self._max_buffer_size:
                pending = self._buffer
                self._buffer = []
        if pending:
            self._write_batch(pending)

    def flush(self) -> None:
        """Persist all buffered records now. Safe to call on an empty buffer."""
        with self._lock:
            pending = self._buffer
            self._buffer = []
        if pending:
            self._write_batch(pending)

    def __len__(self) -> int:
        """Number of records buffered but not yet persisted."""
        with self._lock:
            return len(self._buffer)

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

    async def _write_batch(self, pending: list[WritePair]) -> None:
        """Persist one batch; re-buffer it if the backend fails.

        A transient write_many error must not silently drop audit records:
        the batch returns to the front of the buffer so the next flush
        retries it, preserving order against concurrent adds.
        """
        try:
            await self._backend.write_many(pending)
        except Exception:
            async with self._lock:
                self._buffer = pending + self._buffer
            raise

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
            await self._write_batch(pending)

    async def flush(self) -> None:
        """Persist all buffered records now. Safe to call on an empty buffer."""
        async with self._lock:
            pending = self._buffer
            self._buffer = []
        if pending:
            await self._write_batch(pending)

    def __len__(self) -> int:
        """Number of records buffered but not yet persisted.

        No lock: on the event loop a synchronous read cannot be preempted.
        """
        return len(self._buffer)

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
