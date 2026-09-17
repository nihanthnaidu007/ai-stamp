from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import datetime, timezone
from types import TracebackType

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from aistamp.models import (
    AuditReport,
    ProvenanceRecord,
    PurgeAnchor,
    QueryFilters,
)
from aistamp.store.backend import (
    _apply_record_to_orm,
    _build_query_statements,
    _filters_to_dict,
    _orm_to_model,
    _page_from_rows,
    _record_to_orm,
    _register_sqlite_pragmas,
    _retention_cutoff,
    _to_naive_utc,
    finalize_overwrite_allowed,
)
from aistamp.store.schema import Base, ProvenanceRecordORM, PurgeAnchorORM

logger = logging.getLogger("aistamp.store")


class AsyncStoreBackend(ABC):
    """Abstract interface for async provenance store backends."""

    @abstractmethod
    async def write(self, record: ProvenanceRecord, hmac_signature: str | None) -> None:
        """Persist a provenance record with its HMAC signature."""
        ...

    @abstractmethod
    async def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        """Retrieve a record by content_id. Returns (record, hmac) or None."""
        ...

    @abstractmethod
    async def query(self, filters: QueryFilters) -> AuditReport:
        """Query records with filters. Returns a paginated AuditReport.

        Results are deterministically ordered by (timestamp, id).
        """
        ...

    @abstractmethod
    async def create_tables(self) -> None:
        """Create all tables. For development and testing only."""
        ...

    @abstractmethod
    async def finalize(
        self,
        content_id: str,
        record: ProvenanceRecord,
        hmac_signature: str | None = None,
    ) -> None:
        """Replace the record identified by content_id with its final state.

        Async counterpart of the write-ahead finalize: persist a PENDING
        record before the provider call, then finalize it with the outcome.
        If no row exists, the record is inserted instead.
        """
        ...

    @abstractmethod
    async def write_many(
        self, items: Sequence[tuple[ProvenanceRecord, str | None]]
    ) -> None:
        """Persist a batch of (record, hmac_signature) pairs in one transaction."""
        ...

    @abstractmethod
    async def purge(self, retention_days: int, *, now: datetime | None = None) -> int:
        """Delete records older than retention_days. Returns the deleted count."""
        ...

    @abstractmethod
    async def list_purge_anchors(self) -> list[PurgeAnchor]:
        """Return the retention journal (purge anchors), oldest first."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release the underlying connection pool. Idempotent."""
        ...

    async def dispose(self) -> None:
        """Alias for close()."""
        await self.close()

    async def __aenter__(self) -> AsyncStoreBackend:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()


class _AsyncSQLAlchemyBackend(AsyncStoreBackend):
    """Shared async SQLAlchemy implementation. Not part of the public API."""

    def __init__(self, async_engine: AsyncEngine) -> None:
        self._engine = async_engine
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def create_tables(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def write(self, record: ProvenanceRecord, hmac_signature: str | None) -> None:
        async with self._session_factory() as session:
            session.add(_record_to_orm(record, hmac_signature))
            await session.commit()

    async def finalize(
        self,
        content_id: str,
        record: ProvenanceRecord,
        hmac_signature: str | None = None,
    ) -> None:
        if record.content_id != content_id:
            raise ValueError(
                f"record.content_id {record.content_id!r} does not match "
                f"content_id {content_id!r}"
            )
        async with self._session_factory() as session:
            stmt = select(ProvenanceRecordORM).where(
                ProvenanceRecordORM.content_id == content_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if finalize_overwrite_allowed(row, record, hmac_signature):
                if row is None:
                    session.add(_record_to_orm(record, hmac_signature))
                else:
                    _apply_record_to_orm(row, record, hmac_signature)
            await session.commit()

    async def write_many(
        self, items: Sequence[tuple[ProvenanceRecord, str | None]]
    ) -> None:
        pairs = list(items)
        if not pairs:
            return
        async with self._session_factory() as session:
            for record, hmac in pairs:
                row = ProvenanceRecordORM()
                _apply_record_to_orm(row, record, hmac)
                session.add(row)
            await session.commit()

    async def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        async with self._session_factory() as session:
            stmt = select(ProvenanceRecordORM).where(
                ProvenanceRecordORM.content_id == content_id
            )
            result = await session.execute(stmt)
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return _orm_to_model(row), row.hmac_signature

    async def query(self, filters: QueryFilters) -> AuditReport:
        async with self._session_factory() as session:
            stmt, count_stmt = _build_query_statements(filters)
            data_result = await session.execute(stmt)
            rows = data_result.scalars().all()
            count_result = await session.execute(count_stmt)
            total = count_result.scalar_one()
            records, next_cursor = _page_from_rows(rows, filters)
            return AuditReport(
                records=records,
                total_count=int(total),
                generated_at=datetime.now(timezone.utc),
                filters_applied=_filters_to_dict(filters),
                next_cursor=next_cursor,
            )

    async def purge(self, retention_days: int, *, now: datetime | None = None) -> int:
        """Delete records older than retention_days. Returns the deleted count.

        Every purge writes a PurgeAnchor in the SAME transaction as the
        deletes (security audit P1-5) — see the sync backend for the full
        chain-detectability rationale.
        """
        cutoff = _retention_cutoff(retention_days, now)
        run_at = _to_naive_utc(now if now is not None else datetime.now(timezone.utc))
        async with self._session_factory() as session:
            doomed_hashes: list[str | None] = list(
                (
                    await session.execute(
                        select(ProvenanceRecordORM.prev_hash)
                        .where(ProvenanceRecordORM.timestamp < cutoff)
                        .order_by(
                            ProvenanceRecordORM.timestamp.asc(),
                            ProvenanceRecordORM.id.asc(),
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not doomed_hashes:
                return 0
            session.add(
                PurgeAnchorORM(
                    purged_before=cutoff,
                    purged_count=len(doomed_hashes),
                    deleted_prev_hashes=doomed_hashes,
                    anchor_created_at=run_at,
                )
            )
            await session.execute(
                delete(ProvenanceRecordORM).where(
                    ProvenanceRecordORM.timestamp < cutoff
                )
            )
            await session.commit()
            return len(doomed_hashes)

    async def list_purge_anchors(self) -> list[PurgeAnchor]:
        """Return the retention journal (oldest anchor first)."""
        async with self._session_factory() as session:
            rows = (
                (
                    await session.execute(
                        select(PurgeAnchorORM).order_by(PurgeAnchorORM.id.asc())
                    )
                )
                .scalars()
                .all()
            )
            return [
                PurgeAnchor(
                    id=row.id,
                    purged_before=row.purged_before.replace(tzinfo=timezone.utc),
                    purged_count=row.purged_count,
                    deleted_prev_hashes=list(row.deleted_prev_hashes or []),
                    anchor_created_at=row.anchor_created_at.replace(
                        tzinfo=timezone.utc
                    ),
                    signature=row.signature,
                )
                for row in rows
            ]

    async def close(self) -> None:
        await self._engine.dispose()


class AsyncSQLiteBackend(_AsyncSQLAlchemyBackend):
    """
    Async SQLite backend using aiosqlite.

    Requires ``aiosqlite`` (installed via the ``[dev]`` extra). Intended
    primarily for testing — use ``AsyncPostgreSQLBackend`` in production.
    Connections run with WAL journaling and a busy timeout for production
    postures (check_same_thread is unnecessary — aiosqlite owns its thread).
    """

    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///./aistamp.db",
        busy_timeout_ms: int = 5000,
    ) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(database_url, echo=False)
        _register_sqlite_pragmas(engine.sync_engine, busy_timeout_ms)
        super().__init__(engine)

    async def __aenter__(self) -> AsyncSQLiteBackend:
        return self


class AsyncPostgreSQLBackend(_AsyncSQLAlchemyBackend):
    """
    Async PostgreSQL backend using asyncpg.

    Requires ``asyncpg`` (installed via ``pip install "ai-stamp[postgres]"``).
    Connection string format: ``postgresql+asyncpg://user:pass@host:port/dbname``.
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )
        super().__init__(engine)

    async def __aenter__(self) -> AsyncPostgreSQLBackend:
        return self
