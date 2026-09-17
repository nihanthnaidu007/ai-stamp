from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
)

from aistamp.models import (
    AuditReport,
    ProvenanceRecord,
    QueryFilters,
)
from aistamp.store.backend import (
    _build_query_statements,
    _filters_to_dict,
    _orm_to_model,
    _record_to_orm,
)
from aistamp.store.schema import Base, ProvenanceRecordORM

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
        """Query records with filters. Returns a paginated AuditReport."""
        ...

    @abstractmethod
    async def create_tables(self) -> None:
        """Create all tables. For development and testing only."""
        ...


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
            records = [_orm_to_model(r) for r in rows]
            return AuditReport(
                records=records,
                total_count=int(total),
                generated_at=datetime.now(timezone.utc),
                filters_applied=_filters_to_dict(filters),
            )


class AsyncSQLiteBackend(_AsyncSQLAlchemyBackend):
    """
    Async SQLite backend using aiosqlite.

    Intended primarily for testing — use ``AsyncPostgreSQLBackend`` in
    production.
    """

    def __init__(
        self,
        database_url: str = "sqlite+aiosqlite:///./aistamp.db",
    ) -> None:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(database_url, echo=False)
        super().__init__(engine)


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
