from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Engine, Select, func, select
from sqlalchemy.orm import Session

from aistamp.models import (
    AuditReport,
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
)
from aistamp.store.schema import Base, ProvenanceRecordORM

# TODO(phase-v2): Redis backend for high-throughput event streaming


class StoreBackend(ABC):
    @abstractmethod
    def write(self, record: ProvenanceRecord, hmac_signature: str | None) -> None:
        """Persist a provenance record with its HMAC signature."""
        ...

    @abstractmethod
    def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        """
        Retrieve a record by content_id.
        Returns (ProvenanceRecord, hmac_signature) or None if not found.
        hmac_signature is None when no signature was stored.
        """
        ...

    @abstractmethod
    def query(self, filters: QueryFilters) -> AuditReport:
        """Query records with filters. Returns a paginated AuditReport."""
        ...

    @abstractmethod
    def create_tables(self) -> None:
        """Create all tables. Dev/testing only — use Alembic in production."""
        ...


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC for DB storage.

    SQLAlchemy's plain ``DateTime`` strips timezone info on storage. To keep
    sign/verify deterministic across a DB roundtrip we always store naive UTC
    and re-attach ``timezone.utc`` on read.
    """
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


def _orm_to_model(row: ProvenanceRecordORM) -> ProvenanceRecord:
    pii_result = (
        PIIResult.model_validate(row.pii_result) if row.pii_result is not None else None
    )
    policy_decision = (
        PolicyDecision.model_validate(row.policy_decision)
        if row.policy_decision is not None
        else None
    )
    ts = row.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ProvenanceRecord(
        content_id=row.content_id,
        app_id=row.app_id,
        feature_id=row.feature_id,
        user_id=row.user_id,
        model=row.model,
        prompt_hash=row.prompt_hash,
        response_hash=row.response_hash,
        prompt_tokens=row.prompt_tokens,
        response_tokens=row.response_tokens,
        latency_ms=row.latency_ms,
        timestamp=ts,
        status=RecordStatus(row.status),
        pii_result=pii_result,
        policy_decision=policy_decision,
    )


def _record_to_orm(
    record: ProvenanceRecord, hmac_signature: str | None
) -> ProvenanceRecordORM:
    return ProvenanceRecordORM(
        content_id=record.content_id,
        app_id=record.app_id,
        feature_id=record.feature_id,
        user_id=record.user_id,
        model=record.model,
        prompt_hash=record.prompt_hash,
        response_hash=record.response_hash,
        hmac_signature=hmac_signature,
        prompt_tokens=record.prompt_tokens,
        response_tokens=record.response_tokens,
        latency_ms=record.latency_ms,
        timestamp=_to_naive_utc(record.timestamp),
        status=record.status.value,
        pii_result=(
            record.pii_result.model_dump(mode="json")
            if record.pii_result is not None
            else None
        ),
        policy_decision=(
            record.policy_decision.model_dump(mode="json")
            if record.policy_decision is not None
            else None
        ),
    )


def _build_filter_conditions(filters: QueryFilters) -> list[Any]:
    conditions: list[Any] = []
    if filters.content_id is not None:
        conditions.append(ProvenanceRecordORM.content_id == filters.content_id)
    if filters.user_id is not None:
        conditions.append(ProvenanceRecordORM.user_id == filters.user_id)
    if filters.app_id is not None:
        conditions.append(ProvenanceRecordORM.app_id == filters.app_id)
    if filters.feature_id is not None:
        conditions.append(ProvenanceRecordORM.feature_id == filters.feature_id)
    if filters.model is not None:
        conditions.append(ProvenanceRecordORM.model == filters.model)
    if filters.status is not None:
        conditions.append(ProvenanceRecordORM.status == filters.status.value)
    if filters.pii_severity is not None:
        conditions.append(
            ProvenanceRecordORM.pii_result["highest_severity"].as_string()
            == filters.pii_severity.value
        )
    if filters.policy_decision is not None:
        conditions.append(
            ProvenanceRecordORM.policy_decision["action"].as_string()
            == filters.policy_decision.value
        )
    if filters.from_dt is not None:
        conditions.append(ProvenanceRecordORM.timestamp >= filters.from_dt)
    if filters.to_dt is not None:
        conditions.append(ProvenanceRecordORM.timestamp <= filters.to_dt)
    return conditions


def _build_query_statements(
    filters: QueryFilters,
) -> tuple[Select[tuple[ProvenanceRecordORM]], Select[tuple[int]]]:
    conditions = _build_filter_conditions(filters)
    stmt: Select[tuple[ProvenanceRecordORM]] = select(ProvenanceRecordORM)
    count_stmt: Select[tuple[int]] = select(func.count()).select_from(
        ProvenanceRecordORM
    )
    for cond in conditions:
        stmt = stmt.where(cond)
        count_stmt = count_stmt.where(cond)
    stmt = stmt.limit(filters.limit).offset(filters.offset)
    return stmt, count_stmt


def _filters_to_dict(filters: QueryFilters) -> dict[str, Any]:
    return {
        k: (v.value if isinstance(v, (RecordStatus, PIISeverity, PolicyAction)) else v)
        for k, v in asdict(filters).items()
        if v is not None
    }


class _SyncSQLAlchemyBackend(StoreBackend):
    """Shared sync SQLAlchemy implementation. Not part of the public API."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_tables(self) -> None:
        Base.metadata.create_all(self._engine)

    def write(self, record: ProvenanceRecord, hmac_signature: str | None) -> None:
        with Session(self._engine) as session:
            session.add(_record_to_orm(record, hmac_signature))
            session.commit()

    def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        with Session(self._engine) as session:
            stmt = select(ProvenanceRecordORM).where(
                ProvenanceRecordORM.content_id == content_id
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                return None
            record = _orm_to_model(row)
            return record, row.hmac_signature

    def query(self, filters: QueryFilters) -> AuditReport:
        with Session(self._engine) as session:
            stmt, count_stmt = _build_query_statements(filters)
            rows = session.execute(stmt).scalars().all()
            total = session.execute(count_stmt).scalar_one()
            records = [_orm_to_model(r) for r in rows]
            return AuditReport(
                records=records,
                total_count=int(total),
                generated_at=datetime.now(timezone.utc),
                filters_applied=_filters_to_dict(filters),
            )


class SQLiteBackend(_SyncSQLAlchemyBackend):
    def __init__(self, database_url: str = "sqlite:///./aistamp.db") -> None:
        from sqlalchemy import create_engine

        super().__init__(create_engine(database_url))


class PostgreSQLBackend(_SyncSQLAlchemyBackend):
    """
    PostgreSQL-backed provenance store using SQLAlchemy sync engine.

    Requires psycopg2-binary: ``pip install "ai-stamp[postgres]"``.
    Uses connection pooling via SQLAlchemy's QueuePool with ``pool_pre_ping``
    to guard against stale pooled connections.
    """

    def __init__(
        self,
        database_url: str,
        pool_size: int = 5,
        max_overflow: int = 10,
    ) -> None:
        from sqlalchemy import create_engine

        engine = create_engine(
            database_url,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_pre_ping=True,
        )
        super().__init__(engine)
