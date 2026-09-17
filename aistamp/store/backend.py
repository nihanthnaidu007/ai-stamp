from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from types import TracebackType
from typing import Any, Sequence

from sqlalchemy import (
    Engine,
    Select,
    and_,
    delete,
    event,
    func,
    or_,
    select,
)
from sqlalchemy import ColumnElement
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
        """Query records with filters. Returns a paginated AuditReport.

        Results are deterministically ordered by (timestamp, id).
        """
        ...

    @abstractmethod
    def create_tables(self) -> None:
        """Create all tables. Dev/testing only — use Alembic in production."""
        ...

    @abstractmethod
    def finalize(
        self,
        content_id: str,
        record: ProvenanceRecord,
        hmac_signature: str | None = None,
    ) -> None:
        """Replace the record identified by content_id with its final state.

        Used with write-ahead auditing: persist a PENDING record before the
        provider call, then finalize it with the outcome. If no row exists
        (e.g. the PENDING write was lost), the record is inserted instead.
        """
        ...

    @abstractmethod
    def write_many(
        self, items: Sequence[tuple[ProvenanceRecord, str | None]]
    ) -> None:
        """Persist a batch of (record, hmac_signature) pairs in one transaction."""
        ...

    @abstractmethod
    def purge(
        self, retention_days: int, *, now: datetime | None = None
    ) -> int:
        """Delete records older than retention_days. Returns the deleted count."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release the underlying connection pool. Idempotent."""
        ...

    def dispose(self) -> None:
        """Alias for close()."""
        self.close()

    def __enter__(self) -> StoreBackend:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _to_naive_utc(dt: datetime) -> datetime:
    """Normalize a datetime to naive UTC for DB storage.

    SQLAlchemy's plain ``DateTime`` strips timezone info on storage. To keep
    sign/verify deterministic across a DB roundtrip we always store naive UTC
    and re-attach ``timezone.utc`` on read. Naive datetimes are treated as UTC.
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
        key_id=row.key_id,
        sig_algo=row.sig_algo,
        record_version=row.record_version,
        prev_hash=row.prev_hash,
        scope_sequence=row.scope_sequence,
        error_type=row.error_type,
        error_message=row.error_message,
    )


def _apply_record_to_orm(
    target: ProvenanceRecordORM,
    record: ProvenanceRecord,
    hmac_signature: str | None,
) -> None:
    """Copy every mapped field from a model onto an ORM instance.

    Single source of truth for the model<->ORM mapping, shared by inserts
    (_record_to_orm) and write-ahead finalization.
    """
    target.content_id = record.content_id
    target.app_id = record.app_id
    target.feature_id = record.feature_id
    target.user_id = record.user_id
    target.model = record.model
    target.prompt_hash = record.prompt_hash
    target.response_hash = record.response_hash
    target.hmac_signature = hmac_signature
    target.prompt_tokens = record.prompt_tokens
    target.response_tokens = record.response_tokens
    target.latency_ms = record.latency_ms
    target.timestamp = _to_naive_utc(record.timestamp)
    target.status = record.status.value
    target.pii_result = (
        record.pii_result.model_dump(mode="json")
        if record.pii_result is not None
        else None
    )
    target.policy_decision = (
        record.policy_decision.model_dump(mode="json")
        if record.policy_decision is not None
        else None
    )
    target.key_id = record.key_id
    target.sig_algo = record.sig_algo
    target.record_version = record.record_version
    target.prev_hash = record.prev_hash
    target.scope_sequence = record.scope_sequence
    target.error_type = record.error_type
    target.error_message = record.error_message


def _record_to_orm(
    record: ProvenanceRecord, hmac_signature: str | None
) -> ProvenanceRecordORM:
    row = ProvenanceRecordORM()
    _apply_record_to_orm(row, record, hmac_signature)
    return row


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


_CURSOR_SEPARATOR = "|"


def _encode_cursor(row: ProvenanceRecordORM) -> str:
    """Encode the (timestamp, id) keyset position as an opaque cursor."""
    return f"{row.timestamp.isoformat()}{_CURSOR_SEPARATOR}{row.id}"


def _decode_cursor(cursor: str) -> tuple[datetime, int]:
    timestamp_part, separator, id_part = cursor.rpartition(_CURSOR_SEPARATOR)
    if not separator:
        raise ValueError(f"Malformed pagination cursor: {cursor!r}")
    try:
        timestamp = datetime.fromisoformat(timestamp_part)
        row_id = int(id_part)
    except ValueError as exc:
        raise ValueError(f"Malformed pagination cursor: {cursor!r}") from exc
    return timestamp, row_id


def _build_keyset_condition(filters: QueryFilters) -> ColumnElement[bool] | None:
    """Strict (timestamp, id) tuple comparison for keyset pagination."""
    keyset: tuple[datetime, int] | None = None
    if filters.cursor is not None:
        keyset = _decode_cursor(filters.cursor)
    elif filters.after_timestamp is not None and filters.after_id is not None:
        keyset = (_to_naive_utc(filters.after_timestamp), filters.after_id)
    if keyset is None:
        return None
    after_ts, after_id = keyset
    return or_(
        ProvenanceRecordORM.timestamp > after_ts,
        and_(
            ProvenanceRecordORM.timestamp == after_ts,
            ProvenanceRecordORM.id > after_id,
        ),
    )


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
    keyset_condition = _build_keyset_condition(filters)
    if keyset_condition is not None:
        stmt = stmt.where(keyset_condition)
    # Deterministic total order: the audit trail reads identically no
    # matter which backend or page size produced it.
    stmt = stmt.order_by(
        ProvenanceRecordORM.timestamp.asc(),
        ProvenanceRecordORM.id.asc(),
    )
    # Fetch one extra row to detect has-more without a second query.
    stmt = stmt.limit(filters.limit + 1)
    return stmt, count_stmt


def _page_from_rows(
    rows: Sequence[ProvenanceRecordORM], filters: QueryFilters
) -> tuple[list[ProvenanceRecord], str | None]:
    has_more = len(rows) > filters.limit
    page_rows = rows[: filters.limit] if has_more else list(rows)
    records = [_orm_to_model(r) for r in page_rows]
    next_cursor = _encode_cursor(page_rows[-1]) if has_more and page_rows else None
    return records, next_cursor


def _filters_to_dict(filters: QueryFilters) -> dict[str, Any]:
    return {
        k: (v.value if isinstance(v, (RecordStatus, PIISeverity, PolicyAction)) else v)
        for k, v in filters.model_dump().items()
        if v is not None
    }


def _retention_cutoff(retention_days: int, now: datetime | None) -> datetime:
    if retention_days < 0:
        raise ValueError("retention_days must be >= 0")
    reference = now if now is not None else datetime.now(timezone.utc)
    return _to_naive_utc(reference) - timedelta(days=retention_days)


def _register_sqlite_pragmas(sync_engine: Engine, busy_timeout_ms: int) -> None:
    """Attach WAL journaling and a busy timeout to every new SQLite connection.

    Works for both sync sqlite3 and aiosqlite engines (pass ``engine.sync_engine``
    for the latter — the adapted DBAPI connection presents a sync interface).
    """

    @event.listens_for(sync_engine, "connect")
    def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        try:
            # WAL lets audit reads proceed during writes; the busy timeout
            # turns writer contention into a bounded wait instead of an
            # immediate "database is locked" error.
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
        finally:
            cursor.close()


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

    def finalize(
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
        with Session(self._engine) as session:
            stmt = select(ProvenanceRecordORM).where(
                ProvenanceRecordORM.content_id == content_id
            )
            row = session.execute(stmt).scalar_one_or_none()
            if row is None:
                session.add(_record_to_orm(record, hmac_signature))
            else:
                _apply_record_to_orm(row, record, hmac_signature)
            session.commit()

    def write_many(
        self, items: Sequence[tuple[ProvenanceRecord, str | None]]
    ) -> None:
        pairs = list(items)
        if not pairs:
            return
        with Session(self._engine) as session:
            session.add_all(
                [_record_to_orm(record, hmac) for record, hmac in pairs]
            )
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
            records, next_cursor = _page_from_rows(rows, filters)
            return AuditReport(
                records=records,
                total_count=int(total),
                generated_at=datetime.now(timezone.utc),
                filters_applied=_filters_to_dict(filters),
                next_cursor=next_cursor,
            )

    def purge(self, retention_days: int, *, now: datetime | None = None) -> int:
        cutoff = _retention_cutoff(retention_days, now)
        with Session(self._engine) as session:
            result = session.execute(
                delete(ProvenanceRecordORM).where(
                    ProvenanceRecordORM.timestamp < cutoff
                )
            )
            session.commit()
            return int(result.rowcount or 0)

    def close(self) -> None:
        self._engine.dispose()


class SQLiteBackend(_SyncSQLAlchemyBackend):
    def __init__(
        self,
        database_url: str = "sqlite:///./aistamp.db",
        busy_timeout_ms: int = 5000,
    ) -> None:
        from sqlalchemy import create_engine

        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
        )
        _register_sqlite_pragmas(engine, busy_timeout_ms)
        super().__init__(engine)

    def __enter__(self) -> SQLiteBackend:
        return self


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

    def __enter__(self) -> PostgreSQLBackend:
        return self
