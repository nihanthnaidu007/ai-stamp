"""Retention enforcement: delete provenance records past their retention window.

Operates directly on the store schema because StoreBackend does not yet
expose a delete API; revisit if the storage track adds one. The deletion
targets the same naive-UTC timestamp column the backend writes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session

from aistamp.store.schema import ProvenanceRecordORM


def enforce_retention(
    database_url: str,
    older_than_days: int,
    app_id: str | None = None,
    dry_run: bool = False,
) -> int:
    """Delete records older than ``older_than_days``; return how many.

    With ``dry_run=True`` no rows are deleted and the count of records that
    WOULD be deleted is returned. ``older_than_days`` must be positive.
    """
    if older_than_days <= 0:
        raise ValueError("older_than_days must be a positive integer.")

    # The store persists timestamps as naive UTC; compare in the same terms.
    cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).replace(
        tzinfo=None
    )
    conditions = [ProvenanceRecordORM.timestamp < cutoff]
    if app_id is not None:
        conditions.append(ProvenanceRecordORM.app_id == app_id)

    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            if dry_run:
                stmt = (
                    select(func.count())
                    .select_from(ProvenanceRecordORM)
                    .where(*conditions)
                )
                count = session.execute(stmt).scalar_one()
                return int(count)
            result = session.execute(
                delete(ProvenanceRecordORM).where(*conditions)
            )
            session.commit()
            cursor = cast("CursorResult[Any]", result)
            return int(cursor.rowcount or 0)
    finally:
        engine.dispose()
