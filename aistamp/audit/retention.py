"""Retention enforcement routed through the store's anchored purge API.

Direct bulk deletes are forbidden (storage security P1-5): the provenance
hash chain is global, so an unanchored gap left by an ordinary retention
run is indistinguishable from tampering evidence. Deletion therefore goes
through ``StoreBackend.purge``, which writes the purge anchor in the same
transaction as the deletes. The dry-run count is a read-only SELECT with
exactly the purge predicate, so what it reports is what a real
enforcement run deletes.
"""

from __future__ import annotations

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from aistamp.store.backend import (
    PostgreSQLBackend,
    SQLiteBackend,
    _retention_cutoff,  # keep the dry-run count in lockstep with purge
)
from aistamp.store.schema import ProvenanceRecordORM

_AnySyncBackend = SQLiteBackend | PostgreSQLBackend


def _backend_for(database_url: str) -> _AnySyncBackend:
    if database_url.startswith("postgresql"):
        return PostgreSQLBackend(database_url)
    return SQLiteBackend(database_url)


def enforce_retention(
    database_url: str,
    older_than_days: int,
    *,
    dry_run: bool = False,
) -> int:
    """Delete (or, with ``dry_run=True``, count) records past the window.

    Retention is chain-global by design: the purge anchor's integrity
    guarantee covers every record, so scoping a purge to one app would
    leave the other apps' chain positions unanchored and indistinguishable
    from tampering. ``older_than_days`` must be positive.
    """
    if older_than_days <= 0:
        raise ValueError("older_than_days must be a positive integer.")

    if dry_run:
        # Read-only: no rows change, so no anchor is written.
        cutoff = _retention_cutoff(older_than_days, None)
        engine = create_engine(database_url)
        try:
            with Session(engine) as session:
                count = session.execute(
                    select(func.count())
                    .select_from(ProvenanceRecordORM)
                    .where(ProvenanceRecordORM.timestamp < cutoff)
                ).scalar_one()
                return int(count)
        finally:
            engine.dispose()

    backend = _backend_for(database_url)
    try:
        return backend.purge(older_than_days)
    finally:
        backend.dispose()
