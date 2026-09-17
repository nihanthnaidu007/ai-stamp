"""Retention purge with anchors — deleting history without breaking the audit.

Retention in v0.2 is chain-aware: ``backend.purge(retention_days)`` deletes
records older than the cutoff and, in the SAME transaction, writes a
``PurgeAnchor`` listing the chain positions (``prev_hash`` values) it
removed. A later chain gap is either anchored (legitimate retention) or
unexplained (tampering evidence) — deletion is never silent.

This script demonstrates:

1. Writing three signed records at staggered ages (90 / 40 / 0 days old)
   straight through ``SQLiteBackend.write`` — the same write path the
   client pipeline uses.
2. ``purge(retention_days=30, now=...)`` with a fixed ``now`` for a
   reproducible demo: the 90- and 40-day records go, today's survives.
3. Reading the retention journal with ``list_purge_anchors()`` — one anchor
   with ``purged_count=2`` and the removed chain positions.
4. Post-purge integrity: the survivor still verifies; the purged content
   IDs are gone from the store and ``verify_record`` raises
   ``RecordNotFoundError`` for them.

Run from the repo root:

    python examples/05_retention_purge.py

Expected output (content_ids and hashes vary per run):

    records written : 3 (ages 90, 40, 0 days)
    purged          : 2 record(s)
    surviving       : 1
    -- retention journal --
    anchor #1 purged_before=2026-08-18T12:00:00+00:00 purged_count=2
      chain positions removed: 2
    -- post-purge integrity --
    survivor verifies : True
    purged record gone: True
    verify purged id  : RecordNotFoundError (as documented)

The provider is fake (records are built directly), so nothing here touches
the network.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from pydantic import SecretStr

from aistamp import (
    ProvenanceRecord,
    QueryFilters,
    RecordNotFoundError,
    RecordStatus,
    SQLiteBackend,
    generate_content_id,
    hash_content,
    sign_record,
    verify_record,
)

# Demo-only. Load the real key from your secret store in production.
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")

RETENTION_DAYS = 30


def write_aged_record(
    backend: SQLiteBackend,
    *,
    prompt: str,
    response: str,
    age_days: int,
    now: datetime,
) -> ProvenanceRecord:
    """Write one signed record dated ``age_days`` in the past."""
    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="cookbook",
        feature_id="retention",
        user_id="demo_user",
        model="gpt-4o-mini",
        prompt_hash=hash_content(prompt),
        response_hash=hash_content(response),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=12.5,
        timestamp=now - timedelta(days=age_days),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )
    backend.write(record, sign_record(record, SECRET_KEY))
    return record


def main() -> None:
    db_dir = Path(tempfile.mkdtemp(prefix="aistamp-05-"))
    backend = SQLiteBackend(f"sqlite:///{db_dir / 'retention.db'}")
    backend.create_tables()

    now = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)
    aged: list[tuple[ProvenanceRecord, str]] = []
    for i, age_days in enumerate((90, 40, 0), start=1):
        response = f"archived response {i}"
        record = write_aged_record(
            backend,
            prompt=f"archived request {i}",
            response=response,
            age_days=age_days,
            now=now,
        )
        aged.append((record, response))
    print(f"records written : {len(aged)} (ages 90, 40, 0 days)")

    deleted = backend.purge(RETENTION_DAYS, now=now)
    surviving = backend.query(QueryFilters())
    print(f"purged          : {deleted} record(s)")
    print(f"surviving       : {surviving.total_count}")

    print("-- retention journal --")
    anchors = backend.list_purge_anchors()
    for anchor in anchors:
        print(
            f"anchor #{anchor.id} purged_before={anchor.purged_before.isoformat()}"
            f" purged_count={anchor.purged_count}"
        )
        print(f"  chain positions removed: {len(anchor.deleted_prev_hashes)}")

    print("-- post-purge integrity --")
    survivor, survivor_text = aged[-1]
    verdict = verify_record(survivor.content_id, survivor_text, backend, SECRET_KEY)
    print(f"survivor verifies : {verdict.verified}")

    purged_id = aged[0][0].content_id
    print(f"purged record gone: {backend.get(purged_id) is None}")
    try:
        verify_record(purged_id, "archived response 1", backend, SECRET_KEY)
    except RecordNotFoundError:
        print("verify purged id  : RecordNotFoundError (as documented)")

    if deleted != 2 or len(anchors) != 1 or not verdict.verified:
        raise RuntimeError("retention demo did not behave as documented")


if __name__ == "__main__":
    main()
