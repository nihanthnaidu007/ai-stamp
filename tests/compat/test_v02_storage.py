"""v0.2 storage-and-tamper-evidence compatibility checks.

Covers retention purge anchors (one anchor per purge, written in the same
transaction as the deletes), the record_version 2 envelope, and purge-aware
chain verification — all active on main via the tamper-evidence track
(PR #3). V2-14 (record_version 1 envelope) is intentionally obsolete: the
merged default moved to 2, so V2-16 supersedes it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import _features
import pytest
import pytest_asyncio

from aistamp.client import AsyncProvenanceClient, ProvenanceClient
from aistamp.config import Config
from aistamp.fingerprint import generate_content_id, hash_content, record_hash
from aistamp.models import ProvenanceRecord, PurgeAnchor, RecordStatus
from aistamp.store import SQLiteBackend
from aistamp.store.async_backend import AsyncSQLiteBackend

_SECRET = "compat-kit-secret-key-0-2-32-chars!!"
_MODEL = "gpt-4o-mini"
# Six-record chain scenario: positions 0-2 predate the purge cutoff, 3-5
# survive it (mirrors tests/test_hash_chaining.py's anchored-head setup).
_TS_OLD = datetime(2024, 1, 1, tzinfo=timezone.utc)
_TS_NEW = datetime(2024, 1, 3, tzinfo=timezone.utc)
_PURGE_NOW = datetime(2024, 1, 4, tzinfo=timezone.utc)


def _config() -> Config:
    return Config(secret_key=_SECRET, database_url="sqlite:///:memory:")


def _aware(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt


def _sync_client(backend: SQLiteBackend) -> ProvenanceClient:
    def llm(prompt: str) -> str:
        return f"echo: {prompt}"

    return ProvenanceClient(
        llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )


def _async_client(backend: AsyncSQLiteBackend) -> AsyncProvenanceClient:
    async def llm(prompt: str) -> str:
        return f"echo: {prompt}"

    return AsyncProvenanceClient(
        llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )


def _stamp_three(backend: SQLiteBackend) -> list[str]:
    client = _sync_client(backend)
    return [client.stamp(f"record {i}", _MODEL).content_id for i in range(3)]


@pytest.fixture
def backend() -> SQLiteBackend:
    b = SQLiteBackend("sqlite:///:memory:")
    b.create_tables()
    return b


@pytest_asyncio.fixture
async def async_backend() -> AsyncSQLiteBackend:
    b = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    await b.create_tables()
    yield b
    await b._engine.dispose()


# V2-12 ----------------------------------------------------------------------
def test_v2_12_sync_purge_writes_one_anchor_per_purge(backend: SQLiteBackend) -> None:
    content_ids = _stamp_three(backend)
    now = datetime.now(timezone.utc) + timedelta(days=2)
    purged = backend.purge(1, now=now)
    assert purged == 3
    anchors = backend.list_purge_anchors()
    assert len(anchors) == 1
    anchor = anchors[0]
    assert isinstance(anchor, PurgeAnchor)
    assert anchor.purged_count == 3
    cutoff = now - timedelta(days=1)
    assert _aware(anchor.purged_before) == _aware(cutoff)
    for content_id in content_ids:
        assert backend.get(content_id) is None


# V2-13 ----------------------------------------------------------------------
@pytest.mark.asyncio
async def test_v2_13_async_purge_writes_one_anchor_per_purge(
    async_backend: AsyncSQLiteBackend,
) -> None:
    client = _async_client(async_backend)
    content_ids = [
        (await client.stamp(f"record {i}", _MODEL)).content_id for i in range(3)
    ]
    now = datetime.now(timezone.utc) + timedelta(days=2)
    purged = await async_backend.purge(1, now=now)
    assert purged == 3
    anchors = await async_backend.list_purge_anchors()
    assert len(anchors) == 1
    assert anchors[0].purged_count == 3
    for content_id in content_ids:
        assert await async_backend.get(content_id) is None


# V2-14 ----------------------------------------------------------------------
def test_v2_14_fresh_records_carry_v1_envelope(backend: SQLiteBackend) -> None:
    """New records keep the byte-compatible v1 envelope (record_version 1).

    Superseded once tamper PR #3 flips the default to record_version 2 —
    V2-16 takes over then, and this check retires.
    """
    if _features.record_version_default() != 1:
        pytest.skip("V2-14 obsolete after tamper PR #3 lands: superseded by V2-16")
    record = _sync_client(backend).stamp("envelope", _MODEL).record
    assert record.record_version == 1
    assert record.key_id == "default"
    assert record.sig_algo == "HMAC-SHA256"


# V2-15 (pending until tamper-evidence purges and chain verification agree) ----
def test_v2_15_purge_aware_chain_verification(backend: SQLiteBackend) -> None:
    """After a purge that wrote an anchor, chain verification must not flag
    the purged positions as MISSING — retention and chain verification have
    to agree. The merged tamper-evidence track resolves this internally via
    the purge journal; the check is PENDING on branches without it.
    """
    verify_chain = _features.verify_chain()
    if verify_chain is None or not _features.is_purge_aware():
        _features.skip_pending_pr(3, "purge-aware chain verification (verify_chain)")

    scope = ("compat-app", "compat-feat")
    prev_hash: str | None = None
    for sequence in range(6):
        record = _chain_record(
            scope,
            scope_sequence=sequence,
            prev_hash=prev_hash,
            timestamp=_TS_OLD if sequence < 3 else _TS_NEW,
        )
        backend.write(record, None)
        prev_hash = record_hash(record)

    backend.purge(1, now=_PURGE_NOW)

    result = verify_chain(backend, scope)
    assert result.valid is True
    assert result.issues == []
    assert result.anchored_gaps == 1
    assert result.records_checked == 3


def _chain_record(
    scope: tuple[str, str],
    *,
    scope_sequence: int,
    prev_hash: str | None,
    timestamp: datetime,
) -> ProvenanceRecord:
    app_id, feature_id = scope
    return ProvenanceRecord(
        content_id=generate_content_id(),
        app_id=app_id,
        feature_id=feature_id,
        user_id="user",
        model=_MODEL,
        prompt_hash=hash_content("prompt"),
        response_hash=hash_content("response"),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=timestamp,
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
        scope_sequence=scope_sequence,
        prev_hash=prev_hash,
    )


# V2-16 (tamper-evidence track, PR #3) ----------------------------------------
def test_v2_16_record_version_2_envelope_on_new_records(backend: SQLiteBackend) -> None:
    """Tamper PR #3 flips the envelope: new records ship record_version 2."""
    if _features.record_version_default() != 2:
        _features.skip_pending_pr(3, "record_version 2 envelope default")
    record = _sync_client(backend).stamp("envelope", _MODEL).record
    assert record.record_version == 2
    assert record.key_id == "default"
    assert record.sig_algo == "HMAC-SHA256"
