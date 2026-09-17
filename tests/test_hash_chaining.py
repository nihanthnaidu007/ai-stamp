"""Hash-chaining (optional mode) tests: per-scope chains via prev_hash +
scope_sequence, verify_chain structural detection of missing/reordered records,
and rotation interplay."""

from __future__ import annotations

from datetime import datetime, timezone

from aistamp.fingerprint import (
    ChainIssueKind,
    build_chain_link,
    generate_content_id,
    hash_content,
    record_hash,
    rotate_secret,
    sign_record,
    verify_chain,
)
from aistamp.models import ProvenanceRecord, RecordStatus
from aistamp.store import SQLiteBackend

_OLD_KEY = "old-key-" + "a" * 32
_NEW_KEY = "new-key-" + "b" * 32
_SCOPE: tuple[str, str] = ("chain-app", "chain-feature")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    scope: tuple[str, str],
    *,
    scope_sequence: int | None = None,
    prev_hash: str | None = None,
    response_text: str = "response",
) -> ProvenanceRecord:
    app_id, feature_id = scope
    return ProvenanceRecord(
        content_id=generate_content_id(),
        app_id=app_id,
        feature_id=feature_id,
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("prompt"),
        response_hash=hash_content(response_text),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
        scope_sequence=scope_sequence,
        prev_hash=prev_hash,
    )


def _write_record(backend: SQLiteBackend, record: ProvenanceRecord) -> None:
    backend.write(record, sign_record(record, _OLD_KEY))


def _write_chain_record(
    backend: SQLiteBackend,
    scope: tuple[str, str],
    *,
    scope_sequence: int,
    prev_hash: str | None,
) -> ProvenanceRecord:
    record = _make_record(scope, scope_sequence=scope_sequence, prev_hash=prev_hash)
    _write_record(backend, record)
    return record


def _write_chained(
    backend: SQLiteBackend,
    scope: tuple[str, str],
    count: int,
) -> list[ProvenanceRecord]:
    """Write a contiguous, correctly linked chain of `count` records."""
    written: list[ProvenanceRecord] = []
    prev_hash: str | None = None
    for sequence in range(count):
        record = _write_chain_record(
            backend, scope, scope_sequence=sequence, prev_hash=prev_hash
        )
        written.append(record)
        prev_hash = record_hash(record)
    return written


# ---------------------------------------------------------------------------
# Group 1 — build_chain_link (writer side)
# ---------------------------------------------------------------------------


def test_build_chain_link_starts_empty_scope_at_zero(
    sqlite_backend: SQLiteBackend,
) -> None:
    link = build_chain_link(sqlite_backend, _SCOPE)
    assert link.scope_sequence == 0
    assert link.prev_hash is None


def test_build_chain_link_advances_over_written_records(
    sqlite_backend: SQLiteBackend,
) -> None:
    written = _write_chained(sqlite_backend, _SCOPE, 2)
    link = build_chain_link(sqlite_backend, _SCOPE)
    assert link.scope_sequence == 2
    assert link.prev_hash == record_hash(written[-1])


def test_build_chain_link_is_scoped(sqlite_backend: SQLiteBackend) -> None:
    # Chains are per (app, feature): records in other scopes do not advance ours.
    _write_chained(sqlite_backend, ("other-app", "chain-feature"), 3)
    link = build_chain_link(sqlite_backend, _SCOPE)
    assert link.scope_sequence == 0
    assert link.prev_hash is None


# ---------------------------------------------------------------------------
# Group 2 — verify_chain on intact and empty chains
# ---------------------------------------------------------------------------


def test_verify_chain_accepts_intact_chain(sqlite_backend: SQLiteBackend) -> None:
    _write_chained(sqlite_backend, _SCOPE, 3)
    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is True
    assert result.records_checked == 3
    assert result.issues == []
    assert result.app_id == "chain-app"
    assert result.feature_id == "chain-feature"


def test_verify_chain_accepts_empty_scope(sqlite_backend: SQLiteBackend) -> None:
    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is True
    assert result.records_checked == 0


def test_verify_chain_counts_but_ignores_unchained_records(
    sqlite_backend: SQLiteBackend,
) -> None:
    _write_chained(sqlite_backend, _SCOPE, 2)
    loose = _make_record(_SCOPE, response_text="loose")
    _write_record(sqlite_backend, loose)

    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is True
    assert result.records_checked == 2
    assert result.unchained_records == 1


# ---------------------------------------------------------------------------
# Group 3 — verify_chain detects MISSING records
# ---------------------------------------------------------------------------


def test_verify_chain_detects_sequence_gap(sqlite_backend: SQLiteBackend) -> None:
    # The record with scope_sequence 1 was removed. The surviving records are
    # individually signed and intact — only the chain exposes the deletion.
    _write_chain_record(sqlite_backend, _SCOPE, scope_sequence=0, prev_hash=None)
    phantom = _make_record(_SCOPE, scope_sequence=1)
    _write_chain_record(
        sqlite_backend,
        _SCOPE,
        scope_sequence=2,
        prev_hash=record_hash(phantom),
    )

    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is False
    gaps = [
        issue
        for issue in result.issues
        if issue.kind == ChainIssueKind.MISSING and "expected 1" in issue.detail
    ]
    assert len(gaps) == 1


def test_verify_chain_detects_chain_starting_past_zero(
    sqlite_backend: SQLiteBackend,
) -> None:
    # The head of the chain itself is gone.
    _write_chain_record(sqlite_backend, _SCOPE, scope_sequence=1, prev_hash=None)
    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is False
    assert any(
        issue.kind == ChainIssueKind.MISSING and "expected 0" in issue.detail
        for issue in result.issues
    )


def test_verify_chain_detects_dangling_head_link(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Head claims a predecessor that no longer exists in the scope.
    _write_chain_record(sqlite_backend, _SCOPE, scope_sequence=0, prev_hash="c" * 64)
    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is False
    dangling = [
        issue
        for issue in result.issues
        if issue.kind == ChainIssueKind.MISSING
        and "no earlier record exists" in issue.detail
    ]
    assert len(dangling) == 1


# ---------------------------------------------------------------------------
# Group 4 — verify_chain detects REORDERED / inconsistent links
# ---------------------------------------------------------------------------


def test_verify_chain_detects_broken_link(sqlite_backend: SQLiteBackend) -> None:
    records = _write_chained(sqlite_backend, _SCOPE, 3)
    # Attacker alters the middle record's content without the signing key:
    # the successor's prev_hash no longer matches the altered record's hash.
    tampered = records[1].model_copy(update={"response_hash": hash_content("stolen")})
    sqlite_backend.update_record(tampered, sign_record(records[1], _OLD_KEY))

    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is False
    broken = [i for i in result.issues if i.kind == ChainIssueKind.REORDERED]
    assert len(broken) == 1
    assert broken[0].content_id == records[2].content_id
    assert broken[0].sequence == 2


def test_verify_chain_detects_duplicate_sequence(
    sqlite_backend: SQLiteBackend,
) -> None:
    records = _write_chained(sqlite_backend, _SCOPE, 2)
    # A second record claims scope_sequence 1 with a perfectly valid link to
    # the head — the only defect is the duplicated position.
    duplicate = _make_record(
        _SCOPE, scope_sequence=1, prev_hash=record_hash(records[0])
    )
    _write_record(sqlite_backend, duplicate)

    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is False
    dupes = [i for i in result.issues if i.kind == ChainIssueKind.REORDERED]
    assert len(dupes) == 1
    assert "claimed by multiple records" in dupes[0].detail


# ---------------------------------------------------------------------------
# Group 5 — chain x rotation interplay
# ---------------------------------------------------------------------------


def test_rotation_re_sign_preserves_chain_integrity(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Flagship interplay: envelope relabeling during rotation must not break
    # chain links, because prev_hash chains canonical content (which excludes
    # key_id/sig_algo), not signatures.
    _write_chained(sqlite_backend, _SCOPE, 3)
    before = verify_chain(sqlite_backend, _SCOPE)
    assert before.valid is True

    report = rotate_secret(_OLD_KEY, _NEW_KEY, "v2", sqlite_backend, re_sign=True)
    assert report.records_re_signed == 3

    after = verify_chain(sqlite_backend, _SCOPE)
    assert after.valid is True
    assert after.issues == []


def test_chain_detection_does_not_require_key_material(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Structural chain checks complement per-record HMACs: a scope can be
    # audited for missing/reordered records without holding any secret keys.
    _write_chained(sqlite_backend, _SCOPE, 2)
    result = verify_chain(sqlite_backend, _SCOPE)
    assert result.valid is True
