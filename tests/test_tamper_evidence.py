"""Tamper-evidence v2 tests: envelope, keyring verification, rotation,
canonicalization agility, async verification, and 0.1.x byte-compatibility."""

from __future__ import annotations

import hashlib
import hmac
import json
import threading
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType

import pytest

from aistamp.fingerprint import (
    CanonicalizationError,
    FingerprintError,
    RecordNotFoundError,
    RotationError,
    SignatureStatus,
    UnsupportedAlgorithmError,
    UnsupportedRecordVersionError,
    generate_content_id,
    hash_content,
    record_hash,
    rotate_secret,
    sign_record,
    verify_record,
    verify_record_async,
)
from aistamp.models import (
    AuditReport,
    ProvenanceRecord,
    PurgeAnchor,
    QueryFilters,
    RecordStatus,
    VerificationResult,
)
from aistamp.store import AsyncSQLiteBackend, SQLiteBackend, StoreBackend

_OLD_KEY = "old-key-" + "a" * 32
_NEW_KEY = "new-key-" + "b" * 32
_NEW_KEY_ID = "v2"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    content_id: str | None = None,
    response_text: str | None = "response",
    key_id: str = "default",
    sig_algo: str = "HMAC-SHA256",
    record_version: int = 2,
    scope_sequence: int | None = None,
    prev_hash: str | None = None,
    app_id: str = "app",
    feature_id: str = "feat",
) -> ProvenanceRecord:
    # response_text=None produces a record without a response hash (None sentinel).
    return ProvenanceRecord(
        content_id=content_id or generate_content_id(),
        app_id=app_id,
        feature_id=feature_id,
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("prompt"),
        response_hash=(
            hash_content(response_text) if response_text is not None else None
        ),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
        key_id=key_id,
        sig_algo=sig_algo,
        scope_sequence=scope_sequence,
        prev_hash=prev_hash,
        record_version=record_version,
    )


def _write_signed(
    backend: SQLiteBackend,
    record: ProvenanceRecord,
    key: str,
) -> str:
    signature = sign_record(record, key)
    backend.write(record, signature)
    return signature


class _NoUpdateBackend(StoreBackend):
    """Minimal backend without update_record (inherits the unsupported default)."""

    def __init__(self, record: ProvenanceRecord, signature: str) -> None:
        self._record = record
        self._signature = signature

    def write(self, record: ProvenanceRecord, hmac_signature: str | None) -> None:
        raise NotImplementedError("not used in this test")

    def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        if content_id == self._record.content_id:
            return self._record, self._signature
        return None

    def query(self, filters: QueryFilters) -> AuditReport:
        return AuditReport(
            records=[self._record],
            total_count=1,
            generated_at=datetime.now(timezone.utc),
            filters_applied={},
        )

    def create_tables(self) -> None:
        raise NotImplementedError("not used in this test")

    # Storage v2 widened the abstract protocol; these are never called in
    # this test — the stub exists to inherit update_record's unsupported
    # default and prove rotate_secret refuses re-signing on it.
    def finalize(
        self,
        content_id: str,
        record: ProvenanceRecord,
        hmac_signature: str | None = None,
    ) -> None:
        raise NotImplementedError("not used in this test")

    def write_many(self, items: Sequence[tuple[ProvenanceRecord, str | None]]) -> None:
        raise NotImplementedError("not used in this test")

    def purge(self, retention_days: int, *, now: datetime | None = None) -> int:
        raise NotImplementedError("not used in this test")

    def list_purge_anchors(self) -> list[PurgeAnchor]:
        raise NotImplementedError("not used in this test")

    def close(self) -> None:
        raise NotImplementedError("not used in this test")


class _VanishingBackend(_NoUpdateBackend):
    """Backend whose get() always misses — simulates a record deleted mid-scan."""

    def get(self, content_id: str) -> tuple[ProvenanceRecord, str | None] | None:
        return None


# ---------------------------------------------------------------------------
# Group 1 — signature envelope on ProvenanceRecord
# ---------------------------------------------------------------------------


def test_envelope_fields_default_for_v0_1_records() -> None:
    # Defaults now produce a v2 record (the audit-mandated secure default);
    # v1 remains available explicitly for 0.1.x byte-compatible signatures.
    record = _make_record()
    assert record.key_id == "default"
    assert record.sig_algo == "HMAC-SHA256"
    assert record.record_version == 2
    legacy = _make_record(record_version=1)
    assert legacy.record_version == 1


def test_verification_result_exposes_key_id(sqlite_backend: SQLiteBackend) -> None:
    # The result must surface the key identity the record was signed under.
    record = _make_record(key_id="v1")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(
        record.content_id,
        "response",
        sqlite_backend,
        _OLD_KEY,
        keyring={"v1": _OLD_KEY},
    )
    assert result.key_id == "v1"


def test_unknown_sig_algo_fails_signing_loudly() -> None:
    # No silent fallback: an algorithm this version does not implement must raise.
    record = _make_record(sig_algo="HMAC-SHA3-256")
    with pytest.raises(UnsupportedAlgorithmError):
        sign_record(record, _OLD_KEY)


def test_unknown_sig_algo_fails_verification_loudly(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Envelope-metadata tampering: an attacker rewrites the stored sig_algo to
    # an algorithm this version does not implement. Verification must fail
    # loudly rather than silently mis-verify under a different algorithm.
    record = _make_record(key_id="v1")
    signature = _write_signed(sqlite_backend, record, _OLD_KEY)
    flipped = record.model_copy(update={"sig_algo": "HMAC-SHA3-256"})
    sqlite_backend.update_record(flipped, signature)

    with pytest.raises(UnsupportedAlgorithmError):
        verify_record(
            record.content_id,
            "response",
            sqlite_backend,
            _OLD_KEY,
            keyring={"v1": _OLD_KEY},
        )


# ---------------------------------------------------------------------------
# Group 2 — keyring verification states
# ---------------------------------------------------------------------------


def test_keyring_verifies_active_and_retired_keys(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Historical records signed under a retired key must still verify once the
    # keyring holds both the active and the retired key.
    old_record = _make_record(key_id="v1", response_text="old-text")
    new_record = _make_record(key_id=_NEW_KEY_ID, response_text="new-text")
    _write_signed(sqlite_backend, old_record, _OLD_KEY)
    _write_signed(sqlite_backend, new_record, _NEW_KEY)

    keyring: dict[str, str] = {"v1": _OLD_KEY, _NEW_KEY_ID: _NEW_KEY}
    old_result = verify_record(
        old_record.content_id, "old-text", sqlite_backend, _NEW_KEY, keyring=keyring
    )
    new_result = verify_record(
        new_record.content_id, "new-text", sqlite_backend, _NEW_KEY, keyring=keyring
    )
    assert old_result.verified is True
    assert old_result.key_id == "v1"
    assert new_result.verified is True
    assert new_result.key_id == _NEW_KEY_ID


def test_keyring_without_record_key_yields_unknown_key_status(
    sqlite_backend: SQLiteBackend,
) -> None:
    # A keyring that lacks the signing key must report UNKNOWN_KEY, not tampering.
    record = _make_record(key_id="v1")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(
        record.content_id,
        "response",
        sqlite_backend,
        _NEW_KEY,
        keyring={_NEW_KEY_ID: _NEW_KEY},
    )
    assert result.signature_status == SignatureStatus.UNKNOWN_KEY
    assert result.hmac_valid is False
    assert result.verified is False


def test_tampered_signature_reports_invalid(
    sqlite_backend: SQLiteBackend,
) -> None:
    # An attacker who rewrites stored content (without the key) leaves the old
    # signature in place: the result must distinguish tampering from unknown keys.
    record = _make_record(key_id="v1")
    signature = _write_signed(sqlite_backend, record, _OLD_KEY)

    tampered = record.model_copy(update={"response_hash": hash_content("stolen")})
    sqlite_backend.update_record(tampered, signature)

    result = verify_record(
        record.content_id,
        "stolen",
        sqlite_backend,
        _OLD_KEY,
        keyring={"v1": _OLD_KEY},
    )
    assert result.signature_status == SignatureStatus.INVALID
    assert result.hmac_valid is False
    assert result.verified is False


def test_unsigned_record_reports_unsigned(sqlite_backend: SQLiteBackend) -> None:
    record = _make_record(response_text="text")
    sqlite_backend.write(record, None)
    result = verify_record(
        record.content_id,
        "text",
        sqlite_backend,
        _OLD_KEY,
        keyring={"default": _OLD_KEY},
    )
    assert result.signature_status == SignatureStatus.UNSIGNED
    assert result.hmac_valid is False


def test_keyring_omitted_preserves_v0_1_behavior(sqlite_backend: SQLiteBackend) -> None:
    # Without a keyring the bare secret_key argument is used, exactly as 0.1.x.
    record = _make_record(response_text="text")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(record.content_id, "text", sqlite_backend, _OLD_KEY)
    assert result.verified is True
    assert result.signature_status == SignatureStatus.VALID


def test_verify_record_still_works_positionally(sqlite_backend: SQLiteBackend) -> None:
    # 0.1.x positional call shape must keep working.
    record = _make_record(response_text="text")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(record.content_id, "text", sqlite_backend, _OLD_KEY)
    assert isinstance(result, VerificationResult)
    assert result.verified is True


def test_valid_signature_with_drifted_content_is_valid_but_unverified(
    sqlite_backend: SQLiteBackend,
) -> None:
    # signature_status describes the signature, not the presented content.
    record = _make_record(key_id="v1", response_text="original")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(
        record.content_id, "other", sqlite_backend, _OLD_KEY, keyring={"v1": _OLD_KEY}
    )
    assert result.signature_status == SignatureStatus.VALID
    assert result.hash_match is False
    assert result.drift_detected is True
    assert result.verified is False


def test_keyring_accepts_readonly_mappings(sqlite_backend: SQLiteBackend) -> None:
    # The keyring parameter is typed as Mapping, so read-only keyrings work.
    record = _make_record(key_id="v1", response_text="text")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    keyring = MappingProxyType({"v1": _OLD_KEY})
    result = verify_record(
        record.content_id, "text", sqlite_backend, _OLD_KEY, keyring=keyring
    )
    assert result.verified is True


# ---------------------------------------------------------------------------
# Group 3 — rotation workflow
# ---------------------------------------------------------------------------


def _seed_mixed_store(backend: SQLiteBackend) -> dict[ProvenanceRecord, str]:
    # Map each seeded record to the original response text it was written with.
    old_a = _make_record(key_id="v1", response_text="a")
    old_b = _make_record(key_id="v1", response_text="b")
    on_new = _make_record(key_id=_NEW_KEY_ID, response_text="c")
    unsigned = _make_record(response_text="d")
    _write_signed(backend, old_a, _OLD_KEY)
    _write_signed(backend, old_b, _OLD_KEY)
    _write_signed(backend, on_new, _NEW_KEY)
    backend.write(unsigned, None)
    return {old_a: "a", old_b: "b", on_new: "c", unsigned: "d"}


def test_rotation_without_re_sign_leaves_records_verifiable(
    sqlite_backend: SQLiteBackend,
) -> None:
    seeded = _seed_mixed_store(sqlite_backend)
    report = rotate_secret(_OLD_KEY, _NEW_KEY, _NEW_KEY_ID, sqlite_backend)

    assert report.records_scanned == 4
    assert report.records_legacy_verified == 2
    assert report.records_re_signed == 0

    # Historical records still verify under the retired key via the keyring.
    keyring: dict[str, str] = {"v1": _OLD_KEY, _NEW_KEY_ID: _NEW_KEY}
    for record, text in seeded.items():
        if record.key_id != "v1":
            continue
        result = verify_record(
            record.content_id, text, sqlite_backend, _NEW_KEY, keyring=keyring
        )
        assert result.verified is True
        assert result.key_id == "v1"


def test_rotation_with_re_sign_rewrites_records_under_new_key(
    sqlite_backend: SQLiteBackend,
) -> None:
    old_a = _make_record(key_id="v1", response_text="a")
    _write_signed(sqlite_backend, old_a, _OLD_KEY)

    report = rotate_secret(
        _OLD_KEY, _NEW_KEY, _NEW_KEY_ID, sqlite_backend, re_sign=True
    )
    assert report.records_scanned == 1
    assert report.records_legacy_verified == 1
    assert report.records_re_signed == 1

    fetched = sqlite_backend.get(old_a.content_id)
    assert fetched is not None
    rotated, _ = fetched
    assert rotated.key_id == _NEW_KEY_ID

    # Verifiable under the new key; the retired key no longer claims it.
    new_result = verify_record(
        old_a.content_id,
        "a",
        sqlite_backend,
        _NEW_KEY,
        keyring={_NEW_KEY_ID: _NEW_KEY},
    )
    assert new_result.verified is True
    assert new_result.key_id == _NEW_KEY_ID

    retired_only = verify_record(
        old_a.content_id,
        "a",
        sqlite_backend,
        _OLD_KEY,
        keyring={"default": _OLD_KEY},
    )
    assert retired_only.signature_status == SignatureStatus.UNKNOWN_KEY


def test_rotation_on_empty_backend_returns_zero_counts(
    sqlite_backend: SQLiteBackend,
) -> None:
    report = rotate_secret(_OLD_KEY, _NEW_KEY, _NEW_KEY_ID, sqlite_backend)
    assert report.records_scanned == 0
    assert report.records_re_signed == 0
    assert report.records_legacy_verified == 0


def test_rotation_rejects_empty_key_material(sqlite_backend: SQLiteBackend) -> None:
    with pytest.raises(RotationError):
        rotate_secret(_OLD_KEY, "", _NEW_KEY_ID, sqlite_backend)
    with pytest.raises(RotationError):
        rotate_secret(_OLD_KEY, _NEW_KEY, "", sqlite_backend)


def test_rotation_re_sign_requires_update_capable_backend(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record(key_id="v1")
    signature = _write_signed(sqlite_backend, record, _OLD_KEY)
    readonly = _NoUpdateBackend(record, signature)

    # Scanning works; only the rewrite path is refused.
    report = rotate_secret(_OLD_KEY, _NEW_KEY, _NEW_KEY_ID, readonly)
    assert report.records_legacy_verified == 1

    with pytest.raises(RotationError, match="update_record"):
        rotate_secret(_OLD_KEY, _NEW_KEY, _NEW_KEY_ID, readonly, re_sign=True)


def test_rotation_fails_loudly_when_record_vanishes_mid_scan(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record(key_id="v1")
    signature = _write_signed(sqlite_backend, record, _OLD_KEY)
    vanishing = _VanishingBackend(record, signature)
    with pytest.raises(RotationError, match="disappeared"):
        rotate_secret(_OLD_KEY, _NEW_KEY, _NEW_KEY_ID, vanishing, re_sign=True)


def test_rotation_ignores_records_not_signed_under_old_key(
    sqlite_backend: SQLiteBackend,
) -> None:
    on_new = _make_record(key_id=_NEW_KEY_ID, response_text="c")
    _write_signed(sqlite_backend, on_new, _NEW_KEY)
    report = rotate_secret(
        _OLD_KEY, _NEW_KEY, _NEW_KEY_ID, sqlite_backend, re_sign=True
    )
    assert report.records_scanned == 1
    assert report.records_legacy_verified == 0
    assert report.records_re_signed == 0


def test_fingerprint_errors_share_a_base_class() -> None:
    # Callers can catch a single FingerprintError around verification flows.
    for exc_type in (RotationError, UnsupportedAlgorithmError):
        assert issubclass(exc_type, FingerprintError)
    assert issubclass(RecordNotFoundError, FingerprintError)


# ---------------------------------------------------------------------------
# Group 4 — canonicalization agility and 0.1.x byte-compatibility
# ---------------------------------------------------------------------------


def _v0_1_canonical_bytes(record: ProvenanceRecord) -> bytes:
    """Byte-exact replica of the 0.1.x canonicalization algorithm."""
    ts = record.timestamp
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    else:
        ts = ts.replace(tzinfo=timezone.utc)
    data = record.model_dump(mode="json")
    legacy_fields = (
        "content_id",
        "app_id",
        "feature_id",
        "user_id",
        "model",
        "prompt_hash",
        "response_hash",
        "prompt_tokens",
        "response_tokens",
        "latency_ms",
        "timestamp",
        "status",
        "pii_result",
        "policy_decision",
    )
    legacy = {field: data[field] for field in legacy_fields}
    legacy["timestamp"] = ts.isoformat()
    return json.dumps(
        legacy, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


def test_signature_is_byte_compatible_with_v0_1() -> None:
    # THE compatibility contract: a record carrying record_version=1 must sign
    # to exactly what 0.1.x code produced for the same content. (v2 records —
    # the new default — intentionally sign a different, envelope-binding
    # payload; see test_v2_signature_binds_envelope_fields.)
    record = _make_record(record_version=1)
    expected = hmac.new(
        _OLD_KEY.encode("utf-8"), _v0_1_canonical_bytes(record), hashlib.sha256
    ).hexdigest()
    assert sign_record(record, _OLD_KEY) == expected


def test_record_hash_is_byte_compatible_with_v0_1() -> None:
    record = _make_record()
    expected = hashlib.sha256(_v0_1_canonical_bytes(record)).hexdigest()
    assert record_hash(record) == expected


def test_envelope_metadata_does_not_shift_content_hash() -> None:
    # key_id/sig_algo live outside the canonical payload: relabeling the
    # envelope never moves the content hash that chain links are built from.
    r1 = _make_record(key_id="default", sig_algo="HMAC-SHA256")
    r2 = r1.model_copy(update={"key_id": "v9"})
    assert record_hash(r1) == record_hash(r2)


def test_unknown_record_version_fails_fast() -> None:
    # A newer producer's records must be refused, not mis-verified. record_hash
    # is intentionally version-agnostic (content fields only — that is what
    # keeps chain links stable across versions and envelope changes), so only
    # signing refuses an unknown version here; verification refuses via the
    # registry lookup in test_unknown_record_version_fails_during_verification.
    future = _make_record().model_copy(update={"record_version": 99})
    with pytest.raises(UnsupportedRecordVersionError):
        sign_record(future, _OLD_KEY)


def test_unknown_record_version_fails_during_verification(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record(key_id="v1")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    corrupted_view = record.model_copy(update={"record_version": 7})
    sqlite_backend.update_record(corrupted_view, sign_record(record, _OLD_KEY))
    with pytest.raises(UnsupportedRecordVersionError):
        verify_record(
            record.content_id,
            "response",
            sqlite_backend,
            _OLD_KEY,
            keyring={"v1": _OLD_KEY},
        )


def test_canonicalization_fails_fast_on_unserializable_value() -> None:
    # Bypassing validation (model_construct) with a non-JSON value must fail
    # loudly instead of being silently str-coerced.
    record = _make_record()
    broken = record.model_construct(latency_ms=object())  # type: ignore[arg-type]
    with pytest.raises(CanonicalizationError):
        sign_record(broken, _OLD_KEY)


# ---------------------------------------------------------------------------
# Group 5 — original_hash None sentinel
# ---------------------------------------------------------------------------


def test_original_hash_is_none_for_records_without_response_hash(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record(response_text=None)
    sqlite_backend.write(record, sign_record(record, _OLD_KEY))
    result = verify_record(record.content_id, "anything", sqlite_backend, _OLD_KEY)
    assert result.original_hash is None


def test_original_hash_equals_stored_response_hash(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record(response_text="text")
    _write_signed(sqlite_backend, record, _OLD_KEY)
    result = verify_record(record.content_id, "text", sqlite_backend, _OLD_KEY)
    assert result.original_hash == record.response_hash


# ---------------------------------------------------------------------------
# Group 6 — async verification twin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_record_async_verifies_signed_record() -> None:
    backend = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    try:
        await backend.create_tables()
        record = _make_record(key_id="v1", response_text="text")
        await backend.write(record, sign_record(record, _OLD_KEY))

        result = await verify_record_async(
            record.content_id,
            "text",
            backend,
            _OLD_KEY,
            keyring={"v1": _OLD_KEY},
        )
        assert result.verified is True
        assert result.key_id == "v1"
        assert result.signature_status == SignatureStatus.VALID
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_verify_record_async_raises_for_unknown_id() -> None:
    backend = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    try:
        await backend.create_tables()
        with pytest.raises(RecordNotFoundError):
            await verify_record_async(str(uuid.uuid4()), "text", backend, _OLD_KEY)
    finally:
        await backend.close()


@pytest.mark.asyncio
async def test_verify_record_async_distinguishes_unknown_key() -> None:
    backend = AsyncSQLiteBackend("sqlite+aiosqlite:///:memory:")
    try:
        await backend.create_tables()
        record = _make_record(key_id="v1", response_text="text")
        await backend.write(record, sign_record(record, _OLD_KEY))

        result = await verify_record_async(
            record.content_id,
            "text",
            backend,
            _NEW_KEY,
            keyring={_NEW_KEY_ID: _NEW_KEY},
        )
        assert result.signature_status == SignatureStatus.UNKNOWN_KEY
        assert result.verified is False
    finally:
        await backend.close()


# ---------------------------------------------------------------------------
# Group 7 — fingerprint-to-store import decoupling
# ---------------------------------------------------------------------------


def test_fingerprint_core_has_no_runtime_store_import() -> None:
    # The store import must be typing-only: no runtime binding for backend
    # types may leak into the fingerprint module namespace.
    import aistamp.fingerprint.core as fp_core

    assert "StoreBackend" not in vars(fp_core)
    assert "AsyncStoreBackend" not in vars(fp_core)


# ---------------------------------------------------------------------------
# Group 8 — v2 payload binding (audit P0-1) and key policy (audit P1-6)
# ---------------------------------------------------------------------------


def test_v2_chain_relink_fails_record_verification(
    sqlite_backend: SQLiteBackend,
) -> None:
    # Replicates the auditor's P0-1 PoV (art_y6PXlLmn): a DB-write attacker
    # rewrites prev_hash + scope_sequence on the last record of a chain while
    # KEEPING the old stored HMAC. v1 payloads let this pass as VALID (only
    # verify_chain noticed); the v2 payload binds the chain fields, so record
    # verification itself must fail.
    prev_hash: str | None = None
    last_content_id = ""
    for sequence in range(3):
        record = _make_record(scope_sequence=sequence, prev_hash=prev_hash)
        _write_signed(sqlite_backend, record, _OLD_KEY)
        prev_hash = record_hash(record)
        last_content_id = record.content_id

    fetched = sqlite_backend.get(last_content_id)
    assert fetched is not None
    last_record, stored_hmac = fetched
    assert stored_hmac is not None

    relabeled = last_record.model_copy(
        update={"prev_hash": "f" * 64, "scope_sequence": 9}
    )
    sqlite_backend.update_record(relabeled, stored_hmac)  # old HMAC kept

    result = verify_record(
        last_content_id,
        "response",
        sqlite_backend,
        _OLD_KEY,
        keyring={"default": _OLD_KEY},
    )
    assert result.verified is False
    assert result.hmac_valid is False
    assert result.signature_status == SignatureStatus.INVALID
    assert result.record_version == 2


def test_v2_signature_binds_envelope_fields(sqlite_backend: SQLiteBackend) -> None:
    # Re-labeling a signature as another key's (or swapping the algorithm tag)
    # with the SAME stored HMAC must fail: the v2 payload covers the envelope.
    record = _make_record()
    signature = sign_record(record, _OLD_KEY)
    _write_signed(sqlite_backend, record, _OLD_KEY)

    relabeled = record.model_copy(update={"key_id": "attacker-key"})
    sqlite_backend.update_record(relabeled, signature)

    result = verify_record(record.content_id, "response", sqlite_backend, _OLD_KEY)
    assert result.signature_status == SignatureStatus.INVALID
    assert result.verified is False


def test_record_hash_identical_across_v1_and_v2() -> None:
    # The chain-link hash must be version-agnostic so v1 and v2 records can
    # share one chain and rotation/re-signing never shifts prev_hash links.
    v1_record = _make_record(record_version=1)
    v2_record = v1_record.model_copy(update={"record_version": 2})
    assert record_hash(v1_record) == record_hash(v2_record)


def test_v1_canonical_payload_stays_byte_pinned() -> None:
    # Golden literals generated from the v1 canonicalizer before the v2 change
    # (fixed record: content_id 00000000-…-01, key "k" * 32). Any drift in the
    # 0.1.x field set or serialization — anywhere, ever — breaks this test.
    record = _make_record(
        content_id="00000000-0000-0000-0000-000000000001",
        record_version=1,
    )
    assert (
        sign_record(record, "k" * 32)
        == "5ab5061086056adf8dd7f95d46fb400d7ea1ff5fd33e295cd3a06941b3067832"
    )
    assert (
        record_hash(record)
        == "1dec377463fe76c5e44824c964d104f69efb521e932a82fce056c94237515570"
    )


def test_rotate_secret_rejects_short_new_key(sqlite_backend: SQLiteBackend) -> None:
    # Audit P1-6: rotating to a guessable key silently downgrades tamper
    # evidence. The floor mirrors Config's secret_key validation.
    record = _make_record()
    _write_signed(sqlite_backend, record, _OLD_KEY)

    with pytest.raises(RotationError, match="at least 32"):
        rotate_secret(_OLD_KEY, "abc", "weak", sqlite_backend)


def test_rotate_secret_key_length_floor_is_configurable(
    sqlite_backend: SQLiteBackend,
) -> None:
    record = _make_record()
    _write_signed(sqlite_backend, record, _OLD_KEY)

    with pytest.raises(RotationError, match="at least 4"):
        rotate_secret(_OLD_KEY, "abc", "id", sqlite_backend, min_key_length=4)
    report = rotate_secret(
        _OLD_KEY, "abcd", "ok", sqlite_backend, re_sign=True, min_key_length=4
    )
    assert report.records_re_signed == 1


def test_rotate_secret_rejects_blank_key_id(sqlite_backend: SQLiteBackend) -> None:
    with pytest.raises(RotationError, match="non-empty key identifier"):
        rotate_secret(_OLD_KEY, _NEW_KEY, "   ", sqlite_backend)


def test_config_rejects_short_secret_key() -> None:
    # Alignment guard for the rotation floor: Config must refuse weak keys too.
    from aistamp.config import Config

    with pytest.raises(ValueError, match="at least 32 characters"):
        Config(secret_key="too-short")


# ---------------------------------------------------------------------------
# Group 10 — update_record concurrency (security audit P1: locked RMW)
# ---------------------------------------------------------------------------


def test_update_record_serializes_concurrent_writers(tmp_path: Path) -> None:
    # update_record is a read-modify-write: without row locking, concurrent
    # updaters (rotation re-signing vs. a concurrent finalize) can interleave
    # their read and write phases. PostgreSQL serializes via SELECT ...
    # FOR UPDATE; SQLite via its database-level write lock. Either way,
    # every writer must land its full state and the surviving row must be
    # signature-consistent — never a torn or half-applied update.
    backend = SQLiteBackend(f"sqlite:///{tmp_path / 'concurrent.db'}")
    backend.create_tables()
    record = _make_record()
    backend.write(record, sign_record(record, _OLD_KEY))

    workers = 8
    rewrites = 5
    barrier = threading.Barrier(workers)
    errors: list[Exception] = []

    def _rewrite(worker: int) -> None:
        try:
            barrier.wait()
            for round_ in range(rewrites):
                fetched = backend.get(record.content_id)
                assert fetched is not None
                current, _ = fetched
                updated = current.model_copy(
                    update={"error_message": f"worker-{worker}-{round_}"}
                )
                backend.update_record(updated, sign_record(updated, _OLD_KEY))
        except Exception as exc:  # surfaced through the assertion below
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(_rewrite, range(workers)))

    assert errors == []
    stored = backend.get(record.content_id)
    assert stored is not None
    final, stored_signature = stored
    # The final row is exactly one worker's last write (round index 4), not a
    # blend of interleaved writes, and its signature matches its payload.
    assert final.error_message is not None
    assert final.error_message.startswith("worker-")
    assert final.error_message.endswith("-4")
    assert stored_signature == sign_record(final, _OLD_KEY)
    backend.close()
