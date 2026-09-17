"""Content fingerprinting, signing, and verification with tamper evidence.

v0.2 design notes (tamper-evidence track):

- **Signature envelope.** Records carry ``key_id`` and ``sig_algo`` next to the
  digest so a verifier knows which key and algorithm produced a signature.
  For v2 records the envelope fields are *inside* the signed payload, so an
  attacker cannot relabel a signature as someone else's. They are still kept
  out of ``record_hash`` (the chain-link value), so key rotation/re-signing
  never shifts ``prev_hash`` links.
- **Canonicalization agility.** ``record_version`` selects the canonicalizer
  (see ``_CANONICALIZERS``). v1 pins the exact 0.1.x field set so signatures
  computed by 0.1.x still verify byte-for-byte. v2 additionally binds the
  envelope and chain fields (``key_id``, ``sig_algo``, ``record_version``,
  ``prev_hash``, ``scope_sequence``) into the HMAC, so rewriting chain links
  without the key fails record-level verification — the P0-1 audit fix.
  Future model changes ship a new record_version + canonicalizer instead of
  silently invalidating history.
- **Keyring verification.** ``verify_record`` accepts an optional keyring
  mapping key_id -> key. When given, the keyring is authoritative: the key used
  is ``keyring[record.key_id]``. When absent, the bare ``secret_key`` argument
  is used, exactly as in 0.1.x.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import SecretStr
from pydantic_core import PydanticSerializationError

from aistamp.models import (
    ChainIssue,
    ChainIssueKind,
    ChainLink,
    ChainVerificationResult,
    ProvenanceRecord,
    PurgeAnchor,
    QueryFilters,
    SignatureStatus,
    VerificationResult,
)

if TYPE_CHECKING:
    from aistamp.store.async_backend import AsyncStoreBackend
    from aistamp.store.backend import StoreBackend


class FingerprintError(Exception):
    """Base class for fingerprint/tamper-evidence errors."""


class RecordNotFoundError(FingerprintError):
    """Raised when verify_record cannot find a record for the given content_id."""

    def __init__(self, content_id: str) -> None:
        super().__init__(f"No provenance record found for content_id: {content_id!r}")
        self.content_id = content_id


class CanonicalizationError(FingerprintError):
    """Raised when a record cannot be serialized to canonical signing form."""


class UnsupportedRecordVersionError(FingerprintError):
    """Raised when no canonicalizer is registered for a record's record_version."""

    def __init__(self, record_version: int) -> None:
        super().__init__(
            f"No canonicalization registered for record_version {record_version!r}; "
            "this verifier is too old for the record, or the record is corrupt."
        )
        self.record_version = record_version


class UnsupportedAlgorithmError(FingerprintError):
    """Raised when a record's sig_algo is not implemented by this verifier."""

    def __init__(self, sig_algo: str) -> None:
        super().__init__(f"Unsupported signature algorithm: {sig_algo!r}")
        self.sig_algo = sig_algo


# ---------------------------------------------------------------------------
# Canonicalization
# ---------------------------------------------------------------------------

# v1 canonical payload = exactly the 0.1.x field set. Byte-identity with 0.1.x
# signatures is what keeps every pre-0.2 record verifiable after the upgrade.
# Newer envelope/chain fields are bound by future record_versions, not by v1.
_V1_CANONICAL_FIELDS: tuple[str, ...] = (
    "app_id",
    "content_id",
    "feature_id",
    "latency_ms",
    "model",
    "pii_result",
    "policy_decision",
    "prompt_hash",
    "prompt_tokens",
    "response_hash",
    "response_tokens",
    "status",
    "timestamp",
    "user_id",
)


def _utc_timestamp(record: ProvenanceRecord) -> str:
    # Normalize the timestamp to UTC ISO so signatures survive a SQLite
    # roundtrip (SQLAlchemy's DateTime column strips tzinfo; the backend
    # re-attaches UTC on read).
    ts = record.timestamp
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    else:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts.isoformat()


def _canonical_bytes_v2(record: ProvenanceRecord) -> bytes:
    """v2 payload: v1 content fields PLUS envelope and chain fields.

    Binding ``key_id``/``sig_algo``/``record_version``/``prev_hash``/
    ``scope_sequence`` into the HMAC means a DB-write attacker who relinks
    the chain (or re-labels a signature) cannot present a valid record —
    audit P0-1. ``record_version`` itself is signed (version-tag-in-payload
    per the audit recommendation) even though it also gates interpretation.
    """
    try:
        data = record.model_dump(mode="json")
    except PydanticSerializationError as exc:
        raise CanonicalizationError(
            f"record contains a value that cannot be serialized: {exc}"
        ) from exc
    payload = {field: data[field] for field in _V1_CANONICAL_FIELDS}
    payload.update(
        {
            "key_id": data["key_id"],
            "sig_algo": data["sig_algo"],
            "record_version": data["record_version"],
            "prev_hash": data["prev_hash"],
            "scope_sequence": data["scope_sequence"],
        }
    )
    payload["timestamp"] = _utc_timestamp(record)
    try:
        # No ``default=`` hook: unknown types must fail fast, not be str-coerced.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise CanonicalizationError(
            f"record contains a value that is not JSON-serializable: {exc}"
        ) from exc
    return canonical.encode("utf-8")


def _canonical_bytes_v1(record: ProvenanceRecord) -> bytes:
    try:
        data = record.model_dump(mode="json")
    except PydanticSerializationError as exc:
        raise CanonicalizationError(
            f"record contains a value that cannot be serialized: {exc}"
        ) from exc
    payload = {field: data[field] for field in _V1_CANONICAL_FIELDS}
    payload["timestamp"] = _utc_timestamp(record)
    try:
        # No ``default=`` hook: unknown types must fail fast, not be str-coerced.
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except TypeError as exc:
        raise CanonicalizationError(
            f"record contains a value that is not JSON-serializable: {exc}"
        ) from exc
    return canonical.encode("utf-8")


# Registry mapping record_version -> canonicalizer. New versions register here;
# verifiers that lack a version fail fast instead of mis-verifying.
_CANONICALIZERS: dict[int, Callable[[ProvenanceRecord], bytes]] = {
    1: _canonical_bytes_v1,
    2: _canonical_bytes_v2,
}


def _canonical_bytes(record: ProvenanceRecord) -> bytes:
    canonicalizer = _CANONICALIZERS.get(record.record_version)
    if canonicalizer is None:
        raise UnsupportedRecordVersionError(record.record_version)
    return canonicalizer(record)


# ---------------------------------------------------------------------------
# Hashing and signing
# ---------------------------------------------------------------------------


def generate_content_id() -> str:
    return str(uuid.uuid4())


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_SIGNATURE_ALGORITHMS: dict[str, Callable[[bytes, bytes], str]] = {
    "HMAC-SHA256": lambda key, msg: hmac_lib.new(key, msg, hashlib.sha256).hexdigest(),
}


def _key_bytes(secret_key: str | SecretStr) -> bytes:
    # Config.secret_key became SecretStr in v0.2; accepting either keeps the
    # signing bytes identical for plain-string callers.
    if isinstance(secret_key, SecretStr):
        return secret_key.get_secret_value().encode("utf-8")
    return secret_key.encode("utf-8")


def _compute_signature(record: ProvenanceRecord, secret_key: str | SecretStr) -> str:
    signer = _SIGNATURE_ALGORITHMS.get(record.sig_algo)
    if signer is None:
        raise UnsupportedAlgorithmError(record.sig_algo)
    return signer(_key_bytes(secret_key), _canonical_bytes(record))


def sign_record(record: ProvenanceRecord, secret_key: str | SecretStr) -> str:
    """Sign a record with HMAC under ``secret_key``.

    The canonical payload is selected by the record's ``record_version``: v2
    records (the default) bind the envelope and chain fields into the HMAC,
    so chain relinking or signature re-labeling fails verification. The
    algorithm comes from the record's ``sig_algo``; the key identity from its
    ``key_id``. Changing ``key_id`` (e.g. during rotation/re-sign) changes the
    v2 signature — that is the point — but never ``record_hash``, so
    ``prev_hash`` links are unaffected.
    """
    return _compute_signature(record, secret_key)


def record_hash(record: ProvenanceRecord) -> str:
    """Content hash of a record: SHA-256 over its *content* canonical form.

    Uses the v1 content field set for every record_version, so the value is
    identical for v1 and v2 records with equal content and is stable across
    ``key_id``/``sig_algo`` changes — safe to use as a ``prev_hash`` chain
    link value that survives key rotation/re-signing. (The v2 *signature*
    binds the envelope; the chain-link hash must not.)
    """
    return hashlib.sha256(_canonical_bytes_v1(record)).hexdigest()


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _evaluate_verification(
    content_id: str,
    record: ProvenanceRecord,
    stored_hmac: str | None,
    current_text: str,
    secret_key: str | SecretStr,
    keyring: Mapping[str, str] | None,
) -> VerificationResult:
    """Pure verification core shared by the sync and async twins."""
    current_hash = hash_content(current_text)
    stored_hash = record.response_hash
    hash_match = stored_hash is not None and current_hash == stored_hash

    if stored_hmac is None:
        signature_status = SignatureStatus.UNSIGNED
        hmac_valid = False
    else:
        # The keyring, when provided, is authoritative: the key is looked up by
        # the record's own key_id. Without a keyring, fall back to the bare
        # secret_key (0.1.x behavior for single-key deployments).
        key: str | SecretStr | None = (
            keyring.get(record.key_id) if keyring is not None else secret_key
        )
        if key is None:
            signature_status = SignatureStatus.UNKNOWN_KEY
            hmac_valid = False
        else:
            expected_hmac = _compute_signature(record, key)
            hmac_valid = hmac_lib.compare_digest(stored_hmac, expected_hmac)
            signature_status = (
                SignatureStatus.VALID if hmac_valid else SignatureStatus.INVALID
            )

    drift_detected = not hash_match
    verified = hash_match and hmac_valid

    return VerificationResult(
        content_id=content_id,
        verified=verified,
        hash_match=hash_match,
        hmac_valid=hmac_valid,
        drift_detected=drift_detected,
        original_hash=record.response_hash,
        current_hash=current_hash,
        key_id=record.key_id,
        signature_status=signature_status,
        record_version=record.record_version,
    )


def verify_record(
    content_id: str,
    current_text: str,
    backend: StoreBackend,
    secret_key: str | SecretStr,
    keyring: Mapping[str, str] | None = None,
) -> VerificationResult:
    """Verify a record's content hash and signature.

    ``keyring`` optionally maps key_id -> secret key (active and retired keys).
    When given, it is authoritative; a record whose key_id is absent from the
    keyring yields ``SignatureStatus.UNKNOWN_KEY``. When omitted, the bare
    ``secret_key`` is used as in 0.1.x.
    """
    result = backend.get(content_id)
    if result is None:
        raise RecordNotFoundError(content_id)

    record, stored_hmac = result
    return _evaluate_verification(
        content_id, record, stored_hmac, current_text, secret_key, keyring
    )


async def verify_record_async(
    content_id: str,
    current_text: str,
    backend: AsyncStoreBackend,
    secret_key: str | SecretStr,
    keyring: Mapping[str, str] | None = None,
) -> VerificationResult:
    """Async twin of verify_record for AsyncStoreBackend adopters."""
    result = await backend.get(content_id)
    if result is None:
        raise RecordNotFoundError(content_id)

    record, stored_hmac = result
    return _evaluate_verification(
        content_id, record, stored_hmac, current_text, secret_key, keyring
    )


# ---------------------------------------------------------------------------
# Hash chaining (optional mode)
# ---------------------------------------------------------------------------

_CHAIN_PAGE_SIZE = 500


def _fetch_scope_records(
    backend: StoreBackend, app_id: str, feature_id: str
) -> list[ProvenanceRecord]:
    """All records in an (app_id, feature_id) scope, across query pages."""
    records: list[ProvenanceRecord] = []
    offset = 0
    while True:
        report = backend.query(
            QueryFilters(
                app_id=app_id,
                feature_id=feature_id,
                limit=_CHAIN_PAGE_SIZE,
                offset=offset,
            )
        )
        records.extend(report.records)
        offset += len(report.records)
        if not report.records or offset >= report.total_count:
            return records



# ---------------------------------------------------------------------------
# Purge-anchor signatures (security audit follow-up: signed retention journal)
# ---------------------------------------------------------------------------

_PURGE_ANCHOR_CONTEXT = b"aistamp.purge-anchor.v1"


def _purge_anchor_bytes(anchor: PurgeAnchor) -> bytes:
    """Canonical byte form of a purge anchor, for HMAC signing.

    The anchor ``id`` is excluded: it is assigned by the database on flush,
    so a pre-persist anchor and its persisted row must produce identical
    bytes. Datetimes are normalized to UTC so signatures created before
    persistence verify against anchors read back from SQLite or PostgreSQL.
    """
    parts = [
        _PURGE_ANCHOR_CONTEXT,
        _utc_datetime_bytes(anchor.purged_before),
        str(anchor.purged_count).encode(),
        json.dumps(
            anchor.deleted_prev_hashes, separators=(",", ":"), sort_keys=True
        ).encode(),
        _utc_datetime_bytes(anchor.anchor_created_at),
    ]
    return b"\x00".join(parts)


def _utc_datetime_bytes(value: datetime) -> bytes:
    """UTC ISO-8601 bytes for a naive-or-aware datetime."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().encode()


def sign_purge_anchor(anchor: PurgeAnchor, secret_key: str | SecretStr) -> str:
    """HMAC-sign a purge anchor so journal rows cannot be forged after the fact.

    Same key material as record signatures: the anchor journal is only as
    trustworthy as the records it vouches for. Callers pass this as
    ``backend.purge(..., anchor_signer=...)`` so the signature is written in
    the same transaction as the deletes.
    """
    key = (
        secret_key.get_secret_value()
        if isinstance(secret_key, SecretStr)
        else secret_key
    )
    return hmac_lib.new(
        key.encode(), _purge_anchor_bytes(anchor), hashlib.sha256
    ).hexdigest()


def verify_purge_anchor(
    anchor: PurgeAnchor,
    secret_key: str | SecretStr,
    *,
    anchor_signer: Callable[[PurgeAnchor], str] | None = None,
) -> bool:
    """True when ``anchor.signature`` is a valid HMAC over the anchor contents.

    Anchors written before signing was enabled carry ``signature=None``;
    they verify as ``False`` (an unsigned voucher proves nothing) unless an
    ``anchor_signer`` callback is supplied to reproduce the expected
    signature for signers that cannot persist it themselves.
    """
    if anchor_signer is not None:
        expected = anchor_signer(anchor)
    elif anchor.signature is None:
        return False
    else:
        key = (
            secret_key.get_secret_value()
            if isinstance(secret_key, SecretStr)
            else secret_key
        )
        expected = hmac_lib.new(
            key.encode(), _purge_anchor_bytes(anchor), hashlib.sha256
        ).hexdigest()
    if anchor.signature is None:
        return False
    return hmac_lib.compare_digest(anchor.signature, expected)


def _purged_back_references(anchors: Sequence[PurgeAnchor]) -> set[str | None]:
    """prev_hash values journaled by purges, including None for purged heads."""
    return {h for anchor in anchors for h in anchor.deleted_prev_hashes}


def _gap_is_anchored(
    anchors: Sequence[PurgeAnchor],
    *,
    surviving_predecessor: ProvenanceRecord | None,
) -> bool:
    """Whether the chain gap in front of the next surviving record is explained
    by a retention purge (security audit P1-5).

    ``purge()`` journals the ``prev_hash`` column of every row it deletes. For
    a purged stretch ``r_{P+1}..r_{S-1}`` between survivors ``r_P`` and ``r_S``,
    those journaled values are ``record_hash(r_P) .. record_hash(r_{S-2})`` —
    the first purged record's back-pointer names the surviving predecessor.
    ``record_hash`` binds content_id, so a match proves r_P's immediate
    successor was legitimately purged; forging it would require a journal row,
    which ``purge()`` only writes in the same transaction as real deletes.
    A purged chain *head* (``r_0``) is journaled as ``None`` (no predecessor).
    """
    journal = _purged_back_references(anchors)
    if surviving_predecessor is None:
        return None in journal
    return record_hash(surviving_predecessor) in journal


def build_chain_link(backend: StoreBackend, scope: tuple[str, str]) -> ChainLink:
    """Produce the link data for the next record in a scope's chain.

    Writers stamp ``scope_sequence`` and ``prev_hash`` from the returned
    ChainLink onto the new record before signing it.
    """
    app_id, feature_id = scope
    sequenced = [
        (r.scope_sequence, r)
        for r in _fetch_scope_records(backend, app_id, feature_id)
        if r.scope_sequence is not None
    ]
    if not sequenced:
        return ChainLink(scope_sequence=0, prev_hash=None)
    sequenced.sort(key=lambda pair: pair[0])
    last_sequence, last_record = sequenced[-1]
    return ChainLink(
        scope_sequence=last_sequence + 1, prev_hash=record_hash(last_record)
    )


def verify_chain(
    backend: StoreBackend, scope: tuple[str, str]
) -> ChainVerificationResult:
    """Structurally verify the per-scope hash chain.

    Detects what per-record signatures cannot: missing records (sequence gaps,
    dangling predecessors) and reordered/inconsistent links (including altered
    records whose neighbors' ``prev_hash`` no longer matches). Records without
    ``scope_sequence`` are not part of any chain and are only counted.

    Retention purges are part of the protocol, not tampering (security audit
    P1-5): a break is *anchored* when the purge journal holds the back-pointer
    of the first purged record — ``record_hash`` of the surviving predecessor,
    or ``None`` when the purged stretch includes the chain head. Anchored gaps
    produce no issues and are counted in ``anchored_gaps``; unexplained breaks
    still report MISSING/REORDERED. The journal stores back-pointers, not the
    purged rows' own hashes, so an anchored gap cannot re-verify the survivor's
    ``prev_hash`` — suppressing that check is by design, not an oversight.
    """
    app_id, feature_id = scope
    records = _fetch_scope_records(backend, app_id, feature_id)
    unchained = sum(1 for r in records if r.scope_sequence is None)
    sequenced = sorted(
        ((r.scope_sequence, r) for r in records if r.scope_sequence is not None),
        key=lambda pair: pair[0],
    )

    issues: list[ChainIssue] = []
    anchored_gaps = 0
    # Fetched lazily: the anchor query only runs when a purge-shaped break
    # actually appears in the walk.
    purge_anchors: list[PurgeAnchor] | None = None

    def _anchors() -> list[PurgeAnchor]:
        nonlocal purge_anchors
        if purge_anchors is None:
            purge_anchors = backend.list_purge_anchors()
        return purge_anchors

    # (scope_sequence, record) of the previous chained record, or None at the
    # head. One variable so the not-None branch narrows both at once.
    previous: tuple[int, ProvenanceRecord] | None = None

    for sequence, rec in sequenced:
        if previous is None:
            # A purge can only remove a head segment (r_0 .. r_{N-1}) when the
            # journal holds a None back-pointer AND the survivor still carries
            # a real pointer to the purged r_{N-1}. A record at sequence 0
            # with a non-None prev_hash, or a head-gap survivor with a null
            # pointer, is not purge-shaped and stays flagged.
            head_gap_anchored = (
                sequence != 0
                and rec.prev_hash is not None
                and _gap_is_anchored(_anchors(), surviving_predecessor=None)
            )
            if head_gap_anchored:
                anchored_gaps += 1
            else:
                if sequence != 0:
                    issues.append(
                        ChainIssue(
                            kind=ChainIssueKind.MISSING,
                            sequence=sequence,
                            content_id=rec.content_id,
                            detail=(
                                f"chain starts at scope_sequence {sequence}, "
                                "expected 0; earlier records are missing from "
                                "this scope"
                            ),
                        )
                    )
                if rec.prev_hash is not None:
                    issues.append(
                        ChainIssue(
                            kind=ChainIssueKind.MISSING,
                            sequence=sequence,
                            content_id=rec.content_id,
                            detail=(
                                "first chained record references prev_hash "
                                f"{rec.prev_hash[:12]}…, but no earlier record exists "
                                "in this scope"
                            ),
                        )
                    )
        else:
            prev_sequence, prev_record = previous
            if sequence == prev_sequence:
                issues.append(
                    ChainIssue(
                        kind=ChainIssueKind.REORDERED,
                        sequence=sequence,
                        content_id=rec.content_id,
                        detail=(
                            f"scope_sequence {sequence} is claimed by multiple records "
                            f"(also by {prev_record.content_id})"
                        ),
                    )
                )
            else:
                has_gap = sequence != prev_sequence + 1
                gap_anchored = has_gap and _gap_is_anchored(
                    _anchors(), surviving_predecessor=prev_record
                )
                if has_gap:
                    if gap_anchored:
                        anchored_gaps += 1
                    else:
                        issues.append(
                            ChainIssue(
                                kind=ChainIssueKind.MISSING,
                                sequence=sequence,
                                content_id=rec.content_id,
                                detail=(
                                    f"scope_sequence gap: expected "
                                    f"{prev_sequence + 1}, found {sequence}"
                                ),
                            )
                        )
                expected_prev = record_hash(prev_record)
                # Constant-time comparison (audit hardening): a chain link
                # names attacker-influenced content, so equality against the
                # recomputed hash must not leak match progress by timing.
                links_match = rec.prev_hash is not None and hmac_lib.compare_digest(
                    rec.prev_hash, expected_prev
                )
                if not links_match and not gap_anchored:
                    # When the gap is anchored, the true predecessor was
                    # purged and the survivor's pointer names purged content
                    # whose hash is unrecoverable — the mismatch is expected.
                    found_prev = (
                        rec.prev_hash[:12] if rec.prev_hash is not None else "null"
                    )
                    issues.append(
                        ChainIssue(
                            kind=ChainIssueKind.REORDERED,
                            sequence=sequence,
                            content_id=rec.content_id,
                            detail=(
                                f"prev_hash {found_prev}… does not match the hash "
                                f"of the preceding record ({expected_prev[:12]}…); "
                                "the record was altered or reordered"
                            ),
                        )
                    )
        previous = (sequence, rec)

    return ChainVerificationResult(
        app_id=app_id,
        feature_id=feature_id,
        valid=not issues,
        records_checked=len(sequenced),
        unchained_records=unchained,
        issues=issues,
        anchored_gaps=anchored_gaps,
    )
