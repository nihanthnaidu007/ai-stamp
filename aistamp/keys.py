"""Key management and rotation workflow for aistamp signatures.

The pinned rotation API the CLI track calls:

    rotate_secret(old_key, new_key, new_key_id, backend, re_sign=False)

Scanning works over the backend abstraction (query + per-record get), so any
sync StoreBackend works. Re-signing additionally requires a backend that
implements ``update_record`` (the SQLAlchemy sync backends do); records are
rewritten in place under the new key_id so rotation never duplicates or
orphans provenance rows.

Note on chain interplay: re-signing rewrites ``key_id`` on the record. The
v2 signature binds ``key_id`` (so re-signed records carry a fresh signature
under the new key — required for tamper evidence), but ``record_hash`` is
computed over content fields only, so ``prev_hash`` chain links are unchanged
by rotation. Verified by tests/test_hash_chaining.py.
"""

from __future__ import annotations

import hmac as hmac_lib
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from aistamp.fingerprint.core import FingerprintError, sign_record
from aistamp.models import AuditReport, QueryFilters

if TYPE_CHECKING:
    from aistamp.store.backend import StoreBackend


class RotationError(FingerprintError):
    """Raised when a key rotation cannot be completed."""


class RotationReport(BaseModel):
    """Outcome counts from a rotate_secret() run."""

    model_config = ConfigDict(frozen=True)

    records_scanned: int
    records_re_signed: int
    # Records confirmed to be signed under old_key. With re_sign=True these
    # were all rewritten under new_key_id (so the two counts are equal);
    # with re_sign=False they remain on the retired key and verification
    # needs the old key in the keyring.
    records_legacy_verified: int


_ROTATION_PAGE_SIZE = 500

# Aligned with Config's secret_key validation: a signing key shorter than
# this is guessable, and rotating to one would silently downgrade tamper
# evidence for everything re-signed afterwards (audit P1-6).
MIN_SIGNING_KEY_LENGTH = 32


def rotate_secret(
    old_key: str,
    new_key: str,
    new_key_id: str,
    backend: StoreBackend,
    re_sign: bool = False,
    *,
    min_key_length: int = MIN_SIGNING_KEY_LENGTH,
) -> RotationReport:
    """Rotate the signing key across a backend.

    Scans every record in the backend, identifies records whose stored
    signature verifies under ``old_key``, and (with ``re_sign=True``) rewrites
    them in place under ``new_key_id`` signed with ``new_key``.

    With ``re_sign=False`` no records are modified — historical verification
    keeps working by adding the old key to the verification keyring as a
    retired key.

    Records signed under other keys, unsigned records, and records whose
    signature does not verify under ``old_key`` are counted as scanned and
    left untouched. An unsupported ``sig_algo`` fails the whole run loudly
    rather than mis-verifying a record.

    ``new_key`` must be at least ``min_key_length`` characters (default 32,
    aligned with ``Config``'s ``secret_key`` rule) and ``new_key_id`` must be
    a non-empty identifier — rotating to a guessable key or colliding id is
    rejected up front instead of silently weakening evidence.
    """
    if not new_key:
        raise RotationError("new_key must be a non-empty secret.")
    if len(new_key) < min_key_length:
        raise RotationError(
            f"new_key must be at least {min_key_length} characters "
            f"(got {len(new_key)}); a shorter signing key would silently "
            "downgrade tamper evidence for re-signed records."
        )
    if not new_key_id or not new_key_id.strip():
        raise RotationError("new_key_id must be a non-empty key identifier.")

    scanned = 0
    re_signed = 0
    legacy_verified = 0

    offset = 0
    while True:
        report: AuditReport = backend.query(
            QueryFilters(limit=_ROTATION_PAGE_SIZE, offset=offset)
        )
        page = report.records
        for record in page:
            scanned += 1
            fetched = backend.get(record.content_id)
            if fetched is None:
                raise RotationError(
                    f"Record {record.content_id!r} disappeared during the "
                    "rotation scan; re-run the rotation."
                )
            _, stored_hmac = fetched
            if stored_hmac is None:
                continue  # unsigned record: nothing to rotate

            expected_old = sign_record(record, old_key)
            if not hmac_lib.compare_digest(stored_hmac, expected_old):
                continue  # signed under a different key; leave untouched

            legacy_verified += 1
            if re_sign:
                re_signed_record = record.model_copy(update={"key_id": new_key_id})
                try:
                    backend.update_record(
                        re_signed_record, sign_record(re_signed_record, new_key)
                    )
                except (ValueError, NotImplementedError) as exc:
                    raise RotationError(
                        f"Failed to re-sign record {record.content_id!r}: {exc}"
                    ) from exc
                re_signed += 1

        offset += len(page)
        if not page or offset >= report.total_count:
            break

    return RotationReport(
        records_scanned=scanned,
        records_re_signed=re_signed,
        records_legacy_verified=legacy_verified,
    )
