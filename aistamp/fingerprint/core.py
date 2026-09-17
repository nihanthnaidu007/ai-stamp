from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import uuid
from datetime import timezone

from pydantic import SecretStr

from aistamp.models import ProvenanceRecord, VerificationResult
from aistamp.store.backend import StoreBackend


class RecordNotFoundError(Exception):
    """Raised when verify_record cannot find a record for the given content_id."""

    def __init__(self, content_id: str) -> None:
        super().__init__(f"No provenance record found for content_id: {content_id!r}")
        self.content_id = content_id


def generate_content_id() -> str:
    return str(uuid.uuid4())


def hash_content(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_bytes(record: ProvenanceRecord) -> bytes:
    # Normalize the timestamp to UTC ISO so signatures survive a SQLite roundtrip
    # (SQLAlchemy's DateTime column strips tzinfo; the backend re-attaches UTC on read).
    ts = record.timestamp
    if ts.tzinfo is not None:
        ts = ts.astimezone(timezone.utc)
    else:
        ts = ts.replace(tzinfo=timezone.utc)
    data = record.model_dump(mode="json")
    data["timestamp"] = ts.isoformat()
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return canonical.encode("utf-8")


def sign_record(record: ProvenanceRecord, secret_key: str | SecretStr) -> str:
    key_bytes = _key_bytes(secret_key)
    msg_bytes = _canonical_bytes(record)
    return hmac_lib.new(key_bytes, msg_bytes, hashlib.sha256).hexdigest()


def _key_bytes(secret_key: str | SecretStr) -> bytes:
    # Config.secret_key became SecretStr in v0.2; accepting either keeps the
    # signing bytes identical for plain-string callers.
    if isinstance(secret_key, SecretStr):
        return secret_key.get_secret_value().encode("utf-8")
    return secret_key.encode("utf-8")


def verify_record(
    content_id: str,
    current_text: str,
    backend: StoreBackend,
    secret_key: str | SecretStr,
) -> VerificationResult:
    result = backend.get(content_id)
    if result is None:
        raise RecordNotFoundError(content_id)

    record, stored_hmac = result
    current_hash = hash_content(current_text)

    if record.response_hash is None:
        hash_match = False
    else:
        hash_match = current_hash == record.response_hash

    if stored_hmac is None:
        hmac_valid = False
    else:
        expected_hmac = sign_record(record, secret_key)
        hmac_valid = hmac_lib.compare_digest(stored_hmac, expected_hmac)

    drift_detected = not hash_match
    verified = hash_match and hmac_valid

    return VerificationResult(
        content_id=content_id,
        verified=verified,
        hash_match=hash_match,
        hmac_valid=hmac_valid,
        drift_detected=drift_detected,
        original_hash=record.response_hash or "",
        current_hash=current_hash,
    )
