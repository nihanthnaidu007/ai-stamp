from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone

import pytest

from aistamp.config import Config
from aistamp.fingerprint import (
    RecordNotFoundError,
    generate_content_id,
    hash_content,
    sign_record,
    verify_record,
)
from aistamp.models import ProvenanceRecord, RecordStatus, VerificationResult
from aistamp.store import SQLiteBackend

# ---------------------------------------------------------------------------
# Group 1 — generate_content_id tests
# ---------------------------------------------------------------------------


def test_generate_content_id_returns_string() -> None:
    # generate_content_id() must return a str, not bytes or UUID object.
    assert isinstance(generate_content_id(), str)


def test_generate_content_id_is_valid_uuid() -> None:
    # The returned string must parse as a valid UUID without raising.
    uuid.UUID(generate_content_id())


def test_generate_content_id_is_unique() -> None:
    # Two successive calls must not return the same value.
    assert generate_content_id() != generate_content_id()


# ---------------------------------------------------------------------------
# Group 2 — hash_content tests
# ---------------------------------------------------------------------------


def test_hash_content_returns_64_char_hex_string() -> None:
    # SHA256 hex digest is always 64 lowercase hex characters.
    h = hash_content("hello")
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_hash_content_is_deterministic() -> None:
    # Same input must always produce the same hash.
    assert hash_content("test") == hash_content("test")


def test_hash_content_differs_for_different_inputs() -> None:
    # Even a one-character change must produce a completely different hash.
    assert hash_content("hello") != hash_content("hello!")


def test_hash_content_handles_empty_string() -> None:
    # Empty string is valid input. Must not raise.
    result = hash_content("")
    assert len(result) == 64


def test_hash_content_handles_unicode() -> None:
    # Non-ASCII text must be hashed without error.
    result = hash_content("こんにちは")
    assert len(result) == 64


def test_hash_content_known_value() -> None:
    # Verify against a known SHA256 value to confirm correct algorithm.
    expected = hashlib.sha256(b"abc").hexdigest()
    assert hash_content("abc") == expected


# ---------------------------------------------------------------------------
# Group 3 — sign_record tests
# ---------------------------------------------------------------------------


def _build_record(
    *,
    content_id: str | None = None,
    response_hash: str | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=content_id or generate_content_id(),
        app_id="app",
        feature_id="feat",
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("prompt"),
        response_hash=(
            response_hash if response_hash is not None else hash_content("response")
        ),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )


def test_sign_record_returns_64_char_hex_string() -> None:
    # HMAC-SHA256 hex digest is always 64 lowercase hex characters.
    sig = sign_record(_build_record(), "a" * 32)
    assert len(sig) == 64
    assert all(c in "0123456789abcdef" for c in sig)


def test_sign_record_is_deterministic() -> None:
    # Same record and same key always produce the same signature.
    record = _build_record()
    assert sign_record(record, "k" * 32) == sign_record(record, "k" * 32)


def test_sign_record_differs_with_different_key() -> None:
    # Different secret keys must produce different signatures.
    record = _build_record()
    assert sign_record(record, "key_a" * 8) != sign_record(record, "key_b" * 8)


def test_sign_record_differs_with_different_record() -> None:
    # Changing any field on the record must produce a different signature.
    r1 = _build_record(content_id="00000000-0000-0000-0000-000000000001")
    r2 = _build_record(content_id="00000000-0000-0000-0000-000000000002")
    assert sign_record(r1, "k" * 32) != sign_record(r2, "k" * 32)


def test_sign_record_sensitive_to_response_hash() -> None:
    # Specifically test that response_hash affects the signature.
    cid = generate_content_id()
    r1 = _build_record(content_id=cid, response_hash=hash_content("a"))
    r2 = _build_record(content_id=cid, response_hash=hash_content("b"))
    assert sign_record(r1, "k" * 32) != sign_record(r2, "k" * 32)


# ---------------------------------------------------------------------------
# Group 4 — verify_record tests (full pipeline)
# ---------------------------------------------------------------------------


def _write_signed(
    backend: SQLiteBackend,
    text: str,
    secret: str,
) -> ProvenanceRecord:
    record = _build_record(response_hash=hash_content(text))
    backend.write(record, sign_record(record, secret))
    return record


def test_verify_record_returns_verification_result(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # verify_record() must return a VerificationResult instance.
    record = _write_signed(sqlite_backend, "hello", sample_config.secret_key)
    result = verify_record(
        record.content_id, "hello", sqlite_backend, sample_config.secret_key
    )
    assert isinstance(result, VerificationResult)


def test_verify_record_verified_true_for_original_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # Verifying against the original response text must return verified=True.
    text = "original response text"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    result = verify_record(
        record.content_id, text, sqlite_backend, sample_config.secret_key
    )
    assert result.verified is True


def test_verify_record_hash_match_true_for_original_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # hash_match must be True when current text hashes to stored response_hash.
    text = "original response text"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    result = verify_record(
        record.content_id, text, sqlite_backend, sample_config.secret_key
    )
    assert result.hash_match is True


def test_verify_record_hmac_valid_true_for_correct_key(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # hmac_valid must be True when verified with the same key used for signing.
    text = "original response text"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    result = verify_record(
        record.content_id, text, sqlite_backend, sample_config.secret_key
    )
    assert result.hmac_valid is True


def test_verify_record_drift_detected_false_for_original_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # drift_detected must be False when content matches.
    text = "original response text"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    result = verify_record(
        record.content_id, text, sqlite_backend, sample_config.secret_key
    )
    assert result.drift_detected is False


def test_verify_record_hash_match_false_for_modified_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # Verifying against modified text must return hash_match=False.
    record = _write_signed(sqlite_backend, "original", sample_config.secret_key)
    result = verify_record(
        record.content_id, "modified", sqlite_backend, sample_config.secret_key
    )
    assert result.hash_match is False


def test_verify_record_drift_detected_true_for_modified_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # drift_detected must be True when hash_match is False.
    record = _write_signed(sqlite_backend, "original", sample_config.secret_key)
    result = verify_record(
        record.content_id, "modified", sqlite_backend, sample_config.secret_key
    )
    assert result.drift_detected is True


def test_verify_record_verified_false_for_modified_text(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # verified must be False when content has drifted.
    record = _write_signed(sqlite_backend, "original", sample_config.secret_key)
    result = verify_record(
        record.content_id, "modified", sqlite_backend, sample_config.secret_key
    )
    assert result.verified is False


def test_verify_record_hmac_valid_false_for_wrong_key(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # Verifying with a different secret key must return hmac_valid=False.
    text = "original"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    wrong_key = "z" * 32
    result = verify_record(record.content_id, text, sqlite_backend, wrong_key)
    assert result.hmac_valid is False


def test_verify_record_verified_false_for_wrong_key(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # verified must be False when HMAC check fails.
    text = "original"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    wrong_key = "z" * 32
    result = verify_record(record.content_id, text, sqlite_backend, wrong_key)
    assert result.verified is False


def test_verify_record_hmac_valid_false_when_no_hmac_stored(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # If record was written with hmac=None, hmac_valid must be False.
    record = _build_record(response_hash=hash_content("text"))
    sqlite_backend.write(record, hmac_signature=None)  # type: ignore[arg-type]
    result = verify_record(
        record.content_id, "text", sqlite_backend, sample_config.secret_key
    )
    assert result.hmac_valid is False


def test_verify_record_raises_record_not_found_for_unknown_id(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # verify_record must raise RecordNotFoundError for an unknown content_id.
    unknown = str(uuid.uuid4())
    with pytest.raises(RecordNotFoundError):
        verify_record(unknown, "text", sqlite_backend, sample_config.secret_key)


def test_verify_record_record_not_found_error_contains_content_id(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # RecordNotFoundError must expose the content_id that was not found.
    unknown = str(uuid.uuid4())
    with pytest.raises(RecordNotFoundError) as exc_info:
        verify_record(unknown, "text", sqlite_backend, sample_config.secret_key)
    assert exc_info.value.content_id == unknown


def test_verify_record_original_hash_in_result(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # VerificationResult.original_hash must equal the stored response_hash.
    text = "original"
    record = _write_signed(sqlite_backend, text, sample_config.secret_key)
    result = verify_record(
        record.content_id, text, sqlite_backend, sample_config.secret_key
    )
    assert result.original_hash == record.response_hash


def test_verify_record_current_hash_in_result(
    sqlite_backend: SQLiteBackend, sample_config: Config
) -> None:
    # VerificationResult.current_hash must equal hash_content(current_text).
    record = _write_signed(sqlite_backend, "original", sample_config.secret_key)
    result = verify_record(
        record.content_id, "current", sqlite_backend, sample_config.secret_key
    )
    assert result.current_hash == hash_content("current")


# ---------------------------------------------------------------------------
# Group 5 — store backend fix verification
# ---------------------------------------------------------------------------


def test_get_returns_none_hmac_for_record_written_without_signature(
    sqlite_backend: SQLiteBackend,
) -> None:
    # get() must return None for the hmac when a record was written with hmac=None.
    record = _build_record()
    sqlite_backend.write(record, hmac_signature=None)  # type: ignore[arg-type]
    result = sqlite_backend.get(record.content_id)
    assert result is not None
    _, hmac_value = result
    assert hmac_value is None


def test_get_return_type_is_tuple_with_optional_hmac(
    sqlite_backend: SQLiteBackend,
) -> None:
    # The second element of get()'s return tuple must be str | None.
    record = _build_record()
    sqlite_backend.write(record, hmac_signature="real_hmac_value")
    result = sqlite_backend.get(record.content_id)
    assert result is not None
    _, hmac_value = result
    assert isinstance(hmac_value, str)
