"""Verify a stamped record's integrity - and catch tampering.

Every record is HMAC-signed at write time. verify_record recomputes the
signature from the current text and compares hashes, so any drift between
what was stamped and what you hold now is detected.

    python examples/verify_record.py
"""

from __future__ import annotations

from aistamp import Config, ProvenanceClient, QueryFilters, SQLiteBackend, verify_record

SECRET_KEY = "replace-me-with-a-random-32+-character-secret"


def record_llm(prompt: str) -> str:
    return f"(recorded model) You asked: {prompt}"


def main() -> None:
    config = Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",
        log_level="INFO",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    client = ProvenanceClient(
        record_llm,
        config=config,
        app_id="demo_app",
        feature_id="verify_demo",
        user_id="demo_user",
        backend=backend,
    )
    original_response = client.chat("What is 2+2?")

    # Pull the content_id back out of the store for this demo.
    record = backend.query(QueryFilters()).records[0]
    content_id = record.content_id

    # 1. Honest re-verification: same text that was stamped.
    clean = verify_record(content_id, original_response, backend, SECRET_KEY)
    print(
        f"Untampered record: verified={clean.verified} "
        f"hash_match={clean.hash_match} hmac_valid={clean.hmac_valid}"
    )

    # 2. Tamper simulation: someone altered the response after stamping.
    tampered = verify_record(
        content_id, original_response + " (edited)", backend, SECRET_KEY
    )
    print(
        f"Tampered record:   verified={tampered.verified} "
        f"hash_match={tampered.hash_match} hmac_valid={tampered.hmac_valid}"
    )


if __name__ == "__main__":
    main()
