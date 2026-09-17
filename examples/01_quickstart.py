"""Quickstart: stamp a call to SQLite and verify its integrity afterwards.

This is the smallest end-to-end tour of ai-stamp v0.2:

1. Wrap a plain callable LLM provider in a ``ProvenanceClient``.
2. Call ``stamp()`` — the v0.2 rich result that carries the response text,
   the ``content_id``, the full ``ProvenanceRecord``, the policy decision,
   and token usage.
3. Verify the stored record with ``verify_record`` — first against the
   original text (``verified: True``), then against a tampered copy
   (``drift_detected: True``).

Run from the repo root:

    python examples/01_quickstart.py

Expected output (content_id and hashes vary per run):

    response text : Paris is the capital of France.
    content_id    : 697bc0bd-...
    status        : COMPLETED
    policy        : none
    token usage   : prompt=None response=None   (callables report no usage)
    -- verification --
    original text : verified=True hash_match=True hmac_valid=True
    tampered text : verified=False drift_detected=True

The provider is a fake callable, so nothing here touches the network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import SecretStr

from aistamp import (
    Config,
    ProvenanceClient,
    SQLiteBackend,
    StampError,
    verify_record,
)

# Demo-only. Load the real key from your secret store in production
# (Config.from_yaml / Config.from_env both support this).
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")


def my_llm(prompt: str) -> str:
    """Stand-in for any LLM: the callable needs no changes for ai-stamp."""
    return "Paris is the capital of France."


def main() -> None:
    db_path = Path(tempfile.mkdtemp(prefix="aistamp-01-")) / "quickstart.db"
    config = Config(
        secret_key=SECRET_KEY,
        database_url=f"sqlite:///{db_path}",
        log_level="WARNING",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    client = ProvenanceClient(
        my_llm,
        config=config,
        app_id="cookbook",
        feature_id="quickstart",
        user_id="demo_user",
        backend=backend,
    )

    result = client.stamp("What is the capital of France?", model="gpt-4o-mini")

    decision = result.record.policy_decision
    print(f"response text : {result.text}")
    print(f"content_id    : {result.content_id}")
    print(f"status        : {result.record.status.value}")
    print(f"policy        : {decision.action.value if decision else 'none'}")
    usage = result.usage
    print(
        "token usage   :"
        f" prompt={usage.prompt_tokens if usage else None}"
        f" response={usage.response_tokens if usage else None}"
    )

    print("-- verification --")
    original = verify_record(
        result.content_id, result.text, backend, config.secret_key
    )
    print(
        f"original text : verified={original.verified}"
        f" hash_match={original.hash_match} hmac_valid={original.hmac_valid}"
    )

    tampered = verify_record(
        result.content_id,
        "Paris is the capital of Germany.",  # someone edited the response
        backend,
        config.secret_key,
    )
    print(
        f"tampered text : verified={tampered.verified}"
        f" drift_detected={tampered.drift_detected}"
    )

    if not (original.verified and tampered.drift_detected):
        raise StampError("unexpected verification outcome in quickstart demo")


if __name__ == "__main__":
    main()
