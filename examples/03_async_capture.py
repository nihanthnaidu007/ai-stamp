"""Async capture on aiosqlite — parallel stamping and a clean aclose().

``AsyncProvenanceClient`` is the event-loop-native twin of
``ProvenanceClient``: PII scans and provider dispatch run off the loop, and
``sqlite:///`` database URLs are transparently upgraded to the aiosqlite
driver so persistence never blocks.

This script demonstrates:

1. An ``async`` callable provider — the async contract is
   ``(str) -> str | Awaitable[str]``.
2. Two ``stamp()`` calls raced with ``asyncio.gather`` — both records land.
3. ``await client.aclose()`` — releasing the backend's aiosqlite worker
   threads and pooled connections when the process keeps running after the
   workload (for example a service's shutdown path).
4. Cross-check: the async-written records verify through the *sync* backend
   on the same file — one database, two drivers.

Run from the repo root:

    python examples/03_async_capture.py

Expected output (content_ids vary per run):

    call 1 text : async answer to: summarize the incident report
    call 2 text : async answer to: draft the customer reply
    content_ids : 4f9a1c2b-... , b7e20d51-...
    both stamped : True
    aclosed      : True
    -- sync cross-check on the same SQLite file --
    record 1 : verified=True
    record 2 : verified=True

The provider is a fake callable, so nothing here touches the network.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pydantic import SecretStr

from aistamp import (
    AsyncProvenanceClient,
    Config,
    SQLiteBackend,
    StampError,
    verify_record,
)

# Demo-only. Load the real key from your secret store in production.
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")


async def my_llm(prompt: str) -> str:
    """Async stand-in for any LLM — e.g. an AsyncOpenAI/AsyncAnthropic call."""
    await asyncio.sleep(0.01)  # simulate provider latency
    return f"async answer to: {prompt}"


async def run() -> None:
    db_dir = Path(tempfile.mkdtemp(prefix="aistamp-03-"))
    config = Config(
        secret_key=SECRET_KEY,
        # Plain sqlite:/// — the async client upgrades this to aiosqlite.
        database_url=f"sqlite:///{db_dir / 'async.db'}",
        log_level="WARNING",
    )

    client = AsyncProvenanceClient(
        my_llm,
        config=config,
        app_id="cookbook",
        feature_id="async-capture",
        user_id="demo_user",
    )

    results = await asyncio.gather(
        client.stamp("summarize the incident report", model="gpt-4o-mini"),
        client.stamp("draft the customer reply", model="gpt-4o-mini"),
    )

    texts = [r.text for r in results]
    content_ids = [r.content_id for r in results]
    print(f"call 1 text : {texts[0]}")
    print(f"call 2 text : {texts[1]}")
    print(f"content_ids : {content_ids[0]} , {content_ids[1]}")
    print(f"both stamped : {len(content_ids) == 2}")

    await client.aclose()
    print("aclosed      : True")

    # Cross-check through the sync driver on the same file: the records the
    # async client wrote are readable (and verifiable) anywhere.
    sync_backend = SQLiteBackend(config.database_url)
    verdicts = [
        verify_record(cid, text, sync_backend, config.secret_key)
        for cid, text in zip(content_ids, texts, strict=True)
    ]
    print("-- sync cross-check on the same SQLite file --")
    for i, verdict in enumerate(verdicts, start=1):
        print(f"record {i} : verified={verdict.verified}")

    if not all(v.verified for v in verdicts):
        raise StampError("async-written records failed sync verification")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
