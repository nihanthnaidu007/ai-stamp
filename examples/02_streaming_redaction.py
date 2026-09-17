"""Streaming capture with redact_before_send — what leaves is what gets stamped.

v0.2 stamps provider streams: ``stamp_stream()`` yields chunks as the
provider produces them and, once the stream is exhausted, ``stream.result``
holds a ``StampResult`` stamped over the concatenation of all chunks.

With ``Config(redact_before_send=True)`` the outbound prompt is scrubbed of
PII before the provider sees it, and the persisted evidence (prompt hash)
covers exactly that redacted text — the v0.2.0 streaming hotfix made this
hold on every streaming path. This script proves both halves:

1. The fake provider records what it actually received → the email in the
   prompt left the process as ``[REDACTED]``.
2. ``hash_content(redacted_prompt) == result.record.prompt_hash`` → the
   audit trail hashes the same bytes the provider saw.

Streaming requires an OpenAI- or Anthropic-SDK-shaped client, so the fake
provider here mimics ``openai.OpenAI`` and is injected under the ``openai``
module name — the same technique the repo's test suite uses. With the real
SDK the code path is identical: pass your ``OpenAI(...)`` instance instead.

Run from the repo root:

    python examples/02_streaming_redaction.py

Expected output (content_id varies per run):

    chunk: Doc 47293 says the refund goes to
    chunk:  jane.doe@corp.example within
    chunk:  5 business days.
    -- stream exhausted --
    final text      : Doc 47293 says the refund goes to jane.doe@corp.example
                      within 5 business days.
    content_id      : 0e8f2d6a-...
    record pii      : 1 match(es), highest severity MEDIUM
    token usage     : prompt=24 response=31
    provider saw    : Contact [REDACTED] about refund Doc 47293
    raw prompt left the process? False
    prompt hash     : matches redacted outbound text = True

The provider is a fake, so nothing here touches the network.
"""

from __future__ import annotations

import sys
import tempfile
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from pydantic import SecretStr

from aistamp import Config, ProvenanceClient, SQLiteBackend, hash_content

# Demo-only. Load the real key from your secret store in production.
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")

PROMPT = "Contact jane.doe@corp.example about refund Doc 47293"

RESPONSE_CHUNKS = [
    "Doc 47293 says the refund goes to",
    " jane.doe@corp.example within",
    " 5 business days.",
]


@dataclass
class FakeCompletions:
    """Records every outbound ``create`` call so we can inspect the prompt."""

    calls: list[dict[str, Any]] = field(default_factory=list)

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        stream = [
            SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=c))],
                usage=None,
            )
            for c in RESPONSE_CHUNKS
        ]
        # Final usage-only chunk (empty choices), as OpenAI sends with
        # stream_options={"include_usage": True}.
        stream.append(
            SimpleNamespace(
                choices=[],
                usage=SimpleNamespace(prompt_tokens=24, completion_tokens=31),
            )
        )
        return iter(stream)


def install_fake_openai() -> tuple[types.ModuleType, FakeCompletions]:
    """Inject a minimal ``openai`` module whose ``OpenAI`` streams."""
    module = types.ModuleType("openai")
    completions = FakeCompletions()

    class _FakeOpenAI:
        def __init__(self, **_kwargs: Any) -> None:
            self.chat = SimpleNamespace(completions=completions)

    # ModuleType carries no statically declared attributes; resolve the
    # attribute dynamically the way the adapter does at import time.
    sdk_module: Any = module
    sdk_module.OpenAI = _FakeOpenAI
    sys.modules["openai"] = module
    return module, completions


def main() -> None:
    fake_module, completions = install_fake_openai()

    db_dir = Path(tempfile.mkdtemp(prefix="aistamp-02-"))
    config = Config(
        secret_key=SECRET_KEY,
        database_url=f"sqlite:///{db_dir / 'streaming.db'}",
        log_level="WARNING",
        redact_before_send=True,  # <- the point of this example
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    openai_cls: Any = sys.modules["openai"].OpenAI
    client = ProvenanceClient(
        openai_cls(),
        config=config,
        app_id="cookbook",
        feature_id="streaming",
        user_id="demo_user",
        backend=backend,
        max_retries=0,
    )

    stream = client.stamp_stream(PROMPT, model="gpt-4o-mini")
    for chunk in stream:
        print(f"chunk: {chunk}")

    result = stream.result
    print("-- stream exhausted --")
    print(f"final text      : {result.text}")
    print(f"content_id      : {result.content_id}")
    record = result.record
    highest = (
        record.pii_result.highest_severity.value
        if record.pii_result and record.pii_result.highest_severity
        else "NONE"
    )
    print(
        f"record pii      : {record.pii_result.match_count if record.pii_result else 0}"
        f" match(es), highest severity {highest}"
    )
    usage = result.usage
    print(
        "token usage     :"
        f" prompt={usage.prompt_tokens if usage else None}"
        f" response={usage.response_tokens if usage else None}"
    )

    # What the fake provider actually received (kwargs of the first call):
    outbound = str(completions.calls[0]["messages"][0]["content"])
    print(f"provider saw    : {outbound}")
    print(f"raw prompt left the process? {PROMPT in outbound}")

    # Evidence parity: the stored prompt hash covers the redacted text.
    parity = record.prompt_hash == hash_content(outbound)
    print(f"prompt hash     : matches redacted outbound text = {parity}")

    if "[REDACTED]" not in outbound or not parity:
        raise RuntimeError("redaction parity check failed in streaming demo")


if __name__ == "__main__":
    main()
