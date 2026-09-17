"""Stamp a real OpenAI chat completion.

Requires the ``openai`` package and an ``OPENAI_API_KEY`` environment
variable:

    pip install openai
    export OPENAI_API_KEY=sk-...
    python examples/openai_stamp.py
"""

from __future__ import annotations

import os
import sys

from aistamp import Config, ProvenanceClient, SQLiteBackend

SECRET_KEY = "replace-me-with-a-random-32+-character-secret"


def main() -> int:
    if "OPENAI_API_KEY" not in os.environ:
        print(
            "Set OPENAI_API_KEY to run this example "
            "(or start with examples/quickstart_callable.py)."
        )
        return 1

    # Imported lazily so the example fails with a clear message, not a
    # traceback, when the SDK is not installed.
    try:
        import openai  # noqa: F401
    except ImportError:
        print("Install the OpenAI SDK first: pip install openai")
        return 1

    config = Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///aistamp_openai_demo.db",
        log_level="INFO",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    client = ProvenanceClient(
        openai.OpenAI(),
        config=config,
        app_id="demo_app",
        feature_id="openai_quickstart",
        user_id=os.environ.get("USER", "demo_user"),
        backend=backend,
    )

    response = client.chat(
        "Explain HMAC signing in one sentence.",
        model="gpt-4o-mini",
    )

    print(f"Response: {response}")
    report = backend.query()
    record = report.records[0]
    print(f"Record {record.content_id}: {record.status.value}")
    print(f"  tokens: {record.prompt_tokens} in / {record.response_tokens} out")
    return 0


if __name__ == "__main__":
    sys.exit(main())
