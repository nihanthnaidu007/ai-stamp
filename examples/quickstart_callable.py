"""Quickstart: stamp an LLM call made through a plain Python callable.

ai-stamp wraps ANY callable LLM provider — start here before wiring real
SDKs. Run with:

    python examples/quickstart_callable.py
"""

from __future__ import annotations

from aistamp import Config, ProvenanceClient, QueryFilters, SQLiteBackend

# Point this at your own secret store; never hardcode production keys.
SECRET_KEY = "replace-me-with-a-random-32+-character-secret"


def my_llm(prompt: str) -> str:
    """Your existing LLM entry point — no changes required by ai-stamp."""
    return f"(simulated model) You asked: {prompt}"


def main() -> None:
    config = Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",  # demo only; use a real file or Postgres
        log_level="INFO",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    client = ProvenanceClient(
        my_llm,
        config=config,
        app_id="demo_app",
        feature_id="quickstart",
        user_id="demo_user",
        backend=backend,
    )

    response = client.chat("What is the capital of France?")

    print(f"Response: {response}")
    report = backend.query(QueryFilters())
    record = report.records[0]
    print(f"Stamped {report.total_count} record:")
    print(f"  content_id:   {record.content_id}")
    print(f"  prompt_hash:  {record.prompt_hash[:16]}...")
    print(f"  model:        {record.model}")
    print(f"  status:       {record.status.value}")


if __name__ == "__main__":
    main()
