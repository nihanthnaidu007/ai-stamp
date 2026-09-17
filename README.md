# ai-stamp

`ai-stamp` is a Python package for AI output provenance, audit trails, and
compliance support. It wraps LLM calls, records what was generated, scans for
PII, stores tamper-evident metadata, and helps verify later whether generated
content has changed.

Use it when an application needs a lightweight audit layer around AI-generated
content, such as summaries, assistant responses, document drafts, support
answers, reports, or other LLM outputs.

## What It Does

- Wraps OpenAI, Anthropic, a generic JSON HTTP endpoint, or a Python callable.
- Records app, feature, user, model, status, token counts, latency, and time.
- Hashes prompts and responses for drift and tamper detection.
- Signs provenance records with HMAC.
- Verifies current text against stored provenance.
- Detects common PII in prompts and responses.
- Supports policy rules that can warn or block risky generations.
- Stores audit records in SQLite or PostgreSQL.
- Exports audit reports as JSON or CSV.
- Provides CLI commands through both `aistamp` and `provenance`.
- Ships type information and packaged Alembic migrations.

PII detection is best-effort only and is not a substitute for certified DLP,
privacy review, or legal/compliance tooling. Stored PII evidence snippets redact
recognized sensitive spans in their local context.

## Install

```bash
pip install ai-stamp
```

For PostgreSQL support:

```bash
pip install "ai-stamp[postgres]"
```

For optional spaCy-based PERSON/ORG detection:

```bash
pip install "ai-stamp[nlp]"
```

For development:

```bash
pip install -e ".[dev,postgres,nlp]"
```

## Quick Start

```python
from aistamp import Config, ProvenanceClient, SQLiteBackend

config = Config(secret_key="replace-with-at-least-32-secret-characters")

backend = SQLiteBackend("sqlite:///./aistamp.db")
backend.create_tables()  # local development only

client = ProvenanceClient(
    lambda prompt: f"Generated answer for: {prompt}",
    config=config,
    app_id="my_app",
    feature_id="summarizer",
    user_id="user_42",
    backend=backend,
)

response = client.chat("Summarize this text", model="local-model")
print(response)
```

Each call stores a provenance record containing hashes, timing, model metadata,
PII findings, policy results, and an HMAC signature.

## Configuration

`Config` can be created directly, loaded from environment variables, or loaded
from YAML.

| Field | Environment variable | Default | Notes |
| --- | --- | --- | --- |
| `secret_key` | `AISTAMP_SECRET_KEY` | required | HMAC signing key, at least 32 characters |
| `database_url` | `AISTAMP_DATABASE_URL` | `sqlite:///./aistamp.db` | SQLAlchemy database URL |
| `log_level` | `AISTAMP_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

Example YAML:

```yaml
secret_key: "a-secret-key-that-is-at-least-32-characters"
database_url: "sqlite:///./aistamp.db"
log_level: "INFO"
```

## Provider Wrapping

OpenAI and Anthropic SDK-style clients are detected automatically when their
common response shapes are returned. A plain Python callable can also be used
for local models, tests, or custom generation code.

For arbitrary JSON-over-HTTP providers, use `GenericHTTPClient`:

```python
from aistamp import Config, GenericHTTPClient, ProvenanceClient

http_llm = GenericHTTPClient(
    "https://example.internal/generate",
    headers={"Authorization": "Bearer token"},
)

client = ProvenanceClient(
    http_llm,
    config=Config(secret_key="replace-with-at-least-32-secret-characters"),
    app_id="my_app",
    feature_id="assistant",
    user_id="user_42",
)

text = client.chat("Hello", model="internal-model")
```

The endpoint receives JSON with `prompt` and `model`. It must return `text` or
`response`. Optional integer fields are `prompt_tokens` and `response_tokens`.

## Async Usage

```python
from aistamp import AsyncProvenanceClient, AsyncSQLiteBackend, Config


async def generate(prompt: str) -> str:
    return f"Generated answer for: {prompt}"


backend = AsyncSQLiteBackend("sqlite+aiosqlite:///./aistamp.db")
await backend.create_tables()

client = AsyncProvenanceClient(
    generate,
    config=Config(secret_key="replace-with-at-least-32-secret-characters"),
    app_id="my_app",
    feature_id="assistant",
    user_id="user_42",
    backend=backend,
)

response = await client.chat("Write a short summary", model="local-async")
```

## PII Detection

Built-in detection covers:

- Email addresses
- US phone numbers
- US SSNs
- Luhn-valid 16-digit card numbers
- API keys
- IPv4 addresses
- IPv6 addresses

Custom PII patterns can be loaded from YAML:

```yaml
patterns:
  - name: employee_id
    regex: "\\bEMP-[0-9]{6}\\b"
    pii_type: CUSTOM
    severity: MEDIUM
```

```python
from aistamp import load_patterns_from_yaml, scan_text

patterns = load_patterns_from_yaml("pii-patterns.yaml")
result = scan_text("Employee EMP-123456 requested access", extra_patterns=patterns)
```

## Policy Rules

Policy rules can warn or block records based on supported conditions such as
model tier and PII severity.

```yaml
model_tiers:
  approved:
    - gpt-4o
    - internal-safe-model

rules:
  - name: block_high_pii
    action: BLOCK
    reason: "High-severity PII is not allowed"
    conditions:
      pii_severity: HIGH

  - name: warn_unapproved_model
    action: WARN
    reason: "Model is not in the approved tier"
    conditions:
      model_tier: unapproved
```

```python
from aistamp import PolicyEngine

policy = PolicyEngine.from_yaml("policy.yaml")
```

## Database Setup

SQLite is suitable for local development and simple deployments:

```python
from aistamp import SQLiteBackend

backend = SQLiteBackend("sqlite:///./aistamp.db")
backend.create_tables()
```

For deployed databases, use the packaged Alembic migrations:

**macOS / Linux**
```bash
export AISTAMP_SECRET_KEY="a-secret-key-that-is-at-least-32-characters"
export AISTAMP_DATABASE_URL="postgresql://user:pass@host/db"
aistamp migrate
```

**Windows (Command Prompt)**
```cmd
set AISTAMP_SECRET_KEY=a-secret-key-that-is-at-least-32-characters
set AISTAMP_DATABASE_URL=postgresql://user:pass@host/db
aistamp migrate
```

**Windows (PowerShell)**
```powershell
$env:AISTAMP_SECRET_KEY="a-secret-key-that-is-at-least-32-characters"
$env:AISTAMP_DATABASE_URL="postgresql://user:pass@host/db"
aistamp migrate
```

Use `SQLiteBackend.create_tables()` only in development or tests.

## CLI

Both `aistamp` and `provenance` run the same CLI.

```bash
provenance config-check
provenance audit --content-id abc123
provenance verify --content-id abc123 --text "current text here"
provenance report --from 2024-01-01 --to 2024-06-30
provenance report --model gpt-4o --pii-severity HIGH --format json
provenance report --policy-decision BLOCK --format csv
provenance scan --file output.txt
provenance migrate
```

Date-only `--to` filters include the full named UTC calendar date.

## Audit And Verification

To verify whether stored content still matches the current text:

```python
from aistamp import Config, SQLiteBackend, verify_record

config = Config(secret_key="replace-with-at-least-32-secret-characters")
backend = SQLiteBackend("sqlite:///./aistamp.db")

result = verify_record(
    content_id="abc123",
    current_text="current text here",
    backend=backend,
    secret_key=config.secret_key,
)

print(result.verified)
print(result.drift_detected)
```

## Package Contents

The distribution includes:

- `aistamp` Python package
- `py.typed` marker for type checkers
- CLI entry points: `aistamp`, `provenance`
- Alembic migration files
- MIT license
- Tests in the source distribution

## Development And Verification

```bash
pytest
ruff check aistamp tests
mypy aistamp
python -m build
python -m twine check dist/*
```

PostgreSQL integration tests require a live test database:

**macOS / Linux**
```bash
export AISTAMP_TEST_POSTGRES_URL="postgresql://user:pass@localhost:5432/aistamp_test"
pytest tests/test_store_postgres.py -rs
```

**Windows (Command Prompt)**
```cmd
set AISTAMP_TEST_POSTGRES_URL=postgresql://user:pass@localhost:5432/aistamp_test
pytest tests/test_store_postgres.py -rs
```

**Windows (PowerShell)**
```powershell
$env:AISTAMP_TEST_POSTGRES_URL="postgresql://user:pass@localhost:5432/aistamp_test"
pytest tests/test_store_postgres.py -rs
```

The PostgreSQL extra installs `psycopg2-binary` and `asyncpg`.

## Examples

Runnable quickstarts live in [`examples/`](examples/) — each is a
self-contained script against the current API:

| Example | Shows |
|---|---|
| [`quickstart_callable.py`](examples/quickstart_callable.py) | Wrapping any callable LLM provider |
| [`openai_stamp.py`](examples/openai_stamp.py) | Wrapping the real OpenAI SDK |
| [`policy_block.py`](examples/policy_block.py) | Blocking a risky call with policy rules |
| [`audit_export.py`](examples/audit_export.py) | Exporting the audit trail as JSON/CSV |
| [`verify_record.py`](examples/verify_record.py) | Detecting tampering with `verify_record` |

```bash
python examples/quickstart_callable.py
```

## Status

`ai-stamp` is currently marked **Beta**: the public API is stable, the test
suite runs on CI across Python 3.10–3.13 (including live PostgreSQL
integration tests), and the packaging pipeline is automated. Review privacy,
security, retention, and compliance requirements before using it in regulated
production workflows.

### Trust and verification roadmap (upcoming in 0.2)

The current release signs every record with a single HMAC secret and verifies
on demand. Hardening work planned for the 0.2 line — **not yet merged**:

- **Key rotation** — versioned signatures with key IDs so verification can
  span multiple signing keys and a rotation event.
- **Retention** — configurable record lifecycle (TTL, purge, evidence
  export) for data-minimizing deployments.
- **Re-verification** — bulk re-verification of historical records against
  current keys and content, with tamper-evidence envelopes and optional
  hash chaining.

Until those land, treat the stored HMAC as an integrity check with a
single-key lifecycle, and plan key handling accordingly.

## License

MIT
