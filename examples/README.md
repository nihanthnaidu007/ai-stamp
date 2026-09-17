# Examples Cookbook — aistamp v0.2.0

Six runnable scripts that walk the v0.2 surface end to end. Every script is
self-contained, uses a **fake provider** (plain callables or an injected
SDK-shaped fake), and writes only to temp directories — nothing here
touches the network or needs real API keys.

Run from the repo root (the installed package or an editable install is
required):

```bash
pip install -e ".[dev]"
python examples/01_quickstart.py
```

| Script | Demonstrates |
|---|---|
| [`01_quickstart.py`](01_quickstart.py) | Wrapping a callable provider; `stamp()` rich results; `verify_record` detecting tampering |
| [`02_streaming_redaction.py`](02_streaming_redaction.py) | `stamp_stream()` chunk capture; `redact_before_send` on the streaming path; prompt-hash/evidence parity |
| [`03_async_capture.py`](03_async_capture.py) | `AsyncProvenanceClient` on aiosqlite; concurrent `stamp()` via `asyncio.gather`; clean `aclose()` |
| [`04_policy_rules.py`](04_policy_rules.py) | `PolicyEngine.from_yaml`; `model_tier` / `pii_severity` conditions; ALLOW / WARN / BLOCK, first match wins |
| [`05_retention_purge.py`](05_retention_purge.py) | `backend.purge(retention_days)` with `PurgeAnchor` journaling; post-purge verification |
| [`06_evidence_pack.py`](06_evidence_pack.py) | `AuditExporter` JSON/CSV; per-record verification verdicts; SHA-256 sealed evidence manifest |

Shared assets:

- [`policy_rules.yaml`](policy_rules.yaml) — the policy file loaded by
  examples 04 and 06 (tiered model rules, severity-gated WARN, pre-call BLOCK).

## What each run leaves behind

- 01–05 write their SQLite databases to fresh temp directories only.
- 06 additionally writes its exports to `./evidence_export/` under the
  **current directory** (`audit_report.json`, `records.csv`,
  `evidence_pack.json` plus a SHA-256 manifest). The directory is safe to
  delete and is not committed.

## Conventions

- Scripts are typed and lint-clean under the repo's own gates
  (`ruff check .` covers this directory in CI; `mypy examples` passes in
  strict mode).
- Demo secret keys are hard-coded **on purpose** — they are placeholders.
  Real deployments load keys from the environment or a secret store via
  `Config.from_env()` / `Config.from_yaml()`; both are shown in the
  scripts.
- 02 fakes the OpenAI SDK by injecting a minimal module under the
  `openai` name — the same technique the test suite uses — because
  streaming dispatch is SDK-shaped by design. Swap in a real
  `openai.OpenAI(...)` instance and the code path is identical.
- Expected-output blocks in each docstring reflect actual runs; content
  IDs, hashes, and timestamps vary per run.
