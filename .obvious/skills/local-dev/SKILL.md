---
name: local-dev
description: How to set up, run, and verify local development for nihanthnaidu007/ai-stamp (Python library + CLI, SQLite default, no server).
---

# local-dev — nihanthnaidu007/ai-stamp

Durable record of the Autobuild LOCAL-DEV onboarding run (2026-09-17, sandbox cmp_VPy6yJbH).

## Shape of the project

`ai-stamp` is a **library + CLI**, not a web app. There is no server process and no
port to health-check. Local dev = virtualenv + editable install + SQLite database.

## Setup (validated order)

1. `python3 -m venv .venv` (sandbox ships Python 3.13.14; package requires >=3.10)
2. `.venv/bin/pip install -e ".[dev,postgres]"` — takes ~20s. Skip the `nlp` extra
   unless PERSON/ORG detection is needed (spaCy is heavy).
3. No env vars needed to install or run tests. Required only for CLI/config flows:
   - `AISTAMP_SECRET_KEY` — any string >= 32 chars; local dummy is fine, it is an
     HMAC signing key, not an external credential.
   - `AISTAMP_DATABASE_URL` — default `sqlite:///./aistamp.db`; use
     `sqlite:////tmp/...` (4 slashes) for an absolute path.
4. No secrets are required for local dev. Nothing to request from the vault.

## Verify

```bash
.venv/bin/python -m pytest -q        # 260 passed, 6 skipped — PG tests self-skip without AISTAMP_TEST_POSTGRES_URL
.venv/bin/ruff check aistamp tests
.venv/bin/mypy aistamp               # strict; keep it that way
```

The 6 `test_store_postgres.py` skips are expected in the sandbox (no Docker, no
Postgres binaries). Postgres is an optional backend; SQLite is the dev default.

## Primary user flow (end-to-end evidence, 2026-09-17)

1. Library: `ProvenanceClient(callable_llm, config, backend=SQLiteBackend(...))`
   → `client.chat("Summarize this text", model="local-model")` → record persisted,
   status `COMPLETED` (content_id 39d37c32-…).
2. CLI against the same DB: `config check` → "Configuration OK"; `audit
   --content-id <id>` → full record; `verify --content-id <id> --text <original>` →
   `verified: YES`; tampered text → `verified: NO, drift_detected: YES`;
   `report --format json` → exports records; `scan --file` → EMAIL + PHONE_US
   matches with redacted snippets.
3. `aistamp migrate` → upgrades fresh SQLite DB to head (0001).

## Gotchas

- `client.chat()` returns a plain `str`. Use `backend.query(QueryFilters())` (or
  `backend.get`) to fetch the stored `ProvenanceRecord` / `content_id`.
- README shows `provenance config-check`; the real command is `aistamp config check`
  (sub-app `config`, command `check`).
- `Config.secret_key` must be >= 32 characters or pydantic validation fails.
- Stale `*.db` files are gitignored, but a leftover `/tmp/aistamp_e2e.db` from a
  previous run can double the `report` count — use a fresh DB path per session.
- No Dockerfile/Compose/Makefile/CI in the repo; do not assume one.

## Snapshot

`x5zqmaj5d5fgetghjk6v:default` captured 2026-09-17T18:07:37.578Z — checkout plus
`.venv/` with editable install (extras: dev, postgres).
