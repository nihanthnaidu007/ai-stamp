# Changelog

All notable changes to ai-stamp are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-17

### Added

**Provenance store v2** (PR #2)
- Deterministic queries — every query path orders by `(timestamp ASC, id ASC)`,
  so offset pages no longer skip or repeat rows.
- Keyset paging for large volumes: `QueryFilters(cursor=...)` (opaque string
  from `AuditReport.next_cursor`) or the typed `after_timestamp`/`after_id`
  pair. Offset paging keeps its 0.1.x behavior, now with deterministic order.
- Crash-safe write-ahead audit: `finalize(content_id, record)` persists a
  `PENDING`-status record before the provider call and finalizes with the
  outcome (upserting if the pending row is missing). Failed calls now leave
  evidence — `error_type` / `error_message` are populated on the record.
- Retention and lifecycle: `purge(retention_days)` on every backend (returns
  the deleted count) with a purge-anchor journal.
- High-throughput stamping: `write_many(pairs)` one-transaction batch insert
  and `BufferedWriter` / `AsyncBufferedWriter` in `aistamp.store`.
- Backend lifecycle: `close()` / `dispose()` plus `with` / `async with`
  support on every backend.
- SQLite production posture: WAL journal mode, `busy_timeout` (default
  5000 ms, tunable), thread-friendly connections.
- Migration `0002` adds pinned provenance fields (`key_id`, `sig_algo`,
  `record_version`, `prev_hash`, `scope_sequence`, `error_type`,
  `error_message`); PostgreSQL `pii_result` / `policy_decision` convert to
  `JSONB` with GIN indexes; composite `(app_id, timestamp)`,
  `(user_id, timestamp)` and `feature_id` indexes on both dialects.

**Client & provider layer v2** (PR #11)
- Rich results: `stamp()` / `chat_detailed()` return
  `StampResult(text, content_id, record, decision, usage)`; `chat()` still
  returns plain `str` (0.1.x contract, test-anchored).
- Per-call `app_id` / `feature_id` / `user_id` overrides;
  `metadata` / `conversation_id` / `request_id` echoed on results and
  persisted on records.
- Provider kwargs passthrough (`system`, `messages`, sampling params) and
  configurable `max_tokens` — the hardcoded Anthropic 1024 is gone,
  with a documented `ANTHROPIC_DEFAULT_MAX_TOKENS = 1024` fallback.
- Sync and async streaming (`stamp_stream`) for OpenAI and Anthropic:
  stamps the final concatenation and captures usage.
- Resilience: retries with full-jitter exponential backoff on 429 / 5xx /
  timeouts; numeric `Retry-After` honored; non-retryable classes fail
  immediately.
- Error taxonomy: `AIStampError` base → `ProviderTimeoutError`,
  `ProviderRateLimitError`, `ProviderAuthError`, `ProviderResponseError`;
  every raised library error carries the `content_id`.
- Real async: `AsyncProvenanceClient` defaults to an aiosqlite backend
  (asyncpg for `postgresql+asyncpg` URLs), blocking I/O off-loop via
  `asyncio.to_thread`, `create_tables()` parity, `aclose()` disposal.
- Trust hooks: `on_persist_error` callback (sync and awaited-async variants)
  replaces the silent persist warning; `secret_key` is now a `SecretStr`.

**PII engine v2** (PR #6)
- Redaction API: `redact_text(text, matches=None, placeholder='[REDACTED]')`
  and `redact_prompt_and_response` — spans clamped and overlaps merged
  before substitution, property-tested so no matched fragment survives.
- Confidence scoring: `PIIMatch.confidence` (0–1, 1.0 = fully validated)
  populated by per-type validators — Luhn (including 15-digit Amex), SSN
  range rules, US phone plausibility, email domain shape, `ipaddress`-based
  IP checks; validator-rejected candidates are kept fail-closed at 0.25.
- Overlap arbitration: longest-match-wins across patterns with deterministic
  positional ordering; raw union semantics preserved behind
  `resolve_overlaps=False`.
- Modern key formats: `sk-proj-`, `sk-ant-api03-`, JWTs (`eyJ...`), GitHub
  `ghp_` / `gho_` / `github_pat_`, and raw Bearer tokens.
- Locale packs: opt-in `locale="INDIA"` (Aadhaar with Verhoeff checksum,
  PAN, +91 mobile) and `locale="EU"` (IBAN mod-97, UK NINO, German
  Steuer-ID).
- Allowlists (global and per-pattern) and `PatternConfig` with
  `locale` / `confidence` / `version` fields; duplicate pattern names are
  rejected at YAML load and `scan_text` assembly instead of shadowing.
- NER hardening: module-level cached spaCy model load; loud WARN-level
  degradation with install instructions when spaCy or the model is missing.

**Continuous integration (first CI for the project)**
- `ci.yml` — ruff lint, mypy strict type-check, pytest matrix on Python
  3.10–3.13 with a PostgreSQL 16 service container that activates the six
  `test_store_postgres.py` integration tests, coverage upload to Codecov,
  and a pip-audit dependency scan.
- `build.yml` — builds the sdist and wheel and validates them with
  `twine check` on every push and PR.
- `publish.yml` — publishes to PyPI on `v*` tags via OpenID Connect trusted
  publishing (no stored API token) with build provenance attestations.
- `dependabot.yml` — weekly dependency updates for pip and GitHub Actions.

**Test infrastructure**
- Adapter tests for the OpenAI and Anthropic sync/async paths via fake SDK
  modules injected through `sys.modules` — previously zero coverage.
- Real-HTTP transport tests for `GenericHTTPClient` against a local HTTP
  server — previously the HTTP path was monkeypatched out of its own tests.
- Hypothesis property tests: hash/verify round-trip, tamper detection,
  wrong-key rejection, redaction span non-leakage, policy severity
  monotonicity.
- pytest-benchmark smoke suite on the PII scanner and pipeline hot paths.
- Optional real-model spaCy test, running only when the `nlp` extra and
  model are installed (`pytest.importorskip`).
- Registered pytest markers (`postgres`, `spacy`, `integration`,
  `benchmark`), strict warning handling (`filterwarnings = error` with
  narrowly scoped documented exceptions), and a coverage ratchet gate.

**Packaging**
- Single-sourced version from `aistamp.__version__` via PEP 621 dynamic
  metadata (previously duplicated in two files).
- Project URLs (homepage, repository, changelog, issues) in package
  metadata.
- Classifiers updated: Python 3.13 added, `Python :: 3 :: Only` added,
  Development Status raised Alpha → Beta.
- Removed the committed root `PKG-INFO` and the vestigial `setup.cfg`;
  `MANIFEST.in` now ships `CHANGELOG.md` in the sdist.

**Documentation**
- Runnable `examples/` quickstarts: callable provider, OpenAI, policy
  block, audit export, and record verification.
- `CONTRIBUTING.md` and `SECURITY.md`.
- README trust-and-verification roadmap (key rotation, retention,
  re-verification — upcoming 0.2 capabilities).

### Changed

- `PolicyViolationError` is re-parented under the new `AIStampError` base;
  old import paths keep working via re-exports.
- `GenericHTTPClient` wraps transport failures (`HTTPError` / `URLError` /
  timeouts) into the error taxonomy with the original chained;
  response-content problems stay `ValueError`-compatible per the 0.1
  contract.
- Provenance clients validate the wrapped LLM client at construction time —
  unsupported types raise `TypeError` before any call can be attributed
  provenance.
- `QueryFilters` is a frozen, validated Pydantic model: `limit` clamped to
  [1, 1000], negative `limit`/`offset` rejected, `cursor` and `after_*`
  paging mutually exclusive.
- Overlap resolution is on by default in scan aggregation: overlapping
  matches arbitrate to the single longest span and `match_count` is
  de-duplicated; the 0.1.x raw union is available via
  `resolve_overlaps=False`.
- Locale-only values no longer fire under the default locale (INDIA / EU
  patterns are opt-in).
- Missing spaCy / model degradation is loud (WARN level with install
  instructions) instead of silent at DEBUG level.
- Built-in pattern set grew 7 → 9 to cover modern API-key formats and
  15-digit Amex cards.
- GitHub Actions dependencies bumped: actions/checkout 7, setup-python 7,
  upload-artifact 7, codecov/codecov-action 6 (#7–#10).

### Fixed

- Alembic `migrations/env.py` now runs
  `fileConfig(disable_existing_loggers=False)` — migrations no longer
  disable every pre-existing logger process-wide, which had muted
  `aistamp.pii` for any app that migrated before scanning.
- Fixed the `API_KEY` pattern token char class; a `Bearer <jwt>` now
  arbitrates to the single longest span.
- 10-digit account numbers no longer surface as plausible US phone numbers.
- `aclose()` on the async client disposes the default backend — an
  undisposed aiosqlite engine previously leaked worker threads for the life
  of the process.

---

## [0.1.0] — 2026-05-30

### Added

**Core provenance pipeline**
- `ProvenanceClient` — wraps OpenAI, Anthropic, and generic HTTP clients.
  Intercepts every LLM call and stamps the exchange with a tamper-evident
  provenance record automatically.
- `AsyncProvenanceClient` — async version for `asyncio`-based applications.
- `GenericHTTPClient` — utility wrapper for any HTTP-based LLM endpoint not
  natively supported by the built-in adapters.

**Content fingerprinting and tamper detection**
- SHA256 hashing of every prompt and response at capture time.
- HMAC-SHA256 signing of the full provenance record with a user-configured
  secret key.
- `verify_record()` — independently verifies content hash and HMAC integrity,
  detecting both content drift and database-level record tampering.

**PII detection**
- Six built-in patterns: `EMAIL`, `PHONE_US`, `SSN`, `CREDIT_CARD`,
  `API_KEY`, `IP_ADDRESS` with severity levels HIGH, MEDIUM, and LOW.
- Luhn algorithm validation for credit card false-positive reduction.
- IPv6 address detection.
- YAML-configurable custom patterns for domain-specific sensitive data.
- Optional spaCy NER integration for PERSON and ORG entity detection.
- Best-effort coverage. Not a substitute for certified DLP tooling.

**Policy rule engine**
- Declarative YAML rules with `model_tier` and `pii_severity` conditions.
- Three actions: `ALLOW`, `WARN`, `BLOCK`.
- `BLOCK` raises `PolicyViolationError` before the LLM call goes out,
  preventing sensitive data from reaching external APIs.
- All outcomes including blocked calls are recorded with full audit detail.

**Provenance store**
- `SQLiteBackend` — zero-config default for development and low-volume use.
- `PostgreSQLBackend` — production backend with connection pooling.
- `AsyncSQLiteBackend` and `AsyncPostgreSQLBackend` for async applications.
- Alembic migrations bundled inside the package at `aistamp/migrations/`.
- Full query interface: filter by user, app, feature, model, status,
  PII severity, policy decision, and time range, with pagination.

**Audit export**
- `AuditExporter` — query the store and export as JSON or CSV.
- JSON export preserves full provenance detail including nested PII and
  policy decision objects.
- CSV export flattens records for spreadsheet and BI tool consumption.

**CLI**
- `aistamp audit` — retrieve a single provenance record by content ID.
- `aistamp verify` — verify content integrity against stored record.
- `aistamp report` — query and export records with filters and format options.
- `aistamp scan` — scan a text file for PII patterns.
- `aistamp config check` — validate configuration.
- `aistamp migrate` — run Alembic migrations against the configured database.
- `provenance` alias available as an alternative to `aistamp`.

**Technical**
- Full type annotations. mypy strict-compatible.
- Zero top-level imports of optional dependencies. All lazy-imported.
- 260 tests across 11 test files.
