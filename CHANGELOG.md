# Changelog

All notable changes to ai-stamp are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Released]

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
