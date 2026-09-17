# Changelog

All notable changes to ai-stamp are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

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
