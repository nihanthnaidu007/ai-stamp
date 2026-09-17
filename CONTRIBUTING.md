# Contributing to ai-stamp

Thank you for helping improve ai-stamp. This document covers setup, the
quality bar, and how to submit changes.

## Development setup

Requires Python 3.10–3.13.

```bash
git clone https://github.com/nihanthnaidu007/ai-stamp
cd ai-stamp
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,postgres,nlp]"
```

(`postgres` installs the PostgreSQL test drivers; `nlp` installs spaCy for the
optional real-model NER test.)

## The quality bar

Every change must pass the same gates CI enforces:

```bash
ruff check .          # lint (zero warnings)
mypy aistamp          # strict mode, configured in pyproject.toml
pytest                # full suite, warnings as errors
```

Additional gates run in CI:

- **Coverage ratchet** — `pyproject.toml` sets `fail_under`. It starts just
  below the measured baseline and only moves up. If your change lowers
  coverage, add tests; do not lower the gate.
- **Build check** — `python -m build && twine check dist/*` must pass; the
  sdist and wheel are validated on every PR.
- **Dependency audit** — `pip-audit` runs against the installed dependency
  tree; new vulnerabilities fail the build.

## Testing guidelines

- Tests live in `tests/` and use plain `pytest`. Async tests use
  `pytest-asyncio` (`@pytest.mark.asyncio`).
- New test files must type-check under strict mypy.
- Markers in use: `postgres` (needs a live PostgreSQL; activated by
  `AISTAMP_TEST_POSTGRES_URL`), `spacy` (needs the `nlp` extra), `integration`,
  and `benchmark`.
- Property-based tests use Hypothesis; keep examples simple and let
  Hypothesis generate the rest.
- Keep tests independent: no cross-test ordering, no shared mutable state.

## Commits and pull requests

- Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes
  (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`).
- Branch from `main`; keep PRs focused on one change.
- Describe **why** the change is needed, not just what it does, and include
  the evidence you ran (test output, benchmark deltas).
- Update `CHANGELOG.md` under `[Unreleased]` for any user-visible change.
- CI must be green before review.

## Reporting issues

For bugs, include the minimal reproduction, expected vs. actual behavior, and
your Python/platform versions. For security-sensitive issues, do **not** open
a public issue — see [SECURITY.md](SECURITY.md).
