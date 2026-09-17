"""v0.2 evidence-pack and CLI-v2 compatibility checks.

Everything in this module ships with policy PR #4 (policy v2, verifiable
exports, automation-grade CLI), so every check gates on feature detection
and SKIPs — never fails — while main lacks the surface. Once PR #4 merges
the gates open automatically and the assertions run for real.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import get_args

import _features
import pytest
from typer.testing import CliRunner

from aistamp.cli.main import app
from aistamp.client import ProvenanceClient
from aistamp.config import Config
from aistamp.fingerprint import hash_content
from aistamp.store import SQLiteBackend

_SECRET = "compat-kit-secret-key-0-2-32-chars!!"
_MODEL = "gpt-4o-mini"

runner = CliRunner()


def _config() -> Config:
    return Config(secret_key=_SECRET, database_url="sqlite:///:memory:")


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def _plain(output: str) -> str:
    """Normalize CLI help output for substring assertions.

    CI renders Typer/Rich help with ANSI color codes at a narrower terminal
    width, which both wraps tokens (``--dry-run`` can fold across lines) and
    interleaves escapes. Strip escapes, drop line breaks, collapse spaces.
    """
    no_ansi = _ANSI_RE.sub("", output).replace("\n", " ")
    return re.sub(r" {2,}", " ", no_ansi)


@pytest.fixture
def backend() -> SQLiteBackend:
    b = SQLiteBackend("sqlite:///:memory:")
    b.create_tables()
    return b


def _client(backend: SQLiteBackend) -> ProvenanceClient:
    def llm(prompt: str) -> str:
        return f"echo: {prompt}"

    return ProvenanceClient(
        llm,
        config=_config(),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )


# V2-17 ----------------------------------------------------------------------
def test_v2_17_evidence_pack_binds_record_and_signature(backend: SQLiteBackend) -> None:
    """An evidence pack proves what was stored AND that the stored signature
    verifies: VALID for an intact record, INVALID after tampering, MISSING
    when no signature was stored. The verdict lives in the pack's
    ``verification`` block.
    """
    build_evidence_pack = _features.build_evidence_pack()
    if build_evidence_pack is None:
        _features.skip_pending_pr(
            4, "evidence pack (aistamp.audit.build_evidence_pack)"
        )
    client = _client(backend)
    result = client.stamp("evidence", _MODEL)
    record, stored_sig = backend.get(result.content_id) or (None, None)
    assert record is not None and stored_sig is not None

    pack = build_evidence_pack(record, stored_sig, _SECRET)
    assert pack["evidence_version"] == 1
    assert pack["record"]["content_id"] == result.content_id
    verification = pack["verification"]
    assert verification["signature_verdict"] == "VALID"
    assert verification["algorithm"] == "HMAC-SHA256"

    tampered = record.model_copy(update={"response_hash": hash_content("altered")})
    bad = build_evidence_pack(tampered, stored_sig, _SECRET)
    assert bad["verification"]["signature_verdict"] == "INVALID"

    unsigned = build_evidence_pack(record, None, _SECRET)
    assert unsigned["verification"]["signature_verdict"] == "MISSING"


# V2-18 (policy/exports track, PR #4) -----------------------------------------
def test_v2_18_four_state_verdicts_in_exports(backend: SQLiteBackend) -> None:
    """Exports carry a four-state signature verdict:
    VALID / INVALID / MISSING / UNVERIFIED. Ships with policy PR #4.
    """
    verdict_fn = _features.signature_verdict()
    if verdict_fn is None:
        _features.skip_pending_pr(4, "four-state signature verdicts in exports")

    client = _client(backend)
    result = client.stamp("verdict", _MODEL)
    record, stored_sig = backend.get(result.content_id) or (None, None)
    assert record is not None and stored_sig is not None

    signature_verdict_type = _audit_attr("SignatureVerdict")
    assert set(get_args(signature_verdict_type)) == {
        "VALID",
        "INVALID",
        "MISSING",
        "UNVERIFIED",
    }
    assert verdict_fn(record, stored_sig, _SECRET) == "VALID"
    tampered = record.model_copy(update={"prompt_hash": hash_content("x")})
    assert verdict_fn(tampered, stored_sig, _SECRET) == "INVALID"
    assert verdict_fn(record, None, _SECRET) == "MISSING"


# V2-19 ------------------------------------------------------------------------
def test_v2_19_legacy_cli_commands_still_present() -> None:
    """The six 0.1 CLI commands stay on the app through the v2 CLI."""
    help_text = _plain(runner.invoke(app, ["--help"]).output)
    for command in ("audit", "verify", "report", "migrate", "scan", "config"):
        assert command in help_text, f"legacy command {command!r} missing from CLI"


# V2-20 (PENDING policy PR #4) -------------------------------------------------
def test_v2_20_cli_version_flag() -> None:
    """`aistamp --version` prints the package version and exits 0."""
    if "--version" not in _plain(runner.invoke(app, ["--help"]).output):
        _features.skip_pending_pr(4, "CLI --version flag")
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert _plain(result.output).startswith("aistamp ")


# V2-21 (PENDING policy PR #4) -------------------------------------------------
def test_v2_21_cli_keys_and_retention_subapps() -> None:
    """`keys rotate` and `retention enforce` (with a dry-run mode) exist."""
    top_help = _plain(runner.invoke(app, ["--help"]).output)
    if "keys" not in top_help or "retention" not in top_help:
        _features.skip_pending_pr(4, "CLI keys/retention sub-apps")
    assert runner.invoke(app, ["keys", "rotate", "--help"]).exit_code == 0
    retention_help = runner.invoke(app, ["retention", "enforce", "--help"])
    assert retention_help.exit_code == 0
    assert "--dry-run" in _plain(retention_help.output)


# V2-22 ----------------------------------------------------------------------
def test_v2_22_cli_evidence_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aistamp evidence --content-id X --output pack.json` writes a pack
    whose verification block carries a four-state signature verdict."""
    if "evidence" not in _plain(runner.invoke(app, ["--help"]).output):
        _features.skip_pending_pr(4, "CLI evidence command")
    db_path = tmp_path / "aistamp.db"
    pack_path = tmp_path / "pack.json"
    content_id = _seed_cli_db(db_path, monkeypatch)
    result = runner.invoke(
        app,
        ["evidence", "--content-id", content_id, "--output", str(pack_path)],
    )
    assert result.exit_code == 0, result.output
    pack = json.loads(pack_path.read_text())
    assert pack["record"]["content_id"] == content_id
    assert pack["verification"]["signature_verdict"] == "VALID"


# V2-23 (PENDING policy PR #4) -------------------------------------------------
def test_v2_23_cli_automation_scan_json_and_report_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`scan --json` emits machine-readable findings; `report` can write a
    file with an export manifest. Ships with policy PR #4.
    """
    scan_help = _plain(runner.invoke(app, ["scan", "--help"]).output)
    report_help = _plain(runner.invoke(app, ["report", "--help"]).output)
    if "--json" not in scan_help or "--manifest" not in report_help:
        _features.skip_pending_pr(4, "automation-grade scan --json / report --manifest")

    db_path = tmp_path / "aistamp.db"
    _seed_cli_db(db_path, monkeypatch)

    pii_file = tmp_path / "leak.txt"
    pii_file.write_text("contact alice@example.com for access\n")
    scan = runner.invoke(app, ["scan", "--file", str(pii_file), "--json"])
    assert scan.exit_code == 0, scan.output
    findings = json.loads(scan.output)
    assert findings["match_count"] >= 1

    report_path = tmp_path / "report.json"
    manifest_path = tmp_path / "manifest.json"
    report = runner.invoke(
        app,
        [
            "report",
            "--format",
            "json",
            "--output",
            str(report_path),
            "--manifest",
            str(manifest_path),
        ],
    )
    assert report.exit_code == 0, report.output
    exported = json.loads(report_path.read_text())
    assert exported["total_count"] >= 1
    assert manifest_path.exists()


# --- helpers ------------------------------------------------------------------


def _audit_attr(name: str) -> object:
    import aistamp.audit as audit_module

    return getattr(audit_module, name)


def _seed_cli_db(db_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Stamp one record through the library into a file-backed SQLite DB and
    point the CLI at it via env vars."""
    database_url = f"sqlite:///{db_path}"
    backend = SQLiteBackend(database_url)
    backend.create_tables()
    client = ProvenanceClient(
        lambda prompt: "ok",
        config=Config(secret_key=_SECRET, database_url=database_url),
        app_id="app",
        feature_id="feat",
        user_id="user",
        backend=backend,
    )
    content_id = client.stamp("cli seed", _MODEL).content_id
    backend.close()
    monkeypatch.setenv("AISTAMP_SECRET_KEY", _SECRET)
    monkeypatch.setenv("AISTAMP_DATABASE_URL", database_url)
    return content_id
