"""CLI v2 tests: version, scan gating, verify sources, report paging/export,
credential masking, evidence, retention, and key-rotation fallbacks."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aistamp.cli.main import app
from aistamp.fingerprint import generate_content_id, hash_content, sign_record
from aistamp.models import ProvenanceRecord, RecordStatus
from aistamp.store import SQLiteBackend

runner = CliRunner()

_SECRET = "test-secret-key-32-characters-min!!"

_PII_TEXT = "Contact support@internal-company.com for help."


def _populate_db(
    db_path: Path,
    *,
    response_text: str = "The answer is 42.",
    feature_id: str = "f",
    timestamp: datetime | None = None,
) -> str:
    backend = SQLiteBackend(f"sqlite:///{db_path}")
    backend.create_tables()
    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="cli_test",
        feature_id=feature_id,
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("a prompt"),
        response_hash=hash_content(response_text),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=timestamp or datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )
    backend.write(record, sign_record(record, _SECRET))
    return record.content_id


def _env(db_path: Path | None = None) -> dict[str, str]:
    env = {"AISTAMP_SECRET_KEY": _SECRET}
    if db_path is not None:
        env["AISTAMP_DATABASE_URL"] = f"sqlite:///{db_path}"
    return env


def _config_yaml(tmp_path: Path, database_url: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(
        f"secret_key: {_SECRET}\ndatabase_url: {database_url}\nlog_level: WARNING\n"
    )
    return path


# ---------------------------------------------------------------------------
# --version
# ---------------------------------------------------------------------------


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.output.startswith("aistamp ")


def test_short_version_flag() -> None:
    result = runner.invoke(app, ["-V"])
    assert result.exit_code == 0
    assert result.output.startswith("aistamp ")


# ---------------------------------------------------------------------------
# scan: stdin, JSON, fail-on gating
# ---------------------------------------------------------------------------


def test_scan_reads_stdin_json() -> None:
    result = runner.invoke(
        app,
        ["scan", "--json"],
        input=_PII_TEXT,
        env=_env(),
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["source"] == "<stdin>"
    assert data["match_count"] >= 1
    assert data["fail_triggered"] is False


def test_scan_fail_on_found_exits_nonzero(tmp_path: Path) -> None:
    pii_file = tmp_path / "pii.txt"
    pii_file.write_text(_PII_TEXT)
    clean_file = tmp_path / "clean.txt"
    clean_file.write_text("Nothing to see here.")

    flagged = runner.invoke(
        app, ["scan", "--file", str(pii_file), "--fail-on", "FOUND"]
    )
    assert flagged.exit_code == 1

    clean = runner.invoke(
        app, ["scan", "--file", str(clean_file), "--fail-on", "FOUND"]
    )
    assert clean.exit_code == 0


def test_scan_fail_on_severity_threshold(tmp_path: Path) -> None:
    pii_file = tmp_path / "pii.txt"
    pii_file.write_text(_PII_TEXT)  # EMAIL -> MEDIUM

    below = runner.invoke(
        app, ["scan", "--file", str(pii_file), "--fail-on", "HIGH"]
    )
    assert below.exit_code == 0

    at = runner.invoke(
        app, ["scan", "--file", str(pii_file), "--fail-on", "MEDIUM"]
    )
    assert at.exit_code == 1


def test_scan_invalid_fail_on_value(tmp_path: Path) -> None:
    f = tmp_path / "x.txt"
    f.write_text("hello")
    result = runner.invoke(
        app, ["scan", "--file", str(f), "--fail-on", "EXTREME"]
    )
    assert result.exit_code == 1
    assert "Invalid --fail-on" in result.output


# ---------------------------------------------------------------------------
# verify: file/stdin sources stay out of shell history
# ---------------------------------------------------------------------------


def test_verify_accepts_file(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    cid = _populate_db(db)
    text_file = tmp_path / "current.txt"
    text_file.write_text("some other content entirely")

    result = runner.invoke(
        app,
        ["verify", "--content-id", cid, "--file", str(text_file)],
        env=_env(db),
    )
    assert result.exit_code == 1  # hash mismatch
    assert "verified:       NO" in result.output
    assert "hmac_valid:     YES" in result.output


def test_verify_accepts_stdin_via_dash(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    cid = _populate_db(db)

    result = runner.invoke(
        app,
        ["verify", "--content-id", cid, "--file", "-"],
        input="stdin content",
        env=_env(db),
    )
    assert result.exit_code == 1
    assert "verified:       NO" in result.output


def test_verify_rejects_both_sources(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    cid = _populate_db(db)
    result = runner.invoke(
        app,
        ["verify", "--content-id", cid, "--text", "x", "--file", "-"],
        input="y",
        env=_env(db),
    )
    assert result.exit_code == 1
    assert "Provide only one of --text, --file" in result.output


def test_verify_requires_a_source(tmp_path: Path) -> None:
    db = tmp_path / "v.db"
    cid = _populate_db(db)
    result = runner.invoke(app, ["verify", "--content-id", cid], env=_env(db))
    assert result.exit_code == 1
    assert "--text" in result.output


# ---------------------------------------------------------------------------
# report: feature filter, offset paging, file export, manifest
# ---------------------------------------------------------------------------


def test_report_filters_by_feature_id(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    _populate_db(db, feature_id="wanted")

    hit = runner.invoke(
        app, ["report", "--feature-id", "wanted", "--format", "json"], env=_env(db)
    )
    assert hit.exit_code == 0
    assert json.loads(hit.output)["total_count"] == 1

    miss = runner.invoke(
        app, ["report", "--feature-id", "other", "--format", "json"], env=_env(db)
    )
    assert json.loads(miss.output)["total_count"] == 0


def test_report_offset_skips_records(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    cid = _populate_db(db)

    full = runner.invoke(app, ["report", "--format", "json"], env=_env(db))
    assert json.loads(full.output)["total_count"] == 1

    paged = runner.invoke(
        app, ["report", "--format", "json", "--offset", "1"], env=_env(db)
    )
    data = json.loads(paged.output)
    assert data["total_count"] == 1
    assert data["records"] == []
    assert cid not in paged.output


def test_report_output_writes_file(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    _populate_db(db)
    out = tmp_path / "report.json"

    result = runner.invoke(
        app, ["report", "--format", "json", "--output", str(out)], env=_env(db)
    )
    assert result.exit_code == 0
    assert "Report written to" in result.output
    data = json.loads(out.read_text())
    assert data["records"][0]["signature_verdict"] == "VALID"


def test_report_manifest_sidecar(tmp_path: Path) -> None:
    db = tmp_path / "r.db"
    _populate_db(db)
    manifest = tmp_path / "manifest.json"

    result = runner.invoke(
        app, ["report", "--manifest", str(manifest)], env=_env(db)
    )
    assert result.exit_code == 0
    data = json.loads(manifest.read_text())
    assert data["manifest_version"] == 1
    assert data["aistamp_version"]


# ---------------------------------------------------------------------------
# config check: credential masking
# ---------------------------------------------------------------------------


def test_config_check_masks_database_url_password(tmp_path: Path) -> None:
    config = _config_yaml(
        tmp_path, "postgresql://audit_user:hunter2@db.example.com:5432/prov"
    )
    result = runner.invoke(app, ["config", "check", "--config", str(config)])
    assert result.exit_code == 0
    assert "hunter2" not in result.output
    assert "***" in result.output
    assert "db.example.com" in result.output


# ---------------------------------------------------------------------------
# evidence command
# ---------------------------------------------------------------------------


def test_evidence_command_outputs_pack(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    cid = _populate_db(db)

    result = runner.invoke(app, ["evidence", "--content-id", cid], env=_env(db))
    assert result.exit_code == 0
    pack = json.loads(result.output)
    assert pack["evidence_version"] == 1
    assert pack["content_id"] == cid
    assert pack["verification"]["signature_verdict"] == "VALID"


def test_evidence_command_writes_file(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    cid = _populate_db(db)
    out = tmp_path / "evidence.json"

    result = runner.invoke(
        app, ["evidence", "--content-id", cid, "--output", str(out)], env=_env(db)
    )
    assert result.exit_code == 0
    assert json.loads(out.read_text())["content_id"] == cid


def test_evidence_command_unknown_id(tmp_path: Path) -> None:
    db = tmp_path / "e.db"
    _populate_db(db)
    result = runner.invoke(
        app, ["evidence", "--content-id", "no-such-id"], env=_env(db)
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# retention enforce
# ---------------------------------------------------------------------------


def test_retention_enforce_dry_run(tmp_path: Path) -> None:
    db = tmp_path / "ret.db"
    old_ts = datetime.now(timezone.utc) - timedelta(days=100)
    _populate_db(db, timestamp=old_ts)
    config = _config_yaml(tmp_path, f"sqlite:///{db}")

    result = runner.invoke(
        app,
        [
            "retention",
            "enforce",
            "--older-than-days",
            "90",
            "--dry-run",
            "--config",
            str(config),
        ],
    )
    assert result.exit_code == 0
    assert "would delete 1 record(s)" in result.output


def test_retention_enforce_missing_window(tmp_path: Path) -> None:
    config = _config_yaml(tmp_path, f"sqlite:///{tmp_path / 'ret.db'}")
    result = runner.invoke(
        app, ["retention", "enforce", "--config", str(config)]
    )
    assert result.exit_code == 1
    assert "--older-than-days is required" in result.output


# ---------------------------------------------------------------------------
# keys rotate
# ---------------------------------------------------------------------------


def test_keys_rotate_missing_new_key_env(tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["keys", "rotate"],
        env={"AISTAMP_SECRET_KEY": _SECRET},
    )
    assert result.exit_code == 1
    assert "AISTAMP_NEW_SECRET_KEY" in result.output


def test_keys_rotate_missing_old_key_env() -> None:
    result = runner.invoke(app, ["keys", "rotate"], env={})
    assert result.exit_code == 1
    assert "AISTAMP_SECRET_KEY" in result.output


def test_keys_rotate_graceful_when_api_missing(tmp_path: Path) -> None:
    # aistamp.keys.rotate_secret is the pinned v0.2.0 API; until the crypto
    # track lands the module the command must fail with a clear message,
    # never a traceback.
    result = runner.invoke(
        app,
        ["keys", "rotate"],
        env={
            "AISTAMP_SECRET_KEY": _SECRET,
            "AISTAMP_NEW_SECRET_KEY": "a-brand-new-secret-key-32-chars!!",
        },
    )
    assert result.exit_code == 1
    assert "not available" in result.output


@pytest.mark.xfail(
    reason="aistamp.keys.rotate_secret is owned by the crypto track;"
    " integration test once the pinned API exists on this branch.",
    strict=False,
)
def test_keys_rotate_end_to_end(tmp_path: Path) -> None:
    db = tmp_path / "k.db"
    _populate_db(db)
    result = runner.invoke(
        app,
        ["keys", "rotate"],
        env={
            "AISTAMP_SECRET_KEY": _SECRET,
            "AISTAMP_NEW_SECRET_KEY": "a-brand-new-secret-key-32-chars!!",
            "AISTAMP_DATABASE_URL": f"sqlite:///{db}",
        },
    )
    assert result.exit_code == 0
    assert "Secret key rotated" in result.output
