from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aistamp.cli.main import app
from aistamp.fingerprint import generate_content_id, hash_content, sign_record
from aistamp.models import (
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.store import SQLiteBackend

runner = CliRunner()


_SECRET = "test-secret-key-32-characters-min!!"


def _populate_db(
    db_path: Path,
    *,
    response_text: str = "The answer is 42.",
    user_id: str = "u",
    model: str = "gpt-4o",
    timestamp: datetime | None = None,
    pii_result: PIIResult | None = None,
    policy_decision: PolicyDecision | None = None,
) -> str:
    """Create a temp SQLite DB with one signed record. Returns the content_id."""
    db_url = f"sqlite:///{db_path}"
    backend = SQLiteBackend(db_url)
    backend.create_tables()
    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="cli_test",
        feature_id="f",
        user_id=user_id,
        model=model,
        prompt_hash=hash_content("a prompt"),
        response_hash=hash_content(response_text),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=timestamp or datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=pii_result,
        policy_decision=policy_decision,
    )
    hmac = sign_record(record, _SECRET)
    backend.write(record, hmac)
    return record.content_id


def _env(db_path: Path | None = None) -> dict[str, str]:
    env = {"AISTAMP_SECRET_KEY": _SECRET}
    if db_path is not None:
        env["AISTAMP_DATABASE_URL"] = f"sqlite:///{db_path}"
    return env


# ---------------------------------------------------------------------------
# Group 1 — config check command
# ---------------------------------------------------------------------------


def test_config_check_succeeds_with_valid_env(tmp_path: Path) -> None:
    # "config check" with valid env vars must exit 0 and print "Configuration OK".
    result = runner.invoke(app, ["config", "check"], env=_env(tmp_path / "x.db"))
    assert result.exit_code == 0
    assert "Configuration OK" in result.output


def test_config_check_fails_without_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # "config check" with no AISTAMP_SECRET_KEY in env must exit 1.
    monkeypatch.delenv("AISTAMP_SECRET_KEY", raising=False)
    result = runner.invoke(app, ["config", "check"], env={})
    assert result.exit_code == 1


def test_config_check_does_not_print_secret_key(tmp_path: Path) -> None:
    # The output must not contain the actual secret key value.
    result = runner.invoke(app, ["config", "check"], env=_env(tmp_path / "x.db"))
    assert _SECRET not in result.output


def test_config_check_prints_database_url(tmp_path: Path) -> None:
    # The database URL must appear in the output.
    db_path = tmp_path / "x.db"
    result = runner.invoke(app, ["config", "check"], env=_env(db_path))
    assert str(db_path) in result.output


# ---------------------------------------------------------------------------
# Group 2 — audit command
# ---------------------------------------------------------------------------


def test_audit_command_returns_record_text(tmp_path: Path) -> None:
    # "audit --content-id <id>" must print the record in text format.
    db = tmp_path / "audit.db"
    cid = _populate_db(db)
    result = runner.invoke(app, ["audit", "--content-id", cid], env=_env(db))
    assert result.exit_code == 0
    assert cid in result.output


def test_audit_command_returns_record_json(tmp_path: Path) -> None:
    # "audit --content-id <id> --format json" must print valid JSON.
    db = tmp_path / "audit.db"
    cid = _populate_db(db)
    result = runner.invoke(
        app, ["audit", "--content-id", cid, "--format", "json"], env=_env(db)
    )
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert data["content_id"] == cid


def test_audit_command_exits_1_for_unknown_id(tmp_path: Path) -> None:
    # "audit --content-id nonexistent" must exit with code 1.
    db = tmp_path / "audit.db"
    _populate_db(db)
    result = runner.invoke(
        app,
        ["audit", "--content-id", "00000000-0000-0000-0000-000000000000"],
        env=_env(db),
    )
    assert result.exit_code == 1


def test_audit_command_exits_1_for_unknown_format(tmp_path: Path) -> None:
    # "audit --content-id <id> --format xml" must exit with code 1.
    db = tmp_path / "audit.db"
    cid = _populate_db(db)
    result = runner.invoke(
        app, ["audit", "--content-id", cid, "--format", "xml"], env=_env(db)
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Group 3 — verify command
# ---------------------------------------------------------------------------


def test_verify_command_returns_verified_true(tmp_path: Path) -> None:
    # Original content exits 0 and reports a verified result.
    db = tmp_path / "verify.db"
    cid = _populate_db(db, response_text="The answer is 42.")
    result = runner.invoke(
        app,
        ["verify", "--content-id", cid, "--text", "The answer is 42."],
        env=_env(db),
    )
    assert result.exit_code == 0
    assert "verified:       YES" in result.output


def test_verify_command_detects_drift(tmp_path: Path) -> None:
    # Modified content exits 1 and reports detected drift.
    db = tmp_path / "verify.db"
    cid = _populate_db(db, response_text="The answer is 42.")
    result = runner.invoke(
        app,
        ["verify", "--content-id", cid, "--text", "Tampered text."],
        env=_env(db),
    )
    assert result.exit_code == 1
    assert "drift_detected: YES" in result.output


def test_verify_command_exits_1_for_unknown_id(tmp_path: Path) -> None:
    # "verify --content-id unknown --text text" must exit 1 with error output.
    db = tmp_path / "verify.db"
    _populate_db(db)
    result = runner.invoke(
        app,
        ["verify", "--content-id", "missing-id", "--text", "anything"],
        env=_env(db),
    )
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# Group 4 — report command
# ---------------------------------------------------------------------------


def test_report_command_text_format(tmp_path: Path) -> None:
    # "report" with no filters must print records in text format.
    db = tmp_path / "report.db"
    _populate_db(db, user_id="u1")
    _populate_db(db, user_id="u2")
    result = runner.invoke(app, ["report"], env=_env(db))
    assert result.exit_code == 0
    assert "total_count: 2" in result.output


def test_report_command_json_format(tmp_path: Path) -> None:
    # "report --format json" must return valid JSON with a "records" array.
    db = tmp_path / "report.db"
    _populate_db(db)
    result = runner.invoke(app, ["report", "--format", "json"], env=_env(db))
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "records" in data
    assert isinstance(data["records"], list)


def test_report_command_csv_format(tmp_path: Path) -> None:
    # "report --format csv" must return CSV with a header row.
    db = tmp_path / "report.db"
    _populate_db(db)
    result = runner.invoke(app, ["report", "--format", "csv"], env=_env(db))
    assert result.exit_code == 0
    assert "content_id" in result.output.splitlines()[0]


def test_report_command_filters_by_model(tmp_path: Path) -> None:
    # "report --model gpt-4o" must return only records matching that model.
    db = tmp_path / "report.db"
    _populate_db(db, model="gpt-4o")
    _populate_db(db, model="claude-3-5-sonnet")
    result = runner.invoke(app, ["report", "--model", "gpt-4o"], env=_env(db))
    assert result.exit_code == 0
    assert "total_count: 1" in result.output


def test_report_command_invalid_status_exits_1(tmp_path: Path) -> None:
    # "report --status INVALID" must exit 1 with a clear error.
    db = tmp_path / "report.db"
    _populate_db(db)
    result = runner.invoke(app, ["report", "--status", "INVALID"], env=_env(db))
    assert result.exit_code == 1


def test_report_command_filters_by_pii_severity(tmp_path: Path) -> None:
    db = tmp_path / "report.db"
    result = PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=PIISeverity.HIGH,
        match_count=1,
    )
    _populate_db(db, pii_result=result)
    _populate_db(db)
    output = runner.invoke(app, ["report", "--pii-severity", "HIGH"], env=_env(db))
    assert output.exit_code == 0
    assert "total_count: 1" in output.output


def test_report_command_filters_by_policy_decision(tmp_path: Path) -> None:
    db = tmp_path / "report.db"
    decision = PolicyDecision(
        action=PolicyAction.BLOCK, rule_name="deny", reason="blocked"
    )
    _populate_db(db, policy_decision=decision)
    _populate_db(db)
    output = runner.invoke(app, ["report", "--policy-decision", "BLOCK"], env=_env(db))
    assert output.exit_code == 0
    assert "total_count: 1" in output.output


def test_report_to_date_includes_whole_day(tmp_path: Path) -> None:
    db = tmp_path / "report.db"
    _populate_db(db, timestamp=datetime(2024, 6, 30, 23, 0, tzinfo=timezone.utc))
    result = runner.invoke(app, ["report", "--to", "2024-06-30"], env=_env(db))
    assert result.exit_code == 0
    assert "total_count: 1" in result.output


def test_migrate_command_creates_schema(tmp_path: Path) -> None:
    db = tmp_path / "migrate.db"
    result = runner.invoke(app, ["migrate"], env=_env(db))
    assert result.exit_code == 0
    assert "Database schema upgraded" in result.output
    assert db.exists()


# ---------------------------------------------------------------------------
# Group 5 — scan command
# ---------------------------------------------------------------------------


def test_scan_command_detects_pii_in_file(tmp_path: Path) -> None:
    # "scan --file <path>" must detect PII and print matches.
    f = tmp_path / "pii.txt"
    f.write_text("Email me at user@example.com")
    result = runner.invoke(app, ["scan", "--file", str(f)])
    assert result.exit_code == 0
    assert "EMAIL" in result.output


def test_scan_command_reports_clean_file(tmp_path: Path) -> None:
    # "scan --file <path>" on a clean file must print "No PII detected."
    f = tmp_path / "clean.txt"
    f.write_text("the weather is nice")
    result = runner.invoke(app, ["scan", "--file", str(f)])
    assert result.exit_code == 0
    assert "No PII detected." in result.output


def test_scan_command_exits_1_for_missing_file(tmp_path: Path) -> None:
    # "scan --file nonexistent.txt" must exit 1.
    missing = tmp_path / "nope.txt"
    result = runner.invoke(app, ["scan", "--file", str(missing)])
    assert result.exit_code == 1


def test_scan_command_uses_extra_patterns(tmp_path: Path) -> None:
    # "scan --file <path> --extra-patterns <yaml>" must apply custom patterns.
    f = tmp_path / "emp.txt"
    f.write_text("Employee EMP-123456 is here")
    yaml_file = tmp_path / "custom.yaml"
    yaml_file.write_text(
        "patterns:\n"
        "  - name: EMPLOYEE_ID\n"
        "    pattern: 'EMP-\\d{6}'\n"
        "    severity: HIGH\n"
    )
    result = runner.invoke(
        app,
        ["scan", "--file", str(f), "--extra-patterns", str(yaml_file)],
    )
    assert result.exit_code == 0
    assert "EMPLOYEE_ID" in result.output


# ---------------------------------------------------------------------------
# Group 6 — general CLI behavior
# ---------------------------------------------------------------------------


def test_cli_no_args_shows_help() -> None:
    # Running aistamp with no arguments must show help text. Typer with
    # no_args_is_help=True exits with code 2 (Click convention for "no args")
    # while still rendering the help; accept either 0 or 2.
    result = runner.invoke(app, [])
    assert result.exit_code in (0, 2)
    assert "Usage" in result.output or "Commands" in result.output


def test_cli_unknown_command_exits_nonzero() -> None:
    # Running "aistamp unknowncmd" must exit non-zero.
    result = runner.invoke(app, ["unknowncmd"])
    assert result.exit_code != 0
