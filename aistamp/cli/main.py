from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Annotated

import typer
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

from aistamp.audit import (
    AuditExporter,
    build_export_manifest,
    enforce_retention,
    signature_verdict,
)
from aistamp.config import Config
from aistamp.fingerprint import RecordNotFoundError, verify_record
from aistamp.models import (
    SEVERITY_RANK,
    PIIMatch,
    PIISeverity,
    PolicyAction,
    QueryFilters,
    RecordStatus,
)
from aistamp.pii import load_patterns_from_yaml, scan_text
from aistamp.store import SQLiteBackend

app = typer.Typer(
    name="aistamp",
    help="ai-stamp: Provenance tracking and compliance audit for AI-generated content.",
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Configuration commands.",
    no_args_is_help=True,
)
app.add_typer(config_app)

keys_app = typer.Typer(
    name="keys",
    help="Secret key management commands.",
    no_args_is_help=True,
)
app.add_typer(keys_app)

retention_app = typer.Typer(
    name="retention",
    help="Data retention commands.",
    no_args_is_help=True,
)
app.add_typer(retention_app)


def _version_callback(value: bool) -> None:
    if value:
        from aistamp.audit import package_version

        typer.echo(f"aistamp {package_version()}")
        raise typer.Exit()


@app.callback()
def _root(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the ai-stamp version and exit.",
        ),
    ] = False,
) -> None:
    """Provenance tracking and compliance audit for AI-generated content."""


def _load_config(config_path: Path | None) -> Config:
    try:
        if config_path:
            return Config.from_yaml(config_path)
        return Config.from_env()
    except (KeyError, FileNotFoundError, ValueError) as e:
        typer.echo(f"Configuration error: {e}", err=True)
        raise typer.Exit(code=1) from None


def _get_backend(config: Config) -> SQLiteBackend:
    return SQLiteBackend(config.database_url)


def _read_file_or_stdin(file: Path | None) -> str:
    """Read text from --file, or from stdin when omitted or '-'."""
    if file is None or str(file) == "-":
        return sys.stdin.read()
    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(code=1)
    return file.read_text(encoding="utf-8")


@app.command()
def audit(
    content_id: Annotated[
        str, typer.Option("--content-id", help="The content_id to retrieve.")
    ],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
    format: Annotated[
        str, typer.Option("--format", help="Output format: 'text' or 'json'.")
    ] = "text",
) -> None:
    """Retrieve the full provenance record for a content ID."""
    config = _load_config(config_path)
    backend = _get_backend(config)

    result = backend.get(content_id)
    if result is None:
        typer.echo(f"No record found for content_id: {content_id}", err=True)
        raise typer.Exit(code=1)

    record, stored_hmac = result

    if format == "json":
        exporter = AuditExporter(backend, secret_key=config.secret_key)
        typer.echo(
            json.dumps(exporter.signed_record_dict(record, stored_hmac), indent=2)
        )
        return

    if format == "text":
        pii_count = record.pii_result.match_count if record.pii_result else 0
        pii_sev = (
            record.pii_result.highest_severity.value
            if (record.pii_result and record.pii_result.highest_severity)
            else "none"
        )
        policy_action = (
            record.policy_decision.action.value if record.policy_decision else "none"
        )
        policy_rule = (
            record.policy_decision.rule_name
            if (record.policy_decision and record.policy_decision.rule_name)
            else "no rule"
        )
        verdict = signature_verdict(record, stored_hmac, config.secret_key)
        typer.echo(f"content_id:    {record.content_id}")
        typer.echo(f"app_id:        {record.app_id}")
        typer.echo(f"feature_id:    {record.feature_id}")
        typer.echo(f"user_id:       {record.user_id}")
        typer.echo(f"model:         {record.model}")
        typer.echo(f"status:        {record.status.value}")
        typer.echo(f"timestamp:     {record.timestamp.isoformat()}")
        typer.echo(f"latency_ms:    {record.latency_ms}")
        typer.echo(f"prompt_hash:   {record.prompt_hash}")
        typer.echo(f"response_hash: {record.response_hash}")
        typer.echo(f"pii_matches:   {pii_count} ({pii_sev})")
        typer.echo(f"policy:        {policy_action} ({policy_rule})")
        typer.echo(f"signature:     {'present' if stored_hmac else 'absent'}")
        typer.echo(f"hmac_verdict:  {verdict}")
        return

    typer.echo(f"Unknown format: {format!r}. Use 'text' or 'json'.", err=True)
    raise typer.Exit(code=1)


@app.command()
def verify(
    content_id: Annotated[
        str, typer.Option("--content-id", help="The content_id to verify against.")
    ],
    text: Annotated[
        str | None,
        typer.Option("--text", help="The current text to verify."),
    ] = None,
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help="Read the text to verify from this file ('-' for stdin)."
            " Alternative to --text so secrets stay out of shell history.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Verify whether content matches its stored provenance record."""
    provided = [
        flag
        for flag, given in (
            ("--text", text is not None),
            ("--file", file is not None),
        )
        if given
    ]
    if len(provided) > 1:
        typer.echo(f"Provide only one of {', '.join(provided)}.", err=True)
        raise typer.Exit(code=1)

    # Verify demands an explicit source: an implicit empty stdin read would
    # silently "verify" against the empty string.
    if text is not None:
        current_text = text
    elif file is not None:
        current_text = _read_file_or_stdin(file)
    else:
        typer.echo(
            "Provide the content to verify via --text or --file ('-' for stdin).",
            err=True,
        )
        raise typer.Exit(code=1)

    config = _load_config(config_path)
    backend = _get_backend(config)

    try:
        result = verify_record(content_id, current_text, backend, config.secret_key)
    except RecordNotFoundError as e:
        typer.echo(f"Verification error: {e}", err=True)
        raise typer.Exit(code=1) from None

    def yn(b: bool) -> str:
        return "YES" if b else "NO"

    typer.echo(f"content_id:     {result.content_id}")
    typer.echo(f"verified:       {yn(result.verified)}")
    typer.echo(f"hash_match:     {yn(result.hash_match)}")
    typer.echo(f"hmac_valid:     {yn(result.hmac_valid)}")
    typer.echo(f"drift_detected: {yn(result.drift_detected)}")
    typer.echo(f"original_hash:  {result.original_hash}")
    typer.echo(f"current_hash:   {result.current_hash}")

    raise typer.Exit(code=0 if result.verified else 1)


@app.command()
def report(
    from_date: Annotated[
        str | None, typer.Option("--from", help="Start date (YYYY-MM-DD).")
    ] = None,
    to_date: Annotated[
        str | None, typer.Option("--to", help="End date (YYYY-MM-DD).")
    ] = None,
    model: Annotated[
        str | None, typer.Option("--model", help="Filter by model name.")
    ] = None,
    user_id: Annotated[
        str | None, typer.Option("--user-id", help="Filter by user ID.")
    ] = None,
    app_id: Annotated[
        str | None, typer.Option("--app-id", help="Filter by app ID.")
    ] = None,
    feature_id: Annotated[
        str | None, typer.Option("--feature-id", help="Filter by feature ID.")
    ] = None,
    status: Annotated[
        str | None,
        typer.Option("--status", help="Filter by status: COMPLETED, BLOCKED, ERROR."),
    ] = None,
    pii_severity: Annotated[
        str | None,
        typer.Option(
            "--pii-severity", help="Filter by PII severity: HIGH, MEDIUM, LOW."
        ),
    ] = None,
    policy_decision: Annotated[
        str | None,
        typer.Option("--policy-decision", help="Filter by policy: ALLOW, WARN, BLOCK."),
    ] = None,
    limit: Annotated[
        int, typer.Option("--limit", help="Maximum records to return.")
    ] = 100,
    offset: Annotated[
        int, typer.Option("--offset", help="Skip this many records before paging.")
    ] = 0,
    format: Annotated[
        str, typer.Option("--format", help="Output format: text, json, csv.")
    ] = "text",
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", help="Write the report to this file instead of stdout."
        ),
    ] = None,
    patterns_file: Annotated[
        Path | None,
        typer.Option(
            "--patterns-file",
            help="Pattern file to record in the export manifest (with --manifest).",
        ),
    ] = None,
    policy_file: Annotated[
        Path | None,
        typer.Option(
            "--policy-file",
            help="Policy file to record in the export manifest (with --manifest).",
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option(
            "--manifest",
            help="Write an export manifest (pattern/policy digests, versions)"
            " to this JSON file for reproducibility.",
        ),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Query and export provenance records."""
    config = _load_config(config_path)
    backend = _get_backend(config)

    from_dt = None
    to_dt = None
    if from_date:
        try:
            from_dt = datetime.fromisoformat(from_date)
            if from_dt.tzinfo is None:
                from_dt = from_dt.replace(tzinfo=timezone.utc)
            else:
                from_dt = from_dt.astimezone(timezone.utc)
        except ValueError as e:
            typer.echo(f"Invalid --from date: {e}", err=True)
            raise typer.Exit(code=1) from None
    if to_date:
        try:
            parsed_to = datetime.fromisoformat(to_date)
            if "T" not in to_date and " " not in to_date:
                parsed_to = datetime.combine(parsed_to.date(), time.max)
            if parsed_to.tzinfo is None:
                to_dt = parsed_to.replace(tzinfo=timezone.utc)
            else:
                to_dt = parsed_to.astimezone(timezone.utc)
        except ValueError as e:
            typer.echo(f"Invalid --to date: {e}", err=True)
            raise typer.Exit(code=1) from None

    parsed_status: RecordStatus | None = None
    if status:
        try:
            parsed_status = RecordStatus(status.upper())
        except ValueError:
            typer.echo(
                f"Invalid --status {status!r}. Must be COMPLETED, BLOCKED, or ERROR.",
                err=True,
            )
            raise typer.Exit(code=1) from None

    parsed_pii_severity: PIISeverity | None = None
    if pii_severity:
        try:
            parsed_pii_severity = PIISeverity(pii_severity.upper())
        except ValueError:
            typer.echo(
                f"Invalid --pii-severity {pii_severity!r}."
                " Must be HIGH, MEDIUM, or LOW.",
                err=True,
            )
            raise typer.Exit(code=1) from None

    parsed_policy: PolicyAction | None = None
    if policy_decision:
        try:
            parsed_policy = PolicyAction(policy_decision.upper())
        except ValueError:
            typer.echo(
                f"Invalid --policy-decision {policy_decision!r}."
                " Must be ALLOW, WARN, or BLOCK.",
                err=True,
            )
            raise typer.Exit(code=1) from None

    filters = QueryFilters(
        model=model,
        user_id=user_id,
        app_id=app_id,
        feature_id=feature_id,
        status=parsed_status,
        pii_severity=parsed_pii_severity,
        policy_decision=parsed_policy,
        from_dt=from_dt,
        to_dt=to_dt,
        limit=limit,
        offset=offset,
    )

    exporter = AuditExporter(backend, secret_key=config.secret_key)
    audit_report = exporter.query(filters)

    if format == "json":
        payload = exporter.to_json(audit_report)
    elif format == "csv":
        payload = exporter.to_csv(audit_report)
    elif format == "text":
        lines = [
            f"total_count: {audit_report.total_count}    "
            f"generated_at: {audit_report.generated_at.isoformat()}"
        ]
        for r in audit_report.records:
            pii_count = r.pii_result.match_count if r.pii_result else 0
            policy_action = (
                r.policy_decision.action.value if r.policy_decision else "none"
            )
            lines.append(
                f"{r.timestamp.isoformat()} | {r.content_id[:8]}... | "
                f"{r.model} | {r.status.value} | pii:{pii_count} | "
                f"policy:{policy_action}"
            )
        payload = "\n".join(lines) + ("\n" if lines else "")
    else:
        typer.echo(
            f"Unknown format: {format!r}. Use 'text', 'json', or 'csv'.",
            err=True,
        )
        raise typer.Exit(code=1)

    if manifest is not None:
        try:
            export_manifest = build_export_manifest(
                pattern_files=([patterns_file] if patterns_file else []),
                policy_file=policy_file,
            )
        except FileNotFoundError as e:
            typer.echo(f"Manifest error: {e}", err=True)
            raise typer.Exit(code=1) from None
        manifest.write_text(
            json.dumps(export_manifest.to_dict(), indent=2), encoding="utf-8"
        )
        typer.echo(f"Manifest written to {manifest}", err=True)

    if output is not None:
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Report written to {output}")
        return

    typer.echo(payload, nl=False)


@app.command("migrate")
def migrate(
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Upgrade the configured database to the latest packaged schema."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    config = _load_config(config_path)
    package_dir = Path(__file__).resolve().parent.parent
    alembic_config = AlembicConfig(str(package_dir / "alembic.ini"))
    alembic_config.set_main_option("script_location", str(package_dir / "migrations"))
    alembic_config.set_main_option("sqlalchemy.url", config.database_url)
    command.upgrade(alembic_config, "head")
    typer.echo("Database schema upgraded to head.")


@app.command()
def evidence(
    content_id: Annotated[
        str,
        typer.Option("--content-id", help="The content_id to assemble evidence for."),
    ],
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Write the evidence pack JSON to this file."),
    ] = None,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Assemble a verifiable evidence pack for one record (compliance hand-off)."""
    config = _load_config(config_path)
    backend = _get_backend(config)
    exporter = AuditExporter(backend, secret_key=config.secret_key)

    try:
        pack = exporter.evidence_pack(content_id)
    except RecordNotFoundError as e:
        typer.echo(f"Evidence error: {e}", err=True)
        raise typer.Exit(code=1) from None

    payload = json.dumps(pack, indent=2, default=str)
    if output is not None:
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Evidence pack written to {output}")
        return
    typer.echo(payload, nl=False)


@app.command()
def scan(
    file: Annotated[
        Path | None,
        typer.Option(
            "--file",
            help="Path to the text file to scan ('-' or omitted reads stdin).",
        ),
    ] = None,
    extra_patterns_path: Annotated[
        Path | None,
        typer.Option(
            "--extra-patterns",
            help="Path to YAML file with custom PII patterns.",
        ),
    ] = None,
    json_output: Annotated[
        bool, typer.Option("--json", help="Output results as JSON.")
    ] = False,
    spacy: Annotated[
        bool,
        typer.Option(
            "--spacy",
            help="Also run spaCy NER (PERSON/ORG) if the nlp extra is installed.",
        ),
    ] = False,
    fail_on: Annotated[
        str | None,
        typer.Option(
            "--fail-on",
            help="Exit 1 when PII is found ('FOUND') or when the highest"
            " severity meets a threshold (LOW, MEDIUM, HIGH).",
        ),
    ] = None,
) -> None:
    """Scan text for PII patterns and report matches."""
    fail_on_upper: str | None = None
    if fail_on is not None:
        fail_on_upper = fail_on.upper()
        if fail_on_upper not in {"FOUND", "LOW", "MEDIUM", "HIGH"}:
            typer.echo(
                f"Invalid --fail-on {fail_on!r}."
                " Must be FOUND, LOW, MEDIUM, or HIGH.",
                err=True,
            )
            raise typer.Exit(code=1)

    text = _read_file_or_stdin(file)

    extra_patterns = None
    if extra_patterns_path:
        try:
            extra_patterns = load_patterns_from_yaml(extra_patterns_path)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Failed to load extra patterns: {e}", err=True)
            raise typer.Exit(code=1) from None

    matches = scan_text(text, extra_patterns=extra_patterns, use_spacy=spacy)
    fail_triggered = False
    if fail_on_upper is not None:
        fail_triggered = _fail_on_triggered(fail_on_upper, matches)

    source = "<stdin>" if (file is None or str(file) == "-") else file.name

    if json_output:
        highest = (
            max((m.severity for m in matches), key=lambda s: SEVERITY_RANK[s])
            if matches
            else None
        )
        typer.echo(
            json.dumps(
                {
                    "source": source,
                    "match_count": len(matches),
                    "highest_severity": highest.value if highest else None,
                    "fail_on": fail_on_upper,
                    "fail_triggered": fail_triggered,
                    "matches": [m.model_dump(mode="json") for m in matches],
                },
                indent=2,
            )
        )
    elif not matches:
        typer.echo("No PII detected.")
    else:
        typer.echo(f"PII scan results for: {source}")
        typer.echo(f"Found {len(matches)} match(es):")
        typer.echo("")
        for m in matches:
            typer.echo(
                f"  [{m.severity.value:<6}] {m.pattern_name:<12} "
                f"position {m.start}-{m.end}   {m.redacted_snippet}"
            )

    if fail_triggered:
        raise typer.Exit(code=1)


def _fail_on_triggered(fail_on: str, matches: list[PIIMatch]) -> bool:
    if fail_on == "FOUND":
        return len(matches) > 0
    if not matches:
        return False
    highest = max(SEVERITY_RANK[m.severity] for m in matches)
    return highest >= SEVERITY_RANK[PIISeverity(fail_on)]


@keys_app.command("rotate")
def keys_rotate(
    old_key_env: Annotated[
        str,
        typer.Option(
            "--old-key-env",
            help="Name of the environment variable holding the current secret key.",
        ),
    ] = "AISTAMP_SECRET_KEY",
    new_key_env: Annotated[
        str,
        typer.Option(
            "--new-key-env",
            help="Name of the environment variable holding the new secret key.",
        ),
    ] = "AISTAMP_NEW_SECRET_KEY",
    key_id: Annotated[
        str,
        typer.Option("--key-id", help="Identifier to assign to the new key."),
    ] = "default",
    re_sign: Annotated[
        bool,
        typer.Option(
            "--re-sign",
            help="Re-sign existing records with the new key where supported.",
        ),
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Rotate the signing secret key (keys are passed via environment variables)."""
    old_key = os.environ.get(old_key_env)
    if not old_key:
        typer.echo(
            f"Environment variable {old_key_env!r} is not set.", err=True
        )
        raise typer.Exit(code=1)
    new_key = os.environ.get(new_key_env)
    if not new_key:
        typer.echo(
            f"Environment variable {new_key_env!r} is not set.", err=True
        )
        raise typer.Exit(code=1)

    try:
        from aistamp.keys import rotate_secret
    except ImportError:
        typer.echo(
            "aistamp.keys.rotate_secret is not available in this build;"
            " key rotation lands with the v0.2.0 crypto integration.",
            err=True,
        )
        raise typer.Exit(code=1) from None

    config = _load_config(config_path)
    backend = _get_backend(config)
    rotate_secret(
        old_key=old_key,
        new_key=new_key,
        new_key_id=key_id,
        backend=backend,
        re_sign=re_sign,
    )
    typer.echo(f"Secret key rotated. New key_id: {key_id}")


@retention_app.command("enforce")
def retention_enforce(
    older_than_days: Annotated[
        int | None,
        typer.Option(
            "--older-than-days",
            help="Delete provenance records older than this many days.",
        ),
    ] = None,
    app_id: Annotated[
        str | None,
        typer.Option("--app-id", help="Restrict enforcement to one app ID."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Report how many records would be deleted without deleting.",
        ),
    ] = False,
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Delete provenance records past their retention window."""
    if older_than_days is None:
        typer.echo(
            "--older-than-days is required (e.g. --older-than-days 90).", err=True
        )
        raise typer.Exit(code=1)
    config = _load_config(config_path)
    try:
        deleted = enforce_retention(
            database_url=config.database_url,
            older_than_days=older_than_days,
            app_id=app_id,
            dry_run=dry_run,
        )
    except ValueError as e:
        typer.echo(f"Retention error: {e}", err=True)
        raise typer.Exit(code=1) from None

    scope = f" for app {app_id!r}" if app_id else ""
    if dry_run:
        typer.echo(
            f"Retention (dry-run): would delete {deleted} record(s){scope}"
            f" older than {older_than_days} days."
        )
    else:
        typer.echo(
            f"Retention: deleted {deleted} record(s){scope}"
            f" older than {older_than_days} days."
        )


@config_app.command("check")
def config_check(
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Validate and display the current configuration (credentials masked)."""
    config = _load_config(config_path)
    typer.echo("Configuration OK")
    typer.echo("")
    typer.echo(f"secret_key:   [SET - {len(config.secret_key)} characters]")
    typer.echo(f"database_url: {_masked_database_url(config.database_url)}")
    typer.echo(f"log_level:    {config.log_level}")


def _masked_database_url(database_url: str) -> str:
    """Render the database URL with any password hidden."""
    try:
        return make_url(database_url).render_as_string(hide_password=True)
    except ArgumentError:
        return "[unparseable database_url — credentials may be present]"
