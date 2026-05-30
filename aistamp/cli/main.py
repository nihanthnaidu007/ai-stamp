from __future__ import annotations

import json
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Annotated

import typer

from aistamp.audit import AuditExporter
from aistamp.config import Config
from aistamp.fingerprint import RecordNotFoundError, verify_record
from aistamp.models import PIISeverity, PolicyAction, QueryFilters, RecordStatus
from aistamp.pii import load_patterns_from_yaml, scan_text
from aistamp.store import SQLiteBackend

# TODO(phase-v2): add rich formatting for better CLI UX

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

    record, _hmac = result

    if format == "json":
        exporter = AuditExporter(backend)
        typer.echo(json.dumps(exporter._record_to_dict(record), indent=2, default=str))
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
        return

    typer.echo(f"Unknown format: {format!r}. Use 'text' or 'json'.", err=True)
    raise typer.Exit(code=1)


@app.command()
def verify(
    content_id: Annotated[
        str, typer.Option("--content-id", help="The content_id to verify against.")
    ],
    text: Annotated[str, typer.Option("--text", help="The current text to verify.")],
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Verify whether content matches its stored provenance record."""
    config = _load_config(config_path)
    backend = _get_backend(config)

    try:
        result = verify_record(content_id, text, backend, config.secret_key)
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
    format: Annotated[
        str, typer.Option("--format", help="Output format: text, json, csv.")
    ] = "text",
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
        status=parsed_status,
        pii_severity=parsed_pii_severity,
        policy_decision=parsed_policy,
        from_dt=from_dt,
        to_dt=to_dt,
        limit=limit,
    )

    exporter = AuditExporter(backend)
    audit_report = exporter.query(filters)

    if format == "json":
        typer.echo(exporter.to_json(audit_report))
        return
    if format == "csv":
        typer.echo(exporter.to_csv(audit_report))
        return
    if format == "text":
        typer.echo(
            f"total_count: {audit_report.total_count}    "
            f"generated_at: {audit_report.generated_at.isoformat()}"
        )
        for r in audit_report.records:
            pii_count = r.pii_result.match_count if r.pii_result else 0
            policy_action = (
                r.policy_decision.action.value if r.policy_decision else "none"
            )
            typer.echo(
                f"{r.timestamp.isoformat()} | {r.content_id[:8]}... | "
                f"{r.model} | {r.status.value} | pii:{pii_count} | "
                f"policy:{policy_action}"
            )
        return

    typer.echo(
        f"Unknown format: {format!r}. Use 'text', 'json', or 'csv'.",
        err=True,
    )
    raise typer.Exit(code=1)


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
def scan(
    file: Annotated[
        Path, typer.Option("--file", help="Path to the text file to scan.")
    ],
    extra_patterns_path: Annotated[
        Path | None,
        typer.Option(
            "--extra-patterns",
            help="Path to YAML file with custom PII patterns.",
        ),
    ] = None,
) -> None:
    """Scan a text file for PII patterns and report matches."""
    if not file.exists():
        typer.echo(f"File not found: {file}", err=True)
        raise typer.Exit(code=1)

    text = file.read_text(encoding="utf-8")

    extra_patterns = None
    if extra_patterns_path:
        try:
            extra_patterns = load_patterns_from_yaml(extra_patterns_path)
        except (FileNotFoundError, ValueError) as e:
            typer.echo(f"Failed to load extra patterns: {e}", err=True)
            raise typer.Exit(code=1) from None

    matches = scan_text(text, extra_patterns=extra_patterns)

    if not matches:
        typer.echo("No PII detected.")
        raise typer.Exit(code=0)

    typer.echo(f"PII scan results for: {file.name}")
    typer.echo(f"Found {len(matches)} match(es):")
    typer.echo("")
    for m in matches:
        typer.echo(
            f"  [{m.severity.value:<6}] {m.pattern_name:<12} "
            f"position {m.start}-{m.end}   {m.redacted_snippet}"
        )


@config_app.command("check")
def config_check(
    config_path: Annotated[
        Path | None, typer.Option("--config", help="Path to YAML config file.")
    ] = None,
) -> None:
    """Validate and display the current configuration."""
    config = _load_config(config_path)
    typer.echo("Configuration OK")
    typer.echo("")
    typer.echo(f"secret_key:   [SET - {len(config.secret_key)} characters]")
    typer.echo(f"database_url: {config.database_url}")
    typer.echo(f"log_level:    {config.log_level}")
