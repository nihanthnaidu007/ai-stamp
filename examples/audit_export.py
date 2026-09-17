"""Query the provenance store and export the audit trail as JSON and CSV.

python examples/audit_export.py
"""

from __future__ import annotations

import json
from pathlib import Path

from aistamp import (
    AuditExporter,
    Config,
    ProvenanceClient,
    QueryFilters,
    SQLiteBackend,
)

SECRET_KEY = "replace-me-with-a-random-32+-character-secret"
EXPORT_DIR = Path("audit_exports")


def compliant_llm(prompt: str) -> str:
    return f"(compliant model) Answering: {prompt}"


def main() -> None:
    config = Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",
        log_level="INFO",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    client = ProvenanceClient(
        compliant_llm,
        config=config,
        app_id="demo_app",
        feature_id="audit_demo",
        user_id="demo_user",
        backend=backend,
    )
    client.chat("First audited request")
    client.chat("Second audited request")

    exporter = AuditExporter(backend)
    report = exporter.query(QueryFilters())

    EXPORT_DIR.mkdir(exist_ok=True)
    json_path = EXPORT_DIR / "audit.json"
    csv_path = EXPORT_DIR / "audit.csv"
    json_path.write_text(exporter.to_json(report))
    csv_path.write_text(exporter.to_csv(report))

    print(f"Exported {report.total_count} records:")
    print(f"  {json_path} ({json_path.stat().st_size} bytes)")
    print(f"  {csv_path} ({csv_path.stat().st_size} bytes)")

    # The JSON structure, for downstream compliance tooling:
    parsed = json.loads(json_path.read_text())
    print(f"Keys per export: {sorted(parsed.keys())}")


if __name__ == "__main__":
    main()
