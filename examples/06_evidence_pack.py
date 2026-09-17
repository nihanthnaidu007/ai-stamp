"""Compliance evidence pack export — records, verdicts, and a hashed manifest.

An evidence pack answers an auditor's question in one artifact: for each
provenance record, what did the system capture, what did policy decide, and
does the content still verify? This script assembles one from a live
stamping session:

1. Stamp three calls through a policy-enabled client (one ALLOW, one WARN
   from ``examples/policy_rules.yaml``, one clean call whose stored text we
   later verify against a *tampered* copy to get a drift verdict).
2. Export the queried audit trail with ``AuditExporter`` as JSON and CSV —
   the two formats auditors receive.
3. Build the pack itself: one entry per record with the policy decision,
   the PII summary, the stored response hash, and a verification verdict
   (``verified`` / ``drift``) from ``verify_record``.
4. Seal it with a manifest: SHA-256 of every exported file, so a reviewer
   can re-hash the directory and confirm nothing changed in transit.

Files are written to ``./evidence_export/`` under the current directory.

Run from the repo root:

    python examples/06_evidence_pack.py

Expected output (content_ids and hashes vary per run):

    stamped 3 record(s): 2 policy-reviewed, 1 drift demonstration
    -- exported files (evidence_export) --
      audit_report.json   sha256=9f2c3a1b8d4e...
      records.csv         sha256=41ba70e2c9d5...
      evidence_pack.json  sha256=c70e5f219a83...
    -- evidence entries --
    record 4f9a1c2b...  status=COMPLETED policy=ALLOW pii=0 verdict=verified
    record b7e20d51...  status=COMPLETED policy=WARN  pii=2 verdict=verified
    record 0e8f2d6a...  status=COMPLETED policy=ALLOW pii=0 verdict=drift

The provider is a fake callable, so nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import SecretStr

from aistamp import (
    AuditExporter,
    Config,
    PolicyEngine,
    ProvenanceClient,
    ProvenanceRecord,
    QueryFilters,
    SQLiteBackend,
    VerificationResult,
    verify_record,
)

# Demo-only. Load the real key from your secret store in production.
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")

POLICY_PATH = Path(__file__).with_name("policy_rules.yaml")
EXPORT_DIR = Path("evidence_export")


def fake_llm(prompt: str) -> str:
    """Fake provider: echoes the prompt, no network."""
    return f"generated in response to: {prompt}"


@dataclass(frozen=True)
class EvidenceEntry:
    """One record's compliance snapshot plus its verification verdict."""

    content_id: str
    model: str
    status: str
    policy_action: str
    policy_rule: str
    pii_match_count: int
    pii_highest_severity: str
    response_hash: str
    verdict: str


def build_entry(
    record: ProvenanceRecord, verdict: VerificationResult
) -> EvidenceEntry:
    """Derive the evidence entry from a stored record + verification."""
    decision = record.policy_decision
    pii = record.pii_result
    return EvidenceEntry(
        content_id=record.content_id,
        model=record.model,
        status=record.status.value,
        policy_action=decision.action.value if decision else "NONE",
        policy_rule=decision.rule_name if decision and decision.rule_name else "-",
        pii_match_count=pii.match_count if pii else 0,
        pii_highest_severity=(
            pii.highest_severity.value if pii and pii.highest_severity else "NONE"
        ),
        response_hash=record.response_hash or "",
        # Hash match + valid HMAC is the only state an auditor should
        # accept; anything else is drift.
        verdict="verified" if verdict.verified else "drift",
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    db_dir = Path(tempfile.mkdtemp(prefix="aistamp-06-"))
    config = Config(
        secret_key=SECRET_KEY,
        database_url=f"sqlite:///{db_dir / 'evidence.db'}",
        # Policy WARN events also log via the `aistamp.policy` logger; ERROR
        # keeps this demo's expected output on stdout only.
        log_level="ERROR",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    engine = PolicyEngine.from_yaml(POLICY_PATH)
    client = ProvenanceClient(
        fake_llm,
        config=config,
        app_id="cookbook",
        feature_id="evidence",
        user_id="demo_user",
        engine=engine,
        backend=backend,
    )

    stamped = [
        client.stamp("summarize the release notes", model="gpt-4o-mini"),
        # SSN in the prompt -> HIGH severity -> warn-high-pii rule.
        client.stamp("patient SSN 123-45-6789 intake", model="gpt-4o-mini"),
        client.stamp("list the integration steps", model="gpt-4o-mini"),
    ]
    print(f"stamped {len(stamped)} record(s): 2 policy-reviewed, 1 drift demonstration")

    # Verify each stored record against its own text — except the third,
    # which we verify against a tampered copy to show a drift verdict.
    third = stamped[2]
    verdicts = [
        verify_record(result.content_id, result.text, backend, config.secret_key)
        for result in stamped[:2]
    ] + [
        verify_record(
            third.content_id,
            third.text + " (edited after the fact)",
            backend,
            config.secret_key,
        )
    ]
    entries = [
        build_entry(result.record, verdict)
        for result, verdict in zip(stamped, verdicts, strict=True)
    ]

    # Audit trail export — the two formats auditors receive.
    exporter = AuditExporter(backend)
    report = exporter.query(QueryFilters(app_id="cookbook"))
    EXPORT_DIR.mkdir(exist_ok=True)
    json_path = EXPORT_DIR / "audit_report.json"
    csv_path = EXPORT_DIR / "records.csv"
    json_path.write_text(exporter.to_json(report), encoding="utf-8")
    csv_path.write_text(exporter.to_csv(report), encoding="utf-8", newline="")

    # The pack itself, sealed with a manifest of file hashes.
    pack = {
        "evidence_pack_version": 1,
        "generated_for": "aistamp examples cookbook",
        "filters_applied": report.filters_applied,
        "entries": [entry.__dict__ for entry in entries],
        "manifest": {
            "audit_report.json": sha256_of(json_path),
            "records.csv": sha256_of(csv_path),
        },
    }
    pack_path = EXPORT_DIR / "evidence_pack.json"
    pack_path.write_text(json.dumps(pack, indent=2), encoding="utf-8")

    print(f"-- exported files ({EXPORT_DIR}) --")
    for path in (json_path, csv_path, pack_path):
        print(f"  {path.name:<20} sha256={sha256_of(path)[:12]}...")
    print("-- evidence entries --")
    for entry in entries:
        print(
            f"record {entry.content_id[:8]}...  status={entry.status}"
            f" policy={entry.policy_action} pii={entry.pii_match_count}"
            f" verdict={entry.verdict}"
        )

    if len(entries) != 3 or entries[-1].verdict != "drift":
        raise RuntimeError("evidence pack demo did not behave as documented")


if __name__ == "__main__":
    main()
