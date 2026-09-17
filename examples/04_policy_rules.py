"""Policy rules from YAML — conditions, evaluation order, and all three actions.

``PolicyEngine.from_yaml`` loads the rule file; attach the engine to a
``ProvenanceClient`` and every call is evaluated twice: pre-call (before the
provider sees anything) and post-call (on the finished record).

This script demonstrates, using ``examples/policy_rules.yaml``:

1. **ALLOW** — a clean prompt on a standard-tier model matches the
   ``allow-standard-tier`` rule; the call proceeds.
2. **WARN** — a prompt with an SSN raises the record's highest PII severity
   to HIGH and matches ``warn-high-pii``; the call still proceeds, but the
   decision is returned on the result and persisted.
3. **BLOCK (pre-call)** — a call with ``model="experimental-9"`` matches
   ``block-experimental-tier`` before dispatch: ``stamp()`` raises
   ``PolicyViolationError``, the provider is never invoked, and a BLOCKED
   record is persisted for the audit trail.

Rules are evaluated in order and the FIRST match wins — that is why the
BLOCK rule is listed before the catch-all ALLOW.

Run from the repo root:

    python examples/04_policy_rules.py

Expected output:

    [1] clean call on standard tier
        decision  : ALLOW (rule: allow-standard-tier)
        status    : COMPLETED
    [2] SSN in prompt on standard tier
        decision  : WARN (rule: warn-high-pii)
        status    : COMPLETED
    [3] experimental-tier call
        BLOCKED pre-call: PolicyViolationError — rule 'block-experimental-tier'
        provider calls  : 2   (call 3 never dispatched)
        persisted status: BLOCKED

The provider is a fake callable, so nothing here touches the network.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from pydantic import SecretStr

from aistamp import (
    Config,
    PolicyEngine,
    PolicyViolationError,
    ProvenanceClient,
    QueryFilters,
    RecordStatus,
    SQLiteBackend,
)
from aistamp.client import StampResult

# Demo-only. Load the real key from your secret store in production.
SECRET_KEY = SecretStr("demo-secret-key-change-me-0123456789abcdef")

POLICY_PATH = Path(__file__).with_name("policy_rules.yaml")


def describe(action: str, rule_name: str | None) -> str:
    return f"{action} (rule: {rule_name})"


def print_decision(result: StampResult) -> None:
    """Print the policy decision recorded on a stamp result."""
    decision = result.record.policy_decision
    if decision is None:
        print("    decision  : NONE")
        return
    print(f"    decision  : {describe(decision.action.value, decision.rule_name)}")


def main() -> None:
    db_dir = Path(tempfile.mkdtemp(prefix="aistamp-04-"))
    config = Config(
        secret_key=SECRET_KEY,
        database_url=f"sqlite:///{db_dir / 'policy.db'}",
        # Policy WARN/BLOCK events also log via the `aistamp.policy` logger;
        # ERROR keeps this demo's expected output on stdout only.
        log_level="ERROR",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    engine = PolicyEngine.from_yaml(POLICY_PATH)
    provider_calls: list[str] = []

    def counting_llm(prompt: str) -> str:
        provider_calls.append(prompt)
        return f"here is what you asked for: {prompt}"

    client = ProvenanceClient(
        counting_llm,
        config=config,
        app_id="cookbook",
        feature_id="policy",
        user_id="demo_user",
        engine=engine,
        backend=backend,
    )

    # [1] ALLOW: clean prompt, standard-tier model.
    result = client.stamp("summarize Q3 revenue", model="gpt-4o-mini")
    print("[1] clean call on standard tier")
    print_decision(result)
    print(f"    status    : {result.record.status.value}")

    # [2] WARN: SSN in the prompt -> highest PII severity HIGH -> warn rule.
    result = client.stamp("my SSN is 123-45-6789", model="gpt-4o-mini")
    print("[2] SSN in prompt on standard tier")
    print_decision(result)
    print(f"    status    : {result.record.status.value}")

    # [3] BLOCK: experimental-tier model is blocked BEFORE the provider call.
    print("[3] experimental-tier call")
    try:
        client.stamp("anything at all", model="experimental-9")
    except PolicyViolationError as exc:
        print(f"    BLOCKED pre-call: PolicyViolationError — rule {exc.rule_name!r}")

    print(f"    provider calls  : {len(provider_calls)}   (call 3 never dispatched)")

    # The blocked attempt is still on the audit trail.
    report = backend.query(QueryFilters(status=RecordStatus.BLOCKED))
    blocked = report.records
    print(f"    persisted status: {blocked[0].status.value if blocked else 'NONE'}")

    if len(provider_calls) != 2 or not blocked:
        raise RuntimeError("policy demo did not behave as documented")


if __name__ == "__main__":
    main()
