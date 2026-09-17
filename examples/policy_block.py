"""Enforce an LLM policy and see a BLOCK decision stop the call.

A PolicyEngine can evaluate model tier and PII severity before the prompt
leaves your process. When a BLOCK rule matches, the call raises
PolicyViolationError and the call is stamped as BLOCKED for the audit trail.

    python examples/policy_block.py
"""

from __future__ import annotations

from aistamp import (
    Config,
    PIISeverity,
    PolicyAction,
    PolicyEngine,
    PolicyViolationError,
    ProvenanceClient,
    QueryFilters,
    RuleConditions,
    RuleConfig,
    SQLiteBackend,
)

SECRET_KEY = "replace-me-with-a-random-32+-character-secret"


def experimental_llm(prompt: str) -> str:
    return "(experimental model) This call should have been blocked."


def main() -> None:
    config = Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",
        log_level="INFO",
    )
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()

    # Block experimental models the moment HIGH-severity PII is present.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                name="block_experimental_with_high_pii",
                conditions=RuleConditions(
                    model_tier="experimental",
                    pii_severity=PIISeverity.HIGH,
                ),
                action=PolicyAction.BLOCK,
            )
        ],
        model_tiers={"test-experimental-model": "experimental"},
    )

    client = ProvenanceClient(
        experimental_llm,
        config=config,
        app_id="demo_app",
        feature_id="policy_demo",
        user_id="demo_user",
        engine=engine,
        backend=backend,
    )

    try:
        # model tier for "test-experimental-model" resolves via model_tiers above
        client.chat(
            "My SSN is 123-45-6789 - what can you tell me about it?",
            model="test-experimental-model",
        )
    except PolicyViolationError as exc:
        print(f"Call blocked by policy: {exc}")

    report = backend.query(QueryFilters())
    for record in report.records:
        decision = record.policy_decision
        action = decision.action.value if decision else "n/a"
        rule = decision.rule_name if decision else "n/a"
        print(f"Record {record.content_id}: status={record.status.value}")
        print(f"  policy: action={action} rule={rule}")


if __name__ == "__main__":
    main()
