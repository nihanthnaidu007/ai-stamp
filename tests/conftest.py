from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aistamp.client import ProvenanceClient
from aistamp.config import Config
from aistamp.fingerprint import sign_record
from aistamp.models import (
    PIIMatch,
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.policy import PolicyEngine, RuleConditions, RuleConfig
from aistamp.store import SQLiteBackend


@pytest.fixture
def sample_config() -> Config:
    return Config(
        secret_key="test-secret-key-for-aistamp-unit-tests-32chars",
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
    )


@pytest.fixture
def sqlite_backend(sample_config: Config) -> SQLiteBackend:
    backend = SQLiteBackend(sample_config.database_url)
    backend.create_tables()
    return backend


@pytest.fixture
def sample_pii_match() -> PIIMatch:
    return PIIMatch(
        pattern_name="EMAIL",
        severity=PIISeverity.MEDIUM,
        start=10,
        end=28,
        redacted_snippet="Contact: [REDACTED] for help",
    )


@pytest.fixture
def sample_pii_result(sample_pii_match: PIIMatch) -> PIIResult:
    return PIIResult(
        prompt_matches=[sample_pii_match],
        response_matches=[],
        highest_severity=PIISeverity.MEDIUM,
        match_count=1,
    )


@pytest.fixture
def sample_policy_decision() -> PolicyDecision:
    return PolicyDecision(
        action=PolicyAction.WARN,
        rule_name="warn_on_medium_pii",
        reason="PII severity MEDIUM detected in prompt",
    )


@pytest.fixture
def sample_provenance_record(
    sample_pii_result: PIIResult,
    sample_policy_decision: PolicyDecision,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=str(uuid.uuid4()),
        app_id="test_app",
        feature_id="test_feature",
        user_id="user_001",
        model="gpt-4o",
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        prompt_tokens=120,
        response_tokens=340,
        latency_ms=812.5,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=sample_pii_result,
        policy_decision=sample_policy_decision,
    )


@pytest.fixture
def custom_yaml_patterns_file(tmp_path: Path) -> Path:
    yaml_content = (
        "patterns:\n"
        "  - name: EMPLOYEE_ID\n"
        '    pattern: "EMP-\\\\d{6}"\n'
        "    severity: HIGH\n"
        '    description: "Internal employee ID"\n'
        "  - name: PROJECT_CODE\n"
        '    pattern: "PROJ-[A-Z]{3}-\\\\d{4}"\n'
        "    severity: MEDIUM\n"
        '    description: "Internal project code"\n'
    )
    path = tmp_path / "custom_patterns.yaml"
    path.write_text(yaml_content)
    return path


@pytest.fixture
def signed_record_in_store(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
    sample_config: Config,
) -> dict:
    hmac = sign_record(sample_provenance_record, sample_config.secret_key)
    sqlite_backend.write(sample_provenance_record, hmac)
    return {"record": sample_provenance_record, "hmac": hmac}


@pytest.fixture
def policy_yaml_file(tmp_path: Path) -> Path:
    yaml_content = (
        "model_tiers:\n"
        "  gpt-4o: approved\n"
        "  gpt-3.5-turbo: approved\n"
        "  claude-3-5-sonnet: approved\n"
        "  test-experimental-model: experimental\n"
        "\n"
        "rules:\n"
        "  - name: block_experimental_with_high_pii\n"
        "    conditions:\n"
        "      model_tier: experimental\n"
        "      pii_severity: HIGH\n"
        "    action: BLOCK\n"
        "\n"
        "  - name: warn_on_medium_pii\n"
        "    conditions:\n"
        "      pii_severity: MEDIUM\n"
        "    action: WARN\n"
        "\n"
        "  - name: warn_experimental_any_pii\n"
        "    conditions:\n"
        "      model_tier: experimental\n"
        "      pii_severity: LOW\n"
        "    action: WARN\n"
    )
    path = tmp_path / "policy.yaml"
    path.write_text(yaml_content)
    return path


@pytest.fixture
def default_engine() -> PolicyEngine:
    return PolicyEngine(
        rules=[
            RuleConfig(
                name="block_experimental_with_high_pii",
                conditions=RuleConditions(
                    model_tier="experimental",
                    pii_severity=PIISeverity.HIGH,
                ),
                action=PolicyAction.BLOCK,
            ),
            RuleConfig(
                name="warn_on_medium_pii",
                conditions=RuleConditions(pii_severity=PIISeverity.MEDIUM),
                action=PolicyAction.WARN,
            ),
            RuleConfig(
                name="warn_experimental_any_pii",
                conditions=RuleConditions(
                    model_tier="experimental",
                    pii_severity=PIISeverity.LOW,
                ),
                action=PolicyAction.WARN,
            ),
        ],
        model_tiers={
            "gpt-4o": "approved",
            "gpt-3.5-turbo": "approved",
            "claude-3-5-sonnet": "approved",
            "test-experimental-model": "experimental",
        },
    )


@pytest.fixture
def approved_record(sample_provenance_record: ProvenanceRecord) -> ProvenanceRecord:
    base = sample_provenance_record.model_dump()
    return ProvenanceRecord(
        **{**base, "model": "gpt-4o", "pii_result": None, "policy_decision": None}
    )


@pytest.fixture
def experimental_high_pii_record(
    sample_provenance_record: ProvenanceRecord,
    sample_pii_result: PIIResult,
) -> ProvenanceRecord:
    if sample_pii_result.highest_severity != PIISeverity.HIGH:
        pii = PIIResult(
            prompt_matches=sample_pii_result.prompt_matches,
            response_matches=sample_pii_result.response_matches,
            highest_severity=PIISeverity.HIGH,
            match_count=sample_pii_result.match_count,
        )
    else:
        pii = sample_pii_result

    base = sample_provenance_record.model_dump()
    return ProvenanceRecord(
        **{
            **base,
            "model": "test-experimental-model",
            "pii_result": pii,
            "policy_decision": None,
        }
    )


@pytest.fixture
def mock_llm_client():
    def _client(prompt: str) -> str:
        return f"Mock response to: {prompt[:30]}"

    return _client


@pytest.fixture
def mock_llm_client_with_pii():
    def _client(prompt: str) -> str:
        return (
            "You can reach our support at support@internal-company.com for assistance."
        )

    return _client


@pytest.fixture
def stamp_config():
    return Config(
        secret_key="stamp-test-secret-key-minimum-32-chars!!",
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
    )


@pytest.fixture
def provenance_client(mock_llm_client, stamp_config):
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    return ProvenanceClient(
        mock_llm_client,
        config=stamp_config,
        app_id="test_app",
        feature_id="test_feature",
        user_id="test_user",
        backend=backend,
    )
