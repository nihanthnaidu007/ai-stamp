from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

import aistamp
from aistamp.config import Config
from aistamp.errors import ConfigError
from aistamp.models import (
    AuditReport,
    PIIMatch,
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    QueryFilters,
    RecordStatus,
    VerificationResult,
)
from aistamp.store import SQLiteBackend

# ---------------------------------------------------------------------------
# Group 1 — Config tests
# ---------------------------------------------------------------------------


def test_config_from_env_loads_correctly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Config.from_env() reads the secret key and applies defaults.
    monkeypatch.setenv("AISTAMP_SECRET_KEY", "a" * 32)
    monkeypatch.delenv("AISTAMP_DATABASE_URL", raising=False)
    monkeypatch.delenv("AISTAMP_LOG_LEVEL", raising=False)
    config = Config.from_env()
    # v0.2: secret_key is a SecretStr; secret_key_value exposes the plaintext.
    assert config.secret_key_value == "a" * 32
    assert "a" * 32 not in repr(config.secret_key)
    assert config.database_url == "sqlite:///./aistamp.db"
    assert config.log_level == "INFO"


def test_config_from_env_raises_on_missing_secret_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # v0.2: from_env() raises a friendly ConfigError (not a bare KeyError)
    # when AISTAMP_SECRET_KEY is not set.
    monkeypatch.delenv("AISTAMP_SECRET_KEY", raising=False)
    with pytest.raises(ConfigError):
        Config.from_env()


def test_config_from_yaml_loads_correctly(tmp_path) -> None:
    # from_yaml() must read a YAML file and construct a valid Config.
    # Use explicit double-quoted scalars so the YAML is unambiguous even for
    # values containing multiple colons (e.g. "sqlite:///:memory:").
    yaml_path = tmp_path / "config.yaml"
    yaml_content = (
        'secret_key: "' + "b" * 32 + '"\n'
        'database_url: "sqlite:///:memory:"\n'
        'log_level: "WARNING"\n'
    )
    yaml_path.write_text(yaml_content)
    config = Config.from_yaml(yaml_path)
    assert config.secret_key_value == "b" * 32
    assert config.database_url == "sqlite:///:memory:"
    assert config.log_level == "WARNING"


def test_config_rejects_short_secret_key() -> None:
    # secret_key under 32 characters must raise ValidationError.
    with pytest.raises(ValidationError):
        Config(secret_key="short", database_url="sqlite:///:memory:")


def test_config_rejects_invalid_log_level() -> None:
    # log_level must be one of DEBUG, INFO, WARNING, ERROR.
    with pytest.raises(ValidationError):
        Config(secret_key="a" * 32, log_level="VERBOSE")


def test_config_is_frozen() -> None:
    # Config is frozen. Setting an attribute must raise ValidationError.
    config = Config(secret_key="a" * 32)
    with pytest.raises(ValidationError):
        config.log_level = "DEBUG"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Group 2 — Model instantiation tests
# ---------------------------------------------------------------------------


def test_pii_match_instantiates() -> None:
    # PIIMatch must construct from valid fields without error.
    m = PIIMatch(
        pattern_name="EMAIL",
        severity=PIISeverity.LOW,
        start=0,
        end=5,
        redacted_snippet="[REDACTED]",
    )
    assert m.pattern_name == "EMAIL"


def test_pii_result_instantiates() -> None:
    # PIIResult must construct from valid fields without error.
    r = PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=None,
        match_count=0,
    )
    assert r.match_count == 0


def test_policy_decision_instantiates() -> None:
    # PolicyDecision must construct from valid fields without error.
    d = PolicyDecision(action=PolicyAction.ALLOW, rule_name=None, reason=None)
    assert d.action == PolicyAction.ALLOW


def test_provenance_record_instantiates(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # ProvenanceRecord must construct from valid fields without error.
    assert sample_provenance_record.model == "gpt-4o"


def test_verification_result_instantiates() -> None:
    # VerificationResult must construct from valid fields without error.
    v = VerificationResult(
        content_id="abc",
        verified=True,
        hash_match=True,
        hmac_valid=True,
        drift_detected=False,
        original_hash="a" * 64,
        current_hash="a" * 64,
    )
    assert v.verified is True


def test_audit_report_instantiates() -> None:
    # AuditReport must construct from valid fields without error.
    r = AuditReport(
        records=[],
        total_count=0,
        generated_at=datetime.now(timezone.utc),
        filters_applied={},
    )
    assert r.total_count == 0


# ---------------------------------------------------------------------------
# Group 3 — Frozen model tests
# ---------------------------------------------------------------------------


def test_provenance_record_is_frozen(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # ProvenanceRecord is write-once. Mutation must raise ValidationError.
    with pytest.raises(ValidationError):
        sample_provenance_record.content_id = "new"  # type: ignore[misc]


def test_pii_result_is_frozen(sample_pii_result: PIIResult) -> None:
    # PIIResult must also reject mutation.
    with pytest.raises(ValidationError):
        sample_pii_result.match_count = 99  # type: ignore[misc]


def test_config_is_frozen_via_assignment() -> None:
    # Config mutation must also raise. Tested separately from config group for clarity.
    config = Config(secret_key="z" * 32)
    with pytest.raises(ValidationError):
        config.database_url = "sqlite:///elsewhere.db"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Group 4 — Store write and read tests
# ---------------------------------------------------------------------------


def test_store_write_succeeds(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # write() must persist a record without raising.
    sqlite_backend.write(sample_provenance_record, hmac_signature="dummy_hmac")


def test_store_get_returns_correct_record(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # get() must return the same record that was written.
    sqlite_backend.write(sample_provenance_record, hmac_signature="dummy_hmac")
    result = sqlite_backend.get(sample_provenance_record.content_id)
    assert result is not None
    record, _ = result
    assert record.content_id == sample_provenance_record.content_id
    assert record.app_id == sample_provenance_record.app_id
    assert record.model == sample_provenance_record.model


def test_store_get_returns_hmac_signature(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # get() returns a tuple of (ProvenanceRecord, hmac_signature).
    sqlite_backend.write(sample_provenance_record, hmac_signature="test_hmac_value")
    result = sqlite_backend.get(sample_provenance_record.content_id)
    assert result is not None
    _, hmac = result
    assert hmac == "test_hmac_value"


def test_store_get_returns_none_for_unknown_id(
    sqlite_backend: SQLiteBackend,
) -> None:
    # get() must return None for a content_id that was never written.
    assert sqlite_backend.get("nonexistent-id") is None


def test_store_get_deserializes_pii_result(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # get() must deserialize pii_result to a PIIResult model.
    sqlite_backend.write(sample_provenance_record, hmac_signature="hmac")
    result = sqlite_backend.get(sample_provenance_record.content_id)
    assert result is not None
    record, _ = result
    assert isinstance(record.pii_result, PIIResult)


def test_store_get_deserializes_policy_decision(
    sqlite_backend: SQLiteBackend,
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # get() must deserialize policy_decision back to a PolicyDecision Pydantic model.
    sqlite_backend.write(sample_provenance_record, hmac_signature="hmac")
    result = sqlite_backend.get(sample_provenance_record.content_id)
    assert result is not None
    record, _ = result
    assert isinstance(record.policy_decision, PolicyDecision)


def _make_record(
    user_id: str = "u1", status: RecordStatus = RecordStatus.COMPLETED
) -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=str(uuid.uuid4()),
        app_id="app",
        feature_id="feat",
        user_id=user_id,
        model="gpt-4o",
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=100.0,
        timestamp=datetime.now(timezone.utc),
        status=status,
        pii_result=None,
        policy_decision=None,
    )


def test_store_query_returns_matching_records(sqlite_backend: SQLiteBackend) -> None:
    # query() with a user_id filter must return only records for that user.
    sqlite_backend.write(_make_record(user_id="alice"), hmac_signature="h")
    sqlite_backend.write(_make_record(user_id="bob"), hmac_signature="h")
    report = sqlite_backend.query(QueryFilters(user_id="alice"))
    assert len(report.records) == 1
    assert report.records[0].user_id == "alice"


def test_store_query_returns_empty_for_no_match(
    sqlite_backend: SQLiteBackend,
) -> None:
    # A query with no matching records returns an empty AuditReport.
    report = sqlite_backend.query(QueryFilters(user_id="nobody"))
    assert report.records == []
    assert report.total_count == 0


def test_store_query_respects_limit(sqlite_backend: SQLiteBackend) -> None:
    # query() must respect the limit field in QueryFilters.
    for _ in range(5):
        sqlite_backend.write(_make_record(user_id="multi"), hmac_signature="h")
    report = sqlite_backend.query(QueryFilters(user_id="multi", limit=2))
    assert len(report.records) == 2
    assert report.total_count == 5


def test_store_query_filters_by_status(sqlite_backend: SQLiteBackend) -> None:
    # query() with a status filter must return only records with that status.
    sqlite_backend.write(
        _make_record(status=RecordStatus.COMPLETED), hmac_signature="h"
    )
    sqlite_backend.write(_make_record(status=RecordStatus.ERROR), hmac_signature="h")
    report = sqlite_backend.query(QueryFilters(status=RecordStatus.COMPLETED))
    assert len(report.records) == 1
    assert report.records[0].status == RecordStatus.COMPLETED


# ---------------------------------------------------------------------------
# Group 5 — Package import tests
# ---------------------------------------------------------------------------


def test_package_imports_cleanly() -> None:
    # The top-level aistamp import must succeed and expose the public surface.
    assert hasattr(aistamp, "__version__")


def test_version_is_string() -> None:
    # __version__ must be a string.
    assert isinstance(aistamp.__version__, str)


def test_all_models_importable_from_top_level() -> None:
    # All models must be importable directly from aistamp.
    from aistamp import (  # noqa: F401
        PIIMatch,
        PIIResult,
        ProvenanceRecord,
        VerificationResult,
    )
