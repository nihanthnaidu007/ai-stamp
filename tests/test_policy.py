from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import datetime, timezone
from pathlib import Path

import pytest

from aistamp.fingerprint import generate_content_id
from aistamp.models import (
    PIIResult,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.policy import (
    PolicyEngine,
    PolicyViolationError,
    RuleConditions,
    RuleConfig,
)
from aistamp.store import SQLiteBackend


def _record(
    *,
    model: str = "gpt-4o",
    pii_result: PIIResult | None = None,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="a",
        feature_id="f",
        user_id="u",
        model=model,
        prompt_hash="a" * 64,
        response_hash="b" * 64,
        prompt_tokens=10,
        response_tokens=10,
        latency_ms=100.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=pii_result,
        policy_decision=None,
    )


def _pii(sev: PIISeverity | None, count: int = 1) -> PIIResult:
    return PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=sev,
        match_count=count,
    )


# ---------------------------------------------------------------------------
# Group 1 — RuleConditions and RuleConfig instantiation
# ---------------------------------------------------------------------------


def test_rule_conditions_all_none_is_valid() -> None:
    # RuleConditions with all fields None must instantiate without error.
    c = RuleConditions()
    assert c.model_tier is None
    assert c.pii_severity is None


def test_rule_conditions_is_frozen() -> None:
    # RuleConditions must reject mutation since it is a frozen dataclass.
    c = RuleConditions()
    with pytest.raises(FrozenInstanceError):
        c.model_tier = "approved"  # type: ignore[misc]


def test_rule_config_instantiates() -> None:
    # A RuleConfig with a name, conditions, and action must instantiate correctly.
    r = RuleConfig(name="r1", conditions=RuleConditions(), action=PolicyAction.ALLOW)
    assert r.name == "r1"


def test_rule_config_is_frozen() -> None:
    # RuleConfig must reject mutation.
    r = RuleConfig(name="r1", conditions=RuleConditions(), action=PolicyAction.ALLOW)
    with pytest.raises(FrozenInstanceError):
        r.name = "r2"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Group 2 — PolicyEngine with no rules
# ---------------------------------------------------------------------------


def test_engine_with_no_rules_returns_allow() -> None:
    # An engine with an empty rules list must return ALLOW for any record.
    decision = PolicyEngine(rules=[], model_tiers={}).evaluate(_record())
    assert decision.action == PolicyAction.ALLOW


def test_engine_no_rules_decision_has_no_rule_name() -> None:
    # The default ALLOW decision must have rule_name=None.
    decision = PolicyEngine(rules=[], model_tiers={}).evaluate(_record())
    assert decision.rule_name is None


def test_engine_no_rules_reason_mentions_default() -> None:
    # The reason string must mention "default" or "No matching rule".
    decision = PolicyEngine(rules=[], model_tiers={}).evaluate(_record())
    assert decision.reason is not None
    assert (
        "default" in decision.reason.lower()
        or "no matching rule" in decision.reason.lower()
    )


# ---------------------------------------------------------------------------
# Group 3 — ALLOW action
# ---------------------------------------------------------------------------


def test_explicit_allow_rule_returns_allow() -> None:
    # A rule with action=ALLOW that matches must return a ALLOW decision.
    engine = PolicyEngine(
        rules=[RuleConfig("allow_all", RuleConditions(), PolicyAction.ALLOW)],
        model_tiers={},
    )
    decision = engine.evaluate(_record())
    assert decision.action == PolicyAction.ALLOW


def test_allow_decision_has_correct_rule_name() -> None:
    # When a named ALLOW rule matches, decision.rule_name must equal that rule's name.
    engine = PolicyEngine(
        rules=[RuleConfig("named_allow", RuleConditions(), PolicyAction.ALLOW)],
        model_tiers={},
    )
    decision = engine.evaluate(_record())
    assert decision.rule_name == "named_allow"


def test_allow_does_not_raise() -> None:
    # evaluate() must not raise for an ALLOW outcome.
    engine = PolicyEngine(
        rules=[RuleConfig("allow", RuleConditions(), PolicyAction.ALLOW)],
        model_tiers={},
    )
    engine.evaluate(_record())


# ---------------------------------------------------------------------------
# Group 4 — WARN action
# ---------------------------------------------------------------------------


def test_warn_rule_returns_warn_decision(default_engine: PolicyEngine) -> None:
    # A record matching a WARN rule must return PolicyDecision with action=WARN.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    decision = default_engine.evaluate(record)
    assert decision.action == PolicyAction.WARN


def test_warn_decision_has_rule_name(default_engine: PolicyEngine) -> None:
    # The returned decision must carry the name of the matched rule.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    decision = default_engine.evaluate(record)
    assert decision.rule_name == "warn_on_medium_pii"


def test_warn_does_not_raise(default_engine: PolicyEngine) -> None:
    # evaluate() must not raise for a WARN outcome. It returns the decision.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    default_engine.evaluate(record)


def test_warn_decision_has_reason(default_engine: PolicyEngine) -> None:
    # The reason field on the decision must be a non-empty string.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    decision = default_engine.evaluate(record)
    assert decision.reason is not None
    assert len(decision.reason) > 0


# ---------------------------------------------------------------------------
# Group 5 — BLOCK action and PolicyViolationError
# ---------------------------------------------------------------------------


def test_block_rule_raises_policy_violation_error(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # A record matching a BLOCK rule must raise PolicyViolationError.
    with pytest.raises(PolicyViolationError):
        default_engine.evaluate(experimental_high_pii_record)


def test_policy_violation_error_has_rule_name(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # The raised exception must expose the rule_name that triggered it.
    with pytest.raises(PolicyViolationError) as exc_info:
        default_engine.evaluate(experimental_high_pii_record)
    assert exc_info.value.rule_name == "block_experimental_with_high_pii"


def test_policy_violation_error_has_decision(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # The raised exception must expose the full PolicyDecision as .decision.
    with pytest.raises(PolicyViolationError) as exc_info:
        default_engine.evaluate(experimental_high_pii_record)
    assert isinstance(exc_info.value.decision, PolicyDecision)


def test_policy_violation_error_decision_action_is_block(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # The PolicyDecision inside the exception must have action=BLOCK.
    with pytest.raises(PolicyViolationError) as exc_info:
        default_engine.evaluate(experimental_high_pii_record)
    assert exc_info.value.decision.action == PolicyAction.BLOCK


def test_policy_violation_error_has_content_id(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # The raised exception must expose the content_id from the evaluated record.
    with pytest.raises(PolicyViolationError) as exc_info:
        default_engine.evaluate(experimental_high_pii_record)
    assert exc_info.value.content_id == experimental_high_pii_record.content_id


def test_policy_violation_error_message_is_informative(
    default_engine: PolicyEngine,
    experimental_high_pii_record: ProvenanceRecord,
) -> None:
    # str(err) must contain the rule name and content_id.
    with pytest.raises(PolicyViolationError) as exc_info:
        default_engine.evaluate(experimental_high_pii_record)
    msg = str(exc_info.value)
    assert "block_experimental_with_high_pii" in msg
    assert experimental_high_pii_record.content_id in msg


# ---------------------------------------------------------------------------
# Group 6 — First matching rule wins
# ---------------------------------------------------------------------------


def test_first_matching_rule_wins_not_last() -> None:
    # When two rules both match, the first one in the list determines the outcome.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "warn_first",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.WARN,
            ),
            RuleConfig(
                "block_second",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.BLOCK,
            ),
        ],
        model_tiers={},
    )
    record = _record(pii_result=_pii(PIISeverity.HIGH))
    decision = engine.evaluate(record)
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "warn_first"


def test_non_matching_rule_is_skipped() -> None:
    # A rule that does not match must be skipped silently.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_experimental",
                RuleConditions(model_tier="experimental"),
                PolicyAction.BLOCK,
            ),
            RuleConfig(
                "warn_medium",
                RuleConditions(pii_severity=PIISeverity.MEDIUM),
                PolicyAction.WARN,
            ),
        ],
        model_tiers={"gpt-4o": "approved"},
    )
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    decision = engine.evaluate(record)
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "warn_medium"


def test_catch_all_rule_matches_everything() -> None:
    # A rule with RuleConditions() (all None) matches every record.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "specific",
                RuleConditions(model_tier="experimental"),
                PolicyAction.BLOCK,
            ),
            RuleConfig("catch_all_warn", RuleConditions(), PolicyAction.WARN),
        ],
        model_tiers={"gpt-4o": "approved"},
    )
    decision = engine.evaluate(_record(model="gpt-4o"))
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "catch_all_warn"


# ---------------------------------------------------------------------------
# Group 7 — model_tier condition
# ---------------------------------------------------------------------------


def test_model_tier_condition_matches_correctly(
    default_engine: PolicyEngine,
) -> None:
    # A rule with model_tier="experimental" must fire for experimental models.
    record = _record(model="test-experimental-model", pii_result=_pii(PIISeverity.LOW))
    decision = default_engine.evaluate(record)
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "warn_experimental_any_pii"


def test_model_tier_condition_does_not_fire_for_approved(
    default_engine: PolicyEngine,
) -> None:
    # The BLOCK rule must not fire for a model in the approved tier.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.HIGH))
    # block rule needs experimental+HIGH; gpt-4o is approved, so block must skip.
    # warn_on_medium_pii will still fire because HIGH >= MEDIUM.
    decision = default_engine.evaluate(record)
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "warn_on_medium_pii"


def test_model_tier_unknown_does_not_match_experimental() -> None:
    # A model not in the tier map resolves to "unknown" tier.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_experimental",
                RuleConditions(model_tier="experimental"),
                PolicyAction.BLOCK,
            ),
        ],
        model_tiers={},
    )
    record = _record(model="unmapped-model")
    decision = engine.evaluate(record)
    assert decision.action == PolicyAction.ALLOW


def test_model_tier_comparison_is_case_insensitive() -> None:
    # model_tier="Experimental" in conditions must match tier "experimental" in config.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "warn_exp",
                RuleConditions(model_tier="Experimental"),
                PolicyAction.WARN,
            ),
        ],
        model_tiers={"some-model": "experimental"},
    )
    record = _record(model="some-model")
    decision = engine.evaluate(record)
    assert decision.action == PolicyAction.WARN


# ---------------------------------------------------------------------------
# Group 8 — pii_severity condition (>= semantics)
# ---------------------------------------------------------------------------


def _eng_with_severity(
    threshold: PIISeverity, action: PolicyAction = PolicyAction.WARN
) -> PolicyEngine:
    return PolicyEngine(
        rules=[
            RuleConfig(
                "rule",
                RuleConditions(pii_severity=threshold),
                action,
            ),
        ],
        model_tiers={},
    )


def test_pii_severity_high_condition_fires_for_high() -> None:
    # A rule with pii_severity=HIGH must fire when highest_severity is HIGH.
    engine = _eng_with_severity(PIISeverity.HIGH)
    decision = engine.evaluate(_record(pii_result=_pii(PIISeverity.HIGH)))
    assert decision.action == PolicyAction.WARN


def test_pii_severity_medium_condition_fires_for_high() -> None:
    # A rule with pii_severity=MEDIUM must also fire when highest_severity is HIGH.
    engine = _eng_with_severity(PIISeverity.MEDIUM)
    decision = engine.evaluate(_record(pii_result=_pii(PIISeverity.HIGH)))
    assert decision.action == PolicyAction.WARN


def test_pii_severity_medium_condition_fires_for_medium() -> None:
    # A rule with pii_severity=MEDIUM must fire when highest_severity is MEDIUM.
    engine = _eng_with_severity(PIISeverity.MEDIUM)
    decision = engine.evaluate(_record(pii_result=_pii(PIISeverity.MEDIUM)))
    assert decision.action == PolicyAction.WARN


def test_pii_severity_medium_condition_does_not_fire_for_low() -> None:
    # A rule with pii_severity=MEDIUM must NOT fire when highest_severity is LOW.
    engine = _eng_with_severity(PIISeverity.MEDIUM)
    decision = engine.evaluate(_record(pii_result=_pii(PIISeverity.LOW)))
    assert decision.action == PolicyAction.ALLOW


def test_pii_severity_condition_does_not_fire_when_pii_result_is_none() -> None:
    # A rule with pii_severity condition must not fire when record.pii_result is None.
    engine = _eng_with_severity(PIISeverity.LOW)
    decision = engine.evaluate(_record(pii_result=None))
    assert decision.action == PolicyAction.ALLOW


def test_pii_severity_condition_does_not_fire_when_no_matches() -> None:
    # A rule with pii_severity condition must not fire when highest_severity is None.
    engine = _eng_with_severity(PIISeverity.LOW)
    empty_pii = PIIResult(
        prompt_matches=[],
        response_matches=[],
        highest_severity=None,
        match_count=0,
    )
    decision = engine.evaluate(_record(pii_result=empty_pii))
    assert decision.action == PolicyAction.ALLOW


# ---------------------------------------------------------------------------
# Group 9 — Combined conditions (AND logic)
# ---------------------------------------------------------------------------


def test_both_conditions_must_match() -> None:
    # A rule with model_tier=experimental AND pii_severity=HIGH must not fire
    # when only one condition matches.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_exp_high",
                RuleConditions(
                    model_tier="experimental",
                    pii_severity=PIISeverity.HIGH,
                ),
                PolicyAction.BLOCK,
            ),
        ],
        model_tiers={"exp-m": "experimental", "ok-m": "approved"},
    )
    # 1. experimental model + LOW pii → no fire
    d1 = engine.evaluate(_record(model="exp-m", pii_result=_pii(PIISeverity.LOW)))
    assert d1.action == PolicyAction.ALLOW
    # 2. approved model + HIGH pii → no fire
    d2 = engine.evaluate(_record(model="ok-m", pii_result=_pii(PIISeverity.HIGH)))
    assert d2.action == PolicyAction.ALLOW
    # 3. experimental + HIGH → fire
    with pytest.raises(PolicyViolationError):
        engine.evaluate(_record(model="exp-m", pii_result=_pii(PIISeverity.HIGH)))


def test_single_condition_rule_ignores_other_dimension() -> None:
    # A rule with only pii_severity set must fire regardless of model tier.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "warn_any_high",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.WARN,
            ),
        ],
        model_tiers={"approved-m": "approved"},
    )
    decision = engine.evaluate(
        _record(model="approved-m", pii_result=_pii(PIISeverity.HIGH))
    )
    assert decision.action == PolicyAction.WARN


# ---------------------------------------------------------------------------
# Group 10 — from_yaml loading
# ---------------------------------------------------------------------------


def test_from_yaml_returns_policy_engine(policy_yaml_file: Path) -> None:
    # PolicyEngine.from_yaml must return a PolicyEngine instance.
    assert isinstance(PolicyEngine.from_yaml(policy_yaml_file), PolicyEngine)


def test_from_yaml_loads_model_tiers(policy_yaml_file: Path) -> None:
    # Loaded engine must resolve tiers from YAML correctly.
    engine = PolicyEngine.from_yaml(policy_yaml_file)
    record = _record(model="test-experimental-model", pii_result=_pii(PIISeverity.HIGH))
    with pytest.raises(PolicyViolationError):
        engine.evaluate(record)


def test_from_yaml_loads_rules(policy_yaml_file: Path) -> None:
    # Loaded engine must have the correct number of rules.
    engine = PolicyEngine.from_yaml(policy_yaml_file)
    assert len(engine._rules) == 3  # type: ignore[attr-defined]


def test_from_yaml_raises_file_not_found(tmp_path: Path) -> None:
    # from_yaml must raise FileNotFoundError for a nonexistent path.
    with pytest.raises(FileNotFoundError):
        PolicyEngine.from_yaml(tmp_path / "missing.yaml")


def test_from_yaml_empty_rules_and_tiers_valid(tmp_path: Path) -> None:
    # A YAML with empty rules list and empty model_tiers must produce a valid engine.
    p = tmp_path / "empty.yaml"
    p.write_text("rules: []\nmodel_tiers: {}\n")
    engine = PolicyEngine.from_yaml(p)
    assert isinstance(engine, PolicyEngine)


def test_from_yaml_missing_rules_key_uses_empty_list(tmp_path: Path) -> None:
    # A YAML with only model_tiers and no rules key must produce a valid engine.
    p = tmp_path / "tiers_only.yaml"
    p.write_text("model_tiers:\n  m: approved\n")
    engine = PolicyEngine.from_yaml(p)
    decision = engine.evaluate(_record(model="m"))
    assert decision.action == PolicyAction.ALLOW


def test_from_yaml_raises_on_invalid_action(tmp_path: Path) -> None:
    # A rule with action="EXPLODE" must raise ValueError on load.
    p = tmp_path / "bad_action.yaml"
    p.write_text("rules:\n  - name: r\n    action: EXPLODE\n")
    with pytest.raises(ValueError, match="action"):
        PolicyEngine.from_yaml(p)


def test_from_yaml_raises_on_invalid_pii_severity(tmp_path: Path) -> None:
    # A rule with pii_severity="CRITICAL" in conditions must raise ValueError on load.
    p = tmp_path / "bad_sev.yaml"
    p.write_text(
        "rules:\n"
        "  - name: r\n"
        "    action: WARN\n"
        "    conditions:\n"
        "      pii_severity: CRITICAL\n"
    )
    with pytest.raises(ValueError, match="pii_severity"):
        PolicyEngine.from_yaml(p)


def test_from_yaml_raises_on_missing_rule_name(tmp_path: Path) -> None:
    # A rule entry without a name must raise ValueError with the entry index.
    p = tmp_path / "no_name.yaml"
    p.write_text("rules:\n  - action: WARN\n")
    with pytest.raises(ValueError, match="index 0"):
        PolicyEngine.from_yaml(p)


def test_from_yaml_rule_without_conditions_is_catch_all(tmp_path: Path) -> None:
    # A rule entry with no conditions key must load as RuleConditions() (all None).
    p = tmp_path / "catch_all.yaml"
    p.write_text("rules:\n  - name: catch\n    action: WARN\n")
    engine = PolicyEngine.from_yaml(p)
    decision = engine.evaluate(_record())
    assert decision.action == PolicyAction.WARN
    assert decision.rule_name == "catch"


# ---------------------------------------------------------------------------
# Group 11 — PolicyDecision integration with store
# ---------------------------------------------------------------------------


def test_policy_decision_survives_store_roundtrip(
    sqlite_backend: SQLiteBackend,
    default_engine: PolicyEngine,
) -> None:
    # A real PolicyDecision must survive write/read through the store.
    record = _record(model="gpt-4o", pii_result=_pii(PIISeverity.MEDIUM))
    decision = default_engine.evaluate(record)
    assert decision.action == PolicyAction.WARN

    base = record.model_dump()
    record_with_decision = ProvenanceRecord(
        **{
            **base,
            "pii_result": record.pii_result,
            "policy_decision": decision,
        }
    )
    sqlite_backend.write(record_with_decision, hmac_signature="h")
    fetched = sqlite_backend.get(record_with_decision.content_id)
    assert fetched is not None
    fetched_record, _ = fetched
    assert isinstance(fetched_record.policy_decision, PolicyDecision)
    assert fetched_record.policy_decision.action == PolicyAction.WARN
