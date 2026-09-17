"""Policy engine v2 tests: condition validation, v2 conditions, priorities,
modes, decision transparency, and the predicate hook."""
from __future__ import annotations

from pathlib import Path

import pytest

from aistamp.models import PIISeverity, PolicyAction, ProvenanceRecord
from aistamp.policy import (
    MAX_POLICY_REGEX_LENGTH,
    PolicyEngine,
    PolicyError,
    PolicyMode,
    PolicyViolationError,
    RuleConditions,
    RuleConfig,
)


def _engine(
    *rules: RuleConfig, mode: PolicyMode = PolicyMode.FIRST_MATCH
) -> PolicyEngine:
    return PolicyEngine(rules=list(rules), model_tiers={}, mode=mode)


def _write_policy(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# Unknown keys are rejected at from_yaml (the silent allow-all/global-block bug)
# ---------------------------------------------------------------------------


def test_from_yaml_rejects_unknown_condition_key(
    tmp_path: Path, sample_provenance_record: ProvenanceRecord
) -> None:
    # "pii_severityy" is a typo; 0.1.x silently dropped it and the rule
    # became a catch-all. It must fail loudly instead.
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: typo_rule\n"
        "    conditions:\n"
        "      pii_severityy: HIGH\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(ValueError, match="unknown condition key.*pii_severityy"):
        PolicyEngine.from_yaml(path)


def test_from_yaml_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    path = _write_policy(tmp_path, "modes: first_match\nrules: []\n")
    with pytest.raises(ValueError, match="unknown top-level key.*'modes'"):
        PolicyEngine.from_yaml(path)


def test_from_yaml_rejects_unknown_rule_key(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path,
        "rules:\n  - name: r\n    action: WARN\n    priorityy: 5\n",
    )
    with pytest.raises(ValueError, match="unknown key.*'priorityy'"):
        PolicyEngine.from_yaml(path)


def test_error_message_names_the_rule(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: scope_guard\n"
        "    conditions:\n"
        "      applicaton_id: a\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(ValueError, match="scope_guard.*applicaton_id"):
        PolicyEngine.from_yaml(path)


# ---------------------------------------------------------------------------
# Conditions v2
# ---------------------------------------------------------------------------


def test_feature_id_condition(sample_provenance_record: ProvenanceRecord) -> None:
    engine = _engine(
        RuleConfig(
            name="feat_a_only",
            conditions=RuleConditions(feature_id="test_feature"),
            action=PolicyAction.WARN,
        )
    )
    assert engine.evaluate(sample_provenance_record).action is PolicyAction.WARN

    other = sample_provenance_record.model_copy(update={"feature_id": "other"})
    decision = engine.evaluate(other)
    assert decision.action is PolicyAction.ALLOW
    assert decision.rule_name is None


def test_app_id_condition(sample_provenance_record: ProvenanceRecord) -> None:
    engine = _engine(
        RuleConfig(
            name="app_scope",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.WARN,
        )
    )
    assert engine.evaluate(sample_provenance_record).rule_name == "app_scope"
    other = sample_provenance_record.model_copy(update={"app_id": "elsewhere"})
    assert engine.evaluate(other).action is PolicyAction.ALLOW


def test_user_id_condition(sample_provenance_record: ProvenanceRecord) -> None:
    engine = _engine(
        RuleConfig(
            name="user_scope",
            conditions=RuleConditions(user_id="user_001"),
            action=PolicyAction.WARN,
        )
    )
    assert engine.evaluate(sample_provenance_record).rule_name == "user_scope"


def test_condition_accept_list_allowlist(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="allowlist",
            conditions=RuleConditions(feature_id=["feat_x", "test_feature"]),
            action=PolicyAction.WARN,
        )
    )
    assert engine.evaluate(sample_provenance_record).rule_name == "allowlist"


def test_pii_match_count_min(sample_provenance_record: ProvenanceRecord) -> None:
    # sample record has exactly 1 PII match.
    at_one = _engine(
        RuleConfig(
            name="min1",
            conditions=RuleConditions(pii_match_count_min=1),
            action=PolicyAction.WARN,
        )
    )
    at_two = _engine(
        RuleConfig(
            name="min2",
            conditions=RuleConditions(pii_match_count_min=2),
            action=PolicyAction.WARN,
        )
    )
    assert at_one.evaluate(sample_provenance_record).rule_name == "min1"
    assert at_two.evaluate(sample_provenance_record).action is PolicyAction.ALLOW


def test_pii_types_condition_is_case_insensitive(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="email_guard",
            conditions=RuleConditions(pii_types=["email"]),
            action=PolicyAction.WARN,
        )
    )
    assert engine.evaluate(sample_provenance_record).rule_name == "email_guard"

    ssn_only = _engine(
        RuleConfig(
            name="ssn_guard",
            conditions=RuleConditions(pii_types=["SSN"]),
            action=PolicyAction.WARN,
        )
    )
    assert ssn_only.evaluate(sample_provenance_record).action is PolicyAction.ALLOW


def test_model_regex_uses_fullmatch(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    family = _engine(
        RuleConfig(
            name="gpt4_family",
            conditions=RuleConditions(model_regex="gpt-4.*"),
            action=PolicyAction.WARN,
        )
    )
    assert family.evaluate(sample_provenance_record).rule_name == "gpt4_family"

    exact = _engine(
        RuleConfig(
            name="exact_gpt4",
            conditions=RuleConditions(model_regex="gpt-4"),
            action=PolicyAction.WARN,
        )
    )
    assert exact.evaluate(sample_provenance_record).action is PolicyAction.ALLOW


def test_conditions_combine_with_and_logic(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    both = _engine(
        RuleConfig(
            name="and_rule",
            conditions=RuleConditions(
                app_id="test_app",
                pii_severity=PIISeverity.HIGH,
            ),
            action=PolicyAction.WARN,
        )
    )
    # severity is MEDIUM on the sample record, so the AND fails.
    assert both.evaluate(sample_provenance_record).action is PolicyAction.ALLOW


# ---------------------------------------------------------------------------
# Priority + modes
# ---------------------------------------------------------------------------


def test_priority_beats_declaration_order(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # Both rules match; the higher-priority BLOCK must win even though the
    # WARN rule was declared first.
    engine = _engine(
        RuleConfig(
            name="low_priority_warn",
            conditions=RuleConditions(feature_id="test_feature"),
            action=PolicyAction.WARN,
            priority=0,
        ),
        RuleConfig(
            name="high_priority_block",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.BLOCK,
            priority=10,
        ),
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        engine.evaluate(sample_provenance_record)
    assert excinfo.value.rule_name == "high_priority_block"


def test_first_match_default_mode_stops_at_first(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="first_warn",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.WARN,
        ),
        RuleConfig(
            name="second_block",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.BLOCK,
        ),
    )
    decision = engine.evaluate(sample_provenance_record)
    assert decision.action is PolicyAction.WARN
    assert decision.evaluated_rules == ["first_warn"]


def test_evaluate_all_keeps_trace_and_most_restrictive_wins(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="warn_rule",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.WARN,
        ),
        RuleConfig(
            name="block_rule",
            conditions=RuleConditions(feature_id="test_feature"),
            action=PolicyAction.BLOCK,
        ),
        mode=PolicyMode.EVALUATE_ALL,
    )
    with pytest.raises(PolicyViolationError) as excinfo:
        engine.evaluate(sample_provenance_record)
    decision = excinfo.value.decision
    assert decision.action is PolicyAction.BLOCK
    assert decision.rule_name == "block_rule"
    # The trace must name every rule that matched, in evaluation order.
    assert decision.evaluated_rules == ["warn_rule", "block_rule"]
    assert "block_rule" in decision.reason


def test_evaluate_all_allow_never_beats_warn(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="allow_rule",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.ALLOW,
        ),
        RuleConfig(
            name="warn_rule",
            conditions=RuleConditions(feature_id="test_feature"),
            action=PolicyAction.WARN,
        ),
        mode=PolicyMode.EVALUATE_ALL,
    )
    decision = engine.evaluate(sample_provenance_record)
    assert decision.action is PolicyAction.WARN
    assert decision.rule_name == "warn_rule"
    assert decision.evaluated_rules == ["allow_rule", "warn_rule"]


# ---------------------------------------------------------------------------
# Decision transparency
# ---------------------------------------------------------------------------


def test_decision_carries_matched_conditions_and_timestamp(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="app_scope",
            conditions=RuleConditions(app_id="test_app"),
            action=PolicyAction.WARN,
        )
    )
    decision = engine.evaluate(sample_provenance_record)
    assert decision.decided_at is not None
    assert decision.matched_conditions
    assert isinstance(decision.matched_conditions, dict)
    assert all(isinstance(v, str) for v in decision.evaluated_rules)


def test_no_match_decision_has_empty_trace(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    engine = _engine(
        RuleConfig(
            name="never",
            conditions=RuleConditions(feature_id="nope"),
            action=PolicyAction.WARN,
        )
    )
    decision = engine.evaluate(sample_provenance_record)
    assert decision.action is PolicyAction.ALLOW
    assert decision.evaluated_rules == []
    assert decision.matched_conditions == {}
    assert decision.decided_at is not None


# ---------------------------------------------------------------------------
# Predicate hook
# ---------------------------------------------------------------------------


def test_predicate_hook_fires(sample_provenance_record: ProvenanceRecord) -> None:
    engine = _engine(
        RuleConfig(
            name="predicate_warn",
            conditions=RuleConditions(predicate="operator:truth"),
            action=PolicyAction.WARN,
        )
    )
    decision = engine.evaluate(sample_provenance_record)
    assert decision.action is PolicyAction.WARN
    assert decision.rule_name == "predicate_warn"


def test_bad_predicate_ref_fails_at_yaml_load(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: bad_hook\n"
        "    conditions:\n"
        "      predicate: no_such_module_xyz:missing\n"
        "    action: WARN\n",
    )
    with pytest.raises(ValueError, match="no_such_module_xyz"):
        PolicyEngine.from_yaml(path)


# ---------------------------------------------------------------------------
# ReDoS: operator model_regex patterns are rejected at load, never evaluate
# ---------------------------------------------------------------------------


_ANCHOR = chr(36)  # the "$" end-anchor, spelled via chr(36) for transport safety
_REDOS_REGEX = "(a+)+" + _ANCHOR  # the audit's catastrophic PoV pattern


def test_from_yaml_rejects_redos_pattern(tmp_path: Path) -> None:
    # The audit's PoV pattern (a+)+ anchored to end-of-string used to load
    # fine and hung evaluate() for over five seconds on a 29-character
    # model string.
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: redos\n"
        "    conditions:\n"
        f"      model_regex: '{_REDOS_REGEX}'\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(PolicyError, match="backtracking|ReDoS"):
        PolicyEngine.from_yaml(path)


def test_from_yaml_rejects_redos_via_alternation(tmp_path: Path) -> None:
    # '(a|b+)+' is the same catastrophic shape with the nested quantifier
    # inside one alternation branch.
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: redos\n"
        "    conditions:\n"
        "      model_regex: '(a|b+)+'\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(PolicyError, match="backtracking|ReDoS"):
        PolicyEngine.from_yaml(path)


def test_from_yaml_rejects_oversized_regex(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: long\n"
        "    conditions:\n"
        f"      model_regex: '{'a' * (MAX_POLICY_REGEX_LENGTH + 1)}'\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(PolicyError, match="safety limit"):
        PolicyEngine.from_yaml(path)


def test_safe_regexes_still_load_and_match(
    tmp_path: Path, sample_provenance_record: ProvenanceRecord
) -> None:
    # Plain alternation under a quantifier is safe and must keep working.
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: model_guard\n"
        "    conditions:\n"
        "      model_regex: '(foo|bar)='\n"
        "    action: WARN\n",
    )
    path.write_text(path.read_text().replace("(foo|bar)=", "gpt-4.*"))
    engine = PolicyEngine.from_yaml(path)
    hit = sample_provenance_record.model_copy(update={"model": "gpt-4o-mini"})
    decision = engine.evaluate(hit)
    assert decision.action is PolicyAction.WARN
    assert decision.matched_conditions.get("model_regex") == "gpt-4o-mini"
    miss = sample_provenance_record.model_copy(update={"model": "claude-3"})
    assert engine.evaluate(miss).matched_conditions.get("model_regex") is None


def test_redos_pov_never_reaches_evaluate(tmp_path: Path) -> None:
    # The audit PoV input: pattern accepted at load, then evaluate() hung
    # on model="a"*28+"!". Load must raise before any evaluation is possible.
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: redos\n"
        "    conditions:\n"
        f"      model_regex: '{_REDOS_REGEX}'\n"
        "    action: BLOCK\n",
    )
    with pytest.raises(PolicyError):
        PolicyEngine.from_yaml(path)  # never returns an engine to evaluate with


def test_evaluate_rejects_redos_on_direct_conditions(
    sample_provenance_record: ProvenanceRecord,
) -> None:
    # Defense in depth: conditions constructed directly in code bypass
    # from_yaml, so evaluate() itself must route through the safety check.
    engine = _engine(
        RuleConfig(
            name="direct_redos",
            conditions=RuleConditions(model_regex="(a+)+$"),
            action=PolicyAction.WARN,
        )
    )
    record = sample_provenance_record.model_copy(
        update={"model": "a" * 28 + "!"}
    )
    with pytest.raises(PolicyError, match="backtracking|ReDoS"):
        engine.evaluate(record)


def test_policy_error_is_value_error_for_0_1_x_callers(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path,
        "rules:\n"
        "  - name: r\n"
        "    conditions:\n"
        "      model_regex: '(a+)+'\n"
        "    action: WARN\n",
    )
    # 0.1.x callers catch ValueError for invalid policies.
    with pytest.raises(ValueError):
        PolicyEngine.from_yaml(path)
