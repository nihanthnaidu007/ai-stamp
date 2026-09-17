from __future__ import annotations

import importlib
import logging
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import yaml

from aistamp.errors import AIStampError
from aistamp.models import (
    SEVERITY_RANK,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
)
from aistamp.policy.regex_safety import compiled_safe_regex, ensure_safe_regex
from aistamp.policy.rules import (
    PolicyError,
    PolicyMode,
    RuleConditions,
    RuleConfig,
)

logger = logging.getLogger("aistamp.policy")


_VALID_ACTIONS = {"ALLOW", "WARN", "BLOCK"}
_VALID_SEVERITIES = {"HIGH", "MEDIUM", "LOW"}
_VALID_TOP_LEVEL_KEYS = {"model_tiers", "rules", "mode"}
_VALID_RULE_KEYS = {"name", "action", "conditions", "priority"}
_VALID_CONDITION_KEYS = {
    "model_tier",
    "pii_severity",
    "feature_id",
    "app_id",
    "user_id",
    "pii_match_count_min",
    "pii_types",
    "model_regex",
    "predicate",
}
_ACTION_RESTRICTIVENESS = {
    PolicyAction.ALLOW: 0,
    PolicyAction.WARN: 1,
    PolicyAction.BLOCK: 2,
}
_PREDICATE_REF_SEPARATOR = ":"


class PolicyViolationError(AIStampError):
    """
    Raised by PolicyEngine.evaluate() when a rule with action=BLOCK is matched.

    Since 0.2 this derives from :class:`aistamp.errors.AIStampError` so
    ``except AIStampError`` also catches policy blocks. The old import paths
    (``aistamp.policy.PolicyViolationError``, ``aistamp.PolicyViolationError``)
    keep working.

    Attributes:
        rule_name:  name of the rule that triggered the block.
        decision:   the full PolicyDecision that caused the error.
        content_id: content_id from the ProvenanceRecord being evaluated.
    """

    def __init__(
        self,
        rule_name: str,
        decision: PolicyDecision,
        content_id: str,
    ) -> None:
        super().__init__(
            f"Policy rule {rule_name!r} blocked content_id={content_id!r}. "
            f"Reason: {decision.reason}",
            content_id=content_id,
        )
        self.rule_name = rule_name
        self.decision = decision
        self.content_id = content_id


def _as_str_list(value: str | list[str], *, field: str, rule_name: str) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    raise ValueError(
        f"Rule {rule_name!r} condition {field!r} must be a string"
        f" or a list of strings, got {type(value).__name__}."
    )


def _matches_allowlist(accepted: str | list[str], record_value: str) -> bool:
    accepted_values = accepted if isinstance(accepted, list) else [accepted]
    return record_value in accepted_values


def _parse_allowlist(
    rule_name: str, field: str, value: Any
) -> str | list[str] | None:
    """Validate and normalize a str-or-list-of-str condition value."""
    if value is None:
        return None
    normalized = _as_str_list(value, field=field, rule_name=rule_name)
    if len(normalized) == 1:
        return normalized[0]
    return normalized


def _matches_allowlist_ci(accepted: str | list[str], record_value: str) -> bool:
    """Case-insensitive allowlist match (used for model tiers, per 0.1.x)."""
    accepted_values = accepted if isinstance(accepted, list) else [accepted]
    lowered = record_value.lower()
    return lowered in (v.lower() for v in accepted_values)


def _load_predicate(ref: str) -> Callable[[ProvenanceRecord], bool]:
    """Resolve a ``"module.path:qualname"`` reference to a callable.

    The policy file is an operator-controlled artifact, so importing the
    referenced module is a trusted operation — but it is still validated
    eagerly at load time so a bad reference fails loudly instead of silently
    disabling the rule at evaluation time.
    """
    if ref.count(_PREDICATE_REF_SEPARATOR) != 1:
        raise ValueError(
            f"Invalid predicate reference {ref!r}: expected the form"
            f" 'module.path:qualname'."
        )
    module_name, qualname = ref.split(_PREDICATE_REF_SEPARATOR)
    if not module_name or not qualname:
        raise ValueError(
            f"Invalid predicate reference {ref!r}: module and callable name"
            " must both be non-empty."
        )
    try:
        obj: Any = importlib.import_module(module_name)
        for attr in qualname.split("."):
            obj = getattr(obj, attr)
    except (ImportError, AttributeError) as e:
        raise ValueError(f"Predicate {ref!r} could not be imported: {e}") from e
    if not callable(obj):
        raise ValueError(f"Predicate {ref!r} must resolve to a callable.")
    return cast("Callable[[ProvenanceRecord], bool]", obj)


def _conditions_match(
    conditions: RuleConditions,
    model_tier: str | None,
    record: ProvenanceRecord,
    predicate_fn: Callable[[ProvenanceRecord], bool] | None,
) -> tuple[bool, dict[str, Any]]:
    """Check all set conditions against a record.

    Returns ``(matched, matched_conditions)`` where ``matched_conditions``
    maps each condition that was set to the concrete record-side value that
    satisfied it — the auditable fact, not the rule-side threshold.
    """
    matched: dict[str, Any] = {}

    if conditions.model_tier is not None:
        if model_tier is None:
            return False, {}
        if not _matches_allowlist_ci(conditions.model_tier, model_tier):
            return False, {}
        matched["model_tier"] = model_tier

    if conditions.pii_severity is not None:
        if record.pii_result is None or record.pii_result.highest_severity is None:
            return False, {}
        if (
            SEVERITY_RANK[record.pii_result.highest_severity]
            < SEVERITY_RANK[conditions.pii_severity]
        ):
            return False, {}
        matched["pii_severity"] = record.pii_result.highest_severity.value

    if conditions.feature_id is not None:
        if not _matches_allowlist(conditions.feature_id, record.feature_id):
            return False, {}
        matched["feature_id"] = record.feature_id

    if conditions.app_id is not None:
        if not _matches_allowlist(conditions.app_id, record.app_id):
            return False, {}
        matched["app_id"] = record.app_id

    if conditions.user_id is not None:
        if not _matches_allowlist(conditions.user_id, record.user_id):
            return False, {}
        matched["user_id"] = record.user_id

    if conditions.pii_match_count_min is not None:
        if record.pii_result is None:
            return False, {}
        if record.pii_result.match_count < conditions.pii_match_count_min:
            return False, {}
        matched["pii_match_count_min"] = record.pii_result.match_count

    if conditions.pii_types is not None:
        if record.pii_result is None:
            return False, {}
        record_types = {
            m.pattern_name.upper()
            for m in (
                record.pii_result.prompt_matches
                + record.pii_result.response_matches
            )
        }
        accepted = {t.upper() for t in conditions.pii_types}
        intersection = sorted(record_types & accepted)
        if not intersection:
            return False, {}
        matched["pii_types"] = intersection

    if conditions.model_regex is not None:
        # Defense in depth: also guards conditions constructed directly in
        # code, and compiles once per pattern instead of per call.
        if compiled_safe_regex(conditions.model_regex).fullmatch(
            record.model
        ) is None:
            return False, {}
        matched["model_regex"] = record.model

    if conditions.predicate is not None:
        if predicate_fn is None or not predicate_fn(record):
            return False, {}
        matched["predicate"] = conditions.predicate

    return True, matched


class PolicyEngine:
    """
    Evaluates a ProvenanceRecord against a sequence of policy rules.

    Rules are ordered by explicit ``priority`` (higher first, declaration
    order breaks ties). In the default ``first_match`` mode the first
    matching rule determines the outcome; in ``evaluate_all`` mode every
    rule is evaluated, the full trace of matched rules is recorded, and the
    most restrictive matched action wins (BLOCK > WARN > ALLOW). If no rule
    matches, the default action is ALLOW.
    """

    def __init__(
        self,
        rules: list[RuleConfig],
        model_tiers: dict[str, str],
        mode: PolicyMode | str = PolicyMode.FIRST_MATCH,
    ) -> None:
        self._mode = mode if isinstance(mode, PolicyMode) else PolicyMode(mode)
        self._rules: list[RuleConfig] = [
            rule
            for _, rule in sorted(
                enumerate(rules), key=lambda pair: (-pair[1].priority, pair[0])
            )
        ]
        self._model_tiers = model_tiers
        self._predicate_cache: dict[str, Callable[[ProvenanceRecord], bool]] = {}

    def _predicate_for(
        self, conditions: RuleConditions
    ) -> Callable[[ProvenanceRecord], bool] | None:
        if conditions.predicate is None:
            return None
        if conditions.predicate not in self._predicate_cache:
            self._predicate_cache[conditions.predicate] = _load_predicate(
                conditions.predicate
            )
        return self._predicate_cache[conditions.predicate]

    def evaluate(self, record: ProvenanceRecord) -> PolicyDecision:
        decided_at = datetime.now(timezone.utc)
        model_tier = self._model_tiers.get(record.model, "unknown")

        matched_rules: list[tuple[RuleConfig, dict[str, Any]]] = []
        for rule in self._rules:
            fired, values = _conditions_match(
                rule.conditions,
                model_tier,
                record,
                self._predicate_for(rule.conditions),
            )
            if fired:
                matched_rules.append((rule, values))
                if self._mode is PolicyMode.FIRST_MATCH:
                    break

        if not matched_rules:
            return PolicyDecision(
                action=PolicyAction.ALLOW,
                rule_name=None,
                reason="No matching rule. Default action applied.",
                matched_conditions={},
                evaluated_rules=[],
                decided_at=decided_at,
            )

        evaluated_rules = [rule.name for rule, _ in matched_rules]

        if self._mode is PolicyMode.FIRST_MATCH:
            deciding_rule, matched_conditions = matched_rules[0]
            action = deciding_rule.action
            reason = (
                f"Matched rule {deciding_rule.name!r} with action {action.value}"
            )
        else:
            action = max(
                (rule.action for rule, _ in matched_rules),
                key=lambda a: _ACTION_RESTRICTIVENESS[a],
            )
            deciding_rule, matched_conditions = next(
                (r, v) for r, v in matched_rules if r.action == action
            )
            reason = (
                f"Evaluate-all mode: {len(matched_rules)} rule(s) matched"
                f" ({', '.join(evaluated_rules)}); most restrictive action"
                f" {action.value} applied from rule {deciding_rule.name!r}"
            )

        decision = PolicyDecision(
            action=action,
            rule_name=deciding_rule.name,
            reason=reason,
            matched_conditions=matched_conditions,
            evaluated_rules=evaluated_rules,
            decided_at=decided_at,
        )

        if action == PolicyAction.WARN:
            logger.warning(
                "Policy WARN for content_id=%s rule=%r model_tier=%s",
                record.content_id,
                deciding_rule.name,
                model_tier,
            )
        if action == PolicyAction.BLOCK:
            logger.warning(
                "Policy BLOCK for content_id=%s rule=%r model_tier=%s",
                record.content_id,
                deciding_rule.name,
                model_tier,
            )
            raise PolicyViolationError(
                rule_name=deciding_rule.name,
                decision=decision,
                content_id=record.content_id,
            )

        return decision

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyEngine:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        with p.open("r") as f:
            data: Any = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ValueError(
                f"Policy file {path} must contain a YAML mapping at the top level."
            )

        unknown_top = sorted(set(data) - _VALID_TOP_LEVEL_KEYS)
        if unknown_top:
            raise ValueError(
                f"Policy file {path} has unknown top-level key(s)"
                f" {', '.join(repr(k) for k in unknown_top)}."
                f" Valid keys: {', '.join(sorted(_VALID_TOP_LEVEL_KEYS))}."
            )

        model_tiers_raw: Any = data.get("model_tiers") or {}
        if not isinstance(model_tiers_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str)
            for k, v in model_tiers_raw.items()
        ):
            raise ValueError("'model_tiers' must be a mapping of model name to tier.")

        mode_raw = data.get("mode")
        try:
            mode = PolicyMode.FIRST_MATCH if mode_raw is None else PolicyMode(mode_raw)
        except ValueError:
            raise ValueError(
                f"Invalid mode {mode_raw!r}. Must be one of: first_match, evaluate_all."
            ) from None

        raw_rules: Any = data.get("rules") or []
        if not isinstance(raw_rules, list):
            raise ValueError("'rules' must be a list of rule entries.")

        parsed_rules: list[RuleConfig] = []
        for idx, entry in enumerate(raw_rules):
            parsed_rules.append(cls._parse_rule(idx, entry))

        return cls(rules=parsed_rules, model_tiers=model_tiers_raw, mode=mode)

    @classmethod
    def _parse_rule(cls, idx: int, entry: Any) -> RuleConfig:
        if not isinstance(entry, dict):
            raise ValueError(f"Rule entry at index {idx} must be a mapping.")
        if "name" not in entry:
            raise ValueError(
                f"Rule entry at index {idx} is missing required 'name' field."
            )
        name = entry["name"]

        unknown_rule_keys = sorted(set(entry) - _VALID_RULE_KEYS)
        if unknown_rule_keys:
            raise ValueError(
                f"Rule {name!r} has unknown key(s)"
                f" {', '.join(repr(k) for k in unknown_rule_keys)}."
                f" Valid keys: {', '.join(sorted(_VALID_RULE_KEYS))}."
            )

        if "action" not in entry:
            raise ValueError(f"Rule {name!r} is missing required 'action' field.")
        action_raw = str(entry["action"]).upper()
        if action_raw not in _VALID_ACTIONS:
            raise ValueError(
                f"Rule {name!r} has invalid action {entry['action']!r}."
                " Must be one of: ALLOW, WARN, BLOCK."
            )
        action = PolicyAction(action_raw)

        priority_raw = entry.get("priority", 0)
        if isinstance(priority_raw, bool) or not isinstance(priority_raw, int):
            raise ValueError(f"Rule {name!r} priority must be an integer.")
        priority = int(priority_raw)

        conditions = cls._parse_conditions(name, entry.get("conditions"))
        return RuleConfig(
            name=name, conditions=conditions, action=action, priority=priority
        )

    @classmethod
    def _parse_conditions(cls, name: str, cond_raw: Any) -> RuleConditions:
        if cond_raw is None:
            cond_raw = {}
        if not isinstance(cond_raw, dict):
            raise ValueError(f"Rule {name!r} conditions must be a mapping.")

        unknown_condition_keys = sorted(set(cond_raw) - _VALID_CONDITION_KEYS)
        if unknown_condition_keys:
            raise ValueError(
                f"Rule {name!r} has unknown condition key(s)"
                f" {', '.join(repr(k) for k in unknown_condition_keys)}."
                f" Valid condition keys: {', '.join(sorted(_VALID_CONDITION_KEYS))}."
                " Unknown conditions are rejected because a mistyped key would"
                " silently widen the rule."
            )

        model_tier_raw = cond_raw.get("model_tier")
        model_tier: str | list[str] | None = None
        if model_tier_raw is not None:
            model_tier = _as_str_list(
                model_tier_raw, field="model_tier", rule_name=name
            )
            if len(model_tier) == 1:
                model_tier = model_tier[0]

        pii_severity: PIISeverity | None = None
        if cond_raw.get("pii_severity") is not None:
            sev_raw = str(cond_raw["pii_severity"]).upper()
            if sev_raw not in _VALID_SEVERITIES:
                raise ValueError(
                    f"Rule {name!r} has invalid pii_severity"
                    f" {cond_raw['pii_severity']!r}."
                    " Must be one of: HIGH, MEDIUM, LOW."
                )
            pii_severity = PIISeverity(sev_raw)

        feature_id = _parse_allowlist(name, "feature_id", cond_raw.get("feature_id"))
        app_id = _parse_allowlist(name, "app_id", cond_raw.get("app_id"))
        user_id = _parse_allowlist(name, "user_id", cond_raw.get("user_id"))

        pii_match_count_min: int | None = None
        if cond_raw.get("pii_match_count_min") is not None:
            raw = cond_raw["pii_match_count_min"]
            if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
                raise ValueError(
                    f"Rule {name!r} condition 'pii_match_count_min' must be a"
                    f" non-negative integer, got {raw!r}."
                )
            pii_match_count_min = int(raw)

        pii_types: list[str] | None = None
        if cond_raw.get("pii_types") is not None:
            pii_types = _as_str_list(
                cond_raw["pii_types"], field="pii_types", rule_name=name
            )

        model_regex: str | None = None
        if cond_raw.get("model_regex") is not None:
            model_regex = cond_raw["model_regex"]
            if not isinstance(model_regex, str):
                raise ValueError(
                    f"Rule {name!r} condition 'model_regex' must be a string."
                )
            try:
                re.compile(model_regex)
            except re.error as e:
                raise ValueError(
                    f"Rule {name!r} condition 'model_regex' has invalid regex: {e}"
                ) from e
            # Load-time ReDoS guard: an unsafe pattern must be rejected here
            # so it can never reach evaluate(), where it runs against
            # attacker-influenced model strings.
            try:
                ensure_safe_regex(model_regex)
            except PolicyError as e:
                raise PolicyError(
                    f"Rule {name!r} condition 'model_regex' is unsafe: {e}"
                ) from e

        predicate: str | None = None
        if cond_raw.get("predicate") is not None:
            predicate = cond_raw["predicate"]
            if not isinstance(predicate, str):
                raise ValueError(
                    f"Rule {name!r} condition 'predicate' must be a string"
                    " reference of the form 'module.path:qualname'."
                )
            _load_predicate(predicate)

        return RuleConditions(
            model_tier=model_tier,
            pii_severity=pii_severity,
            feature_id=feature_id,
            app_id=app_id,
            user_id=user_id,
            pii_match_count_min=pii_match_count_min,
            pii_types=pii_types,
            model_regex=model_regex,
            predicate=predicate,
        )
