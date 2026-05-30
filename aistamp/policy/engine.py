from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from aistamp.models import (
    SEVERITY_RANK,
    PIISeverity,
    PolicyAction,
    PolicyDecision,
    ProvenanceRecord,
)
from aistamp.policy.rules import RuleConditions, RuleConfig

logger = logging.getLogger("aistamp.policy")


_VALID_ACTIONS = {"ALLOW", "WARN", "BLOCK"}
_VALID_SEVERITIES = {"LOW", "MEDIUM", "HIGH"}


class PolicyViolationError(Exception):
    """
    Raised by PolicyEngine.evaluate() when a rule with action=BLOCK is matched.

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
            f"Reason: {decision.reason}"
        )
        self.rule_name = rule_name
        self.decision = decision
        self.content_id = content_id


def _conditions_match(
    conditions: RuleConditions,
    model_tier: str | None,
    record: ProvenanceRecord,
) -> bool:
    if conditions.model_tier is not None:
        if model_tier is None:
            return False
        if model_tier.lower() != conditions.model_tier.lower():
            return False

    if conditions.pii_severity is not None:
        if record.pii_result is None:
            return False
        if record.pii_result.highest_severity is None:
            return False
        if (
            SEVERITY_RANK[record.pii_result.highest_severity]
            < SEVERITY_RANK[conditions.pii_severity]
        ):
            return False

    return True


class PolicyEngine:
    """
    Evaluates a ProvenanceRecord against a sequence of policy rules.

    Rules are evaluated in order. The first matching rule determines the
    outcome. If no rule matches, the default action is ALLOW.
    """

    # TODO(phase-v2): add condition types beyond model_tier and pii_severity
    # (e.g. user_id allowlist, feature_id allowlist, token-count thresholds).

    def __init__(
        self,
        rules: list[RuleConfig],
        model_tiers: dict[str, str],
    ) -> None:
        self._rules = rules
        self._model_tiers = model_tiers

    def evaluate(self, record: ProvenanceRecord) -> PolicyDecision:
        model_tier = self._model_tiers.get(record.model, "unknown")

        for rule in self._rules:
            if not _conditions_match(rule.conditions, model_tier, record):
                continue

            decision = PolicyDecision(
                action=rule.action,
                rule_name=rule.name,
                reason=f"Matched rule {rule.name!r} with action {rule.action.value}",
            )

            if rule.action == PolicyAction.WARN:
                logger.warning(
                    "Policy WARN for content_id=%s rule=%r model_tier=%s",
                    record.content_id,
                    rule.name,
                    model_tier,
                )
                return decision

            if rule.action == PolicyAction.BLOCK:
                logger.warning(
                    "Policy BLOCK for content_id=%s rule=%r model_tier=%s",
                    record.content_id,
                    rule.name,
                    model_tier,
                )
                raise PolicyViolationError(
                    rule_name=rule.name,
                    decision=decision,
                    content_id=record.content_id,
                )

            return decision

        return PolicyDecision(
            action=PolicyAction.ALLOW,
            rule_name=None,
            reason="No matching rule. Default action applied.",
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> PolicyEngine:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        with p.open("r") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}

        model_tiers: dict[str, str] = data.get("model_tiers") or {}
        raw_rules: list[dict[str, Any]] = data.get("rules") or []

        parsed_rules: list[RuleConfig] = []
        for idx, entry in enumerate(raw_rules):
            if "name" not in entry:
                raise ValueError(
                    f"Rule entry at index {idx} is missing required 'name' field."
                )
            name = entry["name"]

            if "action" not in entry:
                raise ValueError(f"Rule {name!r} is missing required 'action' field.")
            action_raw = str(entry["action"]).upper()
            if action_raw not in _VALID_ACTIONS:
                raise ValueError(
                    f"Rule {name!r} has invalid action {entry['action']!r}."
                    " Must be one of: ALLOW, WARN, BLOCK."
                )
            action = PolicyAction(action_raw)

            cond_raw = entry.get("conditions") or {}
            model_tier = cond_raw.get("model_tier")

            pii_severity: PIISeverity | None = None
            if "pii_severity" in cond_raw and cond_raw["pii_severity"] is not None:
                sev_raw = str(cond_raw["pii_severity"]).upper()
                if sev_raw not in _VALID_SEVERITIES:
                    raise ValueError(
                        f"Rule {name!r} has invalid pii_severity"
                        f" {cond_raw['pii_severity']!r}."
                        " Must be one of: HIGH, MEDIUM, LOW."
                    )
                pii_severity = PIISeverity(sev_raw)

            conditions = RuleConditions(
                model_tier=model_tier,
                pii_severity=pii_severity,
            )
            parsed_rules.append(
                RuleConfig(name=name, conditions=conditions, action=action)
            )

        return cls(rules=parsed_rules, model_tiers=model_tiers)
