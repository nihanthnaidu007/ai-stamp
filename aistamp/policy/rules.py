from __future__ import annotations

from dataclasses import dataclass

from aistamp.models import PIISeverity, PolicyAction


@dataclass(frozen=True)
class RuleConditions:
    """Conditions that trigger a policy rule.

    All fields are optional. A field set to ``None`` is treated as
    "match anything" for that dimension. A ``RuleConditions()`` with all
    fields ``None`` matches every record and is a valid catch-all rule.

    Field semantics when set:
        - ``model_tier``: matches when the evaluated record's model resolves
          to this tier string (case-insensitive comparison).
        - ``pii_severity``: matches when ``record.pii_result.highest_severity``
          is greater than or equal to this level (LOW < MEDIUM < HIGH).
    """

    model_tier: str | None = None
    pii_severity: PIISeverity | None = None


@dataclass(frozen=True)
class RuleConfig:
    name: str
    conditions: RuleConditions
    action: PolicyAction
