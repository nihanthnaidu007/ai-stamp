from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aistamp.models import PIISeverity, PolicyAction


class PolicyMode(str, Enum):
    """How the engine combines multiple matching rules.

    - ``FIRST_MATCH``: the highest-priority matching rule decides (0.1.x
      behavior, extended with explicit priority). Default.
    - ``EVALUATE_ALL``: every rule is evaluated and the most restrictive
      matched action wins (BLOCK > WARN > ALLOW); the full trace of matched
      rules is kept on the decision.
    """

    FIRST_MATCH = "first_match"
    EVALUATE_ALL = "evaluate_all"


@dataclass(frozen=True)
class RuleConditions:
    """Conditions that trigger a policy rule.

    All fields are optional. A field set to ``None`` is treated as
    "match anything" for that dimension. A ``RuleConditions()`` with all
    fields ``None`` matches every record and is a valid catch-all rule.

    All set conditions must match (AND logic) for the rule to fire.

    Field semantics when set:
        - ``model_tier``: matches when the evaluated record's model resolves
          to this tier string (case-insensitive comparison). Accepts a single
          tier string or a list of acceptable tiers.
        - ``pii_severity``: matches when ``record.pii_result.highest_severity``
          is greater than or equal to this level (LOW < MEDIUM < HIGH).
        - ``feature_id`` / ``app_id`` / ``user_id``: exact (case-sensitive)
          match against the record field. Accepts a single string or a list
          of acceptable strings (allowlist).
        - ``pii_match_count_min``: matches when
          ``record.pii_result.match_count`` is greater than or equal to this
          value. Requires a ``pii_result`` on the record.
        - ``pii_types``: matches when at least one PII match on the record has
          a pattern name in this list (case-insensitive comparison, e.g.
          ``["EMAIL", "SSN"]``).
        - ``model_regex``: matches when the record's model name fully matches
          this regular expression (``re.fullmatch`` semantics — write
          ``gpt-4.*`` to cover model families).
        - ``predicate``: reference to an importable callable
          ``"module.path:qualname"`` accepting the ``ProvenanceRecord`` and
          returning a truthy value when the condition holds. Only use this for
          logic that cannot be expressed with the built-in conditions; the
          callable runs with the privileges of the process loading the policy.
    """

    model_tier: str | list[str] | None = None
    pii_severity: PIISeverity | None = None
    feature_id: str | list[str] | None = None
    app_id: str | list[str] | None = None
    user_id: str | list[str] | None = None
    pii_match_count_min: int | None = None
    pii_types: list[str] | None = None
    model_regex: str | None = None
    predicate: str | None = None


@dataclass(frozen=True)
class RuleConfig:
    """A single policy rule.

    ``priority`` orders rules within an engine: higher priority evaluates
    first (rules with equal priority keep their declaration order). It only
    affects which rule decides in ``FIRST_MATCH`` mode.
    """

    name: str
    conditions: RuleConditions
    action: PolicyAction
    priority: int = 0
