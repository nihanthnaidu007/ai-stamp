"""Static safety analysis for operator-supplied regular expressions.

Policy ``model_regex`` patterns run against attacker-influenced model
strings on every ``evaluate()`` call. CPython's ``re`` engine backtracks,
so a pattern with nested quantifiers — ``(a+)+$`` — can hang evaluation on
a non-matching input of a few dozen characters (ReDoS). Patterns are
analyzed once when a policy is loaded so an unsafe pattern can never reach
``evaluate()``; evaluation routes through the same analysis as defense in
depth for directly constructed conditions.
"""

from __future__ import annotations

import functools
import re
from typing import Any

from aistamp.policy.rules import PolicyError

# CPython's sre parse tree is the precise way to see quantifier nesting.
# It is a private module with no public type stubs; every tree node is an
# (op, av) tuple, hence the narrow Any at the walker boundary below.
try:
    import re._parser as _sre_parser
except ImportError:  # pragma: no cover - non-CPython fallback heuristic
    _sre_parser = None

MAX_POLICY_REGEX_LENGTH = 256


def _has_nested_quantifier(pattern: str) -> bool:
    """Report whether a quantifier is applied to a quantified group.

    This is the canonical catastrophic-backtracking shape: ``(a+)+`` can
    backtrack exponentially. Possessive/atomic forms are also flagged —
    the analysis is deliberately conservative for a safety control.
    """
    if _sre_parser is None:  # pragma: no cover - non-CPython fallback
        # Quantifier directly after a quantified group: "(a+)+", "(a*)?".
        return re.search(r"[*+?]\s*\)[+*{]", pattern) is not None

    repeat_ops = {
        _sre_parser.MAX_REPEAT,
        _sre_parser.MIN_REPEAT,
    }
    possessive = getattr(_sre_parser, "POSSESSIVE_REPEAT", None)
    if possessive is not None:
        repeat_ops.add(possessive)

    def walk(entries: Any, inside_repeat: bool) -> bool:
        for op, av in entries:
            if op in repeat_ops:
                if inside_repeat:
                    return True
                # av = (min, max, subpattern)
                if walk(av[2], True):
                    return True
            elif op == _sre_parser.SUBPATTERN:
                # av = (group, name, add_flags, subpattern)
                if walk(av[3], inside_repeat):
                    return True
            elif op == _sre_parser.BRANCH:
                # av = (has_captures, alternatives)
                if any(walk(branch, inside_repeat) for branch in av[1]):
                    return True
            elif op in (_sre_parser.ASSERT, _sre_parser.ASSERT_NOT):
                # av = (direction, subpattern)
                if walk(av[1], inside_repeat):
                    return True
        return False

    try:
        tree = _sre_parser.parse(pattern)
    except re.error:
        # Invalid regexes are reported by re.compile with a precise message.
        return False
    return walk(tree, False)


def ensure_safe_regex(pattern: str) -> None:
    """Raise PolicyError when a pattern risks catastrophic backtracking.

    Called when policy YAML is loaded so an unsafe pattern never reaches
    evaluate(); evaluate() routes through the same check for conditions
    constructed directly in code.
    """
    if len(pattern) > MAX_POLICY_REGEX_LENGTH:
        raise PolicyError(
            f"Regular expression exceeds the {MAX_POLICY_REGEX_LENGTH}-character"
            f" safety limit (got {len(pattern)} characters):"
            f" {pattern[:64]!r}..."
        )
    if _has_nested_quantifier(pattern):
        raise PolicyError(
            f"Regular expression {pattern!r} nests quantifiers (for example"
            " '(a+)+'), which can hang evaluation on non-matching input"
            " (catastrophic backtracking / ReDoS). Rewrite it without a"
            " quantifier applied to a quantified group — e.g. anchor a"
            " bounded repetition or use an explicit alternation."
        )


@functools.lru_cache(maxsize=256)
def compiled_safe_regex(pattern: str) -> re.Pattern[str]:
    """Compile a pattern after safety analysis, caching the result.

    evaluate() uses this instead of re.fullmatch on the raw pattern so
    regexes are analyzed once and compiled once, not per record.
    """
    ensure_safe_regex(pattern)
    return re.compile(pattern)
