"""Feature detection for the v0.2 compatibility kit.

Checks for surfaces that are *not yet on main* SKIP (never fail) with a
reason naming the PR that ships the feature. Detection is import- and
behavior-based, so a pending check activates automatically once its
dependency merges — the kit never needs a manual switch.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _try_import(module: str, attr: str) -> Any | None:
    try:
        mod = import_module(module)
    except ImportError:
        return None
    return getattr(mod, attr, None)


def build_evidence_pack() -> Any | None:
    """``build_evidence_pack`` — ships with policy PR #4."""
    return _try_import("aistamp.audit", "build_evidence_pack")


def signature_verdict() -> Any | None:
    """``signature_verdict`` / ``SignatureVerdict`` — ships with policy PR #4."""
    return _try_import("aistamp.audit", "signature_verdict")


def verify_chain() -> Any | None:
    """``verify_chain`` — ships with tamper PR #3."""
    return _try_import("aistamp.fingerprint", "verify_chain")


def record_version_default() -> int | None:
    """Default ``record_version`` for newly created records."""
    model = _try_import("aistamp.models", "ProvenanceRecord")
    if model is None:
        return None
    return int(model.model_fields["record_version"].default)


def is_purge_aware() -> bool:
    """True when ``verify_chain`` exists and resolves purge gaps itself.

    The merged tamper-evidence implementation treats retention purges as
    part of the protocol: when a purge-shaped chain break is explained by
    the purge journal (``backend.list_purge_anchors``), it is anchored —
    reported via ``ChainVerificationResult.anchored_gaps``, not as an
    issue. The absence of that field is what makes V2-15 PENDING rather
    than FAILING.
    """
    if verify_chain() is None:
        return False
    result_model = _try_import("aistamp.models", "ChainVerificationResult")
    return result_model is not None and "anchored_gaps" in result_model.model_fields


def cli_feature_available(app: Any, runner: Any, args: list[str], needle: str) -> bool:
    """True when ``aistamp <args> --help`` output contains *needle*."""
    result = runner.invoke(app, [*args, "--help"])
    return needle in result.output


def skip_pending_pr(pr: int, feature: str) -> None:
    """Skip with the canonical pending marker the standalone runner parses."""
    import pytest

    pytest.skip(f"PENDING (PR #{pr}): {feature}")
