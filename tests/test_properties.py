"""Property-based tests (hypothesis) for the fingerprint, PII, and policy cores.

Properties verified here:
- hash/verify round trip: signing + verifying arbitrary content never fails
  for honest content, while tampered content or a wrong key is always caught.
- redaction: scan_text's redacted snippets never contain the matched span.
- policy severity monotonicity: a BLOCK rule at severity S blocks exactly the
  records whose highest PII severity ranks at or above S.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from aistamp.fingerprint import hash_content, sign_record, verify_record
from aistamp.models import (
    SEVERITY_RANK,
    PIIResult,
    PIISeverity,
    PolicyAction,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.pii import scan_text
from aistamp.policy import (
    PolicyEngine,
    PolicyViolationError,
    RuleConditions,
    RuleConfig,
)
from aistamp.store import SQLiteBackend

# Text that always survives utf-8 encoding (excludes lone surrogates).
utf8_text = st.text(max_size=300, alphabet=st.characters(codec="utf-8"))

# Alphabetic+space filler: cannot match any built-in PII pattern.
safe_filler = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz ", min_size=0, max_size=80
).filter(lambda s: s.strip() != "" or s == "")

SEVERITIES = list(PIISeverity)


def _make_record(response_text: str, prompt_text: str = "p") -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id=str(uuid.uuid4()),
        app_id="prop_app",
        feature_id="prop_feature",
        user_id="prop_user",
        model="prop-model",
        prompt_hash=hash_content(prompt_text),
        response_hash=hash_content(response_text),
        prompt_tokens=None,
        response_tokens=None,
        latency_ms=None,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )


# --- hash / verify round trip ------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(text=utf8_text, secret=utf8_text.filter(lambda s: len(s) >= 1))
@example(text="", secret="k" * 40)
@example(text="pïî q-réš ✅", secret="🔑" * 33)
def test_hash_verify_round_trip_accepts_honest_content(text: str, secret: str) -> None:
    backend = SQLiteBackend("sqlite:///:memory:")
    backend.create_tables()
    record = _make_record(text)
    backend.write(record, sign_record(record, secret))

    result = verify_record(record.content_id, text, backend, secret)

    assert result.hash_match is True
    assert result.hmac_valid is True
    assert result.drift_detected is False
    assert result.verified is True


@settings(max_examples=50, deadline=None)
@given(
    original=utf8_text,
    tampered=utf8_text,
    secret=utf8_text.filter(lambda s: len(s) >= 1),
)
def test_hash_verify_flags_tampered_content(
    original: str, tampered: str, secret: str
) -> None:
    assume(original != tampered)
    backend = SQLiteBackend("sqlite:///:memory:")
    backend.create_tables()
    record = _make_record(original)
    backend.write(record, sign_record(record, secret))

    result = verify_record(record.content_id, tampered, backend, secret)

    assert result.hash_match is False
    assert result.drift_detected is True
    assert result.verified is False


@settings(max_examples=50, deadline=None)
@given(
    text=utf8_text,
    secret=utf8_text.filter(lambda s: len(s) >= 1),
    wrong_secret=utf8_text.filter(lambda s: len(s) >= 1),
)
def test_hash_verify_rejects_wrong_key(
    text: str, secret: str, wrong_secret: str
) -> None:
    assume(secret != wrong_secret)
    backend = SQLiteBackend("sqlite:///:memory:")
    backend.create_tables()
    record = _make_record(text)
    backend.write(record, sign_record(record, secret))

    result = verify_record(record.content_id, text, backend, wrong_secret)

    # Content is intact but the signature does not match the presented key.
    assert result.hash_match is True
    assert result.hmac_valid is False
    assert result.verified is False


@given(text=utf8_text)
def test_hash_content_is_deterministic_sha256_hex(text: str) -> None:
    digest = hash_content(text)
    assert digest == hash_content(text)
    assert len(digest) == 64
    assert all(char in "0123456789abcdef" for char in digest)


# --- redaction: snippets never leak the matched span -------------------------


@given(
    prefix=safe_filler,
    digits=st.integers(min_value=0, max_value=999_999),
    suffix=safe_filler,
)
@example(prefix="", digits=0, suffix="")
def test_redacted_snippets_never_leak_matched_spans(
    prefix: str, digits: int, suffix: str
) -> None:
    email = f"user{digits}@example.com"
    text = f"{prefix} {email} {suffix}"

    matches = scan_text(text)

    assert matches, "the constructed email must be detected"
    for match in matches:
        matched_span = text[match.start : match.end]
        assert matched_span not in match.redacted_snippet, (
            f"leaked {matched_span!r} in {match.redacted_snippet!r}"
        )
        assert "[REDACTED]" in match.redacted_snippet


# --- policy severity monotonicity --------------------------------------------


def _record_with_severity(severity: PIISeverity) -> ProvenanceRecord:
    return _make_record("response").model_copy(
        update={
            "pii_result": PIIResult(
                prompt_matches=[],
                response_matches=[],
                highest_severity=severity,
                match_count=1,
            )
        }
    )


@given(
    rule_severity=st.sampled_from(SEVERITIES),
    record_severity=st.sampled_from(SEVERITIES),
)
def test_block_rule_severity_is_monotonic(
    rule_severity: PIISeverity, record_severity: PIISeverity
) -> None:
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                name="block_at_severity",
                conditions=RuleConditions(pii_severity=rule_severity),
                action=PolicyAction.BLOCK,
            )
        ],
        model_tiers={"prop-model": "approved"},
    )
    record = _record_with_severity(record_severity)
    should_block = SEVERITY_RANK[record_severity] >= SEVERITY_RANK[rule_severity]

    if should_block:
        try:
            engine.evaluate(record)
        except PolicyViolationError:
            pass
        else:  # pragma: no cover - assertion guard
            raise AssertionError(
                f"expected BLOCK: rule={rule_severity} record={record_severity}"
            )
    else:
        decision = engine.evaluate(record)
        assert decision.action == PolicyAction.ALLOW
