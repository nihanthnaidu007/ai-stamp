"""Tests for PII v2: redaction, confidence, overlap arbitration, modern
key formats, locale packs, allowlists, and NER hardening."""

from __future__ import annotations

import logging
import random
import re
import sys
import types
from itertools import pairwise
from pathlib import Path

import pytest

from aistamp.models import PIIMatch, PIIResult, PIISeverity
from aistamp.pii import (
    BUILT_IN_PATTERNS,
    LOCALE_PACKS,
    NERConfig,
    PatternConfig,
    get_locale_patterns,
    load_patterns_from_yaml,
    redact_prompt_and_response,
    redact_text,
    scan_prompt_and_response,
    scan_text,
)
from aistamp.pii.validators import VALIDATOR_REJECTED_CONFIDENCE

# ---------------------------------------------------------------------------
# Test samples. GitHub-style tokens are built at runtime — realistic token
# literals must never appear in committed source (secret scanners and push
# protection reject them).
# ---------------------------------------------------------------------------

SK_PROJ_KEY = "sk-proj-abcDEF123ghiJKL456mnoPQR789stuVWX012345"
SK_ANT_KEY = "sk-ant-api03-tuvwXYZA0123BCDE4567FGHI9012JKLM3456MNOP7890QRSTefgh"
SK_LEGACY_KEY = "sk-abcdefghijklmnopqrstuvwxyz1234567890"
JWT_SAMPLE = (
    "eyJhbGciOiJIUzI1NiJ9.eyJmb28iOiJiYXIifQ.4x_H4fakeSignatureChars123"
)
GH_CLASSIC_TOKEN = "gh" + "p_" + "ABCDEFGH" + "IJKLMNOP" + "QRST" + "12"
GH_OAUTH_TOKEN = "gh" + "o_" + "Z" * 24
GH_PAT_TOKEN = "github" + "_pat_" + "a1B2c3D4e5F6" + "g7H8i9J0k1L2m3N4o5P6q7R8"
AADHAAR_VALID = "2345 6789 1238"
AADHAAR_VALID_CONTIGUOUS = "234567891238"
IBAN_VALID = "DE89 3704 0044 0532 0130 00"
IBAN_VALID_CONTIGUOUS = "DE89370400440532013000"
IBAN_INVALID = "DE89 3704 0044 0532 0130 01"
NINO_VALID = "AB123456C"
STEUER_CLEAN = "65432109850"
STEUER_NO_PAIR = "12345678901"


def matches_named(text: str, name: str, locale: str | None = None) -> list[PIIMatch]:
    """scan_text matches filtered by pattern name (locale pack optional)."""
    return [m for m in scan_text(text, locale=locale) if m.pattern_name == name]


# ---------------------------------------------------------------------------
# Group 1 — modern API key formats
# ---------------------------------------------------------------------------


def test_sk_proj_key_detected() -> None:
    matches = matches_named(f"key {SK_PROJ_KEY} end", "API_KEY")
    assert len(matches) == 1


def test_sk_ant_api03_key_detected() -> None:
    matches = matches_named(f"key {SK_ANT_KEY} end", "API_KEY")
    assert len(matches) == 1


def test_modern_key_char_class_captures_full_token() -> None:
    # The fixed char class must include '-' and '_' inside the token body.
    text = f"key {SK_PROJ_KEY} end"
    matches = [m for m in scan_text(text) if m.pattern_name == "API_KEY"]
    assert len(matches) == 1
    assert text[matches[0].start : matches[0].end] == SK_PROJ_KEY


def test_legacy_sk_key_still_detected() -> None:
    # 0.1.x compatibility: bare sk- keys keep matching.
    matches = scan_text(f"key {SK_LEGACY_KEY} end")
    assert any(m.pattern_name == "API_KEY" for m in matches)


def test_github_classic_token_detected() -> None:
    text = f"token {GH_CLASSIC_TOKEN} and {GH_OAUTH_TOKEN}"
    matches = [m for m in scan_text(text) if m.pattern_name == "API_KEY"]
    assert len(matches) == 2


def test_github_fine_grained_pat_detected() -> None:
    text = f"token {GH_PAT_TOKEN}"
    assert any(m.pattern_name == "API_KEY" for m in scan_text(text))


def test_jwt_detected() -> None:
    matches = scan_text(f"auth {JWT_SAMPLE} end")
    assert any(m.pattern_name == "JWT" for m in matches)


def test_jwt_severity_is_high() -> None:
    matches = [m for m in scan_text(f"auth {JWT_SAMPLE}") if m.pattern_name == "JWT"]
    assert matches[0].severity == PIISeverity.HIGH


def test_bearer_token_detected() -> None:
    matches = scan_text("Authorization: Bearer abcdefghijklmnopqrstuvwx123456")
    assert any(m.pattern_name == "API_KEY" for m in matches)


def test_bearer_jwt_prefers_longer_bearer_span() -> None:
    # "Bearer <jwt>" matches both the Bearer arm and the JWT pattern;
    # arbitration must report a single, longest (Bearer) match.
    text = f"Authorization: Bearer {JWT_SAMPLE}"
    matches = scan_text(text)
    assert len(matches) == 1
    assert matches[0].pattern_name == "API_KEY"


def test_all_modern_keys_redacted_without_leak() -> None:
    samples = [
        SK_PROJ_KEY,
        SK_ANT_KEY,
        SK_LEGACY_KEY,
        GH_CLASSIC_TOKEN,
        GH_OAUTH_TOKEN,
        GH_PAT_TOKEN,
        JWT_SAMPLE,
        "Bearer abcdefghijklmnopqrstuvwx123456",
        "AKIAIOSFODNN7EXAMPLE",
    ]
    for sample in samples:
        text = f"payload: {sample} trailing"
        out = redact_text(text)
        assert sample not in out, f"leaked {sample[:12]}..."
        assert out.startswith("payload: [REDACTED]")


# ---------------------------------------------------------------------------
# Group 2 — confidence scoring
# ---------------------------------------------------------------------------


def test_credit_card_confidence_full() -> None:
    matches = matches_named("Card 4111 1111 1111 1111", "CREDIT_CARD")
    assert matches[0].confidence == 1.0


def test_amex_15_digit_detected() -> None:
    matches = matches_named("Card 3782 822463 10005", "CREDIT_CARD")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_amex_15_digit_contiguous_detected() -> None:
    matches = matches_named("Card 378282246310005", "CREDIT_CARD")
    assert len(matches) == 1


def test_ssn_valid_confidence_full() -> None:
    matches = [m for m in scan_text("SSN 123-45-6789") if m.pattern_name == "SSN"]
    assert matches[0].confidence == 1.0


def test_ssn_900_series_rejected() -> None:
    # 900-999 area series was never issued (used by ITINs instead).
    # Fail-closed (audit P1-4): kept at rejected confidence, not dropped.
    matches = matches_named("SSN 987-65-4321", "SSN")
    assert len(matches) == 1
    assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_ssn_000_and_666_areas_rejected() -> None:
    for text in ("SSN 000-12-3456", "SSN 666-12-3456"):
        matches = matches_named(text, "SSN")
        assert len(matches) == 1
        assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_ssn_group_00_and_serial_0000_rejected() -> None:
    for text in ("SSN 123-00-4567", "SSN 123-45-0000"):
        matches = matches_named(text, "SSN")
        assert len(matches) == 1
        assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_phone_account_number_low_confidence() -> None:
    # 10-digit account numbers keep phone *shape* but fail NANP
    # plausibility: they surface only at rejected confidence (audit P1-4
    # fail-closed — redaction covers them, policy filters on confidence).
    for text in ("Account 1002003004", "Account 100-200-3004"):
        matches = matches_named(text, "PHONE_US")
        assert len(matches) == 1
        assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_phone_implausible_exchange_low_confidence() -> None:
    # Existing 0.1.x sample: still a match, downgraded, not dropped.
    matches = matches_named("Call 555-123-4567", "PHONE_US")
    assert len(matches) == 1
    assert matches[0].confidence == 0.5


def test_rejected_candidate_redacted_fail_closed() -> None:
    # Audit P1-4 PoV: a Luhn-failing card next to a valid email. The card
    # span must be redacted AND absent raw from every persisted snippet.
    raw = "email john.doe@corp.example card 1234-5678-9012-3456 thanks"
    card = "1234-5678-9012-3456"
    matches = scan_text(raw)
    assert any(m.pattern_name == "CREDIT_CARD" for m in matches)

    assert card not in redact_text(raw)

    for match in matches:
        assert card not in match.redacted_snippet


def test_snippet_fill_never_embeds_raw_neighbor_matches() -> None:
    # General snippet guarantee (audit P1-4): no match's raw value — nor a
    # fragment of one straddling the context-window edge — may appear raw
    # in another match's persisted snippet.
    text = "pad user1@example.com mid user2@example.com tail"
    matches = scan_text(text)
    assert len(matches) == 2
    for match in matches:
        assert "@example.com" not in match.redacted_snippet
        for other in matches:
            if other is match:
                continue
            assert text[other.start : other.end] not in match.redacted_snippet


def test_phone_fictional_555_line_low_confidence() -> None:
    matches = matches_named("Call 212-555-0199", "PHONE_US")
    assert len(matches) == 1
    assert matches[0].confidence == 0.7


def test_phone_plausible_full_confidence() -> None:
    matches = matches_named("Call (415) 735-0238", "PHONE_US")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_phone_plausible_ordered_above_implausible() -> None:
    plausible = scan_text("Call (415) 735-0238")[0].confidence
    implausible = scan_text("Call 555-123-4567")[0].confidence
    assert plausible > implausible


def test_email_degenerate_shape_low_confidence() -> None:
    matches = [m for m in scan_text("mail a@b.com") if m.pattern_name == "EMAIL"]
    assert len(matches) == 1
    assert matches[0].confidence == 0.7


def test_email_normal_full_confidence() -> None:
    matches = matches_named("mail alice@example.com", "EMAIL")
    assert matches[0].confidence == 1.0


def test_email_confidence_ordering() -> None:
    normal = scan_text("mail alice@example.com")[0].confidence
    degenerate = scan_text("mail a@b.com")[0].confidence
    assert normal > degenerate


def test_public_ip_full_private_ip_lower() -> None:
    public = scan_text("host 8.8.8.8")[0]
    private = scan_text("host 192.168.1.100")[0]
    assert public.confidence == 1.0
    assert private.confidence == 0.7
    assert public.confidence > private.confidence


def test_api_key_bearer_lower_than_vendor_prefixed() -> None:
    vendor = scan_text(f"key {SK_PROJ_KEY}")[0].confidence
    bearer = matches_named(
        "Authorization: Bearer abcdefghijklmnopqrstuvwx123456", "API_KEY"
    )[0].confidence
    assert vendor == 1.0
    assert bearer == 0.9
    assert vendor > bearer


def test_confidence_within_documented_scale() -> None:
    text = (
        f"mail alice@example.com a@b.com 8.8.8.8 192.168.1.100 "
        f"SSN 123-45-6789 card 4111 1111 1111 1111 key {SK_PROJ_KEY}"
    )
    for match in scan_text(text):
        assert 0.0 < match.confidence <= 1.0, match


# ---------------------------------------------------------------------------
# Group 3 — overlap resolution
# ---------------------------------------------------------------------------


def _email_clone_config() -> PatternConfig:
    return PatternConfig(
        name="EMAIL_CLONE",
        pattern=r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        severity=PIISeverity.HIGH,
    )


def test_equal_span_competing_patterns_single_match() -> None:
    text = "email john@example.com here"
    matches = scan_text(text, extra_patterns=[_email_clone_config()])
    assert len(matches) == 1
    # Tie on span -> higher severity wins (EMAIL_CLONE is HIGH).
    assert matches[0].pattern_name == "EMAIL_CLONE"


def test_longest_match_wins_nested() -> None:
    wrapper = PatternConfig(
        name="CUSTOM_ID",
        pattern=r"ID:\s*\d{3}-\d{2}-\d{4}",
        severity=PIISeverity.HIGH,
    )
    text = "ref ID: 123-45-6789 ok"
    matches = scan_text(text, extra_patterns=[wrapper])
    assert len(matches) == 1
    assert matches[0].pattern_name == "CUSTOM_ID"
    assert text[matches[0].start : matches[0].end] == "ID: 123-45-6789"


def test_results_sorted_by_position() -> None:
    text = "a@b.com then 123-45-6789 then 8.8.8.8"
    starts = [m.start for m in scan_text(text)]
    assert starts == sorted(starts)


def test_resolve_overlaps_false_returns_raw_union() -> None:
    text = "email john@example.com here"
    raw = scan_text(
        text, extra_patterns=[_email_clone_config()], resolve_overlaps=False
    )
    assert len(raw) == 2


def test_match_count_not_double_counted() -> None:
    result = scan_prompt_and_response(
        "email john@example.com here",
        "",
        extra_patterns=[_email_clone_config()],
    )
    assert result.match_count == 1
    assert len(result.prompt_matches) == 1


def test_two_separate_emails_both_kept() -> None:
    matches = matches_named("From a@x.com to b@y.com", "EMAIL")
    assert len(matches) == 2


def test_no_overlapping_spans_after_arbitration() -> None:
    text = (
        f"mail alice@example.com SSN 123-45-6789 card 4111 1111 1111 1111 "
        f"key {SK_PROJ_KEY} host 8.8.8.8"
    )
    matches = scan_text(text)
    ordered = sorted(matches, key=lambda m: m.start)
    for first, second in pairwise(ordered):
        assert first.end <= second.start


# ---------------------------------------------------------------------------
# Group 4 — redaction
# ---------------------------------------------------------------------------


def test_redact_text_convenience_scans_first() -> None:
    text = "Contact alice@example.com or SSN 123-45-6789"
    assert redact_text(text) == redact_text(text, scan_text(text))
    out = redact_text(text)
    assert "alice@example.com" not in out
    assert "123-45-6789" not in out
    assert "[REDACTED]" in out


def test_redact_text_custom_placeholder() -> None:
    out = redact_text("mail alice@example.com end", placeholder="<PII>")
    assert "<PII>" in out
    assert "[REDACTED]" not in out


def test_redact_text_empty_placeholder_deletes() -> None:
    out = redact_text("mail alice@example.com end", placeholder="")
    assert out == "mail  end"


def test_redact_text_union_of_overlapping_matches() -> None:
    # Hand-built overlapping matches: redaction must merge spans (union)
    # so no fragment of the matched value survives.
    text = "token abcdefghijklmnopqrstuv"
    matches = [
        PIIMatch(
            pattern_name="A",
            severity=PIISeverity.HIGH,
            start=6,
            end=20,
            redacted_snippet="",
        ),
        PIIMatch(
            pattern_name="B",
            severity=PIISeverity.LOW,
            start=10,
            end=26,
            redacted_snippet="",
        ),
    ]
    out = redact_text(text, matches)
    # Union of [6,20) and [10,26) -> one span [6,26); the trailing "uv"
    # (26..28) lies outside both matches and legitimately survives.
    assert out == "token [REDACTED]uv"
    for value in ("abcdefgh", "klmnopqrstuv"):
        assert value not in out


def test_redact_text_clamps_out_of_range_matches() -> None:
    text = "short"
    matches = [
        PIIMatch(
            pattern_name="X",
            severity=PIISeverity.HIGH,
            start=-10,
            end=99,
            redacted_snippet="",
        )
    ]
    assert redact_text(text, matches) == "[REDACTED]"


def test_redact_text_drops_empty_spans() -> None:
    text = "untouched"
    matches = [
        PIIMatch(
            pattern_name="X",
            severity=PIISeverity.LOW,
            start=3,
            end=3,
            redacted_snippet="",
        )
    ]
    assert redact_text(text, matches) == text


def test_redact_text_type_error_for_non_string() -> None:
    with pytest.raises(TypeError):
        redact_text(12345)  # type: ignore[arg-type]


def test_redact_text_empty_string() -> None:
    assert redact_text("") == ""


def test_redact_prompt_and_response_scrubs_both() -> None:
    prompt, response = redact_prompt_and_response(
        "mail alice@example.com",
        "SSN 123-45-6789 on file",
    )
    assert "alice@example.com" not in prompt
    assert "123-45-6789" not in response
    assert "[REDACTED]" in prompt
    assert "[REDACTED]" in response


def test_redaction_never_leaks_random_interleavings() -> None:
    # Deterministic property-style fuzz: random interleavings of known PII
    # values with plain filler must come out fully scrubbed.
    rng = random.Random(20260917)
    pii_pool = [
        "alice@example.com",
        "bob.smith@corp.io",
        "123-45-6789",
        "4111 1111 1111 1111",
        SK_PROJ_KEY,
        SK_ANT_KEY,
        "8.8.8.8",
        "(415) 735-0238",
        JWT_SAMPLE,
    ]  # Locale-only values (Aadhaar, IBAN) are covered by dedicated tests;
    # redact_text scans with the default locale, so they must not appear here.
    filler_words = ["alpha", "beta", "gamma", "delta", "epsilon", "note", "log"]
    for _ in range(300):
        values = rng.sample(pii_pool, rng.randint(1, 4))
        parts: list[str] = []
        for value in values:
            parts.append(" ".join(rng.sample(filler_words, rng.randint(1, 3))))
            parts.append(value)
        parts.append(rng.choice(filler_words))
        text = " ".join(parts)

        redacted = redact_text(text)
        for value in values:
            assert value not in redacted, f"leaked {value!r} in {redacted!r}"


def test_redaction_never_leaks_raw_union_mode() -> None:
    text = "email john@example.com here"
    raw_matches = scan_text(
        text, extra_patterns=[_email_clone_config()], resolve_overlaps=False
    )
    assert len(raw_matches) == 2
    out = redact_text(text, raw_matches)
    assert "john@example.com" not in out


# ---------------------------------------------------------------------------
# Group 5 — locale packs
# ---------------------------------------------------------------------------


def test_aadhaar_valid_grouped_detected() -> None:
    matches = matches_named(f"ID {AADHAAR_VALID}", "AADHAAR", locale="INDIA")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0
    assert matches[0].severity == PIISeverity.HIGH


def test_aadhaar_valid_contiguous_detected() -> None:
    matches = matches_named(
        f"ID {AADHAAR_VALID_CONTIGUOUS}", "AADHAAR", locale="INDIA"
    )
    assert len(matches) == 1


def test_aadhaar_bad_checksum_rejected() -> None:
    # Fail-closed (audit P1-4): bad Verhoeff stays a match at rejected
    # confidence so redaction still covers the span.
    matches = matches_named("ID 2345 6789 1237", "AADHAAR", locale="INDIA")
    assert len(matches) == 1
    assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_aadhaar_not_detected_without_locale() -> None:
    assert not any(
        m.pattern_name == "AADHAAR" for m in scan_text(f"ID {AADHAAR_VALID}")
    )


def test_pan_detected() -> None:
    matches = matches_named("PAN ABCPD1234F", "PAN", locale="INDIA")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_pan_unknown_holder_type_lower_confidence() -> None:
    matches = matches_named("PAN ABXYZ1234F", "PAN", locale="INDIA")
    assert len(matches) == 1
    assert matches[0].confidence == 0.6


def test_india_mobile_detected() -> None:
    for sample in ("+91 9876543210", "09876543210", "9876543210"):
        matches = matches_named(f"call {sample}", "INDIA_MOBILE", locale="INDIA")
        assert len(matches) == 1, sample


def test_india_mobile_invalid_prefix_rejected() -> None:
    matches = scan_text("call 5876543210", locale="INDIA")
    assert not any(m.pattern_name == "INDIA_MOBILE" for m in matches)


def test_iban_valid_grouped_detected() -> None:
    matches = matches_named(f"pay {IBAN_VALID}", "IBAN", locale="EU")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_iban_valid_contiguous_detected() -> None:
    matches = matches_named(f"pay {IBAN_VALID_CONTIGUOUS}", "IBAN", locale="EU")
    assert len(matches) == 1


def test_iban_bad_checksum_rejected() -> None:
    # Fail-closed (audit P1-4): mod-97 failure stays a match at rejected
    # confidence so redaction still covers the span.
    matches = matches_named(f"pay {IBAN_INVALID}", "IBAN", locale="EU")
    assert len(matches) == 1
    assert matches[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


def test_nino_detected() -> None:
    matches = matches_named(f"NINO {NINO_VALID}", "NINO", locale="EU")
    assert len(matches) == 1


def test_nino_reserved_prefixes_rejected() -> None:
    for prefix in ("BG", "GB", "NK", "TN", "ZZ"):
        text = f"NINO {prefix}123456A"
        assert not any(
            m.pattern_name == "NINO" for m in scan_text(text, locale="EU")
        ), prefix


def test_steuer_id_clean_structure_full_confidence() -> None:
    matches = matches_named(f"tax {STEUER_CLEAN}", "STEUER_ID", locale="EU")
    assert len(matches) == 1
    assert matches[0].confidence == 1.0


def test_steuer_id_without_pair_lower_confidence() -> None:
    # Never dropped — the pair rule modulates confidence only.
    matches = matches_named(f"tax {STEUER_NO_PAIR}", "STEUER_ID", locale="EU")
    assert len(matches) == 1
    assert matches[0].confidence == 0.5


def test_unknown_locale_raises() -> None:
    with pytest.raises(ValueError, match="Unknown locale"):
        scan_text("hello", locale="MARS")


def test_get_locale_patterns_case_insensitive() -> None:
    assert get_locale_patterns("india") == LOCALE_PACKS["INDIA"]
    assert get_locale_patterns("eu") == LOCALE_PACKS["EU"]


def test_locale_pack_patterns_are_tagged() -> None:
    for name, pack in LOCALE_PACKS.items():
        for pattern in pack:
            assert pattern.locale == name


def test_locale_kwarg_works_on_scan_prompt_and_response() -> None:
    result = scan_prompt_and_response(f"ID {AADHAAR_VALID}", "", locale="INDIA")
    assert any(m.pattern_name == "AADHAAR" for m in result.prompt_matches)


# ---------------------------------------------------------------------------
# Group 6 — allowlists and pattern config
# ---------------------------------------------------------------------------


def test_global_allowlist_exact_value() -> None:
    assert scan_text("server 10.0.0.1", allowlist=["10.0.0.1"]) == []
    assert any(
        m.pattern_name == "IP_ADDRESS"
        for m in scan_text("server 8.8.8.8", allowlist=["10.0.0.1"])
    )


def test_global_allowlist_regex_entries() -> None:
    matches = scan_text(
        "servers 10.0.0.1 and 10.0.0.9 and 8.8.8.8",
        allowlist=[r"regex:10\.0\.0\.\d+"],
    )
    names = [m.pattern_name for m in matches]
    assert names == ["IP_ADDRESS"]  # only 8.8.8.8 survives
    assert matches[0].redacted_snippet.count("[REDACTED]") == 1


def test_global_allowlist_regex_is_fullmatch() -> None:
    # A regex entry must not partially allow unrelated values.
    matches = scan_text("server 110.0.0.1", allowlist=[r"regex:10\.0\.0\.1"])
    assert any(m.pattern_name == "IP_ADDRESS" for m in matches)


def test_per_pattern_allowlist() -> None:
    # Per-pattern allowlists scope to that pattern config only; a unique
    # name keeps the built-in EMAIL pattern out of the assertion.
    scoped = PatternConfig(
        name="WORK_EMAIL",
        pattern=r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}",
        severity=PIISeverity.MEDIUM,
        allowlist=("user@example.com",),
    )
    matches = scan_text(
        "from user@example.com and admin@example.com",
        extra_patterns=[scoped],
        # Raw union: identical spans across pattern names would otherwise be
        # arbitracted down to one, hiding the per-pattern allowlist effect.
        resolve_overlaps=False,
    )
    work_emails = [m for m in matches if m.pattern_name == "WORK_EMAIL"]
    assert len(work_emails) == 1
    assert "admin@example.com" not in work_emails[0].redacted_snippet


def test_invalid_allowlist_regex_raises_loudly() -> None:
    with pytest.raises(re.error):
        scan_text("server 10.0.0.1", allowlist=["regex:([bad"])


def test_pattern_config_new_fields_default() -> None:
    config = PatternConfig(name="X", pattern="X", severity=PIISeverity.LOW)
    assert config.locale is None
    assert config.confidence == 1.0
    assert config.version == 1
    assert config.allowlist == ()


def test_pattern_config_invalid_confidence_raises() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PatternConfig(name="X", pattern="X", severity=PIISeverity.LOW, confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        PatternConfig(name="X", pattern="X", severity=PIISeverity.LOW, confidence=0.0)


def test_pattern_config_invalid_version_raises() -> None:
    with pytest.raises(ValueError, match="version"):
        PatternConfig(name="X", pattern="X", severity=PIISeverity.LOW, version=0)


def test_pattern_config_invalid_regex_raises() -> None:
    with pytest.raises(ValueError, match="regex"):
        PatternConfig(name="X", pattern="(?P<oops>", severity=PIISeverity.LOW)


def test_yaml_new_fields_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "patterns.yaml"
    path.write_text(
        "patterns:\n"
        "  - name: BADGE\n"
        '    pattern: "BDG-\\\\d{4}"\n'
        "    severity: LOW\n"
        "    locale: INDIA\n"
        "    confidence: 0.9\n"
        "    version: 3\n"
        "    allowlist:\n"
        '      - "BDG-0001"\n'
    )
    patterns = load_patterns_from_yaml(path)
    assert len(patterns) == 1
    pattern = patterns[0]
    assert pattern.locale == "INDIA"
    assert pattern.confidence == 0.9
    assert pattern.version == 3
    assert pattern.allowlist == ("BDG-0001",)


def test_yaml_duplicate_pattern_names_raise(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text(
        "patterns:\n"
        "  - name: X\n"
        '    pattern: "A"\n'
        "    severity: LOW\n"
        "  - name: X\n"
        '    pattern: "B"\n'
        "    severity: HIGH\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_patterns_from_yaml(path)


def test_extra_pattern_shadowing_builtin_rejected() -> None:
    # A custom pattern reusing a built-in name would silently race the
    # original during arbitration and split per-pattern config lookups.
    shadow = PatternConfig(
        name="CREDIT_CARD",
        pattern=r"\b\d{16}\b",
        severity=PIISeverity.HIGH,
    )
    with pytest.raises(ValueError, match="CREDIT_CARD"):
        scan_text("Card 4111111111111111", extra_patterns=[shadow])


def test_extra_pattern_shadowing_locale_pattern_rejected() -> None:
    shadow = PatternConfig(
        name="AADHAAR",
        pattern=r"\b\d{12}\b",
        severity=PIISeverity.HIGH,
    )
    with pytest.raises(ValueError, match="AADHAAR"):
        scan_text("ID 2346 2678 9001", extra_patterns=[shadow], locale="INDIA")


def test_duplicate_extra_pattern_names_rejected() -> None:
    dup = PatternConfig(
        name="DOUBLE_ENTRY",
        pattern=r"x",
        severity=PIISeverity.LOW,
    )
    with pytest.raises(ValueError, match="DOUBLE_ENTRY"):
        scan_text("x", extra_patterns=[dup, dup])


def test_distinct_extra_pattern_names_still_accepted() -> None:
    # Guard the rejection rule against over-blocking: a unique extra name
    # next to built-ins and a locale pack must scan normally.
    extra = PatternConfig(
        name="EMPLOYEE_ID",
        pattern=r"EMP-\d{6}",
        severity=PIISeverity.LOW,
    )
    matches = scan_text(
        "Employee EMP-123456, id 2346 2678 9001",
        extra_patterns=[extra],
        locale="INDIA",
    )
    names = {m.pattern_name for m in matches}
    assert "EMPLOYEE_ID" in names
    assert "AADHAAR" in names


def test_yaml_confidence_out_of_range_raises(tmp_path: Path) -> None:
    path = tmp_path / "conf.yaml"
    path.write_text(
        "patterns:\n  - name: X\n    pattern: \"A\"\n    severity: LOW\n"
        "    confidence: 2.0\n"
    )
    with pytest.raises(ValueError, match="confidence"):
        load_patterns_from_yaml(path)


def test_yaml_confidence_non_number_raises(tmp_path: Path) -> None:
    path = tmp_path / "conf_str.yaml"
    path.write_text(
        "patterns:\n  - name: X\n    pattern: \"A\"\n    severity: LOW\n"
        "    confidence: high\n"
    )
    with pytest.raises(ValueError, match="confidence"):
        load_patterns_from_yaml(path)


def test_yaml_version_non_integer_raises(tmp_path: Path) -> None:
    path = tmp_path / "ver.yaml"
    path.write_text(
        "patterns:\n  - name: X\n    pattern: \"A\"\n    severity: LOW\n"
        "    version: v2\n"
    )
    with pytest.raises(ValueError, match="version"):
        load_patterns_from_yaml(path)


def test_yaml_allowlist_respected_in_scan(tmp_path: Path) -> None:
    path = tmp_path / "allow.yaml"
    path.write_text(
        "patterns:\n"
        "  - name: BADGE\n"
        '    pattern: "BDG-\\\\d{4}"\n'
        "    severity: LOW\n"
        "    allowlist:\n"
        '      - "BDG-0001"\n'
    )
    patterns = load_patterns_from_yaml(path)
    assert scan_text("badge BDG-0001", extra_patterns=patterns) == []
    assert any(
        m.pattern_name == "BADGE"
        for m in scan_text("badge BDG-0002", extra_patterns=patterns)
    )


def test_compiled_pattern_cache_reuse() -> None:
    from aistamp.pii import scanner as scanner_mod

    probe = PatternConfig(
        name="CACHE_PROBE", pattern=r"CACHE-\d+", severity=PIISeverity.LOW
    )
    scanner_mod._compile_pattern.cache_clear()
    first = scanner_mod._compile_pattern(probe)
    second = scanner_mod._compile_pattern(probe)
    assert first is second


def test_built_in_patterns_still_compile() -> None:
    assert len(BUILT_IN_PATTERNS) >= 9
    for pattern in BUILT_IN_PATTERNS:
        re.compile(pattern.pattern)


# ---------------------------------------------------------------------------
# Group 7 — NER hardening
# ---------------------------------------------------------------------------


class _FakeEntity:
    def __init__(self, label: str, start: int, end: int) -> None:
        self.label_ = label
        self.start_char = start
        self.end_char = end


class _FakeDoc:
    def __init__(self, entities: list[_FakeEntity]) -> None:
        self.ents = entities


class _FakeNLP:
    def __init__(self, entities: list[_FakeEntity]) -> None:
        self._entities = entities

    def __call__(self, text: str) -> _FakeDoc:
        return _FakeDoc(self._entities)


def test_spacy_missing_model_warns_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    from aistamp.pii import ner as ner_mod

    def boom(model_name: str) -> object:
        raise OSError(f"model {model_name} not found")

    monkeypatch.setattr(ner_mod, "load_nlp", boom)
    with caplog.at_level(logging.WARNING, logger="aistamp.pii"):
        matches = scan_text("email alice@example.com", use_spacy=True)
    warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warning_records, "expected a WARN when NER is requested but unavailable"
    assert "spacy download" in warning_records[0].getMessage()
    # Regex results still returned; no exception.
    assert [m.pattern_name for m in matches] == ["EMAIL"]


def test_spacy_missing_library_warns_loudly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "spacy", None)  # forces ImportError
    config = NERConfig(model_name="unavailable-model")
    with caplog.at_level(logging.WARNING, logger="aistamp.pii"):
        matches = scan_text(
            "email alice@example.com", use_spacy=True, ner_config=config
        )
    assert any(r.levelno == logging.WARNING for r in caplog.records)
    assert [m.pattern_name for m in matches] == ["EMAIL"]


def test_spacy_model_loaded_once_per_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Inject a fake spacy module so the test does not depend on spaCy being
    # installed; load_nlp's import finds it via sys.modules either way.
    from aistamp.pii import ner as ner_mod

    calls = {"count": 0}

    def fake_load(model_name: str) -> _FakeNLP:
        calls["count"] += 1
        return _FakeNLP([])

    fake_spacy = types.ModuleType("spacy")
    fake_spacy.load = fake_load  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)
    ner_mod.load_nlp.cache_clear()
    try:
        ner_mod.scan_with_ner("one", NERConfig())
        ner_mod.scan_with_ner("two", NERConfig())
        assert calls["count"] == 1
    finally:
        ner_mod.load_nlp.cache_clear()


def test_ner_config_custom_labels_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    from aistamp.pii import ner as ner_mod

    entities = [
        _FakeEntity("PERSON", 0, 5),
        _FakeEntity("LOC", 6, 9),
    ]
    monkeypatch.setattr(ner_mod, "load_nlp", lambda name: _FakeNLP(entities))
    config = NERConfig(labels=frozenset({"PERSON", "LOC"}))
    matches = ner_mod.scan_with_ner("Alice in Paris", config)
    assert [m.pattern_name for m in matches] == ["PERSON", "LOC"]

    persons_only = NERConfig(labels=frozenset({"PERSON"}))
    matches = ner_mod.scan_with_ner("Alice in Paris", persons_only)
    assert [m.pattern_name for m in matches] == ["PERSON"]


def test_ner_config_severity_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    from aistamp.pii import ner as ner_mod

    entities = [_FakeEntity("PERSON", 0, 5)]
    monkeypatch.setattr(ner_mod, "load_nlp", lambda name: _FakeNLP(entities))
    config = NERConfig(
        labels=frozenset({"PERSON"}),
        severity_by_label={"PERSON": PIISeverity.HIGH},
    )
    matches = ner_mod.scan_with_ner("Alice", config)
    assert matches[0].severity == PIISeverity.HIGH


def test_ner_default_confidence(monkeypatch: pytest.MonkeyPatch) -> None:
    from aistamp.pii import ner as ner_mod

    entities = [_FakeEntity("PERSON", 0, 5)]
    monkeypatch.setattr(ner_mod, "load_nlp", lambda name: _FakeNLP(entities))
    matches = ner_mod.scan_with_ner("Alice", NERConfig())
    assert matches[0].confidence == 0.85


def test_scan_text_with_ner_config_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from aistamp.pii import ner as ner_mod

    entities = [_FakeEntity("PERSON", 0, 5)]
    monkeypatch.setattr(ner_mod, "load_nlp", lambda name: _FakeNLP(entities))
    matches = scan_text("Alice met Bob", use_spacy=True)
    persons = [m for m in matches if m.pattern_name == "PERSON"]
    assert len(persons) == 1
    assert persons[0].confidence == 0.85
    assert "Alice" not in persons[0].redacted_snippet


# ---------------------------------------------------------------------------
# Group 8 — 0.1.x compatibility
# ---------------------------------------------------------------------------


def test_scan_text_positional_signature_unchanged() -> None:
    custom = PatternConfig(
        name="EMP", pattern=r"EMP-\d+", severity=PIISeverity.LOW
    )
    matches = scan_text("Employee EMP-123", [custom], False)
    assert any(m.pattern_name == "EMP" for m in matches)


def test_pii_match_confidence_defaults_to_full() -> None:
    match = PIIMatch(
        pattern_name="EMAIL",
        severity=PIISeverity.MEDIUM,
        start=0,
        end=5,
        redacted_snippet="",
    )
    assert match.confidence == 1.0


def test_old_payload_without_confidence_roundtrips() -> None:
    payload = {
        "pattern_name": "EMAIL",
        "severity": "MEDIUM",
        "start": 0,
        "end": 16,
        "redacted_snippet": "[REDACTED]",
    }
    match = PIIMatch.model_validate(payload)
    assert match.confidence == 1.0
    dumped = match.model_dump()
    assert dumped["confidence"] == 1.0


def test_pii_result_with_confidence_survives_roundtrip() -> None:
    result = scan_prompt_and_response("mail alice@example.com", "clean")
    dumped = result.model_dump()
    restored = PIIResult.model_validate(dumped)
    assert restored.prompt_matches[0].confidence == 1.0
    assert restored.match_count == result.match_count
