from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from aistamp.fingerprint import generate_content_id, hash_content
from aistamp.models import (
    PIIResult,
    PIISeverity,
    PIIType,
    ProvenanceRecord,
    RecordStatus,
)
from aistamp.pii import (
    PatternConfig,
    load_patterns_from_yaml,
    scan_prompt_and_response,
    scan_text,
)
from aistamp.pii.validators import VALIDATOR_REJECTED_CONFIDENCE
from aistamp.store import SQLiteBackend

# ---------------------------------------------------------------------------
# Group 1 — scan_text: input validation
# ---------------------------------------------------------------------------


def test_scan_text_returns_list() -> None:
    # scan_text() must always return a list, never None.
    assert isinstance(scan_text("hello"), list)


def test_scan_text_empty_string_returns_empty_list() -> None:
    # Empty string is valid input, not an error. Returns no matches.
    assert scan_text("") == []


def test_scan_text_clean_text_returns_empty_list() -> None:
    # Text with no PII must return an empty list.
    assert scan_text("The weather is nice today.") == []


def test_scan_text_raises_type_error_for_non_string() -> None:
    # Non-string input must raise TypeError.
    with pytest.raises(TypeError):
        scan_text(12345)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Group 2 — scan_text: EMAIL pattern
# ---------------------------------------------------------------------------


def test_email_detected_in_text() -> None:
    # A valid email address embedded in text must produce an EMAIL match.
    matches = scan_text("Contact us at support@example.com for help")
    assert any(m.pattern_name == PIIType.EMAIL.value for m in matches)


def test_email_pattern_name_is_correct() -> None:
    # The pattern_name on the match must be PIIType.EMAIL.value.
    matches = scan_text("user@example.com")
    email_matches = [m for m in matches if m.pattern_name == PIIType.EMAIL.value]
    assert len(email_matches) == 1


def test_email_severity_is_medium() -> None:
    # EMAIL severity must be PIISeverity.MEDIUM.
    matches = scan_text("user@example.com")
    email_matches = [m for m in matches if m.pattern_name == PIIType.EMAIL.value]
    assert email_matches[0].severity == PIISeverity.MEDIUM


def test_email_start_end_are_correct() -> None:
    # start and end on the match must correctly point to the email in the text.
    text = "Contact us at support@example.com for help"
    matches = scan_text(text)
    email_matches = [m for m in matches if m.pattern_name == PIIType.EMAIL.value]
    m = email_matches[0]
    assert text[m.start : m.end] == "support@example.com"


def test_email_redacted_snippet_contains_redacted_marker() -> None:
    # The redacted_snippet must contain "[REDACTED]" and not the original email.
    matches = scan_text("Contact us at support@example.com for help")
    email_matches = [m for m in matches if m.pattern_name == PIIType.EMAIL.value]
    snippet = email_matches[0].redacted_snippet
    assert "[REDACTED]" in snippet
    assert "support@example.com" not in snippet


def test_no_email_in_clean_text() -> None:
    # Text without an email must not produce an EMAIL match.
    matches = scan_text("hello world")
    assert not any(m.pattern_name == PIIType.EMAIL.value for m in matches)


# ---------------------------------------------------------------------------
# Group 3 — scan_text: PHONE_US pattern
# ---------------------------------------------------------------------------


def test_phone_us_detected_standard_format() -> None:
    # "(555) 123-4567" must produce a PHONE_US match.
    matches = scan_text("Call me at (555) 123-4567")
    assert any(m.pattern_name == PIIType.PHONE_US.value for m in matches)


def test_phone_us_detected_dashes_format() -> None:
    # "555-123-4567" must produce a PHONE_US match.
    matches = scan_text("Call 555-123-4567 today")
    assert any(m.pattern_name == PIIType.PHONE_US.value for m in matches)


def test_phone_us_severity_is_medium() -> None:
    # PHONE_US severity must be PIISeverity.MEDIUM.
    matches = scan_text("Call 555-123-4567")
    phone_matches = [m for m in matches if m.pattern_name == PIIType.PHONE_US.value]
    assert phone_matches[0].severity == PIISeverity.MEDIUM


# ---------------------------------------------------------------------------
# Group 4 — scan_text: SSN pattern
# ---------------------------------------------------------------------------


def test_ssn_detected_in_text() -> None:
    # "SSN: 123-45-6789" must produce an SSN match.
    matches = scan_text("SSN: 123-45-6789")
    assert any(m.pattern_name == PIIType.SSN.value for m in matches)


def test_ssn_severity_is_high() -> None:
    # SSN severity must be PIISeverity.HIGH.
    matches = scan_text("SSN: 123-45-6789")
    ssn_matches = [m for m in matches if m.pattern_name == PIIType.SSN.value]
    assert ssn_matches[0].severity == PIISeverity.HIGH


def test_ssn_pattern_name_is_correct() -> None:
    # pattern_name must be PIIType.SSN.value.
    matches = scan_text("123-45-6789")
    ssn_matches = [m for m in matches if m.pattern_name == PIIType.SSN.value]
    assert len(ssn_matches) == 1


# ---------------------------------------------------------------------------
# Group 5 — scan_text: CREDIT_CARD pattern
# ---------------------------------------------------------------------------


def test_credit_card_detected_with_spaces() -> None:
    # A Luhn-valid card number with spaces must produce a match.
    matches = scan_text("Card: 4111 1111 1111 1111")
    assert any(m.pattern_name == PIIType.CREDIT_CARD.value for m in matches)


def test_credit_card_detected_with_dashes() -> None:
    # A Luhn-valid card number with dashes must produce a match.
    matches = scan_text("Card: 4111-1111-1111-1111")
    assert any(m.pattern_name == PIIType.CREDIT_CARD.value for m in matches)


def test_credit_card_severity_is_high() -> None:
    # CREDIT_CARD severity must be PIISeverity.HIGH.
    matches = scan_text("4111-1111-1111-1111")
    cc_matches = [m for m in matches if m.pattern_name == PIIType.CREDIT_CARD.value]
    assert cc_matches[0].severity == PIISeverity.HIGH


def test_invalid_credit_card_fails_luhn_validation() -> None:
    # Fail-closed (audit P1-4): a Luhn-failing candidate stays a match at
    # reduced confidence instead of silently vanishing from redaction.
    matches = scan_text("Card: 1111 1111 1111 1111")
    cards = [m for m in matches if m.pattern_name == PIIType.CREDIT_CARD.value]
    assert len(cards) == 1
    assert cards[0].confidence == VALIDATOR_REJECTED_CONFIDENCE


# ---------------------------------------------------------------------------
# Group 6 — scan_text: API_KEY pattern
# ---------------------------------------------------------------------------


def test_openai_sk_key_detected() -> None:
    # "sk-" followed by 20+ alphanumeric characters must match API_KEY.
    matches = scan_text("Using sk-abcdefghijklmnopqrstuvwxyz1234567890 today")
    assert any(m.pattern_name == PIIType.API_KEY.value for m in matches)


def test_aws_akia_key_detected() -> None:
    # "AKIAIOSFODNN7EXAMPLE" must match API_KEY.
    matches = scan_text("My AKIAIOSFODNN7EXAMPLE key")
    assert any(m.pattern_name == PIIType.API_KEY.value for m in matches)


def test_api_key_severity_is_high() -> None:
    # API_KEY severity must be PIISeverity.HIGH.
    matches = scan_text("AKIAIOSFODNN7EXAMPLE")
    api_matches = [m for m in matches if m.pattern_name == PIIType.API_KEY.value]
    assert api_matches[0].severity == PIISeverity.HIGH


# ---------------------------------------------------------------------------
# Group 7 — scan_text: IP_ADDRESS pattern
# ---------------------------------------------------------------------------


def test_ipv4_detected_in_text() -> None:
    # "Server is at 192.168.1.100" must produce an IP_ADDRESS match.
    matches = scan_text("Server is at 192.168.1.100")
    assert any(m.pattern_name == PIIType.IP_ADDRESS.value for m in matches)


def test_ip_address_severity_is_low() -> None:
    # IP_ADDRESS severity must be PIISeverity.LOW.
    matches = scan_text("192.168.1.100")
    ip_matches = [m for m in matches if m.pattern_name == PIIType.IP_ADDRESS.value]
    assert ip_matches[0].severity == PIISeverity.LOW


def test_ip_address_pattern_name_is_correct() -> None:
    # pattern_name must be PIIType.IP_ADDRESS.value.
    matches = scan_text("10.0.0.1")
    ip_matches = [m for m in matches if m.pattern_name == PIIType.IP_ADDRESS.value]
    assert len(ip_matches) == 1


def test_ipv6_detected_in_text() -> None:
    matches = scan_text("Server is at 2001:db8::1")
    assert any(m.pattern_name == PIIType.IP_ADDRESS.value for m in matches)


# ---------------------------------------------------------------------------
# Group 8 — scan_text: multiple patterns in one call
# ---------------------------------------------------------------------------


def test_multiple_pii_types_in_one_text() -> None:
    # Text containing both an email and an SSN must return matches for both.
    matches = scan_text("Contact john@example.com, SSN 123-45-6789")
    names = {m.pattern_name for m in matches}
    assert PIIType.EMAIL.value in names
    assert PIIType.SSN.value in names
    assert len(matches) >= 2


def test_two_emails_in_text_return_two_matches() -> None:
    # Text containing two email addresses must return two EMAIL matches.
    matches = scan_text("From alice@example.com to bob@example.com")
    email_matches = [m for m in matches if m.pattern_name == PIIType.EMAIL.value]
    assert len(email_matches) == 2


def test_redacted_snippet_does_not_expose_neighboring_pii() -> None:
    matches = scan_text("SSN 123-45-6789 email user@example.com")
    snippets = " ".join(match.redacted_snippet for match in matches)
    assert "123-45-6789" not in snippets
    assert "user@example.com" not in snippets


# ---------------------------------------------------------------------------
# Group 9 — scan_text: custom patterns via extra_patterns
# ---------------------------------------------------------------------------


def test_custom_pattern_via_extra_patterns() -> None:
    # A custom PatternConfig passed to extra_patterns must be applied.
    custom = [
        PatternConfig(
            name="EMPLOYEE_ID",
            pattern=r"EMP-\d{6}",
            severity=PIISeverity.HIGH,
        )
    ]
    matches = scan_text("Employee EMP-123456 is assigned", extra_patterns=custom)
    assert any(m.pattern_name == "EMPLOYEE_ID" for m in matches)


def test_custom_pattern_does_not_affect_built_ins() -> None:
    # Adding a custom pattern must not disable the built-in patterns.
    custom = [
        PatternConfig(
            name="EMPLOYEE_ID",
            pattern=r"EMP-\d{6}",
            severity=PIISeverity.HIGH,
        )
    ]
    matches = scan_text(
        "EMP-123456 and contact alice@example.com",
        extra_patterns=custom,
    )
    names = {m.pattern_name for m in matches}
    assert "EMPLOYEE_ID" in names
    assert PIIType.EMAIL.value in names


def test_custom_pattern_severity_is_respected() -> None:
    # A custom pattern with LOW severity must return LOW on its matches.
    custom = [
        PatternConfig(
            name="LOW_THING",
            pattern=r"LOW-\d+",
            severity=PIISeverity.LOW,
        )
    ]
    matches = scan_text("Item LOW-42", extra_patterns=custom)
    low_matches = [m for m in matches if m.pattern_name == "LOW_THING"]
    assert low_matches[0].severity == PIISeverity.LOW


# ---------------------------------------------------------------------------
# Group 10 — scan_text: spaCy fallback
# ---------------------------------------------------------------------------


def test_spacy_disabled_returns_only_regex_results() -> None:
    # use_spacy=False must not include spaCy NER results.
    matches = scan_text("Alice from Acme Corp emailed me", use_spacy=False)
    # No PERSON/ORG labels should appear in regex-only mode.
    names = {m.pattern_name for m in matches}
    assert "PERSON" not in names
    assert "ORG" not in names


def test_spacy_unavailable_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    # If spaCy is not installed, use_spacy=True must not raise.
    from aistamp.pii import scanner as scanner_mod

    def fake_scan(_text: str, _config: object):
        return []

    monkeypatch.setattr(scanner_mod, "_scan_with_spacy", fake_scan)
    result = scan_text("Some text without PII", use_spacy=True)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Group 11 — load_patterns_from_yaml
# ---------------------------------------------------------------------------


def test_load_patterns_from_yaml_returns_pattern_configs(
    custom_yaml_patterns_file: Path,
) -> None:
    # load_patterns_from_yaml must return a list of PatternConfig objects.
    patterns = load_patterns_from_yaml(custom_yaml_patterns_file)
    assert all(isinstance(p, PatternConfig) for p in patterns)


def test_load_patterns_from_yaml_count_is_correct(
    custom_yaml_patterns_file: Path,
) -> None:
    # The returned list must have the same number of entries as the YAML.
    patterns = load_patterns_from_yaml(custom_yaml_patterns_file)
    assert len(patterns) == 2


def test_load_patterns_from_yaml_severity_parsed_correctly(
    custom_yaml_patterns_file: Path,
) -> None:
    # HIGH severity in YAML must produce PIISeverity.HIGH in the PatternConfig.
    patterns = load_patterns_from_yaml(custom_yaml_patterns_file)
    by_name = {p.name: p for p in patterns}
    assert by_name["EMPLOYEE_ID"].severity == PIISeverity.HIGH
    assert by_name["PROJECT_CODE"].severity == PIISeverity.MEDIUM


def test_load_patterns_from_yaml_pattern_works_in_scan(
    custom_yaml_patterns_file: Path,
) -> None:
    # A pattern loaded from YAML must work when passed as extra_patterns.
    patterns = load_patterns_from_yaml(custom_yaml_patterns_file)
    matches = scan_text("Employee EMP-123456", extra_patterns=patterns)
    assert any(m.pattern_name == "EMPLOYEE_ID" for m in matches)


def test_load_patterns_from_yaml_raises_file_not_found(tmp_path: Path) -> None:
    # load_patterns_from_yaml must raise FileNotFoundError for a nonexistent path.
    with pytest.raises(FileNotFoundError):
        load_patterns_from_yaml(tmp_path / "does_not_exist.yaml")


def test_load_patterns_from_yaml_raises_on_missing_patterns_key(
    tmp_path: Path,
) -> None:
    # YAML without a top-level "patterns" key must raise ValueError.
    path = tmp_path / "bad.yaml"
    path.write_text("settings: []\n")
    with pytest.raises(ValueError):
        load_patterns_from_yaml(path)


def test_load_patterns_from_yaml_raises_on_invalid_severity(tmp_path: Path) -> None:
    # A YAML entry with severity "CRITICAL" must raise ValueError with a clear message.
    path = tmp_path / "bad_sev.yaml"
    path.write_text(
        'patterns:\n  - name: X\n    pattern: "X"\n    severity: CRITICAL\n'
    )
    with pytest.raises(ValueError, match="severity"):
        load_patterns_from_yaml(path)


def test_load_patterns_from_yaml_raises_on_invalid_regex(tmp_path: Path) -> None:
    # A YAML entry with an invalid regex pattern must raise ValueError.
    path = tmp_path / "bad_regex.yaml"
    path.write_text(
        'patterns:\n  - name: X\n    pattern: "(?P<x>"\n    severity: HIGH\n'
    )
    with pytest.raises(ValueError, match="regex"):
        load_patterns_from_yaml(path)


# ---------------------------------------------------------------------------
# Group 12 — scan_prompt_and_response
# ---------------------------------------------------------------------------


def test_scan_prompt_and_response_returns_pii_result() -> None:
    # scan_prompt_and_response must return a PIIResult instance.
    result = scan_prompt_and_response("hi", "hello")
    assert isinstance(result, PIIResult)


def test_prompt_matches_populated() -> None:
    # PII in the prompt must appear in pii_result.prompt_matches.
    result = scan_prompt_and_response("email me at a@b.com", "ok")
    assert len(result.prompt_matches) >= 1


def test_response_matches_populated() -> None:
    # PII in the response must appear in pii_result.response_matches.
    result = scan_prompt_and_response("clean", "SSN 123-45-6789")
    assert len(result.response_matches) >= 1


def test_match_count_is_sum_of_both() -> None:
    # match_count must equal len(prompt_matches) + len(response_matches).
    result = scan_prompt_and_response("email me at a@b.com", "SSN 123-45-6789")
    assert result.match_count == len(result.prompt_matches) + len(
        result.response_matches
    )


def test_highest_severity_is_none_for_clean_text() -> None:
    # Both prompt and response clean → highest_severity must be None.
    result = scan_prompt_and_response("hello", "world")
    assert result.highest_severity is None


def test_highest_severity_reflects_maximum() -> None:
    # If prompt has MEDIUM and response has HIGH, highest_severity must be HIGH.
    result = scan_prompt_and_response(
        "Email me at alice@example.com",
        "SSN on file: 812-65-4321",
    )
    assert result.highest_severity == PIISeverity.HIGH


def test_highest_severity_medium_when_only_medium_matches() -> None:
    # If only MEDIUM matches exist, highest_severity must be MEDIUM.
    result = scan_prompt_and_response(
        "email: a@b.com",
        "phone 555-123-4567",
    )
    assert result.highest_severity == PIISeverity.MEDIUM


def test_clean_text_produces_empty_pii_result() -> None:
    # Clean prompt and response produce no matches.
    result = scan_prompt_and_response("hello", "world")
    assert result.prompt_matches == []
    assert result.response_matches == []
    assert result.match_count == 0


# ---------------------------------------------------------------------------
# Group 13 — PIIResult integration with store
# ---------------------------------------------------------------------------


def test_pii_result_survives_store_roundtrip(sqlite_backend: SQLiteBackend) -> None:
    # A real PIIResult must survive write/read through the store.
    pii_result = scan_prompt_and_response(
        "My email is test@example.com",
        "Got it.",
    )
    record = ProvenanceRecord(
        content_id=generate_content_id(),
        app_id="app",
        feature_id="f",
        user_id="u",
        model="gpt-4o",
        prompt_hash=hash_content("My email is test@example.com"),
        response_hash=hash_content("Got it."),
        prompt_tokens=10,
        response_tokens=5,
        latency_ms=120.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=pii_result,
        policy_decision=None,
    )
    sqlite_backend.write(record, hmac_signature="h")
    result = sqlite_backend.get(record.content_id)
    assert result is not None
    fetched, _ = result
    assert isinstance(fetched.pii_result, PIIResult)
    assert fetched.pii_result.match_count == pii_result.match_count
