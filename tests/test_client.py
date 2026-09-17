from __future__ import annotations

import uuid

import pytest

from aistamp.client import (
    AsyncProvenanceClient,
    GenericHTTPClient,
    ProvenanceClient,
    StampError,
)
from aistamp.config import Config
from aistamp.fingerprint import hash_content
from aistamp.models import (
    PIISeverity,
    PolicyAction,
    QueryFilters,
    RecordStatus,
)
from aistamp.policy import (
    PolicyEngine,
    PolicyViolationError,
    RuleConditions,
    RuleConfig,
)
from aistamp.store import SQLiteBackend
from aistamp.store.schema import ProvenanceRecordORM

# ---------------------------------------------------------------------------
# Group 1 — ProvenanceClient construction
# ---------------------------------------------------------------------------


def test_client_constructs_with_callable(mock_llm_client, stamp_config) -> None:
    # ProvenanceClient must instantiate without error when given a callable.
    client = ProvenanceClient(
        mock_llm_client,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
    )
    assert client is not None


def test_client_creates_default_sqlite_backend_if_none_provided(
    mock_llm_client,
    stamp_config,
) -> None:
    # If backend=None, client must create a SQLiteBackend internally.
    client = ProvenanceClient(
        mock_llm_client,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
    )
    assert client._backend is not None


def test_client_accepts_explicit_backend(mock_llm_client, stamp_config) -> None:
    # A backend passed in constructor must be stored as self._backend.
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = ProvenanceClient(
        mock_llm_client,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )
    assert client._backend is backend


def test_client_accepts_policy_engine(mock_llm_client, stamp_config) -> None:
    # ProvenanceClient must accept a PolicyEngine without error.
    engine = PolicyEngine(rules=[], model_tiers={})
    client = ProvenanceClient(
        mock_llm_client,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        engine=engine,
    )
    assert client._engine is engine


# ---------------------------------------------------------------------------
# Group 2 — chat: input validation
# ---------------------------------------------------------------------------


def test_chat_raises_type_error_for_non_string_prompt(
    provenance_client: ProvenanceClient,
) -> None:
    # chat() must raise TypeError when prompt is not a string.
    with pytest.raises(TypeError):
        provenance_client.chat(123)  # type: ignore[arg-type]


def test_chat_raises_value_error_for_empty_prompt(
    provenance_client: ProvenanceClient,
) -> None:
    # chat() must raise ValueError for an empty string prompt.
    with pytest.raises(ValueError):
        provenance_client.chat("")


def test_chat_raises_value_error_for_whitespace_prompt(
    provenance_client: ProvenanceClient,
) -> None:
    # chat() must raise ValueError for a whitespace-only prompt.
    with pytest.raises(ValueError):
        provenance_client.chat("   \n\t  ")


# ---------------------------------------------------------------------------
# Group 3 — chat: happy path with callable
# ---------------------------------------------------------------------------


def test_chat_returns_string_response(provenance_client: ProvenanceClient) -> None:
    # chat() must return the string produced by the callable client.
    result = provenance_client.chat("hello")
    assert isinstance(result, str)


def test_chat_returns_correct_response_content(
    provenance_client: ProvenanceClient,
) -> None:
    # The returned string must contain the mock response text.
    result = provenance_client.chat("hello")
    assert "Mock response to:" in result


def test_chat_with_explicit_model_uses_that_model(
    provenance_client: ProvenanceClient,
) -> None:
    # If model is passed to chat(), the stored record must use that model.
    provenance_client.chat("hello", model="test-model-v1")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].model == "test-model-v1"


def test_chat_with_no_model_uses_default_for_callable(
    provenance_client: ProvenanceClient,
) -> None:
    # Callable client with no model → stored record must use "unknown" as model.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].model == "unknown"


# ---------------------------------------------------------------------------
# Group 4 — chat: provenance record is written to store
# ---------------------------------------------------------------------------


def test_chat_writes_record_to_store(provenance_client: ProvenanceClient) -> None:
    # After chat(), exactly one record must exist in the store.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert len(report.records) == 1


def test_stored_record_has_correct_app_id(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must carry the app_id passed to the constructor.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].app_id == "test_app"


def test_stored_record_has_correct_user_id(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must carry the user_id passed to the constructor.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].user_id == "test_user"


def test_stored_record_has_completed_status(
    provenance_client: ProvenanceClient,
) -> None:
    # A successful chat() must produce a record with status=COMPLETED.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].status == RecordStatus.COMPLETED


def test_stored_record_has_non_null_prompt_hash(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must have a non-null prompt_hash.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].prompt_hash is not None
    assert len(report.records[0].prompt_hash) == 64


def test_stored_record_has_non_null_response_hash(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must have a non-null response_hash.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].response_hash is not None


def test_stored_record_has_non_null_latency_ms(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must have a non-null latency_ms greater than 0.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    record = report.records[0]
    assert record.latency_ms is not None
    assert record.latency_ms > 0


def test_stored_record_prompt_hash_matches_input(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored prompt_hash must equal hash_content(prompt).
    prompt = "verify prompt hash"
    provenance_client.chat(prompt)
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.records[0].prompt_hash == hash_content(prompt)


# ---------------------------------------------------------------------------
# Group 5 — chat: PII detection in pipeline
# ---------------------------------------------------------------------------


def _client_with(callable_fn, stamp_config) -> ProvenanceClient:
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    return ProvenanceClient(
        callable_fn,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )


def test_pii_result_stored_in_record(mock_llm_client_with_pii, stamp_config) -> None:
    # After chat() with a PII-containing response, record.pii_result must not be None.
    client = _client_with(mock_llm_client_with_pii, stamp_config)
    client.chat("How do I get support?")
    report = client._backend.query(QueryFilters(user_id="u"))
    assert report.records[0].pii_result is not None


def test_pii_in_response_detected(mock_llm_client_with_pii, stamp_config) -> None:
    # The stored pii_result must have response_matches with at least one EMAIL match.
    client = _client_with(mock_llm_client_with_pii, stamp_config)
    client.chat("How do I get support?")
    report = client._backend.query(QueryFilters(user_id="u"))
    pii = report.records[0].pii_result
    assert pii is not None
    assert any(m.pattern_name == "EMAIL" for m in pii.response_matches)


def test_pii_match_count_is_non_zero(mock_llm_client_with_pii, stamp_config) -> None:
    # match_count in stored pii_result must be > 0 for PII-containing response.
    client = _client_with(mock_llm_client_with_pii, stamp_config)
    client.chat("How do I get support?")
    report = client._backend.query(QueryFilters(user_id="u"))
    assert report.records[0].pii_result.match_count > 0


def test_clean_prompt_and_response_pii_result_is_empty(
    provenance_client: ProvenanceClient,
) -> None:
    # A clean prompt and response must still produce a PIIResult but with match_count=0.
    provenance_client.chat("what is the weather")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    pii = report.records[0].pii_result
    assert pii is not None
    assert pii.match_count == 0


# ---------------------------------------------------------------------------
# Group 6 — chat: policy integration
# ---------------------------------------------------------------------------


def _make_client(
    llm_fn,
    stamp_config: Config,
    engine: PolicyEngine | None,
    user_id: str = "u",
) -> ProvenanceClient:
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    return ProvenanceClient(
        llm_fn,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id=user_id,
        backend=backend,
        engine=engine,
    )


def test_policy_warn_does_not_block_chat(
    mock_llm_client_with_pii,
    stamp_config,
) -> None:
    # A WARN rule that fires must not prevent chat() from completing.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "warn_medium",
                RuleConditions(pii_severity=PIISeverity.MEDIUM),
                PolicyAction.WARN,
            )
        ],
        model_tiers={},
    )
    client = _make_client(mock_llm_client_with_pii, stamp_config, engine)
    response = client.chat("How do I get support?")
    assert isinstance(response, str)


def test_policy_warn_is_stored_in_record(
    mock_llm_client_with_pii,
    stamp_config,
) -> None:
    # When a WARN rule fires, the stored record must have policy_decision.action=WARN.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "warn_medium",
                RuleConditions(pii_severity=PIISeverity.MEDIUM),
                PolicyAction.WARN,
            )
        ],
        model_tiers={},
    )
    client = _make_client(mock_llm_client_with_pii, stamp_config, engine)
    client.chat("How do I get support?")
    report = client._backend.query(QueryFilters(user_id="u"))
    decision = report.records[0].policy_decision
    assert decision is not None
    assert decision.action == PolicyAction.WARN


def test_policy_block_on_prompt_pii_raises(stamp_config) -> None:
    # A BLOCK rule on HIGH pii must raise PolicyViolationError when prompt contains SSN.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_high",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.BLOCK,
            )
        ],
        model_tiers={},
    )
    client = _make_client(lambda p: "ok", stamp_config, engine)
    with pytest.raises(PolicyViolationError):
        client.chat("My SSN is 123-45-6789")


def test_policy_block_stores_blocked_record(stamp_config) -> None:
    # Even when BLOCK is raised, a record with status=BLOCKED must be written to store.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_high",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.BLOCK,
            )
        ],
        model_tiers={},
    )
    client = _make_client(lambda p: "ok", stamp_config, engine)
    with pytest.raises(PolicyViolationError):
        client.chat("My SSN is 123-45-6789")
    report = client._backend.query(QueryFilters(user_id="u"))
    assert len(report.records) == 1
    assert report.records[0].status == RecordStatus.BLOCKED


def test_policy_block_on_response_pii_raises(stamp_config) -> None:
    # A BLOCK rule must also fire if the response contains HIGH-severity PII.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_high",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.BLOCK,
            )
        ],
        model_tiers={},
    )

    def ssn_llm(prompt: str) -> str:
        return "Here is your SSN: 555-12-3456"

    client = _make_client(ssn_llm, stamp_config, engine)
    with pytest.raises(PolicyViolationError):
        client.chat("clean prompt with no PII")


# ---------------------------------------------------------------------------
# Group 7 — chat: error handling
# ---------------------------------------------------------------------------


def test_unsupported_client_raises_on_construction(stamp_config) -> None:
    # v0.2: unsupported clients are rejected at construction time (fail fast)
    # with a TypeError describing the supported types.
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    with pytest.raises(TypeError, match="Unsupported LLM client type"):
        ProvenanceClient(
            object(),
            config=stamp_config,
            app_id="a",
            feature_id="f",
            user_id="u",
            backend=backend,
        )


def test_callable_returning_non_string_raises_stamp_error(stamp_config) -> None:
    # A callable that returns an integer must raise StampError.
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = ProvenanceClient(
        lambda p: 42,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )
    with pytest.raises(StampError):
        client.chat("hello")


def test_generic_http_client_is_supported(
    monkeypatch: pytest.MonkeyPatch, stamp_config
) -> None:
    def fake_complete(
        self: GenericHTTPClient, prompt: str, model: str
    ) -> tuple[str, int, int]:
        assert prompt == "hello"
        assert model == "http-model"
        return "http response", 3, 4

    monkeypatch.setattr(GenericHTTPClient, "complete", fake_complete)
    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = ProvenanceClient(
        GenericHTTPClient("https://example.invalid/generate"),
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )
    assert client.chat("hello", model="http-model") == "http response"


# ---------------------------------------------------------------------------
# Group 8 — AsyncProvenanceClient
# ---------------------------------------------------------------------------


def test_async_client_constructs_with_async_callable(stamp_config) -> None:
    # AsyncProvenanceClient must instantiate with an async callable.
    async def async_mock(prompt: str) -> str:
        return "async response"

    client = AsyncProvenanceClient(
        async_mock,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
    )
    assert client is not None


@pytest.mark.asyncio
async def test_async_chat_returns_string(stamp_config) -> None:
    # async chat() must return a string response from an async callable.
    async def async_mock(prompt: str) -> str:
        return "async response"

    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = AsyncProvenanceClient(
        async_mock,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )
    response = await client.chat("hi")
    assert response == "async response"


@pytest.mark.asyncio
async def test_async_chat_writes_record_to_store(stamp_config) -> None:
    # After async chat(), a record must exist in the store.
    async def async_mock(prompt: str) -> str:
        return "async response"

    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = AsyncProvenanceClient(
        async_mock,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="async_u",
        backend=backend,
    )
    await client.chat("hi")
    report = backend.query(QueryFilters(user_id="async_u"))
    assert len(report.records) == 1


@pytest.mark.asyncio
async def test_async_chat_with_sync_callable_works(stamp_config) -> None:
    # AsyncProvenanceClient must handle a plain sync callable.
    def sync_mock(prompt: str) -> str:
        return "sync result"

    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = AsyncProvenanceClient(
        sync_mock,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
    )
    response = await client.chat("hi")
    assert response == "sync result"


@pytest.mark.asyncio
async def test_async_policy_block_raises(stamp_config) -> None:
    # PolicyViolationError must propagate from async chat() when a BLOCK rule fires.
    engine = PolicyEngine(
        rules=[
            RuleConfig(
                "block_high",
                RuleConditions(pii_severity=PIISeverity.HIGH),
                PolicyAction.BLOCK,
            )
        ],
        model_tiers={},
    )

    async def async_mock(prompt: str) -> str:
        return "ok"

    backend = SQLiteBackend(stamp_config.database_url)
    backend.create_tables()
    client = AsyncProvenanceClient(
        async_mock,
        config=stamp_config,
        app_id="a",
        feature_id="f",
        user_id="u",
        backend=backend,
        engine=engine,
    )
    with pytest.raises(PolicyViolationError):
        await client.chat("My SSN is 123-45-6789")


# ---------------------------------------------------------------------------
# Group 9 — end-to-end pipeline verification
# ---------------------------------------------------------------------------


def test_full_pipeline_content_id_is_valid_uuid(
    provenance_client: ProvenanceClient,
) -> None:
    # The record stored after chat() must have a content_id that parses as UUID.
    provenance_client.chat("hello")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    uuid.UUID(report.records[0].content_id)


def test_full_pipeline_fingerprint_verifiable(
    provenance_client: ProvenanceClient,
) -> None:
    # The stored record must be verifiable against the known mock response.
    prompt = "test prompt"
    provenance_client.chat(prompt)
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    record = report.records[0]
    expected_response = f"Mock response to: {prompt[:30]}"
    assert record.response_hash == hash_content(expected_response)


def test_full_pipeline_record_has_hmac_in_store(
    provenance_client: ProvenanceClient,
) -> None:
    # The ORM row for the stored record must have a non-null hmac_signature.
    from sqlalchemy.orm import Session

    provenance_client.chat("hello")
    backend = provenance_client._backend
    with Session(backend._engine) as session:  # type: ignore[attr-defined]
        row = session.query(ProvenanceRecordORM).first()
        assert row is not None
        assert row.hmac_signature is not None
        assert len(row.hmac_signature) == 64


def test_multiple_chat_calls_write_multiple_records(
    provenance_client: ProvenanceClient,
) -> None:
    # Three separate chat() calls must produce three separate records in the store.
    provenance_client.chat("one")
    provenance_client.chat("two")
    provenance_client.chat("three")
    report = provenance_client._backend.query(QueryFilters(user_id="test_user"))
    assert report.total_count == 3
