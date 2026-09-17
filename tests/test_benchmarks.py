"""pytest-benchmark smoke suite for the scanner and pipeline hot paths.

These are smoke benchmarks: they run in every CI job as functional tests
(``--benchmark-disable`` strips the timing machinery) and locally give quick
before/after numbers when a perf regression is suspected.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from aistamp.client._pipeline import build_pre_call_context, finalize_context
from aistamp.fingerprint import hash_content, sign_record
from aistamp.models import ProvenanceRecord, RecordStatus
from aistamp.pii import scan_prompt_and_response, scan_text

pytestmark = pytest.mark.benchmark

# Realistic mixed content: prose plus several PII hits of varying severity.
_SCAN_TEXT = (
    "Support request #4711: the customer reported login issues. "
    "Contact them at user42@example.com or +1 (555) 123-4567. "
    "Reference ticket SSN 123-45-6789 was redacted. Server 10.0.0.7 is up. "
    "Billing card 4111 1111 1111 1111 on file; "
    "API key sk-abcdefghij0123456789 rotated. "
) * 4

_RESPONSE_TEXT = "The agent resolved the issue and closed ticket #4711 successfully."


def _build_record() -> ProvenanceRecord:
    return ProvenanceRecord(
        content_id="bench-0000-0000",
        app_id="bench_app",
        feature_id="bench_feature",
        user_id="bench_user",
        model="bench-model",
        prompt_hash=hash_content("prompt"),
        response_hash=hash_content("response"),
        prompt_tokens=10,
        response_tokens=20,
        latency_ms=1.0,
        timestamp=datetime.now(timezone.utc),
        status=RecordStatus.COMPLETED,
        pii_result=None,
        policy_decision=None,
    )


def test_benchmark_scan_text(benchmark: BenchmarkFixture) -> None:
    matches = benchmark(scan_text, _SCAN_TEXT)
    assert matches


def test_benchmark_scan_prompt_and_response(benchmark: BenchmarkFixture) -> None:
    result = benchmark(scan_prompt_and_response, _SCAN_TEXT, _RESPONSE_TEXT)
    assert result.match_count > 0


def test_benchmark_hash_content(benchmark: BenchmarkFixture) -> None:
    digest = benchmark(hash_content, _SCAN_TEXT)
    assert len(digest) == 64


def test_benchmark_sign_record(benchmark: BenchmarkFixture) -> None:
    record = _build_record()
    hmac = benchmark(sign_record, record, "bench-secret-key-0123456789abcdef")
    assert len(hmac) == 64


def test_benchmark_pipeline_capture(benchmark: BenchmarkFixture) -> None:
    def capture() -> str | None:
        ctx = build_pre_call_context(
            prompt=_SCAN_TEXT,
            model="bench-model",
            app_id="bench_app",
            feature_id="bench_feature",
            user_id="bench_user",
        )
        finalize_context(
            ctx,
            _RESPONSE_TEXT,
            10,
            20,
            None,
            None,
            False,
        )
        return ctx.response_hash

    response_hash = benchmark(capture)
    assert response_hash == hash_content(_RESPONSE_TEXT)
