"""Tests for GenericHTTPClient's real HTTP transport.

Existing client tests monkeypatch ``GenericHTTPClient.complete``; these tests
exercise the actual ``urllib`` round trip against a local threaded HTTP
server, closing the untested-transport gap without external services.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.error import HTTPError

import pytest

from aistamp.client import GenericHTTPClient, ProvenanceClient
from aistamp.config import Config
from aistamp.errors import ProviderResponseError
from aistamp.models import QueryFilters, RecordStatus
from aistamp.store import SQLiteBackend

SECRET_KEY = "test-secret-key-for-aistamp-unit-tests-32chars"


class _Handler(BaseHTTPRequestHandler):
    # Narrow the framework's BaseServer attribute to our server type.
    server: _LLMServer

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.server.last_body = self.rfile.read(length)
        # Header keys arrive case-normalized; store lowercased for asserts.
        self.server.last_headers = {
            key.lower(): value for key, value in self.headers.items()
        }
        payload = json.dumps(self.server.payload).encode("utf-8")
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        # Silence per-request logging; the test suite asserts on state instead.
        pass


class _LLMServer(ThreadingHTTPServer):
    def __init__(
        self,
        *,
        status: int = 200,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.status = status
        self.payload = (
            payload
            if payload is not None
            else {"text": "ok", "prompt_tokens": 3, "response_tokens": 4}
        )
        self.last_body: bytes | None = None
        self.last_headers: dict[str, str] = {}

    @property
    def endpoint(self) -> str:
        address = self.server_address
        host = str(address[0])
        port = int(address[1])
        return f"http://{host}:{port}/v1/generate"

    def shutdown_and_close(self) -> None:
        self.shutdown()
        self.server_close()


@pytest.fixture
def llm_server_factory() -> Iterator[Any]:
    servers: list[_LLMServer] = []

    def _make(**kwargs: Any) -> _LLMServer:
        server = _LLMServer(**kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append(server)
        return server

    yield _make

    for server in servers:
        server.shutdown_and_close()


def _make_config() -> Config:
    return Config(
        secret_key=SECRET_KEY,
        database_url="sqlite:///:memory:",
        log_level="DEBUG",
    )


def _make_backend(config: Config) -> SQLiteBackend:
    backend = SQLiteBackend(config.database_url)
    backend.create_tables()
    return backend


# --- Transport-level behavior (real socket round trip) -----------------------


def test_complete_parses_json_response(llm_server_factory: Any) -> None:
    server = llm_server_factory()
    client = GenericHTTPClient(endpoint=server.endpoint)

    text, prompt_tokens, response_tokens = client.complete("hi there", "test-model")

    assert text == "ok"
    assert prompt_tokens == 3
    assert response_tokens == 4
    assert server.last_body is not None
    assert json.loads(server.last_body.decode("utf-8")) == {
        "prompt": "hi there",
        "model": "test-model",
    }
    assert server.last_headers["content-type"] == "application/json"


def test_complete_accepts_response_key(llm_server_factory: Any) -> None:
    server = llm_server_factory(payload={"response": "alt-text"})
    client = GenericHTTPClient(endpoint=server.endpoint)

    assert client.complete("p", "m") == ("alt-text", None, None)


def test_complete_requires_text_or_response(llm_server_factory: Any) -> None:
    server = llm_server_factory(payload={"unexpected": 1})
    client = GenericHTTPClient(endpoint=server.endpoint)

    with pytest.raises(
        ValueError, match="must include string field 'text' or 'response'"
    ):
        client.complete("p", "m")


def test_custom_headers_are_sent(llm_server_factory: Any) -> None:
    server = llm_server_factory()
    client = GenericHTTPClient(
        endpoint=server.endpoint, headers={"X-Api-Key": "secret-key"}
    )

    client.complete("p", "m")

    assert server.last_headers["x-api-key"] == "secret-key"


def test_server_error_raises_http_error(llm_server_factory: Any) -> None:
    server = llm_server_factory(status=500, payload={"error": "kaboom"})
    client = GenericHTTPClient(endpoint=server.endpoint)

    # v0.2: raw HTTPError is wrapped into the taxonomy (retryable); the
    # original error stays chained for diagnostics.
    with pytest.raises(ProviderResponseError, match="HTTP 500") as excinfo:
        client.complete("p", "m")
    assert isinstance(excinfo.value.__cause__, HTTPError)


# --- Client-level integration over the real HTTP path ------------------------


@pytest.mark.integration
def test_provenance_client_over_real_http(llm_server_factory: Any) -> None:
    server = llm_server_factory()
    config = _make_config()
    backend = _make_backend(config)
    client = ProvenanceClient(
        GenericHTTPClient(endpoint=server.endpoint),
        config=config,
        app_id="test_app",
        feature_id="test_feature",
        user_id="test_user",
        backend=backend,
    )

    response = client.chat("hello http", model="test-model")

    assert response == "ok"
    report = backend.query(QueryFilters())
    assert report.total_count == 1
    record = report.records[0]
    assert record.status == RecordStatus.COMPLETED
    assert record.model == "test-model"
    assert record.prompt_tokens == 3
    assert record.response_tokens == 4


@pytest.mark.integration
def test_provenance_client_http_error_persists_error_record(
    llm_server_factory: Any,
) -> None:
    server = llm_server_factory(status=500, payload={"error": "kaboom"})
    config = _make_config()
    backend = _make_backend(config)
    client = ProvenanceClient(
        GenericHTTPClient(endpoint=server.endpoint),
        config=config,
        app_id="test_app",
        feature_id="test_feature",
        user_id="test_user",
        backend=backend,
    )

    with pytest.raises(ProviderResponseError, match="HTTP 500"):
        client.chat("hello http", model="test-model")

    report = backend.query(QueryFilters())
    assert report.total_count == 1
    record = report.records[0]
    assert record.status == RecordStatus.ERROR
    assert record.response_hash is None
