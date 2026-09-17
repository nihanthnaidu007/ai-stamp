from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from aistamp.errors import (
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderResponseError,
    ProviderTimeoutError,
)


def _parse_retry_after(headers: Any) -> float | None:
    """Extract a ``Retry-After`` delay in seconds when the header is numeric."""
    if headers is None:
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def wrap_http_error(exc: BaseException) -> ProviderError:
    """Map a raw urllib error into the ai-stamp provider error taxonomy."""
    if isinstance(exc, TimeoutError):
        return ProviderTimeoutError(f"Provider request timed out: {exc}")

    if isinstance(exc, HTTPError):
        code = exc.code
        retry_after = _parse_retry_after(exc.headers)
        if code == 429:
            return ProviderRateLimitError(
                f"Provider rate limited the request (HTTP {code}).",
                status_code=code,
                retry_after=retry_after,
            )
        if code in (401, 403):
            return ProviderAuthError(
                f"Provider rejected the credentials (HTTP {code}).",
                status_code=code,
            )
        return ProviderResponseError(
            f"Provider returned HTTP {code}.",
            status_code=code,
        )

    if isinstance(exc, URLError):
        # URLError wraps low-level socket failures; reason carries the cause.
        if isinstance(exc.reason, TimeoutError):
            return ProviderTimeoutError(f"Provider request timed out: {exc.reason}")
        return ProviderResponseError(f"Provider connection failed: {exc.reason}")

    return ProviderResponseError(f"Provider request failed: {exc}")


@dataclass(frozen=True)
class GenericHTTPClient:
    """Minimal JSON-over-HTTP fallback for LLM-compatible endpoints.

    Requests contain ``{"prompt": <text>, "model": <name>}``. Responses must
    include ``text`` or ``response`` and may include ``prompt_tokens`` and
    ``response_tokens``.

    Since 0.2, raw ``HTTPError``/``URLError``/timeout exceptions are wrapped
    into the ai-stamp error taxonomy (see ``aistamp.errors``).
    """

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    def complete(self, prompt: str, model: str) -> tuple[str, int | None, int | None]:
        payload = json.dumps({"prompt": prompt, "model": model}).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                data: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except TimeoutError as exc:  # covers socket.timeout on py3.10+
            raise wrap_http_error(exc) from exc
        except HTTPError as exc:  # must precede URLError: HTTPError subclasses it
            raise wrap_http_error(exc) from exc
        except URLError as exc:
            raise wrap_http_error(exc) from exc

        # Transport failures are wrapped into the library taxonomy; response
        # CONTENT problems stay ValueError-compatible (json.JSONDecodeError is
        # a ValueError subclass), matching the 0.1 contract for callers that
        # catch ValueError around malformed payloads.
        text = data.get("text", data.get("response"))
        if not isinstance(text, str):
            raise ValueError(
                "HTTP response must include string field 'text' or 'response'."
            )
        prompt_tokens = data.get("prompt_tokens")
        response_tokens = data.get("response_tokens")
        return text, prompt_tokens, response_tokens
