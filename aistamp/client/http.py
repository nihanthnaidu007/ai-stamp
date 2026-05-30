from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class GenericHTTPClient:
    """Minimal JSON-over-HTTP fallback for LLM-compatible endpoints.

    Requests contain ``{"prompt": <text>, "model": <name>}``. Responses must
    include ``text`` or ``response`` and may include ``prompt_tokens`` and
    ``response_tokens``.
    """

    endpoint: str
    headers: dict[str, str] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    def complete(self, prompt: str, model: str) -> tuple[str, int | None, int | None]:
        payload = json.dumps({"prompt": prompt, "model": model}).encode("utf-8")
        headers = {"Content-Type": "application/json", **self.headers}
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")
        with urlopen(request, timeout=self.timeout_seconds) as response:
            data: dict[str, Any] = json.loads(response.read().decode("utf-8"))

        text = data.get("text", data.get("response"))
        if not isinstance(text, str):
            raise ValueError(
                "HTTP response must include string field 'text' or 'response'."
            )
        prompt_tokens = data.get("prompt_tokens")
        response_tokens = data.get("response_tokens")
        return text, prompt_tokens, response_tokens
