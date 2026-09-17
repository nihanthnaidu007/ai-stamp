"""Provider-SDK discovery, wrapped-client validation, and shared dispatch helpers.

The OpenAI/Anthropic SDKs are optional dependencies, so everything here imports
them lazily and tolerates their absence. ``sys.modules`` injection of fake
``openai`` / ``anthropic`` modules therefore works for offline testing too.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from aistamp.client.http import GenericHTTPClient
from aistamp.errors import StampError

_OPENAI_SYNC_MODEL_DEFAULT = "gpt-4o"
_ANTHROPIC_SYNC_MODEL_DEFAULT = "claude-3-5-sonnet-20241022"

# The Anthropic API requires max_tokens on every request (the real SDK marks it
# as a required keyword), so the Anthropic dispatch paths fall back to the 0.1
# default when neither the client-level nor a per-call value supplies one. It
# is a documented compatibility default, no longer hardcoded: callers override
# it via the ``max_tokens`` constructor argument or per-call kwarg.
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024


def _import_openai() -> Any:
    try:
        import openai

        return openai
    except ImportError:
        return None


def _import_anthropic() -> Any:
    try:
        import anthropic

        return anthropic
    except ImportError:
        return None


def _openai_client_cls(is_async: bool) -> Any:
    mod = _import_openai()
    if mod is None:
        return None
    return mod.AsyncOpenAI if is_async else mod.OpenAI


def _anthropic_client_cls(is_async: bool) -> Any:
    mod = _import_anthropic()
    if mod is None:
        return None
    return mod.AsyncAnthropic if is_async else mod.Anthropic


def _supported_types_description(is_async: bool) -> str:
    openai_cls = "openai.AsyncOpenAI" if is_async else "openai.OpenAI"
    anthropic_cls = "anthropic.AsyncAnthropic" if is_async else "anthropic.Anthropic"
    return (
        f"{openai_cls}, {anthropic_cls}, GenericHTTPClient, or a callable (str) -> str"
    )


def validate_llm_client(llm_client: Any, *, is_async: bool) -> None:
    """Reject unsupported wrapped clients at construction time (fail fast).

    Async clients additionally reject sync SDK instances — dispatching them
    from the event loop would block it, the exact failure mode v0.2 removes.
    Sync callables are accepted on the async client (they run via to_thread).
    """
    if callable(llm_client) or isinstance(llm_client, GenericHTTPClient):
        return

    openai_cls = _openai_client_cls(is_async)
    if openai_cls is not None and isinstance(llm_client, openai_cls):
        return

    anthropic_cls = _anthropic_client_cls(is_async)
    if anthropic_cls is not None and isinstance(llm_client, anthropic_cls):
        return

    raise TypeError(
        f"Unsupported LLM client type: {type(llm_client).__name__}."
        f" Expected {_supported_types_description(is_async)}."
    )


def resolve_model(llm_client: Any, model: str | None) -> str:
    """Use the caller's model, or the provider's sensible default."""
    if model:
        return model

    openai_cls = _openai_client_cls(is_async=False)
    if openai_cls is not None and isinstance(llm_client, openai_cls):
        return _OPENAI_SYNC_MODEL_DEFAULT
    openai_async_cls = _openai_client_cls(is_async=True)
    if openai_async_cls is not None and isinstance(llm_client, openai_async_cls):
        return _OPENAI_SYNC_MODEL_DEFAULT

    anthropic_cls = _anthropic_client_cls(is_async=False)
    if anthropic_cls is not None and isinstance(llm_client, anthropic_cls):
        return _ANTHROPIC_SYNC_MODEL_DEFAULT
    anthropic_async_cls = _anthropic_client_cls(is_async=True)
    if anthropic_async_cls is not None and isinstance(llm_client, anthropic_async_cls):
        return _ANTHROPIC_SYNC_MODEL_DEFAULT

    return "unknown"


def with_http_timeout(client: GenericHTTPClient, timeout: float | None) -> Any:
    """Return *client* with ``timeout_seconds`` overridden when configured."""
    if timeout is None:
        return client
    return dataclasses.replace(client, timeout_seconds=timeout)


def build_create_kwargs(
    prompt: str,
    model: str,
    provider_kwargs: dict[str, Any],
    *,
    max_tokens: int | None,
    request_timeout: float | None,
    fallback_max_tokens: int | None = None,
) -> dict[str, Any]:
    """Merge caller kwargs with client defaults for one provider ``create`` call.

    An explicit ``messages`` kwarg replaces the default single-user-message
    payload (multi-turn support); the prompt remains what is hashed/stamped.
    ``max_tokens`` and ``timeout`` are client-level defaults; per-call kwargs
    win. ``fallback_max_tokens`` covers providers whose API requires the field
    (Anthropic): when neither the client default nor a per-call kwarg supplies
    it, the fallback is sent so the request stays valid.
    """
    kwargs: dict[str, Any] = dict(provider_kwargs)
    messages = kwargs.pop("messages", None)
    if messages is None:
        messages = [{"role": "user", "content": prompt}]
    if "max_tokens" not in kwargs:
        effective_max_tokens = (
            max_tokens if max_tokens is not None else fallback_max_tokens
        )
        if effective_max_tokens is not None:
            kwargs["max_tokens"] = effective_max_tokens
    if request_timeout is not None and "timeout" not in kwargs:
        kwargs["timeout"] = request_timeout
    kwargs["model"] = model
    kwargs["messages"] = messages
    return kwargs


def parse_openai_response(resp: Any) -> tuple[str, int | None, int | None]:
    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    response_tokens = getattr(usage, "completion_tokens", None) if usage else None
    return text, prompt_tokens, response_tokens


def parse_anthropic_response(resp: Any) -> tuple[str, int | None, int | None]:
    text = resp.content[0].text if resp.content else ""
    usage = getattr(resp, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None) if usage else None
    response_tokens = getattr(usage, "output_tokens", None) if usage else None
    return text, prompt_tokens, response_tokens


def dispatch_sync(
    client: Any,
    prompt: str,
    model: str,
    provider_kwargs: dict[str, Any],
    *,
    max_tokens: int | None,
    request_timeout: float | None,
) -> tuple[str, int | None, int | None]:
    """One synchronous provider call against any supported client type.

    Shared by the sync client directly and by the async client (via
    ``asyncio.to_thread``) for sync SDK instances, callables, and the HTTP
    fallback.
    """
    if isinstance(client, GenericHTTPClient):
        if provider_kwargs:
            raise StampError(
                "GenericHTTPClient has a fixed payload contract and does not"
                " accept extra provider kwargs: " + ", ".join(sorted(provider_kwargs))
            )
        return client.complete(prompt, model)

    openai = _import_openai()
    if openai is not None and isinstance(client, openai.OpenAI):
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
        )
        resp = client.chat.completions.create(**create_kwargs)
        return parse_openai_response(resp)

    anthropic = _import_anthropic()
    if anthropic is not None and isinstance(client, anthropic.Anthropic):
        create_kwargs = build_create_kwargs(
            prompt,
            model,
            provider_kwargs,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            fallback_max_tokens=ANTHROPIC_DEFAULT_MAX_TOKENS,
        )
        resp = client.messages.create(**create_kwargs)
        return parse_anthropic_response(resp)

    if callable(client):
        if provider_kwargs:
            raise StampError(
                "Callable LLM clients accept only the prompt; extra kwargs"
                " cannot be forwarded: " + ", ".join(sorted(provider_kwargs))
            )
        text = client(prompt)
        if not isinstance(text, str):
            raise StampError(
                f"Generic callable must return str, got {type(text).__name__}"
            )
        return text, None, None

    raise StampError(
        f"Unsupported LLM client type: {type(client).__name__}."
        " Expected openai.OpenAI, anthropic.Anthropic, GenericHTTPClient,"
        " or a callable (str) -> str."
    )
