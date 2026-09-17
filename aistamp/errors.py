"""Exception taxonomy for ai-stamp.

Every error the library raises derives from :class:`AIStampError`, so callers
can catch a single base type. Provider failures get dedicated subtypes so
callers can distinguish "rate limited, retry later" from "bug" without parsing
strings.

All errors carry an optional ``content_id`` attribute: the identifier of the
provenance record the error belongs to. This lets callers correlate a raised
error with its audit trail even when the record itself could not be persisted.
"""

from __future__ import annotations

__all__ = [
    "AIStampError",
    "StampError",
    "ProviderError",
    "ProviderTimeoutError",
    "ProviderRateLimitError",
    "ProviderAuthError",
    "ProviderResponseError",
    "ConfigError",
]


class AIStampError(Exception):
    """Base class for every error raised by ai-stamp.

    Attributes:
        content_id: identifier of the provenance record this error belongs to,
            or ``None`` when the error occurred before a record existed (e.g.
            invalid arguments). Populated by the client pipeline as soon as a
            capture context exists.
    """

    def __init__(self, message: str, *, content_id: str | None = None) -> None:
        super().__init__(message)
        self.content_id = content_id


class StampError(AIStampError):
    """
    Raised when ai-stamp encounters an internal error during the stamping pipeline.
    Wraps exceptions from unsupported client types or unexpected failures.

    Since 0.2 this is a subclass of :class:`AIStampError`; the old import paths
    (``aistamp.client.StampError``, ``aistamp.client.sync.StampError``) keep
    working.
    """


class ProviderError(AIStampError):
    """Base class for errors reported by the wrapped LLM provider.

    Attributes:
        status_code: HTTP status code reported by the provider, when known.
        retry_after: provider-advised wait in seconds (from a ``Retry-After``
            header), when known.
    """

    def __init__(
        self,
        message: str,
        *,
        content_id: str | None = None,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, content_id=content_id)
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """The provider did not answer within the allowed time. Retried automatically."""


class ProviderRateLimitError(ProviderError):
    """The provider rate limited the call (HTTP 429). Retried automatically.

    Check ``retry_after`` for a provider-advised backoff, if sent.
    """


class ProviderAuthError(ProviderError):
    """The provider rejected the credentials (HTTP 401/403). Not retried."""


class ProviderResponseError(ProviderError):
    """The provider returned a malformed or non-retryable error response.

    Retried only when ``status_code`` indicates a server-side (5xx) failure.
    """


class ConfigError(AIStampError, ValueError):
    """
    Raised when configuration is missing or invalid.

    Subclasses :class:`ValueError` so existing ``except ValueError`` handlers
    (including the CLI's) keep working after the bare ``KeyError`` /
    ``FileNotFoundError`` they replaced.
    """
