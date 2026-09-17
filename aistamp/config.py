from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

from aistamp.errors import ConfigError

logger = logging.getLogger("aistamp")

# Schemes the bundled SQLAlchemy backends actually support. Anything else fails
# fast here instead of surfacing as an opaque SQLAlchemy ImportError later.
_ALLOWED_DATABASE_SCHEMES = frozenset(
    {
        "sqlite",
        "sqlite+aiosqlite",
        "postgresql",
        "postgresql+asyncpg",
        "postgresql+psycopg2",
        "postgresql+psycopg",
    }
)

# ${VAR} or ${VAR:-fallback}; fallback applies when VAR is unset or empty.
_ENV_VAR_PATTERN = re.compile(
    r"\$\{(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?::-(?P<default>[^}]*))?\}"
)


def _replace_env_var(match: re.Match[str]) -> str:
    name = match.group("name")
    default = match.group("default")
    env_value = os.environ.get(name)
    if env_value:
        return env_value
    if default is not None:
        return default
    raise ConfigError(
        f"Environment variable {name!r} referenced in the configuration is not set."
    )


def interpolate_env_vars(value: Any) -> Any:
    """Recursively interpolate ``${VAR}`` / ``${VAR:-default}`` in string values.

    Applied to every string in a loaded YAML mapping so secrets can stay in the
    environment instead of on disk.
    """
    if isinstance(value, str):
        return _ENV_VAR_PATTERN.sub(_replace_env_var, value)
    if isinstance(value, dict):
        return {k: interpolate_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [interpolate_env_vars(item) for item in value]
    return value


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    secret_key: SecretStr
    database_url: str = "sqlite:///./aistamp.db"
    log_level: str = "INFO"
    # --- v0.2 pinned shared-interface fields (names agreed across all tracks)
    key_id: str = "default"
    redact_before_send: bool = False

    @field_validator("secret_key")
    @classmethod
    def _validate_secret_key(cls, v: SecretStr) -> SecretStr:
        if len(v.get_secret_value()) < 32:
            raise ValueError(
                "secret_key must be at least 32 characters for HMAC security."
            )
        return v

    @field_validator("database_url")
    @classmethod
    def _validate_database_url(cls, v: str) -> str:
        if "://" not in v:
            raise ValueError(
                "database_url must include a scheme, e.g. 'sqlite:///./aistamp.db' "
                "or 'postgresql://user:pass@host/db'."
            )
        scheme = v.split("://", 1)[0]
        if scheme not in _ALLOWED_DATABASE_SCHEMES:
            raise ValueError(
                f"Unsupported database_url scheme {scheme!r}. Supported schemes: "
                + ", ".join(sorted(_ALLOWED_DATABASE_SCHEMES))
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be one of: DEBUG, INFO, WARNING, ERROR")
        return upper

    @property
    def secret_key_value(self) -> str:
        """The secret as a plain string, for signing and verification only."""
        return self.secret_key.get_secret_value()

    @classmethod
    def from_env(cls) -> Config:
        secret = os.environ.get("AISTAMP_SECRET_KEY")
        if not secret:
            raise ConfigError(
                "AISTAMP_SECRET_KEY environment variable is required."
                " Set it to a secret of at least 32 characters."
            )
        kwargs: dict[str, Any] = {"secret_key": secret}
        if "AISTAMP_DATABASE_URL" in os.environ:
            kwargs["database_url"] = os.environ["AISTAMP_DATABASE_URL"]
        if "AISTAMP_LOG_LEVEL" in os.environ:
            kwargs["log_level"] = os.environ["AISTAMP_LOG_LEVEL"]
        if "AISTAMP_KEY_ID" in os.environ:
            kwargs["key_id"] = os.environ["AISTAMP_KEY_ID"]
        if "AISTAMP_REDACT_BEFORE_SEND" in os.environ:
            kwargs["redact_before_send"] = os.environ[
                "AISTAMP_REDACT_BEFORE_SEND"
            ].strip().lower() in {"1", "true", "yes", "on"}
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        p = Path(path)
        if not p.exists():
            raise ConfigError(f"Config file not found: {path}")
        with p.open("r") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"Config file {path} must contain a YAML mapping,"
                f" got {type(data).__name__}."
            )
        if "secret_key" not in data:
            raise ConfigError("YAML config must contain a 'secret_key' key.")
        return cls(**interpolate_env_vars(data))

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Config:
        if config_path is not None:
            return cls.from_yaml(config_path)
        return cls.from_env()


def configure_logging(config: Config) -> None:
    """Apply the log level from Config to the root aistamp logger."""
    logging.getLogger("aistamp").setLevel(config.log_level)
