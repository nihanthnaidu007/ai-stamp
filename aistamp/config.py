from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, field_validator


class Config(BaseModel):
    model_config = ConfigDict(frozen=True)

    secret_key: str
    database_url: str = "sqlite:///./aistamp.db"
    log_level: str = "INFO"

    @field_validator("secret_key")
    @classmethod
    def _validate_secret_key(cls, v: str) -> str:
        if len(v) < 32:
            raise ValueError(
                "secret_key must be at least 32 characters for HMAC security."
            )
        return v

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        upper = v.upper()
        if upper not in {"DEBUG", "INFO", "WARNING", "ERROR"}:
            raise ValueError("log_level must be one of: DEBUG, INFO, WARNING, ERROR")
        return upper

    @classmethod
    def from_env(cls) -> Config:
        if "AISTAMP_SECRET_KEY" not in os.environ:
            raise KeyError("AISTAMP_SECRET_KEY environment variable is required.")
        kwargs: dict[str, Any] = {"secret_key": os.environ["AISTAMP_SECRET_KEY"]}
        if "AISTAMP_DATABASE_URL" in os.environ:
            kwargs["database_url"] = os.environ["AISTAMP_DATABASE_URL"]
        if "AISTAMP_LOG_LEVEL" in os.environ:
            kwargs["log_level"] = os.environ["AISTAMP_LOG_LEVEL"]
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with p.open("r") as f:
            data = yaml.safe_load(f) or {}
        if "secret_key" not in data:
            raise ValueError("YAML config must contain a 'secret_key' key.")
        return cls(**data)

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Config:
        if config_path is not None:
            return cls.from_yaml(config_path)
        return cls.from_env()


def configure_logging(config: Config) -> None:
    """Apply the log level from Config to the root aistamp logger."""
    logging.getLogger("aistamp").setLevel(config.log_level)
