from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# JSONB on PostgreSQL (indexable via GIN for pii_result/policy_decision
# filters), plain JSON elsewhere (SQLite has no JSONB type).
JSONType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class ProvenanceRecordORM(Base):
    __tablename__ = "provenance_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    content_id: Mapped[str] = mapped_column(
        String(36), unique=True, nullable=False, index=True
    )
    app_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    feature_id: Mapped[str] = mapped_column(String(255), nullable=False)
    user_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    prompt_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    response_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    hmac_signature: Mapped[str | None] = mapped_column(String(128), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    response_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    pii_result: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    policy_decision: Mapped[dict[str, Any] | None] = mapped_column(JSONType, nullable=True)
    # --- Pinned v0.2.0 columns (migration 0002) ---
    key_id: Mapped[str] = mapped_column(
        String(64), nullable=False, default="default", server_default="default"
    )
    sig_algo: Mapped[str] = mapped_column(
        String(32), nullable=False, default="HMAC-SHA256", server_default="HMAC-SHA256"
    )
    record_version: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1, server_default="1"
    )
    prev_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    scope_sequence: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    __table_args__ = (
        # Deterministic audit queries filter by app/user and page by
        # (timestamp, id); these composites make that path index-driven.
        Index("ix_provenance_records_app_id_timestamp", "app_id", "timestamp"),
        Index("ix_provenance_records_user_id_timestamp", "user_id", "timestamp"),
        Index("ix_provenance_records_feature_id", "feature_id"),
        # GIN indexes power the JSON containment filters on Postgres;
        # other dialects ignore the postgresql_* kwargs and create a
        # plain index instead.
        Index(
            "ix_provenance_records_pii_result_gin",
            "pii_result",
            postgresql_using="gin",
        ),
        Index(
            "ix_provenance_records_policy_decision_gin",
            "policy_decision",
            postgresql_using="gin",
        ),
    )
