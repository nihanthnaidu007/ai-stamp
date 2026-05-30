"""Initial provenance record schema.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "provenance_records",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("content_id", sa.String(36), nullable=False),
        sa.Column("app_id", sa.String(255), nullable=False),
        sa.Column("feature_id", sa.String(255), nullable=False),
        sa.Column("user_id", sa.String(255), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("prompt_hash", sa.String(64), nullable=False),
        sa.Column("response_hash", sa.String(64), nullable=True),
        sa.Column("hmac_signature", sa.String(128), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("response_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("pii_result", sa.JSON(), nullable=True),
        sa.Column("policy_decision", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("content_id"),
    )
    op.create_index(
        "ix_provenance_records_content_id",
        "provenance_records",
        ["content_id"],
    )
    op.create_index("ix_provenance_records_app_id", "provenance_records", ["app_id"])
    op.create_index("ix_provenance_records_user_id", "provenance_records", ["user_id"])
    op.create_index("ix_provenance_records_model", "provenance_records", ["model"])
    op.create_index(
        "ix_provenance_records_timestamp", "provenance_records", ["timestamp"]
    )
    op.create_index("ix_provenance_records_status", "provenance_records", ["status"])


def downgrade() -> None:
    op.drop_table("provenance_records")
