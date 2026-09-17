"""Storage v2: pinned identity/error columns, composite indexes, JSONB+GIN.

Adds the pinned v0.2.0 shared columns (key_id, sig_algo, record_version,
prev_hash, scope_sequence, error_type, error_message), the composite audit
indexes (app_id, timestamp) and (user_id, timestamp), a feature_id index,
and — PostgreSQL only — JSONB conversion plus GIN indexes for
pii_result/policy_decision filters.

Revision ID: 0002
Revises: 0001
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_TABLE = "provenance_records"

# Pinned shared columns — the full field list other v0.2.0 tracks build on.
_NEW_COLUMNS = [
    sa.Column("key_id", sa.String(64), nullable=False, server_default="default"),
    sa.Column(
        "sig_algo", sa.String(32), nullable=False, server_default="HMAC-SHA256"
    ),
    sa.Column("record_version", sa.Integer(), nullable=False, server_default="1"),
    sa.Column("prev_hash", sa.String(64), nullable=True),
    sa.Column("scope_sequence", sa.Integer(), nullable=True),
    sa.Column("error_type", sa.String(255), nullable=True),
    sa.Column("error_message", sa.Text(), nullable=True),
]

_COMPOSITE_INDEXES = [
    ("ix_provenance_records_app_id_timestamp", ["app_id", "timestamp"]),
    ("ix_provenance_records_user_id_timestamp", ["user_id", "timestamp"]),
    ("ix_provenance_records_feature_id", ["feature_id"]),
]

_JSON_COLUMNS = ("pii_result", "policy_decision")
_JSON_GIN_INDEXES = {
    "pii_result": "ix_provenance_records_pii_result_gin",
    "policy_decision": "ix_provenance_records_policy_decision_gin",
}


def _is_postgres() -> bool:
    return op.get_bind().dialect.name == "postgresql"


def upgrade() -> None:
    for column in _NEW_COLUMNS:
        op.add_column(_TABLE, column)
    for index_name, columns in _COMPOSITE_INDEXES:
        op.create_index(index_name, _TABLE, columns)
    if _is_postgres():
        for json_column in _JSON_COLUMNS:
            op.alter_column(
                _TABLE,
                json_column,
                type_=postgresql.JSONB(),
                postgresql_using=f"{json_column}::jsonb",
            )
        for json_column, index_name in _JSON_GIN_INDEXES.items():
            op.create_index(
                index_name, _TABLE, [json_column], postgresql_using="gin"
            )


def downgrade() -> None:
    if _is_postgres():
        for index_name in _JSON_GIN_INDEXES.values():
            op.drop_index(index_name, table_name=_TABLE)
        for json_column in _JSON_COLUMNS:
            op.alter_column(
                _TABLE,
                json_column,
                type_=sa.JSON(),
                postgresql_using=f"{json_column}::json",
            )
    for index_name, _columns in reversed(_COMPOSITE_INDEXES):
        op.drop_index(index_name, table_name=_TABLE)
    for column in reversed(_NEW_COLUMNS):
        op.drop_column(_TABLE, column.name)
