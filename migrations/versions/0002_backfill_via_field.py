"""backfill via field on existing value_canonical

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-28
"""
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE golden_observations
        SET value_canonical = jsonb_set(value_canonical, '{via}', '"pint"'::jsonb, true)
        WHERE value_canonical IS NOT NULL
          AND jsonb_typeof(value_canonical) = 'object'
          AND NOT (value_canonical ? 'via')
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE golden_observations
        SET value_canonical = value_canonical - 'via' - 'normalization'
        WHERE value_canonical IS NOT NULL
          AND jsonb_typeof(value_canonical) = 'object'
        """
    )
