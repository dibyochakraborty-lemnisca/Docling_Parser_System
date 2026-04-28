"""backfill source_locator.section='table' on existing observations

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-28
"""
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        UPDATE golden_observations
        SET source_locator = jsonb_set(source_locator, '{section}', '"table"'::jsonb, true)
        WHERE source_locator IS NOT NULL
          AND jsonb_typeof(source_locator) = 'object'
          AND NOT (source_locator ? 'section')
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE golden_observations
        SET source_locator = source_locator - 'section' - 'evidence_quote'
        WHERE source_locator IS NOT NULL
          AND jsonb_typeof(source_locator) = 'object'
        """
    )
