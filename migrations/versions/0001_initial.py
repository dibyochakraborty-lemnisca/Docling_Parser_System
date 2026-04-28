"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-04-28
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "experiments",
        sa.Column("experiment_id", sa.String(), primary_key=True),
        sa.Column("name", sa.String()),
        sa.Column("uploaded_by", sa.String()),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
        sa.Column("status", sa.String(), server_default="ingesting"),
        sa.Column("notes", sa.String()),
    )
    op.create_table(
        "source_files",
        sa.Column("file_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", sa.String(), sa.ForeignKey("experiments.experiment_id"), nullable=False),
        sa.Column("filename", sa.String(), nullable=False),
        sa.Column("sha256", sa.String(), nullable=False),
        sa.Column("mime_type", sa.String()),
        sa.Column("size_bytes", sa.BigInteger()),
        sa.Column("page_count", sa.Integer()),
        sa.Column("storage_path", sa.String(), nullable=False),
        sa.Column("parsed_at", postgresql.TIMESTAMP(timezone=True)),
        sa.Column("parse_status", sa.String(), server_default="pending"),
        sa.Column("parse_error", sa.String()),
        sa.UniqueConstraint("experiment_id", "sha256", name="uq_file_dedup"),
    )
    op.create_table(
        "golden_observations",
        sa.Column("observation_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", sa.String(), sa.ForeignKey("experiments.experiment_id"), nullable=False),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("source_files.file_id"), nullable=False),
        sa.Column("column_name", sa.String(), nullable=False),
        sa.Column("raw_header", sa.String(), nullable=False),
        sa.Column("observation_type", sa.String(), server_default="unknown"),
        sa.Column("value_raw", postgresql.JSONB(), nullable=False),
        sa.Column("unit_raw", sa.String()),
        sa.Column("value_canonical", postgresql.JSONB()),
        sa.Column("unit_canonical", sa.String()),
        sa.Column("conversion_status", sa.String(), server_default="not_applicable"),
        sa.Column("source_locator", postgresql.JSONB(), nullable=False),
        sa.Column("mapping_confidence", sa.Numeric()),
        sa.Column("extraction_confidence", sa.Numeric()),
        sa.Column("needs_review", sa.Boolean(), server_default=sa.text("false")),
        sa.Column("extractor_version", sa.String(), nullable=False),
        sa.Column("superseded_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("golden_observations.observation_id")),
        sa.Column("extracted_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_obs_exp_col", "golden_observations", ["experiment_id", "column_name"])
    op.create_index("idx_obs_review", "golden_observations", ["needs_review"], postgresql_where=sa.text("needs_review"))
    op.create_index("idx_obs_active", "golden_observations", ["experiment_id"], postgresql_where=sa.text("superseded_by IS NULL"))
    op.create_table(
        "residual_data",
        sa.Column("residual_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("file_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("source_files.file_id"), nullable=False),
        sa.Column("experiment_id", sa.String(), sa.ForeignKey("experiments.experiment_id"), nullable=False),
        sa.Column("extractor_version", sa.String(), nullable=False),
        sa.Column("payload", postgresql.JSONB(), nullable=False),
        sa.Column("superseded_by", postgresql.UUID(as_uuid=True), sa.ForeignKey("residual_data.residual_id")),
        sa.Column("created_at", postgresql.TIMESTAMP(timezone=True), server_default=sa.text("now()")),
    )
    op.create_index("idx_residual_file", "residual_data", ["file_id"])


def downgrade() -> None:
    op.drop_index("idx_residual_file", table_name="residual_data")
    op.drop_table("residual_data")
    op.drop_index("idx_obs_active", table_name="golden_observations")
    op.drop_index("idx_obs_review", table_name="golden_observations")
    op.drop_index("idx_obs_exp_col", table_name="golden_observations")
    op.drop_table("golden_observations")
    op.drop_table("source_files")
    op.drop_table("experiments")
