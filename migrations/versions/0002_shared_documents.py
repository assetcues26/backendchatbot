"""shared documents

AssetCues product documentation is authored once and read by every customer.
The original predicate required ``doc.tenant_id = user.tenant_id`` exactly,
which made the customer-facing half of the product impossible: every document
belongs to the AssetCues tenant, and every customer belongs to their own.

``is_shared`` relaxes the TENANT dimension only. Sensitivity and the role ACL
still apply in full, so a shared level-3 specification stays invisible to
customers -- their clearance is 2 and they are not in its ACL. Sharing widens
*who can reach* a document, never *what they are cleared for*.

Backfill marks existing internal-tenant documents as shared, which is what
they are: product documentation. Customer-uploaded material stays scoped.

Revision ID: 0002_shared_documents
Revises: 0001_initial
"""

from alembic import op

revision = "0002_shared_documents"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE documents ADD COLUMN is_shared BOOLEAN NOT NULL DEFAULT FALSE"
    )
    op.execute(
        "UPDATE documents SET is_shared = TRUE "
        "WHERE tenant_id IN (SELECT id FROM tenants WHERE kind = 'INTERNAL')"
    )
    op.execute("CREATE INDEX ix_documents_is_shared ON documents (is_shared)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_is_shared")
    op.execute("ALTER TABLE documents DROP COLUMN IF EXISTS is_shared")
