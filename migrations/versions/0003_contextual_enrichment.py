"""contextual enrichment and capability routing

Every AssetCues document is written to the same template, so its boilerplate is
near-identical across capabilities -- sampling the live index, a chunk's nearest
neighbour in a *different* module reached 0.98 cosine similarity on rows like
"| Requirement | Mapped tests | Authoritative definition |". A retriever cannot
separate those, because as text they are not different. What distinguishes them
is the document they sit in, and the chunk carried none of that.

`chunks.context` holds a short passage situating each chunk inside its own
document. It is embedded with the text but never displayed and never citable,
so it can move a chunk to the right place in vector space without any generated
sentence reaching a reader as though the document said it.

The `documents` columns carry the taxonomy the files already declare
(Capability / Module / Product domain) plus a summary, the terms the document
defines, and what separates it from siblings covering similar ground.

Revision ID: 0003_contextual_enrichment
Revises: 0002_shared_documents
"""

from alembic import op

revision = "0003_contextual_enrichment"
down_revision = "0002_shared_documents"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("ALTER TABLE chunks ADD COLUMN context TEXT NOT NULL DEFAULT ''")

    op.execute(
        """
        ALTER TABLE documents
            ADD COLUMN capability VARCHAR(200) NOT NULL DEFAULT '',
            ADD COLUMN module_declared VARCHAR(200) NOT NULL DEFAULT '',
            ADD COLUMN product_domain VARCHAR(200) NOT NULL DEFAULT '',
            ADD COLUMN summary TEXT NOT NULL DEFAULT '',
            ADD COLUMN key_terms TEXT[] NOT NULL DEFAULT '{}',
            ADD COLUMN distinguishing_points TEXT[] NOT NULL DEFAULT '{}',
            ADD COLUMN enriched_at TIMESTAMP WITH TIME ZONE
        """
    )

    # Routing filters on capability, so it needs an index of its own.
    op.execute("CREATE INDEX ix_documents_capability ON documents (capability)")

    # Seed capability from the folder-derived module so routing has something
    # sane to work with before the first enrichment pass runs.
    op.execute("UPDATE documents SET capability = module WHERE module <> ''")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_capability")
    op.execute(
        """
        ALTER TABLE documents
            DROP COLUMN IF EXISTS capability,
            DROP COLUMN IF EXISTS module_declared,
            DROP COLUMN IF EXISTS product_domain,
            DROP COLUMN IF EXISTS summary,
            DROP COLUMN IF EXISTS key_terms,
            DROP COLUMN IF EXISTS distinguishing_points,
            DROP COLUMN IF EXISTS enriched_at
        """
    )
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS context")
