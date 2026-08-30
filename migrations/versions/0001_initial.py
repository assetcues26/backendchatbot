"""initial schema

Creates the full schema: tenancy, identity, documents, chunks with pgvector
embeddings and a generated tsvector column, ACL tables, audit log.

Two things worth knowing about this migration:

* ``CREATE EXTENSION vector`` must run before the chunks table, which has a
  ``vector(1536)`` column. On Supabase the extension is available but not
  enabled by default.
* The HNSW index on ``chunks.embedding`` uses ``vector_cosine_ops`` because
  the retrieval query orders by ``<=>`` (cosine distance). An index built with
  a different operator class is silently ignored by the planner.

Revision ID: 0001_initial
Revises:
"""

from alembic import op

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- enum types -------------------------------------------------
    op.execute("CREATE TYPE tenant_kind AS ENUM ('INTERNAL', 'CUSTOMER')")
    op.execute("CREATE TYPE doc_status AS ENUM ('PROCESSING', 'PENDING_REVIEW', 'APPROVED', 'ARCHIVED', 'FAILED')")
    op.execute("CREATE TYPE grant_effect AS ENUM ('ALLOW', 'DENY')")

    # --- tables and indexes ------------------------------------------
    op.execute(
        """
        CREATE TABLE audit_log (
            id UUID NOT NULL,
            event_type VARCHAR(64) NOT NULL,
            actor_user_id UUID,
            actor_email VARCHAR(320) NOT NULL,
            tenant_id UUID,
            actor_role_keys TEXT[] NOT NULL,
            document_id UUID,
            severity VARCHAR(16) NOT NULL,
            detail JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id)
        )
        """
    )
    op.execute("CREATE INDEX ix_audit_log_actor_user_id ON audit_log (actor_user_id)")
    op.execute("CREATE INDEX ix_audit_log_created_at ON audit_log (created_at)")
    op.execute("CREATE INDEX ix_audit_log_event_type ON audit_log (event_type)")
    op.execute("CREATE INDEX ix_audit_log_severity ON audit_log (severity)")
    op.execute("CREATE INDEX ix_audit_log_tenant_id ON audit_log (tenant_id)")
    op.execute(
        """
        CREATE TABLE document_versions (
            id UUID NOT NULL,
            document_id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            version INTEGER NOT NULL,
            content_sha256 VARCHAR(64) NOT NULL,
            title VARCHAR(500) NOT NULL,
            sensitivity SMALLINT,
            role_keys TEXT[] NOT NULL,
            action VARCHAR(32) NOT NULL,
            note TEXT NOT NULL,
            created_by UUID,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id)
        )
        """
    )
    op.execute("CREATE INDEX ix_document_versions_document_id ON document_versions (document_id)")
    op.execute("CREATE INDEX ix_document_versions_tenant_id ON document_versions (tenant_id)")
    op.execute(
        """
        CREATE TABLE roles (
            id SERIAL NOT NULL,
            key VARCHAR(32) NOT NULL,
            name VARCHAR(100) NOT NULL,
            description TEXT NOT NULL,
            clearance SMALLINT NOT NULL,
            is_internal BOOLEAN NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT ck_role_clearance CHECK (clearance BETWEEN 1 AND 4)
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_roles_key ON roles (key)")
    op.execute(
        """
        CREATE TABLE system_state (
            id INTEGER NOT NULL,
            acl_version BIGINT NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT ck_system_state_singleton CHECK (id = 1)
        )
        """
    )
    op.execute(
        """
        CREATE TABLE tenants (
            id UUID NOT NULL,
            slug VARCHAR(64) NOT NULL,
            name VARCHAR(200) NOT NULL,
            kind tenant_kind NOT NULL,
            is_active BOOLEAN NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id)
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_tenants_slug ON tenants (slug)")
    op.execute(
        """
        CREATE TABLE documents (
            id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            title VARCHAR(500) NOT NULL,
            source_filename VARCHAR(500) NOT NULL,
            source_key VARCHAR(700) NOT NULL,
            module VARCHAR(200) NOT NULL,
            doc_type VARCHAR(100) NOT NULL,
            content_sha256 VARCHAR(64) NOT NULL,
            version INTEGER NOT NULL,
            byte_size BIGINT NOT NULL,
            storage_path VARCHAR(1000) NOT NULL,
            sensitivity SMALLINT NOT NULL,
            status doc_status NOT NULL,
            declared_audience TEXT[] NOT NULL,
            suggested_role_keys TEXT[] NOT NULL,
            suggested_sensitivity SMALLINT,
            classifier_rationale TEXT NOT NULL,
            uploaded_by UUID,
            approved_by UUID,
            approved_at TIMESTAMP WITH TIME ZONE,
            error_message TEXT NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT uq_document_tenant_source UNIQUE (tenant_id, source_key),
            CONSTRAINT ck_doc_sensitivity CHECK (sensitivity BETWEEN 1 AND 4),
            FOREIGN KEY(tenant_id) REFERENCES tenants (id) ON DELETE RESTRICT
        )
        """
    )
    op.execute("CREATE INDEX ix_documents_content_sha256 ON documents (content_sha256)")
    op.execute("CREATE INDEX ix_documents_source_key ON documents (source_key)")
    op.execute("CREATE INDEX ix_documents_status ON documents (status)")
    op.execute("CREATE INDEX ix_documents_status_tenant ON documents (status, tenant_id)")
    op.execute("CREATE INDEX ix_documents_tenant_id ON documents (tenant_id)")
    op.execute(
        """
        CREATE TABLE users (
            id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            email VARCHAR(320) NOT NULL,
            display_name VARCHAR(200) NOT NULL,
            is_active BOOLEAN NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            last_seen_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id),
            FOREIGN KEY(tenant_id) REFERENCES tenants (id) ON DELETE RESTRICT
        )
        """
    )
    op.execute("CREATE UNIQUE INDEX ix_users_email ON users (email)")
    op.execute("CREATE INDEX ix_users_is_active ON users (is_active)")
    op.execute("CREATE INDEX ix_users_tenant_id ON users (tenant_id)")
    op.execute(
        """
        CREATE TABLE access_requests (
            id UUID NOT NULL,
            user_id UUID NOT NULL,
            question TEXT NOT NULL,
            justification TEXT NOT NULL,
            status VARCHAR(24) NOT NULL,
            resolved_by UUID,
            resolved_at TIMESTAMP WITH TIME ZONE,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX ix_access_requests_status ON access_requests (status)")
    op.execute("CREATE INDEX ix_access_requests_user_id ON access_requests (user_id)")
    op.execute(
        """
        CREATE TABLE chunks (
            id UUID NOT NULL,
            document_id UUID NOT NULL,
            tenant_id UUID NOT NULL,
            ordinal INTEGER NOT NULL,
            heading_path TEXT NOT NULL,
            text TEXT NOT NULL,
            text_sha256 VARCHAR(64) NOT NULL,
            token_count INTEGER NOT NULL,
            embedding VECTOR(1536),
            tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (id),
            CONSTRAINT uq_chunk_doc_ordinal UNIQUE (document_id, ordinal),
            FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX ix_chunks_document_id ON chunks (document_id)")
    op.execute("CREATE INDEX ix_chunks_tenant_id ON chunks (tenant_id)")
    op.execute("CREATE INDEX ix_chunks_text_sha256 ON chunks (text_sha256)")
    op.execute("CREATE INDEX ix_chunks_tsv ON chunks USING gin (tsv)")
    op.execute(
        """
        CREATE TABLE document_acl (
            document_id UUID NOT NULL,
            role_id INTEGER NOT NULL,
            granted_by UUID,
            granted_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (document_id, role_id),
            FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE,
            FOREIGN KEY(role_id) REFERENCES roles (id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        CREATE TABLE user_document_grants (
            id UUID NOT NULL,
            user_id UUID NOT NULL,
            document_id UUID NOT NULL,
            effect grant_effect NOT NULL,
            reason TEXT NOT NULL,
            granted_by UUID,
            granted_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            expires_at TIMESTAMP WITH TIME ZONE,
            PRIMARY KEY (id),
            CONSTRAINT uq_user_doc_grant UNIQUE (user_id, document_id, effect),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY(document_id) REFERENCES documents (id) ON DELETE CASCADE
        )
        """
    )
    op.execute("CREATE INDEX ix_user_document_grants_document_id ON user_document_grants (document_id)")
    op.execute("CREATE INDEX ix_user_document_grants_user_id ON user_document_grants (user_id)")
    op.execute(
        """
        CREATE TABLE user_roles (
            user_id UUID NOT NULL,
            role_id INTEGER NOT NULL,
            granted_by UUID,
            granted_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            PRIMARY KEY (user_id, role_id),
            FOREIGN KEY(user_id) REFERENCES users (id) ON DELETE CASCADE,
            FOREIGN KEY(role_id) REFERENCES roles (id) ON DELETE CASCADE
        )
        """
    )

    # --- vector index --------------------------------------------------
    # Cosine distance, matching the `<=>` operator in app/rag/retrieval.py.
    op.execute(
        "CREATE INDEX ix_chunks_embedding_hnsw ON chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    # --- singleton row the ACL version counter lives in -----------------
    op.execute("INSERT INTO system_state (id, acl_version) VALUES (1, 1)")


def downgrade() -> None:
    for table in (
        "access_requests",
        "audit_log",
        "user_document_grants",
        "document_acl",
        "document_versions",
        "chunks",
        "documents",
        "user_roles",
        "users",
        "roles",
        "tenants",
        "system_state",
    ):
        op.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
    for enum_name in ("grant_effect", "doc_status", "tenant_kind"):
        op.execute(f"DROP TYPE IF EXISTS {enum_name}")
