"""Folder sync CLI: ``acues-ingest``.

Bulk-loads a directory of documents and, on later runs, reconciles the
database with what is on disk -- adding new files, updating changed ones, and
deleting documents whose source file is gone.

    acues-ingest sync "C:/path/to/Product Doc"
    acues-ingest sync ./docs --auto-approve      # demo bootstrap only
    acues-ingest sync ./docs --dry-run
    acues-ingest enrich              # profile and situate anything unenriched
    acues-ingest enrich --all --force  # rewrite every context from scratch
    acues-ingest status

Idempotent: running it twice makes no embedding calls the second time,
because unchanged content is detected by hash before any model is invoked.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import audit
from app.core.principal import ROLE_ADMIN, Principal
from app.db.models import Chunk, DocStatus, Document, Role
from app.db.seed import INTERNAL_TENANT_SLUG, seed_all, suggest_access
from app.db.session import dispose_engine, get_session_factory
from app.ingest import backfill, pipeline
from app.ingest.parsers import SUPPORTED_SUFFIXES

# A stable identity for actions taken by the command line rather than a person.
SYSTEM_USER_ID = uuid.UUID("00000000-0000-0000-0000-000000000001")


def _system_principal(tenant_id: uuid.UUID, role_ids: dict[str, int]) -> Principal:
    return Principal(
        user_id=SYSTEM_USER_ID,
        email="cli@assetcues.local",
        tenant_id=tenant_id,
        tenant_slug=INTERNAL_TENANT_SLUG,
        role_ids=frozenset({role_ids[ROLE_ADMIN]}),
        role_keys=frozenset({ROLE_ADMIN}),
        clearance=4,
    )


def discover(root: Path) -> list[Path]:
    if not root.exists():
        raise SystemExit(f"error: path does not exist: {root}")
    if root.is_file():
        return [root]
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith("~$")  # Word lock files
    )


def source_key_for(path: Path, root: Path) -> str:
    """Path relative to the sync root, so a rename is a delete plus an add."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


async def cmd_sync(args: argparse.Namespace) -> int:
    root = Path(args.path).resolve()
    files = discover(root)
    if not files:
        print(f"No supported documents found under {root}")
        return 1

    print(f"Found {len(files)} document(s) under {root}\n")

    # A dry run answers "what would this do" and must not require a database,
    # a network, or an API key. It is the first thing someone reaches for when
    # the connection is the thing they are unsure about.
    if args.dry_run:
        for path in files:
            key = source_key_for(path, root)
            module = path.parent.name if path.parent != root else ""
            print(f"  would ingest  {key}" + (f"   [module: {module}]" if module else ""))
        print(f"\n{len(files)} document(s) would be processed. No changes made.")
        return 0

    factory = get_session_factory()
    async with factory() as session:
        tenant_id, role_ids = await seed_all(session)
        principal = _system_principal(tenant_id, role_ids)

        created = updated = unchanged = failed = 0
        reused_total = embedded_total = 0
        seen_keys: set[str] = set()

        for path in files:
            key = source_key_for(path, root)
            seen_keys.add(key)
            module = path.parent.name if path.parent != root else ""

            try:
                result = await pipeline.ingest_path(
                    session,
                    path,
                    tenant_id=tenant_id,
                    source_key=key,
                    module=module,
                    principal=principal,
                    run_classifier=not args.no_classifier,
                )
            except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
                await session.rollback()
                failed += 1
                print(f"  FAILED   {key}: {type(exc).__name__}: {exc}")
                continue

            embedded_total += result.chunks_embedded
            reused_total += result.chunks_reused

            if result.action == "created":
                created += 1
            elif result.action == "updated":
                updated += 1
            else:
                unchanged += 1

            if args.auto_approve and result.action in {"created", "updated"}:
                await _auto_approve(session, result.document_id, path, principal)

            note = ""
            if result.chunks_reused:
                note = f" ({result.chunks_reused} embeddings reused)"
            print(
                f"  {result.action:<9} {key}  "
                f"[{result.chunks_total} chunks]{note}"
            )

            await session.commit()

        removed = 0
        if not args.dry_run and not args.no_delete:
            removed = await _remove_missing(session, tenant_id, seen_keys, principal)
            await session.commit()

    print(
        f"\ncreated={created} updated={updated} unchanged={unchanged} "
        f"deleted={removed} failed={failed}"
    )
    print(f"embeddings computed={embedded_total} reused={reused_total}")
    if not args.auto_approve and (created or updated):
        print(
            "\nDocuments are in PENDING_REVIEW and readable by nobody. "
            "Approve them in the admin panel, or re-run with --auto-approve "
            "to apply the default access matrix."
        )
    return 0 if failed == 0 else 2


async def _auto_approve(
    session: AsyncSession,
    document_id: uuid.UUID,
    path: Path,
    principal: Principal,
) -> None:
    """Apply the seeded access matrix without a human.

    Convenience for bootstrapping the demo from a known-good folder. It is
    off by default and every approval is still written to the audit log, so
    there is a record that a machine and not a person made the decision.
    """
    doc = (
        await session.execute(select(Document).where(Document.id == document_id))
    ).scalar_one()
    sensitivity, role_keys = suggest_access(doc.doc_type, path.name)
    await pipeline.approve_document(session, doc, role_keys, sensitivity, principal)


async def _remove_missing(
    session: AsyncSession,
    tenant_id: uuid.UUID,
    seen_keys: set[str],
    principal: Principal,
) -> int:
    """Delete documents whose source file no longer exists.

    This is the half of the pipeline people forget. Without it, deleting a
    confidential file from the folder leaves it fully retrievable.
    """
    rows = (
        await session.execute(
            select(Document).where(Document.tenant_id == tenant_id)
        )
    ).scalars().all()

    removed = 0
    for doc in rows:
        if doc.source_key in seen_keys:
            continue
        count = await pipeline.delete_document(session, doc, principal)
        print(f"  deleted   {doc.source_key}  [{count} chunks removed]")
        removed += 1
    return removed


async def cmd_enrich(args: argparse.Namespace) -> int:
    """Re-run the enrichment pass over documents already in the database.

    This is what makes a prompt improvement deployable: nobody has to find the
    original folder and re-upload it. It reads the text back out of the chunks,
    profiles the document, writes a context for each chunk and re-embeds only
    what actually changed.

    Access is untouched -- no ACL, sensitivity or status is written -- but every
    cached answer is invalidated at the end, because the retrieval those answers
    were built from has moved.
    """
    factory = get_session_factory()
    async with factory() as session:
        tenant_id, role_ids = await seed_all(session)
        await session.commit()

        documents = await backfill.select_documents(
            session, tenant_id=tenant_id, only_missing=not args.all
        )
        if not documents:
            print(
                "Nothing to enrich. Every document already has a profile; "
                "use --all to re-run them anyway."
            )
            return 0

        print(f"Enriching {len(documents)} document(s)\n")

        if args.dry_run:
            for doc in documents:
                state = "enriched" if doc.enriched_at else "never enriched"
                print(f"  would enrich  {doc.source_key:<52} [{state}]")
            print(
                f"\n{len(documents)} document(s) would be processed. "
                "No changes made."
            )
            return 0

        principal = _system_principal(tenant_id, role_ids)
        siblings = await pipeline.sibling_capabilities(session, tenant_id)

        done = failed = 0
        contexts_total = embedded_total = 0

        for doc in documents:
            try:
                report = await backfill.enrich_stored_document(
                    session, doc, siblings=siblings, force=args.force
                )
            except Exception as exc:  # noqa: BLE001 - one bad document is not a run
                await session.rollback()
                failed += 1
                print(f"  FAILED   {doc.source_key}: {type(exc).__name__}: {exc}")
                continue

            await audit.record(
                session,
                audit.Event.DOC_ENRICHED,
                principal=principal,
                document_id=doc.id,
                title=doc.title,
                capability=report.capability,
                chunks=report.chunks_total,
                contexts=report.contexts_written,
                re_embedded=report.chunks_embedded,
            )
            await session.commit()

            # A capability learned from one document teaches its neighbours.
            if doc.capability and doc.module and doc.capability != doc.module:
                siblings.setdefault(doc.module, doc.capability)

            done += 1
            contexts_total += report.contexts_written
            embedded_total += report.chunks_embedded
            note = f"  {report.message}" if report.message else ""
            print(
                f"  enriched  {doc.source_key:<52} "
                f"[{report.contexts_written}/{report.chunks_total} contexts, "
                f"{report.chunks_embedded} re-embedded]  "
                f"{report.capability or '(no capability)'}{note}"
            )

        if done:
            # Cached answers were built from the old vectors. Retire them.
            await audit.bump_acl_version(session)
            await session.commit()

    print(f"\nenriched={done} failed={failed}")
    print(f"contexts written={contexts_total} embeddings computed={embedded_total}")
    return 0 if failed == 0 else 2


async def cmd_status(_: argparse.Namespace) -> int:
    factory = get_session_factory()
    async with factory() as session:
        total = (
            await session.execute(select(func.count()).select_from(Document))
        ).scalar_one()
        chunks = (
            await session.execute(select(func.count()).select_from(Chunk))
        ).scalar_one()
        orphans = (
            await session.execute(
                select(func.count())
                .select_from(Chunk)
                .where(~Chunk.document_id.in_(select(Document.id)))
            )
        ).scalar_one()

        print(f"documents: {total}")
        print(f"chunks:    {chunks}")
        print(f"orphaned chunks: {orphans}" + ("  <-- BUG" if orphans else "  (good)"))

        print("\nby status:")
        for st in DocStatus:
            n = (
                await session.execute(
                    select(func.count())
                    .select_from(Document)
                    .where(Document.status == st)
                )
            ).scalar_one()
            if n:
                print(f"  {st.value:<16} {n}")

        roles = (await session.execute(select(Role).order_by(Role.key))).scalars().all()
        if roles:
            print("\nroles:")
            for r in roles:
                print(f"  {r.key:<14} clearance={r.clearance}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="acues-ingest",
        description="Sync a folder of documents into the AssetCues assistant.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sync = sub.add_parser("sync", help="ingest a folder and reconcile deletions")
    sync.add_argument("path", help="file or directory to sync")
    sync.add_argument(
        "--auto-approve",
        action="store_true",
        help="apply the default access matrix instead of leaving documents "
        "in the review queue (demo bootstrap only)",
    )
    sync.add_argument(
        "--no-delete",
        action="store_true",
        help="do not remove documents whose source file has disappeared",
    )
    sync.add_argument(
        "--no-classifier",
        action="store_true",
        help="skip the LLM classification pass (no model calls)",
    )
    sync.add_argument("--dry-run", action="store_true", help="list actions only")
    sync.set_defaults(func=cmd_sync)

    enrich = sub.add_parser(
        "enrich",
        help="re-run the enrichment pass over documents already ingested",
    )
    enrich.add_argument(
        "--all",
        action="store_true",
        help="include documents that already have a profile "
        "(default: only those that have never been enriched)",
    )
    enrich.add_argument(
        "--force",
        action="store_true",
        help="rewrite contexts that already exist, instead of keeping them",
    )
    enrich.add_argument("--dry-run", action="store_true", help="list actions only")
    enrich.set_defaults(func=cmd_enrich)

    status = sub.add_parser("status", help="show what is in the database")
    status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    async def run() -> int:
        try:
            result: int = await args.func(args)
            return result
        finally:
            await dispose_engine()

    try:
        sys.exit(asyncio.run(run()))
    except KeyboardInterrupt:
        print("\nInterrupted. Documents already committed are unaffected.")
        sys.exit(130)
    except Exception as exc:  # noqa: BLE001 - this is the top of a CLI
        # An operator running a command should get a sentence, not a
        # traceback. The three failures below are the ones people actually
        # hit on first setup.
        sys.exit(_explain(exc))


def _explain(exc: Exception) -> str:
    name = type(exc).__name__
    text = str(exc)

    if "password authentication failed" in text or name == "InvalidPasswordError":
        return (
            "error: the database rejected the credentials in DATABASE_URL.\n"
            "       Check the password in your .env against Supabase -> "
            "Settings -> Database."
        )
    if name in {"ConnectionRefusedError", "OSError", "TimeoutError"} or (
        "connect" in text.lower() and "refused" in text.lower()
    ):
        return (
            "error: could not reach the database.\n"
            f"       {text}\n"
            "       Check DATABASE_URL, and that the Supabase project is not "
            "paused."
        )
    if "OPENAI_API_KEY" in text:
        return (
            "error: OPENAI_API_KEY is not set.\n"
            "       Add it to .env, or pass --no-classifier to skip all model "
            "calls."
        )
    if "UndefinedTableError" in name or "does not exist" in text:
        return (
            "error: the schema is missing. Run:\n"
            "       python -m alembic upgrade head"
        )
    return f"error: {name}: {text}"


if __name__ == "__main__":
    main()
