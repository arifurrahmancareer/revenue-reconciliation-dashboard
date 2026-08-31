"""
ingest_service.py -- Upload -> parse -> persist, as one transaction.

WHY A SERVICE LAYER AT ALL
  The API layer's job is HTTP: parse the request, check auth, shape the response. The
  domain layer's job is business rules. This layer joins them and owns the transaction
  boundary. Keeping it separate means the ingestion path can be exercised in a test with a
  session and two byte strings -- no TestClient, no multipart encoding.

TRANSACTION SEMANTICS
  One commit at the end. If the payments file turns out to be unparseable after the orders
  file has been read, nothing is written -- there is no half-loaded batch to clean up.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..db.models import DataQualityIssue, Order, Payment, UploadBatch, User
from ..domain.parsing import CsvStructureError, parse_orders_csv, parse_payments_csv
from ..domain.records import ParseResult


class IngestError(Exception):
    """A user-fixable ingestion problem. Mapped to HTTP 400 with the message shown as-is,
    so the message must always name what is wrong AND what to do about it."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _default_label() -> str:
    return f"Upload {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC"


def ingest_files(
    db: Session,
    user: User,
    orders_bytes: bytes,
    payments_bytes: bytes,
    orders_filename: str,
    payments_filename: str,
    label: str | None = None,
) -> tuple[UploadBatch, ParseResult, ParseResult]:
    """
    Parse both files and persist a batch. Returns (batch, orders_result, payments_result).

    Size is checked BEFORE parsing: the parser loads the file into memory, so a 2 GB upload
    would be an out-of-memory kill rather than a validation error.
    """
    settings = get_settings()
    limit = settings.max_upload_bytes

    for name, raw in ((orders_filename, orders_bytes), (payments_filename, payments_bytes)):
        if not raw:
            raise IngestError(f"{name} is empty. Please upload a CSV file with a header row.")
        if len(raw) > limit:
            raise IngestError(
                f"{name} is {len(raw) / 1_048_576:.1f} MB, which exceeds the "
                f"{limit / 1_048_576:.0f} MB upload limit."
            )

    # CsvStructureError -> IngestError: the domain layer raises a domain error, and this
    # layer translates it. The domain never imports HTTP concepts.
    try:
        orders_result = parse_orders_csv(orders_bytes)
    except CsvStructureError as exc:
        raise IngestError(str(exc)) from exc
    try:
        payments_result = parse_payments_csv(payments_bytes)
    except CsvStructureError as exc:
        raise IngestError(str(exc)) from exc

    batch = UploadBatch(
        user_id=user.id,
        label=(label or "").strip() or _default_label(),
        orders_filename=orders_filename[:255],
        payments_filename=payments_filename[:255],
        orders_sha256=_sha256(orders_bytes),
        payments_sha256=_sha256(payments_bytes),
        orders_rows_read=orders_result.rows_read,
        orders_rows_loaded=orders_result.rows_loaded,
        payments_rows_read=payments_result.rows_read,
        payments_rows_loaded=payments_result.rows_loaded,
    )
    db.add(batch)
    db.flush()   # populates batch.id without committing, so children can reference it

    db.add_all([
        Order(
            user_id=user.id, batch_id=batch.id,
            order_key=order.order_key, raw_order_id=order.raw_order_id,
            order_date=order.order_date, customer_email=order.customer_email,
            currency=order.currency, gross_amount=order.gross_amount,
            discount=order.discount, net_amount=order.net_amount,
            status=order.status, source_row=order.source_row,
        )
        for order in orders_result.orders
    ])
    db.add_all([
        Payment(
            user_id=user.id, batch_id=batch.id,
            transaction_ref=payment.transaction_ref, order_key=payment.order_key,
            raw_order_reference=payment.raw_order_reference[:160],
            processed_at=payment.processed_at, currency=payment.currency,
            amount=payment.amount, fee=payment.fee, net_settled=payment.net_settled,
            payment_type=payment.payment_type, status=payment.status,
            source_row=payment.source_row,
        )
        for payment in payments_result.payments
    ])
    db.add_all([
        DataQualityIssue(
            user_id=user.id, batch_id=batch.id, source=issue.source,
            source_row=issue.source_row, identifier=issue.identifier[:120],
            code=issue.code, message=issue.message, dropped=issue.dropped,
        )
        for issue in (orders_result.issues + payments_result.issues)
    ])

    db.commit()
    db.refresh(batch)
    return batch, orders_result, payments_result


# ======================================================================================
# Reads -- every one of them filtered by user_id. That is the tenancy boundary.
# ======================================================================================
def list_batches(db: Session, user: User) -> list[UploadBatch]:
    return list(db.scalars(
        select(UploadBatch)
        .where(UploadBatch.user_id == user.id)
        .order_by(UploadBatch.created_at.desc())
    ))


def get_batch(db: Session, user: User, batch_id: int) -> UploadBatch | None:
    """user_id is part of the WHERE clause, not an assertion afterwards: a request for
    someone else's batch returns None and therefore a 404, which does not even confirm
    that the id exists."""
    return db.scalars(
        select(UploadBatch).where(UploadBatch.id == batch_id, UploadBatch.user_id == user.id)
    ).first()


def get_latest_batch(db: Session, user: User) -> UploadBatch | None:
    return db.scalars(
        select(UploadBatch)
        .where(UploadBatch.user_id == user.id)
        .order_by(UploadBatch.created_at.desc())
        .limit(1)
    ).first()


def get_data_quality_issues(db: Session, user: User, batch_id: int) -> list[DataQualityIssue]:
    return list(db.scalars(
        select(DataQualityIssue)
        .where(DataQualityIssue.user_id == user.id, DataQualityIssue.batch_id == batch_id)
        .order_by(DataQualityIssue.source, DataQualityIssue.source_row)
    ))


def delete_batch(db: Session, user: User, batch_id: int) -> bool:
    """Delete a batch and everything derived from it. Explicit child deletes rather than
    relying on cascades alone, so behaviour is identical on SQLite (where foreign keys are
    off by default) and Postgres."""
    batch = get_batch(db, user, batch_id)
    if batch is None:
        return False
    db.execute(delete(Order).where(Order.batch_id == batch.id, Order.user_id == user.id))
    db.execute(delete(Payment).where(Payment.batch_id == batch.id, Payment.user_id == user.id))
    db.execute(delete(DataQualityIssue).where(
        DataQualityIssue.batch_id == batch.id, DataQualityIssue.user_id == user.id
    ))
    db.delete(batch)   # runs, discrepancies and explanations cascade from here
    db.commit()
    return True
