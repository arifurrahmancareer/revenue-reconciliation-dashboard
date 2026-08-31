"""
models.py -- SQLAlchemy 2.0 ORM models.

SCHEMA PRINCIPLES
  1. EVERY tenant-owned table carries `user_id`. Multi-tenancy is enforced by a WHERE
     clause on every query plus a foreign key, not by hoping the UI filters correctly.
     There is no "admin" bypass in the code, so there is no path that leaks another
     user's data.
  2. MONEY IS Numeric(14,2), never Float. On Postgres this is exact decimal arithmetic.
     14 digits covers 999,999,999,999.99 -- absurd for this dataset, cheap to allow.
  3. RESULTS ARE STORED, NOT RECOMPUTED. A reconciliation run persists its findings AND
     the config that produced them, so a discrepancy from last month can still be
     explained with the tolerances in force at the time. Recomputing on read would make
     history mutable.
  4. RAW VALUES ARE KEPT beside cleaned ones (`raw_order_id`, `raw_order_reference`), so
     the UI can show 'we matched " ord-1801 " to ORD-1801' rather than asking for trust.
  5. CASCADE DELETES from user and batch: deleting an upload must not leave orphaned
     discrepancies pointing at rows that no longer exist.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

from sqlalchemy import (
    JSON, Boolean, DateTime, ForeignKey, Index, Integer, Numeric, String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Numeric(14, 2): the single most important type decision in the schema.
Money = Numeric(14, 2)


def _utcnow() -> datetime:
    """Timezone-aware UTC. Naive server-local timestamps are the reason 'why is this row
    5.5 hours old' bug reports exist."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    # unique + index: the login lookup, and the guard against duplicate accounts.
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True, nullable=False)
    # Only ever a bcrypt hash. The plaintext password is not stored, logged or returned.
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    batches: Mapped[list["UploadBatch"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class UploadBatch(Base):
    """
    One upload of one orders file + one payments file.

    Batches exist so a user can upload twice without the second upload silently merging
    into the first, and so 'reconcile' always means 'reconcile this pair of files'.
    """

    __tablename__ = "upload_batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    label: Mapped[str] = mapped_column(String(120), nullable=False)

    orders_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    payments_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    # SHA-256 of the raw bytes: lets the UI say 'you already uploaded this exact file',
    # and makes a result reproducible against a specific file rather than a filename.
    orders_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    payments_sha256: Mapped[str] = mapped_column(String(64), nullable=False)

    orders_rows_read: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    orders_rows_loaded: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    payments_rows_read: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    payments_rows_loaded: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    user: Mapped[User] = relationship(back_populates="batches")
    orders: Mapped[list["Order"]] = relationship(cascade="all, delete-orphan")
    payments: Mapped[list["Payment"]] = relationship(cascade="all, delete-orphan")
    runs: Mapped[list["ReconRun"]] = relationship(cascade="all, delete-orphan")
    issues: Mapped[list["DataQualityIssue"]] = relationship(cascade="all, delete-orphan")

    __table_args__ = (Index("ix_batches_user_created", "user_id", "created_at"),)


class Order(Base):
    """A cleaned row from orders.csv."""

    __tablename__ = "orders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    batch_id: Mapped[int] = mapped_column(ForeignKey("upload_batches.id", ondelete="CASCADE"), nullable=False)

    order_key: Mapped[str] = mapped_column(String(120), nullable=False)   # normalised join key
    raw_order_id: Mapped[str] = mapped_column(String(120), nullable=False)  # audit trail
    order_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    # Already masked at parse time. The raw address never reaches this table.
    customer_email: Mapped[str | None] = mapped_column(String(320), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    gross_amount: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    discount: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    net_amount: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    status: Mapped[str] = mapped_column(String(40), nullable=False)
    source_row: Mapped[int] = mapped_column(Integer, nullable=False)  # CSV line number

    __table_args__ = (
        # One order per key per batch. The DB enforces what the parser promises -- if a
        # future change let a duplicate through, the insert fails loudly instead of
        # double counting the order value.
        UniqueConstraint("batch_id", "order_key", name="uq_orders_batch_key"),
        Index("ix_orders_user_batch", "user_id", "batch_id"),
    )


class Payment(Base):
    """A cleaned row from payments.csv. Multiple rows per order_key are expected."""

    __tablename__ = "payments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    batch_id: Mapped[int] = mapped_column(ForeignKey("upload_batches.id", ondelete="CASCADE"), nullable=False)

    transaction_ref: Mapped[str] = mapped_column(String(120), nullable=False)
    order_key: Mapped[str] = mapped_column(String(120), nullable=False)
    raw_order_reference: Mapped[str] = mapped_column(String(160), nullable=False)
    # Nullable on purpose: one payment in the sample has no date. Money without a
    # timestamp is still money, so NOT NULL here would have deleted a real transaction.
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=False), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    amount: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    fee: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    net_settled: Mapped[Decimal | None] = mapped_column(Money, nullable=True)
    payment_type: Mapped[str] = mapped_column(String(24), nullable=False)  # charge | refund
    status: Mapped[str] = mapped_column(String(24), nullable=False)        # settled|pending|failed
    source_row: Mapped[int] = mapped_column(Integer, nullable=False)

    __table_args__ = (
        UniqueConstraint("batch_id", "transaction_ref", name="uq_payments_batch_txn"),
        Index("ix_payments_user_batch", "user_id", "batch_id"),
        Index("ix_payments_order_key", "batch_id", "order_key"),  # the join key lookup
    )


class ReconRun(Base):
    """One execution of the engine over one batch."""

    __tablename__ = "recon_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    batch_id: Mapped[int] = mapped_column(ForeignKey("upload_batches.id", ondelete="CASCADE"), nullable=False)

    engine_version: Mapped[str] = mapped_column(String(20), nullable=False)
    # The exact tolerances used. Without this, a stored finding is unexplainable later.
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    # The computed headline figures, so the dashboard is one row read rather than a
    # re-aggregation of thousands of rows on every page load.
    summary_json: Mapped[dict] = mapped_column(JSON, nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    discrepancies: Mapped[list["Discrepancy"]] = relationship(cascade="all, delete-orphan")

    __table_args__ = (Index("ix_runs_user_created", "user_id", "created_at"),)


class Discrepancy(Base):
    """One finding: one rule firing on one order key, within one run."""

    __tablename__ = "discrepancies"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    run_id: Mapped[int] = mapped_column(ForeignKey("recon_runs.id", ondelete="CASCADE"), nullable=False)

    order_key: Mapped[str] = mapped_column(String(120), nullable=False)
    discrepancy_type: Mapped[str] = mapped_column(String(40), nullable=False)
    severity: Mapped[str] = mapped_column(String(20), nullable=False)
    risk_direction: Mapped[str] = mapped_column(String(30), nullable=False)
    rule_id: Mapped[str] = mapped_column(String(40), nullable=False)  # e.g. R04-DUPLICATE-CHARGE

    summary: Mapped[str] = mapped_column(String(400), nullable=False)
    detail: Mapped[str] = mapped_column(Text, nullable=False)

    expected_amount: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    collected_amount: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    delta_amount: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    # 0.00 on secondary findings, so SUM(amount_at_risk) never double counts one order.
    amount_at_risk: Mapped[Decimal] = mapped_column(Money, default=Decimal("0.00"), nullable=False)
    is_primary: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)

    order_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    customer_email_masked: Mapped[str | None] = mapped_column(String(320), nullable=True)
    # The full evidence bundle: transaction refs, dates, amounts, statuses. This is also
    # exactly what the LLM prompt is built from -- no separate, driftable copy.
    evidence_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    explanations: Mapped[list["AiExplanation"]] = relationship(cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_disc_run_type", "run_id", "discrepancy_type"),   # the dashboard filter
        Index("ix_disc_user_run", "user_id", "run_id"),            # the tenancy filter
        Index("ix_disc_risk", "run_id", "risk_direction"),
    )


class DataQualityIssue(Base):
    """
    A file-hygiene problem, stored separately from money discrepancies.

    Keeping these out of `discrepancies` is a deliberate modelling decision: 'this row had
    no email' must never inflate the number a finance team is asked to act on.
    """

    __tablename__ = "data_quality_issues"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    batch_id: Mapped[int] = mapped_column(ForeignKey("upload_batches.id", ondelete="CASCADE"), nullable=False)

    source: Mapped[str] = mapped_column(String(20), nullable=False)     # orders | payments
    source_row: Mapped[int] = mapped_column(Integer, nullable=False)    # CSV line number
    identifier: Mapped[str] = mapped_column(String(120), nullable=False)
    code: Mapped[str] = mapped_column(String(60), nullable=False)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    dropped: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    __table_args__ = (Index("ix_dq_user_batch", "user_id", "batch_id"),)


class AiExplanation(Base):
    """
    A cached LLM explanation for one discrepancy.

    WHY CACHE: identical prompts cost money and add latency every time a reviewer reopens
    a row. The unique constraint on (discrepancy_id, model, prompt_version) means a model
    or prompt change produces a NEW explanation instead of silently overwriting the old
    one -- so results stay comparable across versions.

    `source` records whether the text came from OpenAI or from the deterministic fallback,
    which is how the UI can be honest about what the user is reading.
    """

    __tablename__ = "ai_explanations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    discrepancy_id: Mapped[int] = mapped_column(ForeignKey("discrepancies.id", ondelete="CASCADE"), nullable=False)

    model: Mapped[str] = mapped_column(String(80), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(20), nullable=False)
    temperature: Mapped[float] = mapped_column(Numeric(3, 2), nullable=False)

    what_happened: Mapped[str] = mapped_column(Text, nullable=False)
    likely_cause: Mapped[str | None] = mapped_column(Text, nullable=True)
    recommended_action: Mapped[str] = mapped_column(Text, nullable=False)
    owner_team: Mapped[str | None] = mapped_column(String(60), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)

    source: Mapped[str] = mapped_column(String(20), nullable=False)  # openai | fallback
    was_repaired: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    __table_args__ = (
        UniqueConstraint("discrepancy_id", "model", "prompt_version", name="uq_expl_disc_model"),
        Index("ix_expl_user", "user_id"),
    )
