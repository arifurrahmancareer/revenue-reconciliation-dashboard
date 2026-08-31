"""
records.py -- The cleaned, in-memory shapes the reconciliation engine works with.

These are plain frozen dataclasses, NOT SQLAlchemy models and NOT Pydantic schemas.

WHY (interview answer): the engine is the most valuable and most testable part of this
project, so it must not depend on the web framework or the database. Keeping a pure
domain layer means:
  * `pytest` runs the engine with zero I/O and zero fixtures,
  * the same engine can later run in a batch job, a Lambda, or a notebook,
  * a DB migration can never change reconciliation behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class CleanOrder:
    """One row of orders.csv after cleaning."""

    order_key: str  # normalised join key, e.g. 'ORD-1801'
    raw_order_id: str  # exactly what was in the file, kept for audit/trust
    order_date: datetime | None
    customer_email: str | None
    currency: str
    gross_amount: Decimal | None
    discount: Decimal | None
    net_amount: Decimal | None
    status: str  # normalised lower-case
    source_row: int  # 1-based CSV line number, so a user can find the row in Excel


@dataclass(frozen=True, slots=True)
class CleanPayment:
    """One row of payments.csv after cleaning."""

    transaction_ref: str
    order_key: str  # normalised join key
    raw_order_reference: str  # e.g. ' ord-1801 ' -- proof of what we fixed
    processed_at: datetime | None
    currency: str
    amount: Decimal | None
    fee: Decimal | None
    net_settled: Decimal | None
    payment_type: str  # 'charge' | 'refund' | ...
    status: str  # 'settled' | 'pending' | 'failed' | ...
    source_row: int


@dataclass(frozen=True, slots=True)
class RowIssue:
    """
    A problem found while PARSING a single row (as opposed to a reconciliation
    discrepancy, which is about two rows disagreeing).

    Separating the two is a deliberate product decision: 'this cell is empty' is a data
    hygiene task for the ops team, while 'we charged the customer twice' is money.
    Mixing them into one number would make the headline figure meaningless.
    """

    source: str  # 'orders' | 'payments'
    source_row: int
    identifier: str  # order id / transaction ref, best effort
    code: str  # machine-readable, e.g. 'MISSING_EMAIL'
    message: str  # human sentence for the UI
    dropped: bool = False  # True only when the row could not be used at all


@dataclass(slots=True)
class ParseResult:
    """Output of ingesting one CSV file."""

    orders: list[CleanOrder] = field(default_factory=list)
    payments: list[CleanPayment] = field(default_factory=list)
    issues: list[RowIssue] = field(default_factory=list)
    rows_read: int = 0
    rows_loaded: int = 0
    rows_dropped: int = 0
    duplicate_rows_collapsed: int = 0
