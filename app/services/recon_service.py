"""
recon_service.py -- Runs the engine over a stored batch and persists the results.

BOUNDARY: this module converts ORM rows into the engine's frozen dataclasses, calls the
pure `reconcile()` function, and writes the findings back. The engine itself never imports
SQLAlchemy, so it stays testable and provably free of side effects.

WHY PERSIST FINDINGS INSTEAD OF RECOMPUTING ON READ
  * a stored run is auditable: it keeps the exact config that produced it;
  * the dashboard reads one summary row instead of re-aggregating on every page load;
  * an AI explanation can be attached to a specific finding and stay valid.
"""

from __future__ import annotations

import time
from datetime import datetime
from decimal import Decimal, InvalidOperation

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import DataQualityIssue, Discrepancy, Order, Payment, ReconRun, UploadBatch, User
from ..domain.engine import reconcile
from ..domain.metrics import build_summary, priority_sort_key
from ..domain.records import CleanOrder, CleanPayment, RowIssue
from ..domain.rules import ReconConfig, TYPE_LABELS, DiscrepancyType

# Bumped when rule behaviour changes. Stored on every run so a finding can be traced to
# the version of the rulebook that produced it.
ENGINE_VERSION = "1.0.0"


def _naive(value: datetime | None) -> datetime | None:
    """Strip tzinfo. Postgres returns aware datetimes for timestamptz columns and SQLite
    returns naive ones; comparing the two raises TypeError. The engine works in naive local
    time because both source files are naive."""
    if value is None:
        return None
    return value.replace(tzinfo=None) if value.tzinfo else value


def _to_clean_order(row: Order) -> CleanOrder:
    return CleanOrder(
        order_key=row.order_key, raw_order_id=row.raw_order_id,
        order_date=_naive(row.order_date), customer_email=row.customer_email,
        currency=row.currency, gross_amount=row.gross_amount, discount=row.discount,
        net_amount=row.net_amount, status=row.status, source_row=row.source_row,
    )


def _to_clean_payment(row: Payment) -> CleanPayment:
    return CleanPayment(
        transaction_ref=row.transaction_ref, order_key=row.order_key,
        raw_order_reference=row.raw_order_reference, processed_at=_naive(row.processed_at),
        currency=row.currency, amount=row.amount, fee=row.fee,
        net_settled=row.net_settled, payment_type=row.payment_type, status=row.status,
        source_row=row.source_row,
    )


def run_reconciliation(
    db: Session, user: User, batch: UploadBatch, config: ReconConfig | None = None
) -> ReconRun:
    """Reconcile a batch and store the run, its summary and every finding."""
    started = time.perf_counter()
    config = config or ReconConfig()

    orders = [_to_clean_order(row) for row in db.scalars(
        select(Order).where(Order.user_id == user.id, Order.batch_id == batch.id)
    )]
    payments = [_to_clean_payment(row) for row in db.scalars(
        select(Payment).where(Payment.user_id == user.id, Payment.batch_id == batch.id)
    )]

    output = reconcile(orders, payments, config=config, row_issues=_load_row_issues(db, user, batch))
    summary = build_summary(output)
    duration_ms = int((time.perf_counter() - started) * 1000)

    run = ReconRun(
        user_id=user.id, batch_id=batch.id, engine_version=ENGINE_VERSION,
        config_json=config.as_dict(), summary_json=summary, duration_ms=duration_ms,
    )
    db.add(run)
    db.flush()

    by_key = {result.order_key: result for result in output.results}
    db.add_all([
        Discrepancy(
            user_id=user.id, run_id=run.id, order_key=finding.order_key,
            discrepancy_type=finding.discrepancy_type.value, severity=finding.severity.value,
            risk_direction=finding.risk_direction.value, rule_id=finding.rule_id,
            summary=finding.summary[:400], detail=finding.detail,
            expected_amount=by_key[finding.order_key].expected_amount,
            collected_amount=by_key[finding.order_key].collected_amount,
            delta_amount=by_key[finding.order_key].delta,
            amount_at_risk=finding.amount_at_risk,   # 0.00 on secondary findings
            is_primary=finding.is_primary,
            order_status=by_key[finding.order_key].order_status,
            currency=(by_key[finding.order_key].order_currency
                      or (by_key[finding.order_key].payment_currencies[0]
                          if by_key[finding.order_key].payment_currencies else None)),
            customer_email_masked=by_key[finding.order_key].customer_email_masked,
            evidence_json=finding.evidence,
        )
        for finding in output.findings
    ])
    db.commit()
    db.refresh(run)
    return run


def _load_row_issues(db: Session, user: User, batch: UploadBatch) -> list[RowIssue]:
    return [
        RowIssue(source=row.source, source_row=row.source_row, identifier=row.identifier,
                 code=row.code, message=row.message, dropped=row.dropped)
        for row in db.scalars(select(DataQualityIssue).where(
            DataQualityIssue.user_id == user.id, DataQualityIssue.batch_id == batch.id
        ))
    ]


# ======================================================================================
# Reads
# ======================================================================================
def get_run(db: Session, user: User, run_id: int) -> ReconRun | None:
    return db.scalars(
        select(ReconRun).where(ReconRun.id == run_id, ReconRun.user_id == user.id)
    ).first()


def get_latest_run(db: Session, user: User, batch_id: int | None = None) -> ReconRun | None:
    query = select(ReconRun).where(ReconRun.user_id == user.id)
    if batch_id is not None:
        query = query.where(ReconRun.batch_id == batch_id)
    return db.scalars(query.order_by(ReconRun.created_at.desc(), ReconRun.id.desc()).limit(1)).first()


def list_runs(db: Session, user: User, limit: int = 20) -> list[ReconRun]:
    return list(db.scalars(
        select(ReconRun).where(ReconRun.user_id == user.id)
        .order_by(ReconRun.created_at.desc()).limit(limit)
    ))


def query_discrepancies(
    db: Session, user: User, run_id: int, discrepancy_type: str | None = None,
    severity: str | None = None, search: str | None = None, primary_only: bool = True,
    limit: int = 50, offset: int = 0,
) -> tuple[list[Discrepancy], int]:
    """
    Filtered, paginated drill-down. Returns (items, total_matching).

    Filtering and paging happen in SQL, not in Python: 187 rows would be fine either way,
    but a real payments export is 500k rows and the same code has to survive it.
    """
    conditions = [Discrepancy.user_id == user.id, Discrepancy.run_id == run_id]
    if primary_only:
        # Default view shows one row per order -- the primary finding. Secondary findings
        # are available (primary_only=false) as context, and are always worth 0.00.
        conditions.append(Discrepancy.is_primary.is_(True))
    if discrepancy_type:
        conditions.append(Discrepancy.discrepancy_type == discrepancy_type)
    if severity:
        conditions.append(Discrepancy.severity == severity)
    if search:
        # Parameterised LIKE: the term is bound, never interpolated, so a search box
        # cannot become SQL injection.
        term = f"%{search.strip().lower()}%"
        conditions.append(func.lower(Discrepancy.order_key).like(term)
                          | func.lower(Discrepancy.summary).like(term)
                          | func.lower(Discrepancy.rule_id).like(term))

    total = db.scalar(select(func.count()).select_from(Discrepancy).where(*conditions)) or 0
    items = list(db.scalars(
        select(Discrepancy).where(*conditions)
        # Worst first, deterministically: severity rank, then money, then key.
        .order_by(
            func.lower(Discrepancy.severity) == "critical",
            Discrepancy.amount_at_risk.desc(),
            Discrepancy.order_key,
        )
        .limit(max(1, min(limit, 500))).offset(max(0, offset))
    ))
    # SQL cannot express the 4-level severity ordering portably across SQLite/Postgres
    # without a CASE expression per dialect, so the page is re-sorted in Python. It is one
    # page (<=500 rows), so the cost is negligible and the behaviour is identical on both.
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    items.sort(key=lambda row: (rank.get(row.severity, 9), -row.amount_at_risk, row.order_key))
    return items, total


def get_discrepancy(db: Session, user: User, discrepancy_id: int) -> Discrepancy | None:
    return db.scalars(
        select(Discrepancy).where(Discrepancy.id == discrepancy_id, Discrepancy.user_id == user.id)
    ).first()


def discrepancy_to_llm_payload(discrepancy: Discrepancy, run: ReconRun) -> dict:
    """
    Build the prompt payload from STORED data.

    The client sends only an id, so the explanation always describes what the engine
    actually found -- a caller cannot inject its own amounts into the prompt.
    """
    return {
        "order_key": discrepancy.order_key,
        "discrepancy_type": discrepancy.discrepancy_type,
        "label": TYPE_LABELS.get(DiscrepancyType(discrepancy.discrepancy_type), discrepancy.discrepancy_type)
                 if discrepancy.discrepancy_type in {t.value for t in DiscrepancyType} else discrepancy.discrepancy_type,
        "severity": discrepancy.severity,
        "risk_direction": discrepancy.risk_direction,
        "rule_id": discrepancy.rule_id,
        "summary": discrepancy.summary,
        "detail": discrepancy.detail,
        "expected_amount": str(discrepancy.expected_amount),
        "collected_amount": str(discrepancy.collected_amount),
        "delta_amount": str(discrepancy.delta_amount),
        "amount_at_risk": str(discrepancy.amount_at_risk),
        "currency": discrepancy.currency,
        "order_status": discrepancy.order_status,
        "customer_email_masked": discrepancy.customer_email_masked,   # masked only
        "evidence": discrepancy.evidence_json or {},
        "config": run.config_json or {},
    }


def run_priorities(db: Session, user: User, run_id: int, limit: int = 10) -> list[dict]:
    """Top findings for the 'do this first' panel, read from the stored run."""
    items, _ = query_discrepancies(db, user, run_id, primary_only=True, limit=200, offset=0)
    rank = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    items.sort(key=lambda row: (rank.get(row.severity, 9), -row.amount_at_risk, row.order_key))
    return [
        {
            "order_key": row.order_key,
            "type": row.discrepancy_type,
            "label": TYPE_LABELS.get(DiscrepancyType(row.discrepancy_type), row.discrepancy_type)
                     if row.discrepancy_type in {t.value for t in DiscrepancyType} else row.discrepancy_type,
            "severity": row.severity,
            "risk_direction": row.risk_direction,
            "amount_at_risk": str(row.amount_at_risk),
            "currency": row.currency,
            "summary": row.summary,
        }
        for row in items[:limit]
    ]


def parse_config_overrides(
    amount_tolerance: str | None = None,
    duplicate_window_hours: int | None = None,
    max_settlement_lag_days: int | None = None,
) -> ReconConfig:
    """
    Build a ReconConfig from query parameters, with bounds.

    The bounds are the interesting part. A tolerance of 500.00 would mask every real
    discrepancy in this dataset, so the API refuses it: the tolerance is tunable for
    sensitivity analysis, not a switch for turning the product off.
    """
    kwargs: dict = {}
    if amount_tolerance is not None:
        try:
            tolerance = Decimal(str(amount_tolerance))
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("amount_tolerance must be a decimal number, e.g. 0.05") from exc
        if tolerance < 0 or tolerance > Decimal("5.00"):
            raise ValueError("amount_tolerance must be between 0.00 and 5.00")
        kwargs["amount_tolerance"] = tolerance
    if duplicate_window_hours is not None:
        if not 1 <= duplicate_window_hours <= 720:
            raise ValueError("duplicate_window_hours must be between 1 and 720")
        kwargs["duplicate_window_hours"] = duplicate_window_hours
    if max_settlement_lag_days is not None:
        if not 1 <= max_settlement_lag_days <= 90:
            raise ValueError("max_settlement_lag_days must be between 1 and 90")
        kwargs["max_settlement_lag_days"] = max_settlement_lag_days
    return ReconConfig(**kwargs)
