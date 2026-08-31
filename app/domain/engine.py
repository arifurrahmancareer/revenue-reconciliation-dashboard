"""
engine.py -- THE RECONCILIATION ENGINE. This is the heart of the project.

CONTRACT
  reconcile(orders, payments, config) -> ReconciliationOutput
  * Pure function. No DB, no network, no clock, no randomness, no LLM.
  * Deterministic: same input bytes -> byte-identical output, including ordering
    (every collection is built from `sorted()` keys).
  * Total: never raises on messy data; unusable values become findings, not exceptions.

HOW MATCHING WORKS
  Orders and payments are grouped by a NORMALISED order key (see normalize.py), then each
  key is evaluated by an ordered set of rules. A key can produce several findings, but
  exactly one is marked `is_primary` and only that one carries `amount_at_risk`, so the
  headline exposure can never be double counted.

WHY AN LLM IS NOT INVOLVED
  Matching money must be explainable, repeatable and auditable. A model that answers
  "probably the same order" cannot be reproduced, cannot be unit tested, and cannot be
  defended to a finance team. The LLM in this project only writes prose ABOUT the
  findings this file produces.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal

from .normalize import ZERO, mask_email, quantize
from .records import CleanOrder, CleanPayment, RowIssue
from .rules import (
    CHARGE_TYPES,
    FAILED_PAYMENT_STATUSES,
    NON_FINANCIAL_TYPES,
    ORDER_STATUSES_EXPECTING_NO_MONEY,
    ORDER_STATUSES_EXPECTING_PAYMENT,
    ORDER_STATUSES_PENDING,
    PENDING_PAYMENT_STATUSES,
    PRECEDENCE_INDEX,
    REFUND_TYPES,
    SETTLED_PAYMENT_STATUSES,
    DiscrepancyType,
    ReconConfig,
    RiskDirection,
    Severity,
)


# --------------------------------------------------------------------------------------
# Output shapes
# --------------------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Finding:
    """One thing that is wrong about one order key."""

    order_key: str
    discrepancy_type: DiscrepancyType
    severity: Severity
    risk_direction: RiskDirection
    rule_id: str  # stable id so a row on the dashboard maps to a rule in the README
    summary: str  # one line, safe to show in a table cell
    detail: str  # full sentence(s) with the numbers spelled out
    amount_at_risk: Decimal  # non-zero ONLY when is_primary is True
    is_primary: bool
    evidence: dict[str, object]  # JSON-safe facts, reused by the UI and the LLM prompt


@dataclass(slots=True)
class OrderReconciliation:
    """The full reconciled picture for a single order key (matched or not)."""

    order_key: str
    has_order: bool
    has_payments: bool
    order_status: str | None
    order_currency: str | None
    payment_currencies: list[str]
    customer_email_masked: str
    order_date: datetime | None
    last_payment_date: datetime | None
    expected_amount: Decimal  # what the order system says should be collected
    collected_amount: Decimal  # settled charges - settled refunds
    pending_amount: Decimal  # money promised but not banked
    failed_amount: Decimal  # attempts that definitively did not land
    delta: Decimal  # collected - expected  (+ = over-collected)
    payment_count: int
    settled_charge_count: int
    refund_count: int
    is_reconciled: bool
    primary_type: DiscrepancyType | None
    primary_severity: Severity | None
    amount_at_risk: Decimal
    risk_direction: RiskDirection
    findings: list[Finding] = field(default_factory=list)
    matched_via_normalisation: bool = False  # True when raw refs differed by case/space


@dataclass(slots=True)
class ReconciliationOutput:
    results: list[OrderReconciliation]
    findings: list[Finding]
    config: ReconConfig
    row_issues: list[RowIssue] = field(default_factory=list)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------
def _is_settled(payment: CleanPayment) -> bool:
    return payment.status in SETTLED_PAYMENT_STATUSES


def _is_pending(payment: CleanPayment) -> bool:
    return payment.status in PENDING_PAYMENT_STATUSES


def _is_failed(payment: CleanPayment) -> bool:
    return payment.status in FAILED_PAYMENT_STATUSES


def _is_refund(payment: CleanPayment) -> bool:
    return payment.payment_type in REFUND_TYPES


def _is_charge(payment: CleanPayment) -> bool:
    # Anything not explicitly a refund is treated as a charge. Being permissive here is
    # safer than dropping money we do not recognise.
    return payment.payment_type in CHARGE_TYPES or not _is_refund(payment)


def _amount(payment: CleanPayment) -> Decimal:
    return payment.amount if payment.amount is not None else ZERO


def _money(value: Decimal | None) -> str:
    return f"{(value if value is not None else ZERO):.2f}"


def _expected_amount(order: CleanOrder) -> tuple[Decimal, str]:
    """
    How much money SHOULD be sitting with the business for this order, per the order system.

    This is the single most important judgement call in the engine, so it is explicit:
      completed/fulfilled -> the net amount (we sold something and kept the money)
      cancelled/refunded  -> 0.00 (whatever happened, we should not be holding cash)
      pending/processing  -> 0.00 (we have not asked for the money yet)
      anything unknown    -> net amount, and the unknown status is flagged

    Comparing against an *expectation derived from order status* (rather than blindly
    comparing net_amount to the payment total) is what lets the same subtraction detect
    a cancelled-but-charged order and a completed-but-unpaid order.
    """
    net = order.net_amount if order.net_amount is not None else ZERO
    if order.status in ORDER_STATUSES_EXPECTING_PAYMENT:
        return net, "expects_payment"
    if order.status in ORDER_STATUSES_EXPECTING_NO_MONEY:
        return ZERO, "expects_no_money"
    if order.status in ORDER_STATUSES_PENDING:
        return ZERO, "pending"
    return net, "unknown_status"


def _severity_for_amount(amount: Decimal) -> Severity:
    """
    Money-scaled severity for amount mismatches.

    Thresholds are deliberately blunt and documented rather than clever: in this dataset
    order values run from ~$27 to ~$474, so $100+ is 'a big slice of a large order' and
    $25+ is 'more than a rounding argument'.
    """
    magnitude = abs(amount)
    if magnitude >= Decimal("100"):
        return Severity.HIGH
    if magnitude >= Decimal("25"):
        return Severity.MEDIUM
    return Severity.LOW


# --------------------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------------------
def reconcile(
    orders: list[CleanOrder],
    payments: list[CleanPayment],
    config: ReconConfig | None = None,
    row_issues: list[RowIssue] | None = None,
) -> ReconciliationOutput:
    config = config or ReconConfig()
    tolerance = config.amount_tolerance

    orders_by_key: dict[str, list[CleanOrder]] = defaultdict(list)
    for order in orders:
        orders_by_key[order.order_key].append(order)

    payments_by_key: dict[str, list[CleanPayment]] = defaultdict(list)
    for payment in payments:
        # Payments with no reference at all are bucketed under '' and reported as orphans.
        payments_by_key[payment.order_key].append(payment)

    results: list[OrderReconciliation] = []
    all_findings: list[Finding] = []

    # sorted() -> deterministic output order, which makes API responses and snapshot
    # tests stable. Never iterate a set here.
    for key in sorted(set(orders_by_key) | set(payments_by_key)):
        order_rows = orders_by_key.get(key, [])
        key_payments = sorted(
            payments_by_key.get(key, []),
            # Sort by (date, txn ref) with a sentinel for missing dates so ordering is
            # total and stable even when processed_at is blank.
            key=lambda p: (p.processed_at or datetime.min.replace(tzinfo=None) if p.processed_at is None else p.processed_at, p.transaction_ref),
        )
        order = order_rows[0] if order_rows else None

        findings: list[Finding] = []

        # ---------------- money buckets ------------------------------------------------
        settled_charges = [p for p in key_payments if _is_charge(p) and _is_settled(p)]
        settled_refunds = [p for p in key_payments if _is_refund(p) and _is_settled(p)]
        pending_payments = [p for p in key_payments if _is_pending(p)]
        failed_payments = [p for p in key_payments if _is_failed(p)]

        charge_total = quantize(sum((_amount(p) for p in settled_charges), ZERO))
        refund_total = quantize(sum((_amount(p) for p in settled_refunds), ZERO))
        pending_total = quantize(sum((_amount(p) for p in pending_payments), ZERO))
        failed_total = quantize(sum((_amount(p) for p in failed_payments), ZERO))
        collected = quantize(charge_total - refund_total) if config.net_refunds_against_charges else charge_total

        expected, expectation_kind = _expected_amount(order) if order else (ZERO, "no_order")
        delta = quantize(collected - expected)

        # `absorbed` tracks how much of `delta` has already been explained by a
        # higher-precedence rule. Whatever is left over is a genuine amount mismatch.
        # This is the mechanism that stops the engine reporting the SAME dollar twice.
        absorbed = ZERO
        amount_comparable = True

        # =============================== RULE 1: MISSING ORDER ========================
        if order is None:
            risk = collected if collected > ZERO else pending_total
            findings.append(Finding(
                order_key=key,
                discrepancy_type=DiscrepancyType.MISSING_ORDER,
                severity=Severity.CRITICAL,
                risk_direction=RiskDirection.NEEDS_INVESTIGATION,
                rule_id="R02-ORPHAN-PAYMENT",
                summary=f"Payment received for {key or '(no reference)'} but no such order exists",
                detail=(
                    f"{len(key_payments)} payment(s) totalling {_money(collected)} settled against order reference "
                    f"{key or '(blank)'}, which does not appear in the orders file. Either the order was never "
                    f"exported, the reference is wrong, or this is money taken for something we have no record of "
                    f"selling. Until it is identified, this cash cannot be recognised as revenue."
                ),
                amount_at_risk=risk,
                is_primary=False,
                evidence={
                    "order_key": key,
                    "transactions": [p.transaction_ref for p in key_payments],
                    "settled_total": _money(collected),
                    "payment_currencies": sorted({p.currency for p in key_payments if p.currency}),
                    "processed_dates": [p.processed_at.isoformat() if p.processed_at else None for p in key_payments],
                },
            ))

        # ============================== RULE 2: MISSING PAYMENT =======================
        elif not key_payments:
            if expectation_kind == "expects_payment" and expected > tolerance:
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.MISSING_PAYMENT,
                    severity=Severity.CRITICAL,
                    risk_direction=RiskDirection.REVENUE_AT_RISK,
                    rule_id="R01-NO-PAYMENT",
                    summary=f"Order marked {order.status} for {_money(expected)} has no payment at all",
                    detail=(
                        f"Order {key} ({order.status}, {_money(expected)} {order.currency or ''}) has no payment row of "
                        f"any kind in the payment processor export - not even a failed attempt. The goods were "
                        f"treated as sold but no money was ever requested or received."
                    ),
                    amount_at_risk=expected,
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_status": order.status,
                        "order_net_amount": _money(order.net_amount),
                        "order_date": order.order_date.isoformat() if order.order_date else None,
                        "currency": order.currency,
                        "payment_rows_found": 0,
                    },
                ))
            # NOTE: a cancelled or refunded order with NO payment is CORRECT. Flagging it
            # would be inventing a discrepancy, which the brief explicitly penalises.

        # ======================= RULES 3-7: order and payments both exist =============
        else:
            # ----------------------- RULE 3: CURRENCY MISMATCH ------------------------
            order_ccy = order.currency
            mismatched = sorted({p.currency for p in key_payments if p.currency and order_ccy and p.currency != order_ccy})
            if mismatched:
                # Once the currencies differ, comparing the NUMBERS is meaningless: 210 EUR
                # and 210 USD look equal and are not. We therefore suppress the amount rule
                # for this key and treat the whole settlement as unverified. Guessing an FX
                # rate would be inventing data.
                amount_comparable = False
                exposure = expected if expected > ZERO else collected
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.CURRENCY_MISMATCH,
                    severity=Severity.HIGH,
                    risk_direction=RiskDirection.NEEDS_INVESTIGATION,
                    rule_id="R03-CURRENCY",
                    summary=f"Order is in {order_ccy} but payment is in {', '.join(mismatched)}",
                    detail=(
                        f"Order {key} is priced in {order_ccy} ({_money(order.net_amount)}) while its payment(s) "
                        f"settled in {', '.join(mismatched)} ({_money(charge_total)}). The two amounts are numerically "
                        f"similar, which is the trap: without an FX rate they cannot be compared, so this settlement "
                        f"is unverified rather than merely 'slightly off'. Amount checking is intentionally skipped "
                        f"for this order and the full order value is treated as unverified."
                    ),
                    amount_at_risk=exposure,
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_currency": order_ccy,
                        "payment_currencies": mismatched,
                        "order_net_amount": _money(order.net_amount),
                        "settled_charge_total": _money(charge_total),
                        "transactions": [p.transaction_ref for p in key_payments],
                    },
                ))

            # ----------------------- RULE 4: DUPLICATE PAYMENT ------------------------
            # Signature of a double charge: same order key, same amount, both settled,
            # inside the configured window. Grouping by amount avoids mislabelling a
            # legitimate two-part payment (e.g. 40 + 60) as a duplicate.
            duplicate_extra = ZERO
            duplicate_groups: list[dict[str, object]] = []
            by_amount: dict[Decimal, list[CleanPayment]] = defaultdict(list)
            for payment in settled_charges:
                by_amount[_amount(payment)].append(payment)

            for amount_value in sorted(by_amount):
                group = by_amount[amount_value]
                if len(group) < 2:
                    continue
                dated = [p.processed_at for p in group if p.processed_at is not None]
                within_window = True
                span_hours: float | None = None
                if len(dated) >= 2:
                    span_hours = (max(dated) - min(dated)).total_seconds() / 3600.0
                    within_window = span_hours <= config.duplicate_window_hours
                if not within_window:
                    continue  # far apart: treat as instalments, flag via amount rule instead
                extra = quantize(amount_value * (len(group) - 1))
                duplicate_extra = quantize(duplicate_extra + extra)
                duplicate_groups.append({
                    "amount": _money(amount_value),
                    "count": len(group),
                    "transactions": [p.transaction_ref for p in group],
                    "minutes_apart": round(span_hours * 60, 1) if span_hours is not None else None,
                    "extra_charged": _money(extra),
                })

            if duplicate_extra > ZERO:
                absorbed = quantize(absorbed + duplicate_extra)
                first_group = duplicate_groups[0]
                gap = first_group.get("minutes_apart")
                gap_text = f" {gap} minutes apart" if gap is not None else ""
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.DUPLICATE_PAYMENT,
                    severity=Severity.HIGH,
                    risk_direction=RiskDirection.CUSTOMER_OWED,
                    rule_id="R04-DUPLICATE-CHARGE",
                    summary=f"Charged {int(first_group['count'])}x {first_group['amount']} for one order - {_money(duplicate_extra)} overcharged",
                    detail=(
                        f"Order {key} has {int(first_group['count'])} settled charges of {first_group['amount']} each"
                        f"{gap_text} ({', '.join(first_group['transactions'])}). The order is only worth "
                        f"{_money(expected)}, so {_money(duplicate_extra)} was taken from the customer twice. This is a "
                        f"refund liability and a chargeback risk, not an accounting rounding issue."
                    ),
                    amount_at_risk=duplicate_extra,
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_net_amount": _money(order.net_amount),
                        "duplicate_groups": duplicate_groups,
                        "duplicate_window_hours": config.duplicate_window_hours,
                        "total_settled": _money(charge_total),
                        "extra_charged": _money(duplicate_extra),
                    },
                ))

            # ------------------------ RULE 5: STATUS CONFLICT -------------------------
            # 5a. The order says no money should be held, but money is held.
            if expectation_kind == "expects_no_money" and collected > tolerance:
                partial_refund = refund_total > ZERO and charge_total > refund_total
                absorbed = quantize(absorbed + delta)  # this rule explains the whole delta
                if partial_refund:
                    summary = f"Order marked {order.status} but only {_money(refund_total)} of {_money(charge_total)} was refunded"
                    detail = (
                        f"Order {key} is marked {order.status}, which should leave the business holding nothing, but "
                        f"only {_money(refund_total)} of the {_money(charge_total)} charge was returned. "
                        f"{_money(collected)} is still held against a {order.status} order: either the refund was "
                        f"issued for the wrong amount or the order status is wrong."
                    )
                    rule_id = "R05b-PARTIAL-REFUND"
                else:
                    summary = f"Order is {order.status} but {_money(collected)} was charged and settled"
                    detail = (
                        f"Order {key} is marked {order.status}, yet "
                        f"{', '.join(p.transaction_ref for p in settled_charges) or 'a payment'} settled for "
                        f"{_money(collected)}. The customer has paid for something the store believes it did not "
                        f"sell. This is the highest-priority class of conflict: it is customer-visible, it is a "
                        f"chargeback risk, and no goods may have shipped."
                    )
                    rule_id = "R05a-CANCELLED-BUT-PAID"
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.STATUS_CONFLICT,
                    severity=Severity.CRITICAL if not partial_refund else Severity.HIGH,
                    risk_direction=RiskDirection.CUSTOMER_OWED,
                    rule_id=rule_id,
                    summary=summary,
                    detail=detail,
                    amount_at_risk=quantize(collected),
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_status": order.status,
                        "order_net_amount": _money(order.net_amount),
                        "settled_charge_total": _money(charge_total),
                        "settled_refund_total": _money(refund_total),
                        "net_held": _money(collected),
                        "transactions": [
                            {"ref": p.transaction_ref, "type": p.payment_type, "status": p.status, "amount": _money(p.amount)}
                            for p in key_payments
                        ],
                    },
                ))

            # 5b. The order says money is due, but nothing settled.
            elif expectation_kind == "expects_payment" and not settled_charges and key_payments:
                blocking = pending_payments or failed_payments or key_payments
                state = blocking[0].status
                is_pending_case = bool(pending_payments) and not failed_payments
                absorbed = quantize(absorbed + delta)
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.STATUS_CONFLICT,
                    severity=Severity.CRITICAL if not is_pending_case else Severity.HIGH,
                    risk_direction=RiskDirection.REVENUE_AT_RISK,
                    rule_id="R05c-COMPLETED-BUT-UNPAID",
                    summary=f"Order marked {order.status} but its only payment is {state}",
                    detail=(
                        f"Order {key} is marked {order.status} for {_money(expected)}, but the only payment "
                        f"({blocking[0].transaction_ref}) is {state}, so no money has been banked. "
                        + (
                            "A pending payment may still settle, so this needs watching rather than chasing today."
                            if is_pending_case else
                            "A failed payment will not settle on its own: the order was fulfilled without payment."
                        )
                    ),
                    amount_at_risk=expected,
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_status": order.status,
                        "order_net_amount": _money(order.net_amount),
                        "payment_statuses": sorted({p.status for p in key_payments}),
                        "pending_total": _money(pending_total),
                        "failed_total": _money(failed_total),
                        "transactions": [
                            {"ref": p.transaction_ref, "type": p.payment_type, "status": p.status, "amount": _money(p.amount)}
                            for p in key_payments
                        ],
                    },
                ))

            # 5c. The order still says completed, but the money was handed back.
            elif expectation_kind == "expects_payment" and refund_total > ZERO and collected <= tolerance:
                absorbed = quantize(absorbed + delta)
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.STATUS_CONFLICT,
                    severity=Severity.HIGH,
                    risk_direction=RiskDirection.NEEDS_INVESTIGATION,
                    rule_id="R05d-REFUNDED-BUT-COMPLETED",
                    summary=f"Order still marked {order.status} but was fully refunded ({_money(refund_total)})",
                    detail=(
                        f"Order {key} was charged {_money(charge_total)} and then fully refunded "
                        f"({_money(refund_total)}), leaving {_money(collected)} collected - yet the order system still "
                        f"says {order.status}. The money is correct; the ORDER RECORD is wrong, which means revenue "
                        f"reports and any fulfilment queue driven off order status are both overstated."
                    ),
                    amount_at_risk=quantize(refund_total),
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_status": order.status,
                        "settled_charge_total": _money(charge_total),
                        "settled_refund_total": _money(refund_total),
                        "net_collected": _money(collected),
                        "transactions": [
                            {"ref": p.transaction_ref, "type": p.payment_type, "status": p.status, "amount": _money(p.amount)}
                            for p in key_payments
                        ],
                    },
                ))

            # 5d. Unknown order status: we cannot form an expectation we trust.
            elif expectation_kind == "unknown_status":
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.DATA_QUALITY,
                    severity=Severity.LOW,
                    risk_direction=RiskDirection.NONE,
                    rule_id="R08-UNKNOWN-ORDER-STATUS",
                    summary=f"Unrecognised order status {order.status!r}",
                    detail=(
                        f"Order {key} has status {order.status!r}, which is not in the known vocabulary. It was treated "
                        f"as 'expects payment' for reconciliation, but the rulebook should be extended before trusting it."
                    ),
                    amount_at_risk=ZERO,
                    is_primary=False,
                    evidence={"order_key": key, "order_status": order.status},
                ))

            # ------------------------ RULE 6: AMOUNT MISMATCH -------------------------
            # Only the part of the gap that no earlier rule already explained.
            residual = quantize(delta - absorbed)
            if amount_comparable and abs(residual) > tolerance:
                over = residual > ZERO
                findings.append(Finding(
                    order_key=key,
                    discrepancy_type=DiscrepancyType.AMOUNT_MISMATCH,
                    severity=_severity_for_amount(residual),
                    risk_direction=RiskDirection.CUSTOMER_OWED if over else RiskDirection.REVENUE_AT_RISK,
                    rule_id="R06-AMOUNT",
                    summary=(
                        f"{'Over' if over else 'Under'}-collected {_money(abs(residual))} "
                        f"(order {_money(expected)} vs settled {_money(collected)})"
                    ),
                    detail=(
                        f"Order {key} is worth {_money(expected)} {order.currency or ''} but {_money(collected)} was "
                        f"settled, a difference of {_money(abs(residual))} "
                        f"({'in the store\u2019s favour' if over else 'against the store'}). This is "
                        f"{abs(residual) / (expected if expected > ZERO else Decimal('1')) * 100:.1f}% of the order value "
                        f"and is far outside the {_money(tolerance)} rounding tolerance, so it is a pricing, discount or "
                        f"capture error rather than a floating-point artefact."
                    ),
                    amount_at_risk=quantize(abs(residual)),
                    is_primary=False,
                    evidence={
                        "order_key": key,
                        "order_net_amount": _money(order.net_amount),
                        "order_gross_amount": _money(order.gross_amount),
                        "order_discount": _money(order.discount),
                        "expected_amount": _money(expected),
                        "settled_amount": _money(collected),
                        "difference": _money(residual),
                        "tolerance": _money(tolerance),
                        "direction": "over_collected" if over else "under_collected",
                        "transactions": [
                            {"ref": p.transaction_ref, "type": p.payment_type, "status": p.status, "amount": _money(p.amount)}
                            for p in key_payments
                        ],
                    },
                ))

            # ------------------------- RULE 7: TIMING ANOMALY -------------------------
            # Informational (LOW, zero money) but genuinely useful: a settlement weeks
            # after the order is either a retried payment, a manual capture, or a
            # mis-keyed reference.
            if order.order_date is not None:
                for payment in settled_charges:
                    if payment.processed_at is None:
                        continue
                    lag_days = (payment.processed_at - order.order_date).total_seconds() / 86400.0
                    if lag_days > config.max_settlement_lag_days:
                        findings.append(Finding(
                            order_key=key,
                            discrepancy_type=DiscrepancyType.TIMING_ANOMALY,
                            severity=Severity.LOW,
                            risk_direction=RiskDirection.NONE,
                            rule_id="R07a-LATE-SETTLEMENT",
                            summary=f"Settled {lag_days:.0f} days after the order was placed",
                            detail=(
                                f"Payment {payment.transaction_ref} for order {key} settled {lag_days:.0f} days after the "
                                f"order date, against a normal window of under {config.max_settlement_lag_days} days. "
                                f"The amount matches, so no money is at risk, but a gap this large usually means a "
                                f"retried card, a manual capture, or the reference was reused."
                            ),
                            amount_at_risk=ZERO,
                            is_primary=False,
                            evidence={
                                "order_key": key,
                                "transaction_ref": payment.transaction_ref,
                                "order_date": order.order_date.isoformat(),
                                "processed_at": payment.processed_at.isoformat(),
                                "lag_days": round(lag_days, 2),
                                "threshold_days": config.max_settlement_lag_days,
                            },
                        ))
                    elif lag_days < -config.max_payment_before_order_days:
                        findings.append(Finding(
                            order_key=key,
                            discrepancy_type=DiscrepancyType.TIMING_ANOMALY,
                            severity=Severity.MEDIUM,
                            risk_direction=RiskDirection.NEEDS_INVESTIGATION,
                            rule_id="R07b-PAYMENT-BEFORE-ORDER",
                            summary=f"Payment settled {abs(lag_days):.0f} days BEFORE the order existed",
                            detail=(
                                f"Payment {payment.transaction_ref} settled {abs(lag_days):.0f} days before order {key} "
                                f"was created. That is chronologically impossible, so either a date is wrong or the "
                                f"payment belongs to a different order."
                            ),
                            amount_at_risk=ZERO,
                            is_primary=False,
                            evidence={
                                "order_key": key,
                                "transaction_ref": payment.transaction_ref,
                                "order_date": order.order_date.isoformat(),
                                "processed_at": payment.processed_at.isoformat(),
                                "lag_days": round(lag_days, 2),
                            },
                        ))

        # ---------------- choose the ONE finding that owns the money -------------------
        financial = [f for f in findings if f.discrepancy_type not in NON_FINANCIAL_TYPES]
        ranked = sorted(
            financial or findings,
            key=lambda f: (PRECEDENCE_INDEX[f.discrepancy_type], -f.amount_at_risk, f.rule_id),
        )
        primary = ranked[0] if ranked else None

        rebuilt: list[Finding] = []
        for finding in findings:
            is_primary = primary is not None and finding is primary
            rebuilt.append(Finding(
                order_key=finding.order_key,
                discrepancy_type=finding.discrepancy_type,
                severity=finding.severity,
                risk_direction=finding.risk_direction,
                rule_id=finding.rule_id,
                summary=finding.summary,
                detail=finding.detail,
                # Secondary findings are still shown in the drill-down, but carry 0.00 so
                # summing the table can never exceed the true exposure.
                amount_at_risk=finding.amount_at_risk if is_primary else ZERO,
                is_primary=is_primary,
                evidence=(
                    finding.evidence if is_primary
                    else {**finding.evidence, "risk_attributed_to": primary.discrepancy_type.value if primary else None}
                ),
            ))
        findings = rebuilt
        primary_out = next((f for f in findings if f.is_primary), None)

        matched_via_normalisation = any(
            p.raw_order_reference.strip().upper() != p.raw_order_reference or p.raw_order_reference.strip() != key
            for p in key_payments
        ) and order is not None

        results.append(OrderReconciliation(
            order_key=key,
            has_order=order is not None,
            has_payments=bool(key_payments),
            order_status=order.status if order else None,
            order_currency=order.currency if order else None,
            payment_currencies=sorted({p.currency for p in key_payments if p.currency}),
            customer_email_masked=mask_email(order.customer_email if order else None),
            order_date=order.order_date if order else None,
            last_payment_date=max((p.processed_at for p in key_payments if p.processed_at), default=None),
            expected_amount=expected,
            collected_amount=collected,
            pending_amount=pending_total,
            failed_amount=failed_total,
            delta=delta,
            payment_count=len(key_payments),
            settled_charge_count=len(settled_charges),
            refund_count=len(settled_refunds),
            is_reconciled=primary_out is None,
            primary_type=primary_out.discrepancy_type if primary_out else None,
            primary_severity=primary_out.severity if primary_out else None,
            amount_at_risk=primary_out.amount_at_risk if primary_out else ZERO,
            risk_direction=primary_out.risk_direction if primary_out else RiskDirection.NONE,
            findings=findings,
            matched_via_normalisation=matched_via_normalisation,
        ))
        all_findings.extend(findings)

    return ReconciliationOutput(
        results=results,
        findings=all_findings,
        config=config,
        row_issues=list(row_issues or []),
    )
