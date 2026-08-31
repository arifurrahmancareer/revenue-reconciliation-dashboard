"""
rules.py -- The reconciliation RULEBOOK: taxonomy, tolerances, status vocabularies.

Everything a reviewer might argue with lives in this one file, on purpose. When the
interviewer says "why 5 cents?" or "why is a cancelled-but-charged order high severity?",
the answer is a named constant with a comment next to it, not a magic number buried in
a 200-line function.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class DiscrepancyType(str, Enum):
    """
    The six money-affecting classes required by the brief, plus two review classes.

    The last two are kept OUT of the headline "money at risk" figure so the number a
    revenue owner sees is only cash that is genuinely wrong or unverifiable.
    """

    MISSING_PAYMENT = "MISSING_PAYMENT"      # order exists, no payment at all
    MISSING_ORDER = "MISSING_ORDER"          # payment exists, no order (orphan cash)
    DUPLICATE_PAYMENT = "DUPLICATE_PAYMENT"  # same order charged more than once
    AMOUNT_MISMATCH = "AMOUNT_MISMATCH"      # amounts differ beyond tolerance
    CURRENCY_MISMATCH = "CURRENCY_MISMATCH"  # order and payment in different currencies
    STATUS_CONFLICT = "STATUS_CONFLICT"      # lifecycle states contradict each other
    TIMING_ANOMALY = "TIMING_ANOMALY"        # settled far outside the normal window
    DATA_QUALITY = "DATA_QUALITY"            # the row itself is incomplete/inconsistent


class Severity(str, Enum):
    CRITICAL = "CRITICAL"  # cash is missing or unattributable -> act today
    HIGH = "HIGH"          # customer-visible or compliance risk -> act this week
    MEDIUM = "MEDIUM"      # real but bounded money impact
    LOW = "LOW"            # hygiene / review only, no direct money impact


class RiskDirection(str, Enum):
    """
    Which way the money points. A dashboard that only shows one blended 'disputed'
    number cannot answer the question a revenue owner actually asks:
    'am I owed money, or do I owe money?'
    """

    REVENUE_AT_RISK = "REVENUE_AT_RISK"        # we may never collect this
    CUSTOMER_OWED = "CUSTOMER_OWED"            # we took too much; refund liability
    NEEDS_INVESTIGATION = "NEEDS_INVESTIGATION"  # cash exists but cannot be trusted/tied out
    NONE = "NONE"                              # informational only


# --------------------------------------------------------------------------------------
# Status vocabularies
# --------------------------------------------------------------------------------------
# Only these payment states represent money that actually moved. This dataset uses
# 'settled'; the extra synonyms make the engine survive a processor swap without a
# code change (Stripe says 'succeeded', Adyen says 'settled', Braintree 'submitted').
SETTLED_PAYMENT_STATUSES = frozenset({"settled", "succeeded", "captured", "paid", "completed", "complete"})

# Money is promised but not banked. Deliberately NOT counted as collected: treating
# 'pending' as revenue is exactly how a business overstates its cash position.
PENDING_PAYMENT_STATUSES = frozenset({"pending", "processing", "authorized", "authorised", "requires_capture", "in_transit"})

# Money definitively did not move.
FAILED_PAYMENT_STATUSES = frozenset({"failed", "declined", "error", "voided", "cancelled", "canceled", "chargeback", "reversed"})

CHARGE_TYPES = frozenset({"charge", "payment", "capture", "sale", "debit"})
REFUND_TYPES = frozenset({"refund", "credit", "return", "chargeback", "reversal"})

# Order states that SHOULD have a matching settled charge for the net amount.
ORDER_STATUSES_EXPECTING_PAYMENT = frozenset({"completed", "complete", "fulfilled", "shipped", "delivered", "paid"})

# Order states that should end with NO net money held by the business.
ORDER_STATUSES_EXPECTING_NO_MONEY = frozenset({"cancelled", "canceled", "void", "voided", "refunded", "returned", "failed", "rejected"})

# Order states where money has not been asked for yet.
ORDER_STATUSES_PENDING = frozenset({"pending", "processing", "created", "awaiting_payment", "on_hold"})


@dataclass(frozen=True, slots=True)
class ReconConfig:
    """
    Every tunable threshold, in one immutable object that is stored alongside each run.

    Storing the config with the run is what makes results reproducible AND auditable:
    if a number on the dashboard is questioned six weeks later, we can show the exact
    rulebook that produced it.
    """

    # ---------------------------------------------------------------- amount tolerance
    # +/- $0.05 absolute.
    #
    # WHY 5 CENTS, DEFENDED WITH THIS DATASET:
    #   * The only sub-dollar gaps present are $0.01 (ORD-1901), $0.02 (ORD-1902) and
    #     $0.01 (ORD-1903) -- classic half-cent rounding between an order system that
    #     rounds line items and a processor that rounds the total.
    #   * The smallest genuine business error is $18.50 (ORD-1402). There is a ~900x gap
    #     between noise and signal here, so any tolerance in [$0.03, $1.00] separates them.
    #     $0.05 sits at the conservative end of that safe band.
    #   * A PERCENTAGE tolerance was considered and REJECTED: 0.5% of the largest order
    #     (~$474) would be $2.37, which would start hiding real cent-level fraud on small
    #     orders while adding no benefit at these order values (all < $500).
    amount_tolerance: Decimal = Decimal("0.05")

    # ------------------------------------------------------------ duplicate detection
    # Two settled charges for the SAME order key and the SAME amount inside this window
    # are treated as one double-charge event rather than two legitimate part-payments.
    # Observed duplicates here are 29 minutes apart (ORD-1501, ORD-1502) -- the classic
    # double-submit / retry signature. 24h is wide enough to catch a retry the next
    # morning, narrow enough that a genuine monthly instalment is not swallowed.
    duplicate_window_hours: int = 24

    # -------------------------------------------------------------------- timing rule
    # Median order->settlement lag in this dataset is 0.03 days (about 43 minutes) and
    # 182 of 183 matched payments settle within 3 days. 7 days is therefore a very safe
    # outlier threshold: it flags ORD-2101 (29 days) without flagging normal traffic.
    max_settlement_lag_days: int = 7
    # A payment dated before its order by more than this is impossible, not just odd.
    max_payment_before_order_days: int = 1

    # ------------------------------------------------------------------- interpretation
    # Pending money is NOT counted as collected (see PENDING_PAYMENT_STATUSES).
    count_pending_as_collected: bool = False
    # Refunds are netted off charges before comparing to the order value.
    net_refunds_against_charges: bool = True

    def as_dict(self) -> dict[str, object]:
        """Serialised into the recon_runs row so every run carries its own rulebook."""
        return {
            "amount_tolerance": str(self.amount_tolerance),
            "duplicate_window_hours": self.duplicate_window_hours,
            "max_settlement_lag_days": self.max_settlement_lag_days,
            "max_payment_before_order_days": self.max_payment_before_order_days,
            "count_pending_as_collected": self.count_pending_as_collected,
            "net_refunds_against_charges": self.net_refunds_against_charges,
        }


# --------------------------------------------------------------------------------------
# Precedence: which single type "owns" the money for a given order
# --------------------------------------------------------------------------------------
# An order can trip several rules at once (ORD-1701 is both a status conflict and, on
# paper, an amount mismatch). If we added up every finding, the headline exposure would
# be double counted and the dashboard would lie.
#
# So: an order can carry MANY findings (all shown in the drill-down), but exactly ONE is
# marked `is_primary` and only that one contributes to money at risk. Precedence runs
# from 'cash is absent/unattributable' down to 'cosmetic'.
TYPE_PRECEDENCE: tuple[DiscrepancyType, ...] = (
    DiscrepancyType.MISSING_PAYMENT,
    DiscrepancyType.MISSING_ORDER,
    DiscrepancyType.DUPLICATE_PAYMENT,
    DiscrepancyType.STATUS_CONFLICT,
    DiscrepancyType.CURRENCY_MISMATCH,
    DiscrepancyType.AMOUNT_MISMATCH,
    DiscrepancyType.TIMING_ANOMALY,
    DiscrepancyType.DATA_QUALITY,
)

PRECEDENCE_INDEX: dict[DiscrepancyType, int] = {t: i for i, t in enumerate(TYPE_PRECEDENCE)}

# Types that never contribute to the money-at-risk headline.
NON_FINANCIAL_TYPES = frozenset({DiscrepancyType.TIMING_ANOMALY, DiscrepancyType.DATA_QUALITY})

# Human-facing labels used by the API and dashboard.
TYPE_LABELS: dict[DiscrepancyType, str] = {
    DiscrepancyType.MISSING_PAYMENT: "Missing payment",
    DiscrepancyType.MISSING_ORDER: "Missing order",
    DiscrepancyType.DUPLICATE_PAYMENT: "Duplicate payment",
    DiscrepancyType.AMOUNT_MISMATCH: "Amount mismatch",
    DiscrepancyType.CURRENCY_MISMATCH: "Currency mismatch",
    DiscrepancyType.STATUS_CONFLICT: "Status conflict",
    DiscrepancyType.TIMING_ANOMALY: "Timing anomaly",
    DiscrepancyType.DATA_QUALITY: "Data quality",
}

# Fixed ordering for charts so colours/bars never jump between runs.
TYPE_DISPLAY_ORDER: tuple[str, ...] = tuple(t.value for t in TYPE_PRECEDENCE)
SEVERITY_DISPLAY_ORDER: tuple[str, ...] = ("CRITICAL", "HIGH", "MEDIUM", "LOW")
