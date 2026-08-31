"""
metrics.py -- Turns raw findings into the numbers a revenue owner actually asks for.

The dashboard has to answer three questions in one glance:
  1. How bad is it?            -> money at risk, split by DIRECTION (owed vs at risk)
  2. What kind of problems?    -> counts and value by type
  3. What do I look at first?  -> a deterministic priority ordering

Key anti-lying rule: `money_at_risk` only ever sums PRIMARY findings, so the headline
number always equals the sum of the drill-down table. A dashboard whose total does not
tie to its own rows destroys trust instantly.
"""

from __future__ import annotations

from collections import defaultdict
from decimal import Decimal
from typing import Iterable

from .engine import OrderReconciliation, ReconciliationOutput
from .normalize import ZERO, quantize
from .records import RowIssue
from .rules import (
    NON_FINANCIAL_TYPES,
    PRECEDENCE_INDEX,
    SEVERITY_DISPLAY_ORDER,
    TYPE_DISPLAY_ORDER,
    TYPE_LABELS,
    DiscrepancyType,
    RiskDirection,
    Severity,
)

SEVERITY_RANK = {s: i for i, s in enumerate(SEVERITY_DISPLAY_ORDER)}


def _sum(values: Iterable[Decimal]) -> Decimal:
    return quantize(sum(values, ZERO))


def build_summary(output: ReconciliationOutput) -> dict[str, object]:
    """Compute every headline figure and breakdown in one deterministic pass."""
    results = output.results

    order_rows = [r for r in results if r.has_order]
    payment_rows = [r for r in results if r.has_payments]

    total_order_value = _sum(r.expected_amount for r in order_rows)
    total_collected = _sum(r.collected_amount for r in payment_rows)

    clean = [r for r in results if r.is_reconciled]
    flagged = [r for r in results if not r.is_reconciled]

    # "Reconciled value" = money on orders that tie out perfectly. Deliberately measured
    # on the ORDER side: it is the share of what we believe we sold that we can prove.
    reconciled_value = _sum(r.expected_amount for r in clean if r.has_order)

    # "Disputed value" = the full value of every order/payment touched by a financial
    # finding. It is intentionally LARGER than money at risk: an order can be entirely
    # untrustworthy while only part of it is cash we might lose.
    financially_flagged = [
        r for r in flagged
        if r.primary_type is not None and r.primary_type not in NON_FINANCIAL_TYPES
    ]
    disputed_value = _sum(
        max(r.expected_amount, r.collected_amount) for r in financially_flagged
    )
    money_at_risk = _sum(r.amount_at_risk for r in results)

    by_direction: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for r in results:
        if r.amount_at_risk > ZERO:
            by_direction[r.risk_direction.value] = quantize(by_direction[r.risk_direction.value] + r.amount_at_risk)

    # -------------------------------- by type -----------------------------------------
    type_counts: dict[str, int] = defaultdict(int)
    type_values: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for finding in output.findings:
        type_counts[finding.discrepancy_type.value] += 1
        type_values[finding.discrepancy_type.value] = quantize(
            type_values[finding.discrepancy_type.value] + finding.amount_at_risk
        )

    by_type = [
        {
            "type": type_name,
            "label": TYPE_LABELS[DiscrepancyType(type_name)],
            "count": type_counts.get(type_name, 0),
            "amount_at_risk": str(type_values.get(type_name, ZERO)),
            "is_financial": DiscrepancyType(type_name) not in NON_FINANCIAL_TYPES,
        }
        for type_name in TYPE_DISPLAY_ORDER
        if type_counts.get(type_name, 0) > 0
    ]

    # ------------------------------ by severity ---------------------------------------
    sev_counts: dict[str, int] = defaultdict(int)
    sev_values: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for r in flagged:
        if r.primary_severity is None:
            continue
        sev_counts[r.primary_severity.value] += 1
        sev_values[r.primary_severity.value] = quantize(sev_values[r.primary_severity.value] + r.amount_at_risk)

    by_severity = [
        {
            "severity": name,
            "count": sev_counts.get(name, 0),
            "amount_at_risk": str(sev_values.get(name, ZERO)),
        }
        for name in SEVERITY_DISPLAY_ORDER
        if sev_counts.get(name, 0) > 0
    ]

    match_rate = (len(clean) / len(results) * 100.0) if results else 0.0

    return {
        # ---- volumes
        "total_orders": len(order_rows),
        "total_payment_transactions": sum(r.payment_count for r in results),
        # Every DISTINCT order reference seen in either file. Larger than total_orders
        # because orphaned payments create keys with no order behind them -- and those
        # orphans are exactly what a reconciliation is supposed to surface.
        "total_order_keys": len(results),
        "total_reconciled_keys": len(clean),
        "total_flagged_keys": len(flagged),
        "match_rate_pct": round(match_rate, 2),
        # ---- money
        "total_order_value": str(total_order_value),
        "total_payments_settled": str(total_collected),
        "reconciled_value": str(reconciled_value),
        "disputed_value": str(disputed_value),
        "money_at_risk": str(money_at_risk),
        "revenue_at_risk": str(by_direction.get(RiskDirection.REVENUE_AT_RISK.value, ZERO)),
        "customer_owed": str(by_direction.get(RiskDirection.CUSTOMER_OWED.value, ZERO)),
        "needs_investigation": str(by_direction.get(RiskDirection.NEEDS_INVESTIGATION.value, ZERO)),
        # ---- breakdowns
        "by_type": by_type,
        "by_severity": by_severity,
        # ---- hygiene (kept separate from money on purpose)
        "data_quality_issue_count": len(output.row_issues),
        "data_quality_rows_dropped": sum(1 for i in output.row_issues if i.dropped),
        "config": output.config.as_dict(),
    }


def priority_sort_key(result: OrderReconciliation) -> tuple:
    """
    Deterministic "what do I fix first" ordering:
      severity, then money, then type precedence, then key (a total, stable tiebreak).

    Money alone is the wrong sort: a $27 order that was charged twice is a customer
    complaint waiting to happen, while a $400 rounding query is not urgent.
    """
    severity = result.primary_severity.value if result.primary_severity else "LOW"
    precedence = PRECEDENCE_INDEX[result.primary_type] if result.primary_type else 99
    return (SEVERITY_RANK.get(severity, 99), -result.amount_at_risk, precedence, result.order_key)


def top_priorities(output: ReconciliationOutput, limit: int = 10) -> list[OrderReconciliation]:
    flagged = [r for r in output.results if not r.is_reconciled]
    return sorted(flagged, key=priority_sort_key)[:limit]


def summarise_row_issues(issues: list[RowIssue]) -> list[dict[str, object]]:
    """Group parse-time issues by code for the 'Data quality' panel."""
    grouped: dict[str, dict[str, object]] = {}
    for issue in issues:
        entry = grouped.setdefault(issue.code, {
            "code": issue.code,
            "count": 0,
            "dropped": 0,
            "example": issue.message,
            "examples": [],
        })
        entry["count"] = int(entry["count"]) + 1
        if issue.dropped:
            entry["dropped"] = int(entry["dropped"]) + 1
        examples = entry["examples"]
        assert isinstance(examples, list)
        if len(examples) < 5:
            examples.append({"row": issue.source_row, "id": issue.identifier, "source": issue.source})
    return sorted(grouped.values(), key=lambda e: (-int(e["count"]), str(e["code"])))


def severity_of(value: str | Severity) -> Severity:
    return value if isinstance(value, Severity) else Severity(value)
