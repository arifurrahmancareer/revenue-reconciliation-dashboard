"""
test_engine.py -- Behavioural tests for the reconciliation engine, run against the REAL
sample CSVs rather than hand-built fixtures.

WHY REAL FILES: a fixture I invent only proves the engine agrees with my assumptions. The
supplied files contain the actual quirks -- trailing spaces, mixed date formats, orphaned
payments, a charge-plus-refund pair -- and asserting exact figures against them means a
regression in any rule fails a test instead of quietly changing a number on the dashboard.

Every expected value below was verified by hand against the source rows before being
written down. These are the numbers quoted in the README.

Run:  pytest backend/tests -q
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

# tests/ now sits at the repository root, alongside the app/ package.
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
sys.path.insert(0, str(ROOT))

from app.domain.engine import reconcile  # noqa: E402
from app.domain.metrics import build_summary, top_priorities  # noqa: E402
from app.domain.parsing import parse_orders_csv, parse_payments_csv  # noqa: E402
from app.domain.rules import DiscrepancyType, ReconConfig, RiskDirection, Severity  # noqa: E402


def _dec(value: str) -> Decimal:
    return Decimal(value)


def _run(config: ReconConfig | None = None):
    """Parse both sample files and reconcile them."""
    orders = parse_orders_csv((SAMPLES / "orders.csv").read_bytes())
    payments = parse_payments_csv((SAMPLES / "payments.csv").read_bytes())
    return reconcile(
        orders.orders, payments.payments, config=config,
        row_issues=orders.issues + payments.issues,
    )


def _primary_by_key(output) -> dict:
    """order_key -> its single primary finding."""
    return {f.order_key: f for f in output.findings if f.is_primary}


# ======================================================================================
# Headline figures
# ======================================================================================
def test_headline_totals_are_exact():
    summary = build_summary(_run())

    # 185 order rows read, one exact duplicate collapsed -> 184 loaded.
    assert summary["total_orders"] == 184
    assert summary["total_payment_transactions"] == 187
    # 187 distinct references across BOTH files: 184 orders + 3 orphaned payment refs.
    assert summary["total_order_keys"] == 187
    assert summary["total_reconciled_keys"] == 167
    assert summary["total_flagged_keys"] == 20
    assert summary["match_rate_pct"] == 89.3

    # Order value is what we EXPECTED to collect, so a cancelled or fully refunded order
    # contributes 0.00 (ORD-1701, ORD-1702): we are not owed that money, we are holding it.
    # It is not lost from the report -- it reappears under `customer_owed`, which is the
    # column that actually drives an action.
    assert summary["total_order_value"] == "41854.65"
    assert summary["total_payments_settled"] == "41904.38"
    assert summary["reconciled_value"] == "39773.28"
    assert summary["disputed_value"] == "2827.95"


def test_money_at_risk_splits_exactly_into_three_directions():
    """The headline must equal the sum of its parts, or the dashboard is lying."""
    summary = build_summary(_run())

    assert summary["money_at_risk"] == "2178.43"
    assert summary["revenue_at_risk"] == "787.85"      # we are owed
    assert summary["customer_owed"] == "628.58"        # we owe
    assert summary["needs_investigation"] == "762.00"  # unknown

    parts = (_dec(summary["revenue_at_risk"]) + _dec(summary["customer_owed"])
             + _dec(summary["needs_investigation"]))
    assert parts == _dec(summary["money_at_risk"])


def test_money_at_risk_equals_sum_of_primary_findings():
    """Guards the anti-double-counting rule: only primary findings carry money, so the
    headline always ties to the drill-down table."""
    output = _run()
    summary = build_summary(output)

    primary_total = sum((f.amount_at_risk for f in output.findings if f.is_primary), Decimal("0.00"))
    secondary_total = sum((f.amount_at_risk for f in output.findings if not f.is_primary), Decimal("0.00"))

    assert primary_total == _dec(summary["money_at_risk"])
    assert secondary_total == Decimal("0.00")


def test_amount_at_risk_never_exceeds_the_money_involved():
    """Sanity bound: exposure on one reference cannot exceed the larger of what we expected
    and what we collected. Catches a sign error that would inflate the headline."""
    for result in _run().results:
        assert result.amount_at_risk <= max(result.expected_amount, result.collected_amount)
        assert result.amount_at_risk >= Decimal("0.00")


# ======================================================================================
# Breakdowns
# ======================================================================================
def test_amount_at_risk_by_type_is_exact():
    summary = build_summary(_run())
    by_type = {row["type"]: row for row in summary["by_type"]}

    assert by_type["MISSING_PAYMENT"]["amount_at_risk"] == "392.35"
    assert by_type["MISSING_ORDER"]["amount_at_risk"] == "308.00"
    assert by_type["AMOUNT_MISMATCH"]["amount_at_risk"] == "103.50"
    assert by_type["DUPLICATE_PAYMENT"]["amount_at_risk"] == "248.58"
    assert by_type["CURRENCY_MISMATCH"]["amount_at_risk"] == "355.00"
    assert by_type["STATUS_CONFLICT"]["amount_at_risk"] == "771.00"

    # A timing anomaly is a review item, not exposure: flagged, but worth 0.00.
    assert by_type["TIMING_ANOMALY"]["amount_at_risk"] == "0.00"
    assert by_type["TIMING_ANOMALY"]["is_financial"] is False


def test_primary_findings_per_type_are_exact():
    """One row per order in the default view -- these counts are what the table shows."""
    primaries = _primary_by_key(_run())
    counts: dict[str, int] = {}
    for finding in primaries.values():
        counts[finding.discrepancy_type.value] = counts.get(finding.discrepancy_type.value, 0) + 1

    assert counts == {
        "MISSING_PAYMENT": 4,      # ORD-1201..1204
        "MISSING_ORDER": 3,        # ORD-1301..1303
        "AMOUNT_MISMATCH": 3,      # ORD-1401..1403
        "DUPLICATE_PAYMENT": 2,    # ORD-1501, ORD-1502
        "CURRENCY_MISMATCH": 2,    # ORD-1601, ORD-1602
        "STATUS_CONFLICT": 5,      # ORD-1701, 1702, 1703, 2001, 2002
        "TIMING_ANOMALY": 1,       # ORD-2101
    }
    assert sum(counts.values()) == 20


def test_severity_breakdown_is_exact():
    summary = build_summary(_run())
    by_severity = {row["severity"]: row for row in summary["by_severity"]}

    assert by_severity["CRITICAL"]["count"] == 9
    assert by_severity["CRITICAL"]["amount_at_risk"] == "1185.35"
    assert by_severity["HIGH"]["count"] == 7
    assert by_severity["HIGH"]["amount_at_risk"] == "889.58"
    assert by_severity["MEDIUM"]["count"] == 2
    assert by_severity["MEDIUM"]["amount_at_risk"] == "85.00"
    assert by_severity["LOW"]["count"] == 2
    assert by_severity["LOW"]["amount_at_risk"] == "18.50"


# ======================================================================================
# Rule-by-rule behaviour
# ======================================================================================
def test_missing_payments_are_critical_revenue_at_risk():
    """R01: an order with no payment at all. Full order value is at risk."""
    primaries = _primary_by_key(_run())

    for key, expected in {
        "ORD-1201": "94.87", "ORD-1202": "80.83", "ORD-1203": "59.52", "ORD-1204": "157.13",
    }.items():
        finding = primaries[key]
        assert finding.discrepancy_type is DiscrepancyType.MISSING_PAYMENT
        assert finding.severity is Severity.CRITICAL
        assert finding.risk_direction is RiskDirection.REVENUE_AT_RISK
        assert finding.amount_at_risk == _dec(expected)
        assert finding.rule_id == "R01-NO-PAYMENT"


def test_orphaned_payments_are_flagged_not_dropped():
    """R02: the brief's ORD-1301/1302/1303 -- payments with no matching order.

    Direction is NEEDS_INVESTIGATION, not revenue: we are holding the customer's money, but
    until the missing order is found we cannot say whether we owe it back.
    """
    primaries = _primary_by_key(_run())

    for key, expected in {"ORD-1301": "79.51", "ORD-1302": "78.98", "ORD-1303": "149.51"}.items():
        finding = primaries[key]
        assert finding.discrepancy_type is DiscrepancyType.MISSING_ORDER
        assert finding.severity is Severity.CRITICAL
        assert finding.risk_direction is RiskDirection.NEEDS_INVESTIGATION
        assert finding.amount_at_risk == _dec(expected)


def test_amount_mismatches_use_signed_deltas():
    """R06: delta = collected - expected.

    ORD-1401 is OVERPAID  (+25.00): we hold too much  -> CUSTOMER_OWED.
    ORD-1402 is UNDERPAID (-18.50): we are short      -> REVENUE_AT_RISK.
    ORD-1403 is the brief's 199.01 vs 259.01 case (+60.00, overpaid).

    The sign is what routes the work to refunds or to collections, so it is asserted
    explicitly rather than tested through an absolute value.
    """
    output = _run()
    primaries = _primary_by_key(output)
    results = {r.order_key: r for r in output.results}

    over = primaries["ORD-1401"]
    assert over.discrepancy_type is DiscrepancyType.AMOUNT_MISMATCH
    assert results["ORD-1401"].delta == _dec("25.00")
    assert over.risk_direction is RiskDirection.CUSTOMER_OWED
    assert over.severity is Severity.MEDIUM
    assert over.amount_at_risk == _dec("25.00")

    under = primaries["ORD-1402"]
    assert results["ORD-1402"].delta == _dec("-18.50")
    assert under.risk_direction is RiskDirection.REVENUE_AT_RISK
    assert under.severity is Severity.LOW
    assert under.amount_at_risk == _dec("18.50")

    brief_case = primaries["ORD-1403"]
    assert results["ORD-1403"].expected_amount == _dec("199.01")
    assert results["ORD-1403"].collected_amount == _dec("259.01")
    assert brief_case.amount_at_risk == _dec("60.00")
    assert brief_case.risk_direction is RiskDirection.CUSTOMER_OWED


def test_sub_tolerance_rounding_is_not_flagged():
    """The tolerance's whole purpose: ORD-1901/1902/1903 differ by 0.01-0.02 (the brief's
    68.65 vs 68.63 case). Flagging those would bury the 20 real findings in noise."""
    output = _run()
    results = {r.order_key: r for r in output.results}

    for key in ("ORD-1901", "ORD-1902", "ORD-1903"):
        result = results[key]
        assert result.is_reconciled, f"{key} is within tolerance and must not be flagged"
        assert abs(result.delta) <= _dec("0.05")
        assert result.delta != _dec("0.00")   # there IS a difference; it is just immaterial

    assert results["ORD-1902"].expected_amount == _dec("68.65")
    assert results["ORD-1902"].collected_amount == _dec("68.63")


def test_tolerance_is_configuration_not_a_hard_coded_constant():
    """Set the tolerance to zero and the three rounding orders appear. This is the test that
    proves the threshold is a business input, and it is the demo used in the README."""
    strict = {r.order_key: r for r in _run(ReconConfig(amount_tolerance=Decimal("0.00"))).results}

    for key in ("ORD-1901", "ORD-1902", "ORD-1903"):
        assert not strict[key].is_reconciled
        assert strict[key].primary_type is DiscrepancyType.AMOUNT_MISMATCH

    # ...and every genuine finding still fires. The tolerance changes only the margin.
    assert strict["ORD-1403"].primary_type is DiscrepancyType.AMOUNT_MISMATCH
    assert strict["ORD-1201"].primary_type is DiscrepancyType.MISSING_PAYMENT


def test_duplicate_charges_outrank_the_amount_mismatch_they_cause():
    """R04 vs R06 precedence.

    ORD-1501/1502 were each charged twice. That is BOTH a duplicate payment and an amount
    mismatch -- but 'duplicate payment' is the actionable cause and 'amount mismatch' is
    merely its symptom. Precedence reports the cause and suppresses the symptom, so the
    duplicated money is counted exactly once and the row still says what to do.
    """
    output = _run()
    findings_by_key: dict[str, list] = {}
    for finding in output.findings:
        findings_by_key.setdefault(finding.order_key, []).append(finding)

    for key, at_risk in {"ORD-1501": "119.84", "ORD-1502": "128.74"}.items():
        findings = findings_by_key[key]
        primary = [f for f in findings if f.is_primary]
        assert len(primary) == 1
        assert primary[0].discrepancy_type is DiscrepancyType.DUPLICATE_PAYMENT
        assert primary[0].severity is Severity.HIGH
        assert primary[0].risk_direction is RiskDirection.CUSTOMER_OWED
        # Exposure is the DUPLICATE charge only, not the whole collected amount: the first
        # charge was legitimate.
        assert primary[0].amount_at_risk == _dec(at_risk)

        # The amount-mismatch symptom is suppressed outright rather than recorded twice, so
        # this reference carries exactly one finding.
        assert [f for f in findings if not f.is_primary] == []
        # ...and the evidence still shows the duplicate group that caused it.
        assert primary[0].evidence.get("extra_charged")
        assert primary[0].evidence.get("duplicate_groups")


def test_refunds_are_netted_against_charges():
    """ORD-1702 and ORD-1703 both have a charge and a refund, and they are NOT the same case.

    ORD-1702: refunded status, 240.00 charged, 120.00 refunded -> 120.00 still held.
    ORD-1703: completed status, 99.00 charged, 99.00 refunded  -> nothing collected.

    Summing charges and ignoring the refund sign would report both as fully paid.
    """
    output = _run()
    results = {r.order_key: r for r in output.results}
    primaries = _primary_by_key(output)

    partial = results["ORD-1702"]
    # Expected is 0.00 because the order was refunded: we are not owed anything on it.
    # Collected is 120.00 (240.00 charged minus a 120.00 refund) -- money we are still
    # holding on an order that was supposed to be refunded in full.
    assert partial.expected_amount == _dec("0.00")
    assert partial.collected_amount == _dec("120.00")
    assert partial.refund_count == 1
    assert primaries["ORD-1702"].discrepancy_type is DiscrepancyType.STATUS_CONFLICT
    assert primaries["ORD-1702"].rule_id == "R05b-PARTIAL-REFUND"
    assert primaries["ORD-1702"].severity is Severity.HIGH
    assert primaries["ORD-1702"].risk_direction is RiskDirection.CUSTOMER_OWED
    assert primaries["ORD-1702"].amount_at_risk == _dec("120.00")

    fully_refunded = results["ORD-1703"]
    assert fully_refunded.expected_amount == _dec("99.00")
    assert fully_refunded.collected_amount == _dec("0.00")
    assert primaries["ORD-1703"].discrepancy_type is DiscrepancyType.STATUS_CONFLICT
    assert primaries["ORD-1703"].rule_id == "R05d-REFUNDED-BUT-COMPLETED"
    assert primaries["ORD-1703"].amount_at_risk == _dec("99.00")


def test_currency_mismatch_suppresses_amount_comparison():
    """R03. Both sample cases are 'same number, different currency'.

    Note the direction of the brief's example is reversed in the actual data: ORD-1601 is
    USD in orders and EUR in payments (ORD-1602 is the EUR->USD case).

    Comparing 210 USD to 210 EUR as if they were the same unit would report a perfect match,
    so the engine refuses to compare amounts at all and routes the case to investigation.
    """
    output = _run()
    results = {r.order_key: r for r in output.results}
    primaries = _primary_by_key(output)

    first = primaries["ORD-1601"]
    assert first.discrepancy_type is DiscrepancyType.CURRENCY_MISMATCH
    assert first.severity is Severity.HIGH
    assert first.risk_direction is RiskDirection.NEEDS_INVESTIGATION
    assert first.amount_at_risk == _dec("210.00")
    assert results["ORD-1601"].order_currency == "USD"
    assert "EUR" in results["ORD-1601"].payment_currencies
    # The amounts are numerically identical; only the currency differs.
    assert results["ORD-1601"].delta == _dec("0.00")

    second = primaries["ORD-1602"]
    assert second.discrepancy_type is DiscrepancyType.CURRENCY_MISMATCH
    assert second.amount_at_risk == _dec("145.00")
    assert results["ORD-1602"].order_currency == "EUR"
    assert "USD" in results["ORD-1602"].payment_currencies


def test_four_status_conflict_shapes_are_distinguished():
    """Status conflicts are not one rule. Each shape has a different owner and action, so
    each gets its own rule id, severity and direction."""
    primaries = _primary_by_key(_run())

    # Cancelled but paid: we are holding money for an order that does not exist.
    cancelled = primaries["ORD-1701"]
    assert cancelled.rule_id == "R05a-CANCELLED-BUT-PAID"
    assert cancelled.severity is Severity.CRITICAL
    assert cancelled.risk_direction is RiskDirection.CUSTOMER_OWED
    assert cancelled.amount_at_risk == _dec("175.00")

    # Refunded but still holding part of the money. A DIFFERENT rule from R05a on purpose:
    # the action is "finish the refund", not "reverse a charge that should never have run".
    assert primaries["ORD-1702"].rule_id == "R05b-PARTIAL-REFUND"
    assert primaries["ORD-1702"].severity is Severity.HIGH
    assert primaries["ORD-1702"].risk_direction is RiskDirection.CUSTOMER_OWED

    # Completed but the payment failed: goods delivered, no money. The brief's ORD-2001/2002.
    failed = primaries["ORD-2001"]
    assert failed.rule_id == "R05c-COMPLETED-BUT-UNPAID"
    assert failed.severity is Severity.CRITICAL
    assert failed.risk_direction is RiskDirection.REVENUE_AT_RISK
    assert failed.amount_at_risk == _dec("310.00")

    pending = primaries["ORD-2002"]
    assert pending.rule_id == "R05c-COMPLETED-BUT-UNPAID"
    assert pending.risk_direction is RiskDirection.REVENUE_AT_RISK
    assert pending.amount_at_risk == _dec("67.00")
    # Pending is less severe than outright failed: it may still settle.
    assert pending.severity is Severity.HIGH


def test_timing_anomaly_is_flagged_without_money():
    """ORD-2101 settled 29 days after the order. The amounts tie out exactly, so this is a
    review item: flagged for visibility, contributing 0.00 to exposure."""
    output = _run()
    finding = _primary_by_key(output)["ORD-2101"]
    result = {r.order_key: r for r in output.results}["ORD-2101"]

    assert finding.discrepancy_type is DiscrepancyType.TIMING_ANOMALY
    assert finding.severity is Severity.LOW
    assert finding.risk_direction is RiskDirection.NONE
    assert finding.amount_at_risk == _dec("0.00")
    assert result.expected_amount == result.collected_amount == _dec("190.00")


def test_dirty_references_reconcile_cleanly():
    """The brief's quirk 1. ' ord-1801 ' and 'ord-1802' must match ORD-1801/ORD-1802 and
    then RECONCILE -- normalisation is not enough if the matched pair still gets flagged."""
    output = _run()
    results = {r.order_key: r for r in output.results}

    for key in ("ORD-1801", "ORD-1802"):
        assert key in results
        assert results[key].has_order and results[key].has_payments
        assert results[key].is_reconciled
        assert results[key].matched_via_normalisation is True


# ======================================================================================
# Structural guarantees
# ======================================================================================
def test_engine_is_deterministic():
    """Same input, same output -- byte for byte. This is the property that makes the
    engine defensible: the AI can be non-deterministic, the numbers cannot."""
    first, second = build_summary(_run()), build_summary(_run())
    assert first == second

    keys_first = [(f.order_key, f.rule_id, str(f.amount_at_risk)) for f in _run().findings]
    keys_second = [(f.order_key, f.rule_id, str(f.amount_at_risk)) for f in _run().findings]
    assert keys_first == keys_second


def test_every_flagged_order_has_exactly_one_primary_finding():
    """Invariant behind the headline totals: one primary per flagged reference, and none on
    a reference that reconciled."""
    output = _run()
    for result in output.results:
        primaries = [f for f in result.findings if f.is_primary]
        if result.is_reconciled:
            assert primaries == []
        else:
            assert len(primaries) == 1
            assert primaries[0].discrepancy_type is result.primary_type
            assert primaries[0].severity is result.primary_severity


def test_priority_order_is_worst_first():
    """Severity first, then money. The 310.00 completed-but-unpaid order outranks the
    157.13 missing payment, and both outrank anything MEDIUM."""
    output = _run()
    priorities = top_priorities(output, limit=10)
    keys = [result.order_key for result in priorities]

    assert keys[:5] == ["ORD-2001", "ORD-1701", "ORD-1204", "ORD-1303", "ORD-1201"]
    assert priorities[0].amount_at_risk == _dec("310.00")

    ranks = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    severities = [ranks[result.primary_severity.value] for result in priorities]
    assert severities == sorted(severities)


def test_findings_carry_evidence_for_the_ai_prompt():
    """Every finding must be explainable without re-reading the CSVs: the evidence bundle is
    what the LLM prompt is built from, so an empty one would mean an ungrounded explanation."""
    for finding in _run().findings:
        assert isinstance(finding.evidence, dict)
        assert finding.summary and finding.detail
        assert finding.rule_id
        assert finding.evidence.get("order_key") == finding.order_key
        # Status conflicts are argued from the individual transactions, so those must be
        # present. Other rules carry rule-specific evidence instead (duplicate groups,
        # currency sets, settlement lag), asserted by their own tests above.
        if finding.discrepancy_type is DiscrepancyType.STATUS_CONFLICT:
            assert finding.evidence.get("transactions"), (
                f"{finding.order_key} has no transaction evidence"
            )


def test_config_is_recorded_with_the_results():
    """A stored run must be reproducible: the tolerances travel with the numbers."""
    config = _run().config.as_dict()
    assert config["amount_tolerance"] == "0.05"
    assert config["duplicate_window_hours"] == 24
    assert config["max_settlement_lag_days"] == 7
