"""
test_parsing.py -- Tests for the cleaning layer, run against the REAL sample CSVs.

These tests pin down the answer to "what exactly did you do to my data?" -- the first
question anyone reviewing a reconciliation tool should ask. Each test maps to a quirk in the
brief.

Run:  pytest backend/tests -q
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

# tests/ now sits at the repository root, alongside the app/ package.
ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
sys.path.insert(0, str(ROOT))

from app.domain.engine import reconcile  # noqa: E402
from app.domain.normalize import (  # noqa: E402
    mask_email, normalize_currency, normalize_reference, parse_money, parse_order_datetime,
    parse_payment_datetime,
)
from app.domain.parsing import parse_orders_csv, parse_payments_csv  # noqa: E402

UTC = timezone.utc


def _load():
    orders = parse_orders_csv((SAMPLES / "orders.csv").read_bytes())
    payments = parse_payments_csv((SAMPLES / "payments.csv").read_bytes())
    return orders, payments


# ======================================================================================
# Quirk 1 -- whitespace and case
# ======================================================================================
def test_reference_normalisation_rules():
    """Trim, collapse internal whitespace, uppercase.

    Also covers the two invisible cases that survive a naive `.strip()`: a UTF-8 BOM (which
    Excel loves to add to the first cell) and a non-breaking space.
    """
    assert normalize_reference("ord-1801 ") == "ORD-1801"
    assert normalize_reference(" ord-1802") == "ORD-1802"
    assert normalize_reference("ORD-1802") == "ORD-1802"
    assert normalize_reference("ord-1802") == "ORD-1802"
    assert normalize_reference("\ufeffORD-1000") == "ORD-1000"     # BOM stripped
    assert normalize_reference("ORD\u00a01000") == "ORD 1000"      # NBSP -> normal space
    assert normalize_reference("  ") == ""
    assert normalize_reference(None) == ""


def test_dirty_references_match_their_orders():
    """The two dirty payment references in the file must resolve to real order keys, and the
    raw text must be preserved so the UI can show what was actually in the file."""
    orders, payments = _load()
    order_keys = {order.order_key for order in orders.orders}

    dirty = {
        payment.transaction_ref: payment
        for payment in payments.payments
        if payment.raw_order_reference != payment.order_key
    }
    assert set(dirty) == {"TXN700178", "TXN700179"}

    assert dirty["TXN700178"].raw_order_reference == " ord-1801 "
    assert dirty["TXN700178"].order_key == "ORD-1801"
    assert dirty["TXN700179"].raw_order_reference == "ord-1802"
    assert dirty["TXN700179"].order_key == "ORD-1802"

    # The cleaned keys exist on the orders side, so these payments will match.
    assert {"ORD-1801", "ORD-1802"} <= order_keys


def test_normalisation_is_recorded_as_a_data_quality_issue():
    """Cleaning must be visible. Every normalisation is logged with its CSV row number, so
    the dashboard can show the raw value beside the key it was matched to."""
    _, payments = _load()
    normalised = [i for i in payments.issues if i.code == "ORDER_REFERENCE_NORMALISED"]

    assert len(normalised) == 2
    assert {issue.identifier for issue in normalised} == {"TXN700178", "TXN700179"}
    for issue in normalised:
        assert issue.dropped is False      # normalised, not discarded
        assert issue.source == "payments"
        assert issue.source_row > 1        # a real line number, for the audit trail


# ======================================================================================
# Quirk 2 -- two date formats
# ======================================================================================
def test_date_parsing_is_format_specific_and_day_first():
    """Orders are `YYYY-MM-DD HH:MM:SS`; payments are `DD/MM/YYYY HH:MM`.

    THE CRITICAL CASE is `10/04/2025`. Read day-first it is 10 April; read month-first it is
    4 October -- a six-month error that would corrupt every timing rule. The two files get
    two separate parsers with explicit format lists, so the interpretation is a stated
    decision rather than a library default. Results are timezone-aware UTC, which keeps
    date arithmetic unambiguous.
    """
    assert parse_order_datetime("2025-04-01 10:15:00") == datetime(2025, 4, 1, 10, 15, tzinfo=UTC)
    assert parse_payment_datetime("10/04/2025 00:22") == datetime(2025, 4, 10, 0, 22, tzinfo=UTC)
    # Unambiguous proof of day-first: 25 cannot be a month.
    assert parse_payment_datetime("25/12/2025 23:59") == datetime(2025, 12, 25, 23, 59, tzinfo=UTC)

    # Unparseable input returns None instead of raising: one bad cell must not abort a file.
    assert parse_payment_datetime("") is None
    assert parse_payment_datetime("not a date") is None
    assert parse_order_datetime(None) is None


def test_missing_payment_date_is_kept_not_dropped():
    """A payment with no date is still money. Dropping the row to keep the schema tidy would
    delete a real 155.00 transaction from the reconciliation."""
    _, payments = _load()
    undated = [payment for payment in payments.payments if payment.processed_at is None]

    assert len(undated) == 1
    assert undated[0].transaction_ref == "TXN700187"
    assert undated[0].amount == Decimal("155.00")   # the amount is intact
    assert undated[0].order_key                     # and it can still be matched

    flagged = [i for i in payments.issues if i.code == "MISSING_PROCESSED_AT"]
    assert len(flagged) == 1
    assert flagged[0].identifier == "TXN700187"
    assert flagged[0].dropped is False


# ======================================================================================
# Quirk 3 -- missing fields
# ======================================================================================
def test_missing_email_and_discount_are_flagged_but_loaded():
    """ORD-2201 has no email AND no discount. Both are reported; neither blocks the row,
    because neither is needed to reconcile money.

    Note the blank discount is left as None -- *unknown* -- rather than coerced to 0.00.
    `net_amount` is the authoritative figure for reconciliation, so there is no reason to
    invent a value, and inventing one would hide the gap from the data-quality report.
    """
    orders, _ = _load()
    by_key = {order.order_key: order for order in orders.orders}

    assert "ORD-2201" in by_key
    target = by_key["ORD-2201"]
    assert target.customer_email is None
    assert target.discount is None                       # unknown, not fabricated as 0.00
    assert target.net_amount == Decimal("120.00")        # what reconciliation actually uses

    codes = {(i.code, i.identifier) for i in orders.issues}
    assert ("MISSING_EMAIL", "ORD-2201") in codes
    assert ("MISSING_DISCOUNT", "ORD-2201") in codes


# ======================================================================================
# Quirk 4 -- duplicates
# ======================================================================================
def test_exact_duplicate_order_row_is_collapsed_once():
    """ORD-1004 appears twice, byte-identical. Loading both would double-count its value.

    Only EXACT duplicates are collapsed. Two rows sharing an id with different values are a
    conflict, not a duplicate, and the parser must never silently pick a winner.
    """
    orders, _ = _load()

    assert orders.rows_read == 185
    assert orders.rows_loaded == 184
    assert orders.duplicate_rows_collapsed == 1
    # Collapsed, not "dropped": no information was lost, so it is not counted as a loss.
    assert orders.rows_dropped == 0

    keys = [order.order_key for order in orders.orders]
    assert len(keys) == len(set(keys))       # exactly one row per key
    assert keys.count("ORD-1004") == 1

    collapsed = [i for i in orders.issues if i.code == "DUPLICATE_ORDER_ROW"]
    assert len(collapsed) == 1
    assert collapsed[0].identifier == "ORD-1004"
    assert collapsed[0].dropped is True      # this row did not reach the engine


def test_payments_are_never_collapsed():
    """Multiple payments per order are LEGITIMATE (instalments, retries, refunds).

    De-duplicating them would destroy the very evidence the duplicate-charge rule needs, so
    every payment row is loaded and the engine decides what the pattern means.
    """
    _, payments = _load()

    assert payments.rows_read == 187
    assert payments.rows_loaded == 187
    assert payments.rows_dropped == 0
    assert payments.duplicate_rows_collapsed == 0

    refs = [payment.transaction_ref for payment in payments.payments]
    assert len(refs) == len(set(refs))       # transaction refs are unique in this file

    for key in ("ORD-1501", "ORD-1502", "ORD-1702", "ORD-1703"):
        matching = [p for p in payments.payments if p.order_key == key]
        assert len(matching) == 2, f"{key} must keep both of its payment rows"


# ======================================================================================
# Amounts, currency, PII
# ======================================================================================
def test_parse_money_is_decimal_and_tolerant_of_formatting():
    """Decimal, never float: `0.1 + 0.2 != 0.3` in binary floating point, and this
    application exists to argue about cents.

    Handles thousands separators, currency symbols and accounting-style negatives, because
    all three turn up in exported finance data.
    """
    assert parse_money("199.01") == Decimal("199.01")
    assert parse_money("1,234.56") == Decimal("1234.56")
    assert parse_money("$68.63") == Decimal("68.63")
    assert parse_money("(25.00)") == Decimal("-25.00")     # accounting negative
    assert parse_money("-25.00") == Decimal("-25.00")
    assert isinstance(parse_money("10"), Decimal)

    # Blank means UNKNOWN, not zero. Treating it as zero would silently create a
    # discrepancy that is not in the data.
    assert parse_money("") is None
    assert parse_money("abc") is None
    assert parse_money(None) is None

    # Quantised to 2 places, ROUND_HALF_UP. Bankers' rounding would make this 10.12, which
    # is not how an invoice is rounded.
    assert parse_money("10.125") == Decimal("10.13")


def test_currency_normalisation():
    assert normalize_currency(" usd ") == "USD"
    assert normalize_currency("eur") == "EUR"
    assert normalize_currency("") == ""       # unknown stays empty rather than guessing USD
    assert normalize_currency(None) == ""


def test_email_masking_protects_pii():
    """Addresses are masked before they can reach a finding, an API response, the dashboard
    or an LLM prompt."""
    assert mask_email("karen@example.com") == "k****@example.com"
    assert mask_email("a@example.com") == "a*@example.com"
    # No address at all must not render as an empty string that looks like a bug.
    assert mask_email("") == "(none)"
    assert mask_email(None) == "(none)"


def test_no_raw_address_reaches_the_reconciliation_output():
    """The important half of the PII guarantee: whatever the CSV contains, everything the
    engine emits (and therefore everything the API, the UI and any prompt can see) is masked.
    """
    orders, payments = _load()
    output = reconcile(orders.orders, payments.payments,
                       row_issues=orders.issues + payments.issues)

    raw_addresses = {o.customer_email for o in orders.orders if o.customer_email}
    assert raw_addresses, "sanity: the sample file does contain addresses"

    for result in output.results:
        masked = result.customer_email_masked
        if masked and masked != "(none)":
            assert "*" in masked, f"{result.order_key} exposed an unmasked address"
            assert masked not in raw_addresses


def test_structural_totals_are_stable():
    """Belt-and-braces totals, so a change to the sample files cannot silently invalidate
    every number quoted in the README."""
    orders, payments = _load()

    assert len(orders.orders) == 184
    assert len(payments.payments) == 187

    order_keys = {order.order_key for order in orders.orders}
    payment_keys = {payment.order_key for payment in payments.payments}

    # Three orphaned payment references: the brief's ORD-1301/1302/1303.
    assert payment_keys - order_keys == {"ORD-1301", "ORD-1302", "ORD-1303"}
    # Four orders with no payment at all.
    assert order_keys - payment_keys == {"ORD-1201", "ORD-1202", "ORD-1203", "ORD-1204"}

    # Every key is already normalised: no lowercase or padded keys survive parsing.
    for key in order_keys | payment_keys:
        assert key == key.strip().upper()
