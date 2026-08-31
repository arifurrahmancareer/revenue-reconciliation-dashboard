"""
normalize.py -- Field-level cleaning primitives.

WHY THIS FILE EXISTS (interview answer):
Reconciliation is only as good as the keys you join on. Every messy-data problem in
this dataset is really a *normalisation* problem, so all cleaning lives in one small,
pure, unit-testable module with NO imports from FastAPI, SQLAlchemy or pandas.
That means the exact same code path runs in the API, in the tests, and in scripts.

DESIGN RULES
1. Money is `Decimal`, never `float`. 0.1 + 0.2 != 0.3 in binary floating point, and
   this app's whole job is deciding whether two amounts are equal. Float drift would
   manufacture fake one-cent discrepancies.
2. Every parser is total: it returns `None` for unusable input instead of raising, and
   the caller records a data-quality issue. One bad cell must never kill a 187-row import.
3. Date parsing uses an EXPLICIT per-source format list. We never let a library guess,
   because guessing is how 05/04/2025 silently becomes 4 May instead of 5 April.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

# --------------------------------------------------------------------------------------
# Reference / string normalisation
# --------------------------------------------------------------------------------------

# Whitespace characters that survive naive .strip() calls in real exports:
# non-breaking space, zero-width space, BOM, tabs.
_INVISIBLE = dict.fromkeys(map(ord, "\u00a0\u200b\ufeff\t\r\n"), " ")
_MULTISPACE = re.compile(r"\s+")

# Dash variants an Excel round-trip can introduce (en dash, em dash, minus sign).
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-"})


def clean_text(raw: object) -> str:
    """Collapse a raw CSV cell into a trimmed, single-spaced string. Never returns None."""
    if raw is None:
        return ""
    text = str(raw)
    # NFKC folds full-width and compatibility characters into their ASCII equivalents.
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_INVISIBLE)
    text = _MULTISPACE.sub(" ", text)
    return text.strip()


def normalize_reference(raw: object) -> str:
    """
    Canonical join key for an order.

    Handles the observed quirks in this dataset:
      ' ord-1801 ' -> 'ORD-1801'   (leading/trailing whitespace)
      'ord-1802'   -> 'ORD-1802'   (lower case)
      'ORD\u20131802'   -> 'ORD-1802'   (en dash pasted from a spreadsheet)

    Deliberately NOT done: stripping the 'ORD-' prefix or the hyphen. Two different
    stores could use 'ORD-1' and 'ORD1' as distinct references; over-normalising a key
    creates false matches, which are far more dangerous in finance than false misses.
    """
    text = clean_text(raw).translate(_DASHES)
    # Internal spaces are never meaningful inside a reference ('ORD 1801' == 'ORD1801'
    # is wrong, but 'ORD- 1801' from a wrapped cell is real), so only remove spaces
    # that sit directly next to the hyphen separator.
    text = re.sub(r"\s*-\s*", "-", text)
    return text.upper()


def normalize_enum(raw: object) -> str:
    """Lower-case canonical form for status/type columns ('Settled ' -> 'settled')."""
    return clean_text(raw).lower()


def normalize_currency(raw: object) -> str:
    """ISO-4217-style upper-case currency code. Returns '' when absent."""
    return clean_text(raw).upper()


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def normalize_email(raw: object) -> str | None:
    """
    Lower-cased email, or None when missing/unparseable.

    ORD-2201 in this dataset has an empty email. We keep the order (it still represents
    real revenue) and raise a DATA_QUALITY flag rather than dropping the row -- silently
    discarding revenue is the worst possible outcome for a reconciliation tool.
    """
    text = clean_text(raw).lower()
    if not text:
        return None
    return text if _EMAIL_RE.match(text) else None


def mask_email(email: str | None) -> str:
    """
    Irreversibly shorten an email for logs and for LLM prompts: 'kate.d@example.com'
    -> 'k****@example.com'. Used so customer PII never leaves our backend.
    """
    if not email or "@" not in email:
        return "(none)"
    local, _, domain = email.partition("@")
    head = local[0] if local else "?"
    return f"{head}{'*' * max(len(local) - 1, 1)}@{domain}"


# --------------------------------------------------------------------------------------
# Money
# --------------------------------------------------------------------------------------

CENTS = Decimal("0.01")
ZERO = Decimal("0.00")

# Strip thousands separators and currency symbols an export may include.
_MONEY_STRIP = re.compile(r"[,\s\u00a0$\u20ac\u00a3\u00a5]")


def parse_money(raw: object) -> Decimal | None:
    """
    Parse a monetary cell into a 2dp Decimal, or None if it is blank/unparseable.

    Accepts: '1,234.50', '$99.00', '(18.50)' -> -18.50, '' -> None.
    Uses ROUND_HALF_UP (what accountants expect) rather than Python's default
    banker's rounding.
    """
    text = clean_text(raw)
    if not text:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = _MONEY_STRIP.sub("", text)
    if not text or text in {"-", ".", "-."}:
        return None
    try:
        value = Decimal(text)
    except InvalidOperation:
        return None
    if not value.is_finite():
        return None
    if negative:
        value = -value
    return quantize(value)


def quantize(value: Decimal) -> Decimal:
    """Force a Decimal onto the 2dp money grid."""
    return value.quantize(CENTS, rounding=ROUND_HALF_UP)


# --------------------------------------------------------------------------------------
# Dates
# --------------------------------------------------------------------------------------

# orders.csv -- observed 185/185 rows as 'YYYY-MM-DD HH:MM:SS'.
ORDER_DATE_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
)

# payments.csv -- observed 186/187 rows as 'DD/MM/YYYY HH:MM' (1 blank).
# DAY-FIRST IS PROVEN, NOT ASSUMED: the file contains '13/04/2025', '21/04/2025',
# '30/04/2025' and '22/05/2025'. A value of 13+ in the first position is impossible
# for a month, so the layout must be DD/MM/YYYY. This is exactly the check to state
# out loud in the interview -- guessing here would shift ~2/3 of payment dates.
PAYMENT_DATE_FORMATS = (
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%Y-%m-%d %H:%M:%S",  # tolerated in case the processor changes its export
    "%Y-%m-%d",
)


def _parse_with(formats: tuple[str, ...], raw: object) -> datetime | None:
    text = clean_text(raw)
    if not text:
        return None
    for fmt in formats:
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        # Both exports are naive local timestamps. We attach UTC so every datetime in
        # the system is timezone-aware and comparable; the choice is documented in the
        # README rather than left implicit.
        return parsed.replace(tzinfo=timezone.utc)
    return None


def parse_order_datetime(raw: object) -> datetime | None:
    return _parse_with(ORDER_DATE_FORMATS, raw)


def parse_payment_datetime(raw: object) -> datetime | None:
    """None for the blank `processed_at` on TXN700187 -- flagged, not dropped."""
    return _parse_with(PAYMENT_DATE_FORMATS, raw)


def normalize_header(raw: object) -> str:
    """'  Net Amount ' -> 'net_amount'. Makes ingestion resilient to header drift."""
    text = clean_text(raw).lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")
