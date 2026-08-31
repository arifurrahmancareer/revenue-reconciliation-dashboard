"""
charts.py -- Plotly figure builders. Pure functions: data in, figure out.

WHY PLOTLY (not Altair): native hover tooltips carry the exact amounts, which matters when
the difference between two bars is $0.02, and `st.plotly_chart` needs no extra theming to
look consistent.

THREE DELIBERATE CHART DECISIONS
  1. SEVERITY HAS FIXED COLOURS. Critical is always the same red, everywhere. A palette
     that reassigns colours per chart forces the reader to re-learn the legend each time.
  2. NON-FINANCIAL FINDINGS ARE GREY. A timing anomaly carries no money, so it must not
     look like exposure sitting next to a duplicate charge.
  3. CHARTS PLOT MONEY AT RISK, NOT ROW COUNTS, by default. Four missing payments worth
     $392 matter more than a single mis-currency worth $210 -- counts alone hide that.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import plotly.graph_objects as go

# Colour-blind-safe, and matched to the badge colours used in the table.
SEVERITY_COLORS = {
    "CRITICAL": "#B42318",
    "HIGH": "#DC6803",
    "MEDIUM": "#B54708",
    "LOW": "#475467",
}
TYPE_COLOR = "#1D4ED8"
NON_FINANCIAL_COLOR = "#98A2B3"

DIRECTION_COLORS = {
    "REVENUE_AT_RISK": "#B42318",      # money we are owed
    "CUSTOMER_OWED": "#0B7285",        # money we hold and should return
    "NEEDS_INVESTIGATION": "#7A5AF8",  # unknown until someone looks
}
DIRECTION_LABELS = {
    "REVENUE_AT_RISK": "Revenue at risk",
    "CUSTOMER_OWED": "Owed to customers",
    "NEEDS_INVESTIGATION": "Needs investigation",
}

LAYOUT_DEFAULTS = dict(
    margin=dict(l=10, r=10, t=40, b=10),
    height=320,
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(size=13),
    showlegend=False,
)


def _money(value) -> float:
    """Money arrives as a STRING (see build_summary in app/domain/metrics.py) and Plotly
    needs a float.

    Conversion happens here, at the very last step before rendering, so exact Decimal
    values are preserved everywhere that arithmetic or comparison happens.
    """
    try:
        return float(Decimal(str(value or "0")))
    except (InvalidOperation, ValueError, TypeError):
        return 0.0


def _empty(message: str) -> go.Figure:
    """A clean 'nothing to show' state. An empty axis frame looks like a broken chart."""
    figure = go.Figure()
    figure.add_annotation(text=message, showarrow=False, font=dict(size=14, color="#667085"))
    figure.update_layout(**{**LAYOUT_DEFAULTS, "height": 220},
                         xaxis=dict(visible=False), yaxis=dict(visible=False))
    return figure


def discrepancy_type_chart(by_type: list[dict]) -> go.Figure:
    """Horizontal bars: money at risk by discrepancy type.

    Horizontal because type labels are words ('Duplicate payment'), and vertical bars turn
    those into unreadable 45-degree text.
    """
    if not by_type:
        return _empty("No discrepancies found")

    rows = sorted(by_type, key=lambda row: _money(row.get("amount_at_risk")))
    figure = go.Figure(go.Bar(
        x=[_money(row.get("amount_at_risk")) for row in rows],
        y=[row.get("label", row.get("type", "?")) for row in rows],
        orientation="h",
        marker_color=[
            TYPE_COLOR if row.get("is_financial", True) else NON_FINANCIAL_COLOR for row in rows
        ],
        # The count lives in the hover text, so one chart answers both 'how much' and
        # 'how many' without a second axis.
        customdata=[[row.get("count", 0)] for row in rows],
        hovertemplate="<b>%{y}</b><br>At risk: %{x:,.2f}<br>Orders: %{customdata[0]}<extra></extra>",
        text=[f"{_money(row.get('amount_at_risk')):,.2f}" for row in rows],
        textposition="outside",
        cliponaxis=False,
    ))
    figure.update_layout(**LAYOUT_DEFAULTS, title="Money at risk by discrepancy type",
                         xaxis_title=None, yaxis_title=None)
    return figure


def discrepancy_count_chart(by_type: list[dict]) -> go.Figure:
    """Order counts by type -- the companion view to the money chart above."""
    if not by_type:
        return _empty("No discrepancies found")

    rows = sorted(by_type, key=lambda row: row.get("count", 0))
    figure = go.Figure(go.Bar(
        x=[row.get("count", 0) for row in rows],
        y=[row.get("label", row.get("type", "?")) for row in rows],
        orientation="h",
        marker_color=[
            TYPE_COLOR if row.get("is_financial", True) else NON_FINANCIAL_COLOR for row in rows
        ],
        hovertemplate="<b>%{y}</b><br>Findings: %{x}<extra></extra>",
        text=[row.get("count", 0) for row in rows],
        textposition="outside",
        cliponaxis=False,
    ))
    figure.update_layout(**LAYOUT_DEFAULTS, title="Number of findings by type",
                         xaxis_title=None, yaxis_title=None)
    return figure


def severity_chart(by_severity: list[dict]) -> go.Figure:
    """Donut by severity. A donut (not a pie) leaves room for the total in the centre --
    the headline number and its breakdown in a single glance."""
    if not by_severity:
        return _empty("No discrepancies found")

    total = sum(row.get("count", 0) for row in by_severity)
    figure = go.Figure(go.Pie(
        labels=[row.get("severity") for row in by_severity],
        values=[row.get("count", 0) for row in by_severity],
        hole=0.58,
        sort=False,   # keep CRITICAL -> LOW order; Plotly would otherwise sort by size
        direction="clockwise",
        marker=dict(colors=[SEVERITY_COLORS.get(row.get("severity", ""), "#98A2B3")
                            for row in by_severity]),
        customdata=[[row.get("amount_at_risk", "0.00")] for row in by_severity],
        hovertemplate="<b>%{label}</b><br>Findings: %{value}<br>At risk: %{customdata[0]}<extra></extra>",
        textinfo="label+value",
    ))
    figure.update_layout(
        **{**LAYOUT_DEFAULTS, "showlegend": False},
        title="Findings by severity",
        annotations=[dict(text=f"<b>{total}</b><br>flagged", x=0.5, y=0.5,
                          font=dict(size=16), showarrow=False)],
    )
    return figure


def risk_direction_chart(summary: dict) -> go.Figure:
    """
    Exposure split by DIRECTION -- the chart a finance lead actually acts on.

    'Money at risk' is not one number: money we are owed goes to collections, money we owe
    goes to refunds, and unknowns go to ops. Different teams, different urgency. Collapsing
    them into a single total would hide the only distinction that changes what happens next.
    """
    keys = ["revenue_at_risk", "customer_owed", "needs_investigation"]
    names = ["REVENUE_AT_RISK", "CUSTOMER_OWED", "NEEDS_INVESTIGATION"]
    values = [_money(summary.get(key)) for key in keys]

    if not any(values):
        return _empty("No exposure -- everything reconciles")

    figure = go.Figure(go.Bar(
        x=[DIRECTION_LABELS[name] for name in names],
        y=values,
        marker_color=[DIRECTION_COLORS[name] for name in names],
        text=[f"{value:,.2f}" for value in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
    ))
    figure.update_layout(**LAYOUT_DEFAULTS, title="Exposure by direction",
                         xaxis_title=None, yaxis_title=None)
    return figure


def reconciliation_gauge(match_rate: float) -> go.Figure:
    """Match-rate gauge.

    The bands are the opinionated part: below 90% is red, because in reconciliation a 10%
    unmatched rate is a serious operational problem, not a 'B grade'. A linear 0-100 scale
    with no bands would let 85% look acceptable.
    """
    figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(match_rate or 0),
        number={"suffix": "%", "font": {"size": 30}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#1D4ED8"},
            "steps": [
                {"range": [0, 90], "color": "#FEE4E2"},    # unacceptable
                {"range": [90, 98], "color": "#FEF0C7"},   # needs work
                {"range": [98, 100], "color": "#D1FADF"},  # healthy
            ],
            "threshold": {"line": {"color": "#B42318", "width": 3},
                          "thickness": 0.8, "value": 98},
        },
        title={"text": "Clean match rate", "font": {"size": 14}},
    ))
    figure.update_layout(**{**LAYOUT_DEFAULTS, "height": 260})
    return figure


def value_comparison_chart(summary: dict) -> go.Figure:
    """Order value vs settled payments vs reconciled vs disputed.

    Deliberately four separate bars rather than a stacked total: they are four different
    measures of the same book, and stacking them would imply they sum to something.
    """
    labels = ["Order value", "Payments settled", "Reconciled", "Disputed"]
    values = [
        _money(summary.get("total_order_value")),
        _money(summary.get("total_payments_settled")),
        _money(summary.get("reconciled_value")),
        _money(summary.get("disputed_value")),
    ]
    colors = ["#1D4ED8", "#0B7285", "#039855", "#B42318"]

    figure = go.Figure(go.Bar(
        x=labels, y=values, marker_color=colors,
        text=[f"{value:,.2f}" for value in values],
        textposition="outside", cliponaxis=False,
        hovertemplate="<b>%{x}</b><br>%{y:,.2f}<extra></extra>",
    ))
    figure.update_layout(**LAYOUT_DEFAULTS, title="Value reconciled vs disputed",
                         xaxis_title=None, yaxis_title=None)
    return figure


def top_orders_chart(priorities: list[dict]) -> go.Figure:
    """The work queue as a chart: biggest exposure first, coloured by severity."""
    if not priorities:
        return _empty("Nothing to prioritise")

    rows = list(reversed(priorities))   # Plotly draws bottom-up; reverse => worst on top
    figure = go.Figure(go.Bar(
        x=[_money(row.get("amount_at_risk")) for row in rows],
        y=[row.get("order_key", "?") for row in rows],
        orientation="h",
        marker_color=[SEVERITY_COLORS.get(row.get("severity", ""), "#98A2B3") for row in rows],
        customdata=[[row.get("label") or row.get("type", ""), row.get("severity", "")] for row in rows],
        hovertemplate=("<b>%{y}</b><br>%{customdata[0]} (%{customdata[1]})"
                       "<br>At risk: %{x:,.2f}<extra></extra>"),
        text=[f"{_money(row.get('amount_at_risk')):,.2f}" for row in rows],
        textposition="outside",
        cliponaxis=False,
    ))
    figure.update_layout(**LAYOUT_DEFAULTS, title="Highest-exposure order references",
                         xaxis_title=None, yaxis_title=None)
    return figure
