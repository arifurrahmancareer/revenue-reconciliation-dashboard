"""
dashboard_view.py -- Headline metrics, charts, the work queue, and the drill-down table.

WHY THERE IS NO st.cache_data IN THIS FILE
  The previous version cached HTTP responses, because every widget interaction meant a
  network round trip. Reads are now indexed, page-sized queries against a local session,
  so caching would buy microseconds while introducing the two classic Streamlit bugs:
  stale rows after a re-run, and DetachedInstanceError from ORM objects cached beyond the
  life of their session. Explanations, which ARE expensive, are memoised explicitly in
  `st.session_state["explanations"]`.

STATE THAT MUST SURVIVE A RE-RUN
  Streamlit re-executes this module top-to-bottom on every click, so a variable assigned
  here is gone by the next interaction. Anything that must persist -- which run is being
  viewed, which page of the table, which explanations have been generated -- is written to
  st.session_state by name.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation

import streamlit as st

from ..domain.rules import TYPE_LABELS, DiscrepancyType
from ..services import ingest_service, recon_service
from ..services import explain_service
from .pdf_export import generate_reconciliation_pdf
from . import upload_view

ROWS_PER_PAGE = 10

SEVERITY_BADGE = {
    "CRITICAL": "🔴 CRITICAL",
    "HIGH": "🟠 HIGH",
    "MEDIUM": "🟡 MEDIUM",
    "LOW": "⚪ LOW",
}

DIRECTION_LABELS = {
    "REVENUE_AT_RISK": "Money we are owed",
    "CUSTOMER_OWED": "Money we owe the customer",
    "NEEDS_INVESTIGATION": "Needs investigation",
    "NONE": "No financial impact",
}

TYPE_OPTIONS = ["All types"] + [TYPE_LABELS[t] for t in DiscrepancyType]
TYPE_BY_LABEL = {TYPE_LABELS[t]: t.value for t in DiscrepancyType}
SEVERITY_OPTIONS = ["All severities", "CRITICAL", "HIGH", "MEDIUM", "LOW"]


def _money(value) -> Decimal:
    """Summary money crosses as STRINGS to preserve exact Decimal values in JSON."""
    try:
        return Decimal(str(value or "0"))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def _fmt(value, currency: str = "") -> str:
    amount = _money(value)
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{amount:,.2f}"


# --------------------------------------------------------------------------- header --


def _render_headline(summary: dict, run) -> None:
    st.markdown("#### Headline")

    row_one = st.columns(4)
    row_one[0].metric("Orders", f"{summary.get('total_orders', 0):,}")
    row_one[1].metric("Payment transactions", f"{summary.get('total_payment_transactions', 0):,}")
    row_one[2].metric(
        "Reconciled",
        f"{summary.get('total_reconciled_keys', 0):,}",
        help="Order references where orders and payments agree within tolerance.",
    )
    row_one[3].metric(
        "Flagged",
        f"{summary.get('total_flagged_keys', 0):,}",
        delta=f"{summary.get('match_rate_pct', 0)}% matched",
        delta_color="off",
    )

    row_two = st.columns(4)
    row_two[0].metric("Total order value", _fmt(summary.get("total_order_value")))
    row_two[1].metric("Payments settled", _fmt(summary.get("total_payments_settled")))
    row_two[2].metric(
        "Reconciled value", _fmt(summary.get("reconciled_value")),
        help="Value on order references with no discrepancy.",
    )
    row_two[3].metric(
        "Disputed value", _fmt(summary.get("disputed_value")),
        help="Order value sitting on references that carry at least one finding.",
    )

    # Money at risk is deliberately split three ways: the total alone hides the only
    # distinction that changes who acts on it.
    st.markdown("#### Where the exposure sits")
    row_three = st.columns(4)
    row_three[0].metric("Money at risk", _fmt(summary.get("money_at_risk")))
    row_three[1].metric("We are owed", _fmt(summary.get("revenue_at_risk")))
    row_three[2].metric("We owe customers", _fmt(summary.get("customer_owed")))
    row_three[3].metric("To investigate", _fmt(summary.get("needs_investigation")))

    config = run.config_json or {}
    st.caption(
        f"Run #{run.id} · engine v{run.engine_version} · {run.duration_ms} ms · "
        f"tolerance ±{config.get('amount_tolerance', '0.05')} · "
        f"duplicate window {config.get('duplicate_window_hours', 24)}h · "
        f"settlement lag {config.get('max_settlement_lag_days', 7)}d · "
        f"{run.created_at:%d %b %Y %H:%M}"
    )


# --------------------------------------------------------------------------- charts --


def _render_charts(summary: dict, priorities: list[dict]) -> None:
    from . import charts

    st.markdown("#### Breakdown")

    top_left, top_right = st.columns(2)
    top_left.plotly_chart(
        charts.discrepancy_type_chart(summary.get("by_type", [])), width='stretch'
    )
    top_right.plotly_chart(
        charts.discrepancy_count_chart(summary.get("by_type", [])), width='stretch'
    )

    mid_left, mid_right = st.columns(2)
    mid_left.plotly_chart(
        charts.severity_chart(summary.get("by_severity", [])), width='stretch'
    )
    mid_right.plotly_chart(charts.risk_direction_chart(summary), width='stretch')

    low_left, low_right = st.columns(2)
    low_left.plotly_chart(
        charts.reconciliation_gauge(float(summary.get("match_rate_pct", 0) or 0)),
        width='stretch',
    )
    low_right.plotly_chart(charts.value_comparison_chart(summary), width='stretch')

    if priorities:
        st.plotly_chart(charts.top_orders_chart(priorities), width='stretch')


# ------------------------------------------------------------------------ AI digest --


def _render_digest(db, user, run) -> None:
    digest_key = f"digest_{run.id}"

    with st.container(border=True):
        header, action = st.columns([3, 1])
        header.markdown("**AI run digest**")
        header.caption(
            "Summarises findings the engine already produced. It cannot create, suppress "
            "or re-price a finding."
        )
        if action.button("Summarise this run", key=f"digest_btn_{run.id}",
                         width='stretch'):
            with st.spinner("Thinking..."):
                st.session_state[digest_key] = explain_service.explain_run(db, user, run)

        digest = st.session_state.get(digest_key)
        if digest:
            st.markdown(f"**{digest['headline']}**")
            
            if digest.get("themes"):
                st.markdown("**Themes:**")
                for theme in digest.get("themes", []):
                    st.markdown(f"- {theme}")
            
            if digest.get("priorities"):
                st.markdown("**Next steps:**")
                for priority in digest.get("priorities", []):
                    st.markdown(f"- {priority}")
            
            source = digest.get("source", "fallback")
            if source == "fallback":
                st.caption(
                    "Generated deterministically from the findings (no model configured "
                    "or the model was unavailable)."
                )
            else:
                st.caption(
                    f"Generated by `{digest.get('model')}` at temperature "
                    f"{digest.get('temperature')}. {digest.get('disclaimer', '')}"
                )


# -------------------------------------------------------------------- work queue -----


def _render_priorities(priorities: list[dict]) -> None:
    if not priorities:
        return
    with st.expander("Do these first", expanded=False):
        st.caption(
            "Ordered by severity, then by money. Severity ranks above amount on purpose: "
            "a completed order that was never paid outranks a larger currency mismatch "
            "that is merely unverified."
        )
        st.dataframe(
            [
                {
                    "Order": row["order_key"],
                    "Type": row.get("label") or row.get("type"),
                    "Severity": row.get("severity"),
                    "Direction": DIRECTION_LABELS.get(
                        row.get("risk_direction", ""), row.get("risk_direction", "")
                    ),
                    "At risk": _fmt(row.get("amount_at_risk"), row.get("currency") or ""),
                    "Summary": row.get("summary"),
                }
                for row in priorities
            ],
            width='stretch',
            hide_index=True,
        )


# ------------------------------------------------------------------- drill-down ------


def _render_explanation(payload: dict) -> None:
    st.markdown(f"**What likely happened.** {payload.get('what_happened', '')}")
    if payload.get("likely_cause"):
        st.markdown(f"**Likely cause.** {payload['likely_cause']}")
    st.markdown(f"**Recommended action.** {payload.get('recommended_action', '')}")

    bits = []
    if payload.get("owner_team"):
        bits.append(f"Owner: {payload['owner_team']}")
    if payload.get("confidence"):
        bits.append(f"Confidence: {payload['confidence']}")
    source = payload.get("source", "fallback")
    if source == "openai":
        bits.append(f"{payload.get('model')} @ temp {payload.get('temperature')}")
        if payload.get("was_repaired"):
            bits.append("response needed repair")
    elif source == "cache":
        bits.append("cached")
    else:
        bits.append("deterministic fallback — no model was used")
    if payload.get("latency_ms"):
        bits.append(f"{payload['latency_ms']} ms")
    st.caption(" · ".join(bits))


def _render_row(db, user, run, item, already_explained: bool) -> None:
    badge = SEVERITY_BADGE.get(item.severity, item.severity)
    title = (
        f"{badge}  ·  **{item.order_key}**  ·  "
        f"{TYPE_LABELS.get(DiscrepancyType(item.discrepancy_type), item.discrepancy_type)}"
        f"  ·  {_fmt(item.amount_at_risk, item.currency or '')} at risk"
    )

    with st.expander(title, expanded=False):
        st.markdown(item.summary)
        st.caption(item.detail)

        facts = st.columns(4)
        facts[0].metric("Expected", _fmt(item.expected_amount))
        facts[1].metric("Collected", _fmt(item.collected_amount))
        facts[2].metric("Delta", _fmt(item.delta_amount))
        facts[3].metric(
            "Direction",
            DIRECTION_LABELS.get(item.risk_direction, item.risk_direction),
        )

        st.caption(
            f"Rule `{item.rule_id}` · order status: {item.order_status or 'n/a'} · "
            f"customer: {item.customer_email_masked or '(none)'}"
        )

        if item.evidence_json:
            with st.popover("Evidence"):
                st.json(item.evidence_json)

        explanations = st.session_state.setdefault("explanations", {})
        cached = explanations.get(item.id)

        buttons = st.columns([1, 1, 3])
        clicked = buttons[0].button(
            "Explain with AI", key=f"explain_btn_{item.id}", width='stretch'
        )
        refresh = False
        if cached or already_explained:
            refresh = buttons[1].button(
                "Regenerate", key=f"explain_refresh_{item.id}", width='stretch'
            )

        if clicked or refresh:
            with st.spinner("Thinking..."):
                # explain_discrepancy never raises for model reasons: a timeout, a bad key
                # or unparseable JSON all resolve to the deterministic explanation.
                explanations[item.id] = explain_service.explain_discrepancy(
                    db, user, item, run, refresh=refresh
                )
            cached = explanations[item.id]

        if cached is None and already_explained:
            cached = explain_service.get_cached_explanation(db, user, item.id)
            if cached:
                explanations[item.id] = cached

        if cached:
            st.divider()
            _render_explanation(cached)


def _render_table(db, user, run) -> None:
    st.markdown("#### Every discrepancy")

    filters = st.columns([2, 2, 4, 2])

    type_label = filters[0].selectbox("Type", TYPE_OPTIONS, key="filter_type")
    severity = filters[1].selectbox("Severity", SEVERITY_OPTIONS, key="filter_severity")
    search = filters[2].text_input(
        "Search", key="filter_search", placeholder="Order reference, rule id, or wording"
    )
    primary_only = filters[3].toggle(
        "Primary only", value=True, key="filter_primary",
        help="One row per order: the finding that carries the money. Turn off to see "
             "secondary findings, which are always worth 0.00.",
    )

    # Changing a filter must reset paging, otherwise a user on page 5 of 5 sees an empty
    # table after narrowing the results.
    signature = (type_label, severity, search, primary_only, run.id)
    if st.session_state.get("_filter_signature") != signature:
        st.session_state["_filter_signature"] = signature
        st.session_state["drill_page"] = 0

    page = st.session_state.get("drill_page", 0)

    items, total = recon_service.query_discrepancies(
        db, user, run.id,
        discrepancy_type=TYPE_BY_LABEL.get(type_label),
        severity=None if severity.startswith("All") else severity,
        search=search or None,
        primary_only=primary_only,
        limit=ROWS_PER_PAGE,
        offset=page * ROWS_PER_PAGE,
    )

    if total == 0:
        st.success("No discrepancies match these filters.")
        return

    last_page = max(0, (total - 1) // ROWS_PER_PAGE)
    st.caption(
        f"{total} finding(s) · showing {page * ROWS_PER_PAGE + 1}–"
        f"{min((page + 1) * ROWS_PER_PAGE, total)}"
    )

    already = explain_service.explained_discrepancy_ids(db, user, run.id)
    for item in items:
        _render_row(db, user, run, item, already_explained=item.id in already)

    nav = st.columns([1, 1, 6])
    if nav[0].button("Previous", disabled=page <= 0, width='stretch'):
        st.session_state["drill_page"] = max(0, page - 1)
        st.rerun()
    if nav[1].button("Next", disabled=page >= last_page, width='stretch'):
        st.session_state["drill_page"] = min(last_page, page + 1)
        st.rerun()


# ------------------------------------------------------------------- sensitivity -----


def _render_sensitivity(db, user, run) -> None:
    with st.expander("Re-run with different thresholds", expanded=False):
        st.caption(
            "The thresholds are arguments, not constants. Re-running proves the output is "
            "stable: on this dataset any tolerance between 0.03 and about 18.00 produces "
            "identical results, because the smallest genuine error is 18.50 and the "
            "rounding noise is 0.01–0.02."
        )
        config = run.config_json or {}
        with st.form("rerun_form"):
            cols = st.columns(3)
            tolerance = cols[0].text_input(
                "Amount tolerance", value=str(config.get("amount_tolerance", "0.05"))
            )
            window = cols[1].number_input(
                "Duplicate window (hours)", min_value=1, max_value=720,
                value=int(config.get("duplicate_window_hours", 24)),
            )
            lag = cols[2].number_input(
                "Max settlement lag (days)", min_value=1, max_value=90,
                value=int(config.get("max_settlement_lag_days", 7)),
            )
            submitted = st.form_submit_button("Re-run reconciliation")

        if submitted:
            batch = ingest_service.get_batch(db, user, run.batch_id)
            if batch is None:
                st.error("That upload no longer exists.")
                return
            try:
                overrides = recon_service.parse_config_overrides(
                    amount_tolerance=tolerance,
                    duplicate_window_hours=int(window),
                    max_settlement_lag_days=int(lag),
                )
            except ValueError as exc:
                # Bounds exist so the tolerance stays a sensitivity control and does not
                # become a switch for turning the product off.
                st.error(str(exc))
                return

            with st.spinner("Reconciling..."):
                new_run = recon_service.run_reconciliation(db, user, batch, overrides)
            st.session_state["active_run_id"] = new_run.id
            st.session_state["explanations"] = {}
            # Reset pagination when re-running with different thresholds
            st.session_state["drill_page"] = 0
            st.session_state["_filter_signature"] = None
            st.rerun()


# ------------------------------------------------------------------------- render ----


def _resolve_run(db, user):
    """Which run is on screen: the one just produced, else the newest one."""
    run_id = st.session_state.get("active_run_id")
    if run_id:
        run = recon_service.get_run(db, user, run_id)
        if run is not None:
            return run
        st.session_state["active_run_id"] = None      # deleted underneath us
    return recon_service.get_latest_run(db, user)


def render(db, user) -> None:
    run = _resolve_run(db, user)
    if run is None:
        upload_view.render(db, user)
        return

    st.title("Reconciliation dashboard")

    runs = recon_service.list_runs(db, user, limit=20)
    if len(runs) > 1:
        labels = {
            f"Run #{r.id} · {r.created_at:%d %b %H:%M} · ±{(r.config_json or {}).get('amount_tolerance', '0.05')}": r.id
            for r in runs
        }
        current = next((label for label, rid in labels.items() if rid == run.id), None)
        chosen = st.selectbox(
            "Run", list(labels), index=list(labels).index(current) if current else 0
        )
        if labels[chosen] != run.id:
            st.session_state["active_run_id"] = labels[chosen]
            # Reset pagination state when switching runs
            st.session_state["explanations"] = {}
            st.session_state["drill_page"] = 0
            st.session_state["_filter_signature"] = None
            st.rerun()

    summary = run.summary_json or {}
    priorities = recon_service.run_priorities(db, user, run.id, limit=10)

    _render_headline(summary, run)

    # PDF Download Button
    try:
        pdf_bytes = generate_reconciliation_pdf(run, summary, priorities)
        st.download_button(
            label="📥 Download Executive Summary (PDF)",
            data=pdf_bytes,
            file_name=f"reconciliation_report_run_{run.id}.pdf",
            mime="application/pdf",
            width='content',
            help="Export headline KPIs and priority action items as a PDF report.",
        )
    except Exception as e:
        st.caption(f"PDF generator unavailable: {e}")

    dq_count = summary.get("data_quality_issue_count", 0)
    if dq_count:
        st.caption(
            f"{dq_count} data-quality issue(s) recorded during parsing "
            f"({summary.get('data_quality_rows_dropped', 0)} row(s) dropped). "
            "See the Upload data page for the full audit list."
        )

    st.divider()
    _render_digest(db, user, run)
    _render_priorities(priorities)

    st.divider()
    _render_charts(summary, priorities)

    st.divider()
    _render_table(db, user, run)

    st.divider()
    _render_sensitivity(db, user, run)
