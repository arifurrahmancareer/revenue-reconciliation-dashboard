"""
upload_view.py -- Upload a pair of CSVs, inspect data quality, run the engine.

The view holds no business logic. It reads bytes out of Streamlit's uploader and hands
them to `ingest_service`, exactly as the old HTTP endpoint did -- the parser has always
taken bytes, which is why removing the API layer required no change to it.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..domain.rules import ReconConfig
from ..services import ingest_service, recon_service
from ..services.ingest_service import IngestError


def _clear_dashboard_state() -> None:
    """Clear all dashboard-related session state (filters, pagination, explanations)."""
    st.session_state["explanations"] = {}
    st.session_state["drill_page"] = 0
    st.session_state["_filter_signature"] = None
    st.session_state["filter_type"] = "All types"
    st.session_state["filter_severity"] = "All"
    st.session_state["filter_search"] = ""
    st.session_state["filter_primary"] = True


# AI explanations for data quality issues - deterministic insights for each issue type
_DATA_QUALITY_EXPLANATIONS = {
    "ORDER_REFERENCE_NORMALISED": {
        "what_happened": "The order reference had leading/trailing whitespace or inconsistent casing.",
        "likely_cause": "Manual data entry, export formatting inconsistencies, or system-to-system conversion issues.",
        "recommended_action": "Check the source system for data quality controls. Consider adding trimming/normalization to the export process.",
        "owner_team": "Engineering / Data Quality",
        "confidence": "high",
    },
    "DUPLICATE_ORDER_ROW": {
        "what_happened": "The exact same order row appeared multiple times in the export.",
        "likely_cause": "Export process ran twice, data pipeline reprocessing, or duplicate rows in source system.",
        "recommended_action": "Verify the export timestamp. Check if a recent re-export or data sync caused the duplication.",
        "owner_team": "Data Quality / Engineering",
        "confidence": "high",
    },
    "DUPLICATE_ORDER_ID": {
        "what_happened": "The same order ID appeared with different values (both rows were kept).",
        "likely_cause": "Order was updated between first and second export, or multiple currencies/amounts for one order.",
        "recommended_action": "Reconcile which row is current. If amendment, manually adjust the record to reflect the final state.",
        "owner_team": "Finance / Operations",
        "confidence": "medium",
    },
    "DUPLICATE_TRANSACTION_REF": {
        "what_happened": "A payment transaction reference appeared more than once.",
        "likely_cause": "Duplicate payment processing, export including both pending and settled versions, or system error.",
        "recommended_action": "Contact payments team to verify if duplicate is a system error or legitimate retry. Remove duplicates from the export.",
        "owner_team": "Payments Ops",
        "confidence": "high",
    },
    "MISSING_EMAIL": {
        "what_happened": "An order record has no customer email address.",
        "likely_cause": "Guest checkout, email field was optional, or data corruption during export.",
        "recommended_action": "Optional if guest checkouts are permitted. Consider making email mandatory in the source system to improve reconciliation.",
        "owner_team": "Customer Support / Operations",
        "confidence": "medium",
    },
    "MISSING_DISCOUNT": {
        "what_happened": "A discount amount field was blank and was left unknown rather than assumed zero.",
        "likely_cause": "Field was optional or null in the database. Discounts are complex: null !== zero.",
        "recommended_action": "Check the source system. If discounts are possible, populate the field (use 0 if none). If not, document the assumption.",
        "owner_team": "Finance / Engineering",
        "confidence": "medium",
    },
    "MISSING_PROCESSED_AT": {
        "what_happened": "A payment record has no processed date.",
        "likely_cause": "Payment is pending or the date field was not populated at export time.",
        "recommended_action": "If pending, re-export after settlement. If missing from settled payments, contact payments provider for the actual date.",
        "owner_team": "Payments Ops",
        "confidence": "high",
    },
    "UNPARSEABLE_DATE": {
        "what_happened": "A date field could not be parsed using any known date format.",
        "likely_cause": "Unusual date format in the export, non-Gregorian calendar, or data corruption.",
        "recommended_action": "Check the source system's date format. Standardize to ISO 8601 (YYYY-MM-DD) or common format in the export.",
        "owner_team": "Engineering / Data Quality",
        "confidence": "high",
    },
    "MISSING_AMOUNT": {
        "what_happened": "An amount field was blank or could not be parsed as a number.",
        "likely_cause": "Field was optional, system exported a non-numeric value, or currency/decimal separator mismatch.",
        "recommended_action": "Verify the column uses consistent decimal separators. Ensure all amounts are populated. Check for currency conversion issues.",
        "owner_team": "Finance / Engineering",
        "confidence": "high",
    },
    "ROW_ARITHMETIC_MISMATCH": {
        "what_happened": "An order's arithmetic didn't add up: gross - discount ≠ net.",
        "likely_cause": "Rounding differences, tax included/excluded inconsistency, or data entry error.",
        "recommended_action": "Check the source system for how gross/discount/net are calculated. Tolerance for rounding is typical but should be small (<0.01).",
        "owner_team": "Finance / Operations",
        "confidence": "medium",
    },
    "MISSING_ORDER_REFERENCE": {
        "what_happened": "A payment record has no order reference to link it to an order.",
        "likely_cause": "Orphaned payment, payment made outside an order, or export error.",
        "recommended_action": "Investigate the orphaned payment. If valid, manually link or document as out-of-scope for reconciliation.",
        "owner_team": "Finance / Payments Ops",
        "confidence": "high",
    },
    "MISSING_IDENTIFIER": {
        "what_happened": "A row has no usable identifier (order reference or transaction ID).",
        "likely_cause": "Both identifier fields were empty or null in the source.",
        "recommended_action": "This row cannot be reconciled. Check the source system for why identifier is missing and correct it.",
        "owner_team": "Engineering / Data Quality",
        "confidence": "high",
    },
}

SAMPLES_DIR = Path(__file__).resolve().parents[2] / "data" / "samples"

# Plain-English labels for the machine codes the parser emits. A reviewer should never
# have to look up what MISSING_PROCESSED_AT means.
ISSUE_LABELS = {
    "ORDER_REFERENCE_NORMALISED": "Reference needed cleaning (case or whitespace)",
    "DUPLICATE_ORDER_ROW": "Identical order row appeared twice (collapsed)",
    "DUPLICATE_ORDER_ID": "Same order id with different values (kept both)",
    "DUPLICATE_TRANSACTION_REF": "Same transaction reference appeared twice",
    "MISSING_EMAIL": "Order has no customer email",
    "MISSING_DISCOUNT": "Discount was blank (left unknown, not assumed zero)",
    "MISSING_PROCESSED_AT": "Payment has no processed date",
    "UNPARSEABLE_DATE": "Date could not be parsed in any known format",
    "MISSING_AMOUNT": "Amount was blank or unparseable",
    "ROW_ARITHMETIC_MISMATCH": "gross - discount does not equal net",
    "MISSING_ORDER_REFERENCE": "Payment has no order reference",
    "MISSING_IDENTIFIER": "Row has no usable identifier",
}


def _parse_summary(label: str, result) -> None:
    st.markdown(f"**{label}**")
    cols = st.columns(4)
    cols[0].metric("Rows read", result.rows_read)
    cols[1].metric("Rows loaded", result.rows_loaded)
    cols[2].metric("Dropped", result.rows_dropped)
    cols[3].metric("Duplicates collapsed", result.duplicate_rows_collapsed)


def _run_engine(db, user, batch, config: ReconConfig | None = None) -> None:
    with st.spinner("Reconciling..."):
        run = recon_service.run_reconciliation(db, user, batch, config)
    
    # Update active batch/run and clear all dashboard state
    st.session_state["active_batch_id"] = batch.id
    st.session_state["active_run_id"] = run.id
    _clear_dashboard_state()
    
    st.cache_data.clear()          # stored results changed; cached reads are now stale
    st.rerun()


def _ingest(db, user, orders_bytes, payments_bytes, orders_name, payments_name, label):
    try:
        batch, orders_result, payments_result = ingest_service.ingest_files(
            db, user, orders_bytes, payments_bytes, orders_name, payments_name, label or None
        )
    except IngestError as exc:
        # Expected, user-fixable problems: wrong file, missing column, file too large.
        st.error(str(exc))
        return None
    except Exception as exc:                    # noqa: BLE001 - never show a raw traceback
        st.error(f"Could not read those files: {exc}")
        return None

    st.success(f"Loaded batch **{batch.label}** (#{batch.id}).")
    _parse_summary("Orders", orders_result)
    _parse_summary("Payments", payments_result)
    st.cache_data.clear()
    return batch


def render(db, user) -> None:
    st.title("Upload data")
    st.caption(
        "Two CSVs: an orders export and a payments export. Both are parsed, cleaned and "
        "stored against your account only."
    )

    # ---------------------------------------------------------------- upload form ----
    with st.form("upload_form", clear_on_submit=False):
        col_a, col_b = st.columns(2)
        orders_file = col_a.file_uploader("orders.csv", type=["csv"], key="orders_upload")
        payments_file = col_b.file_uploader("payments.csv", type=["csv"], key="payments_upload")
        label = st.text_input(
            "Label (optional)", placeholder="e.g. April export", max_chars=120
        )
        submitted = st.form_submit_button("Upload and reconcile", width='stretch')

    if submitted:
        if orders_file is None or payments_file is None:
            st.error("Please choose both files.")
        else:
            batch = _ingest(
                db, user,
                orders_file.getvalue(), payments_file.getvalue(),
                orders_file.name, payments_file.name, label,
            )
            if batch is not None:
                _run_engine(db, user, batch)

    # ------------------------------------------------------------- sample dataset ----
    if SAMPLES_DIR.exists():
        st.divider()
        st.markdown("#### No files handy?")
        st.caption(
            "Loads the bundled sample export (184 orders, 187 payments) so the dashboard "
            "can be evaluated in one click."
        )
        if st.button("Load the sample dataset", width='content'):
            try:
                orders_bytes = (SAMPLES_DIR / "orders.csv").read_bytes()
                payments_bytes = (SAMPLES_DIR / "payments.csv").read_bytes()
            except OSError as exc:
                st.error(f"Sample files are missing from this deployment: {exc}")
            else:
                batch = _ingest(
                    db, user, orders_bytes, payments_bytes,
                    "orders.csv", "payments.csv", "Sample dataset",
                )
                if batch is not None:
                    _run_engine(db, user, batch)

    # ------------------------------------------------------------------- batches ----
    st.divider()
    st.markdown("#### Your uploads")

    batches = ingest_service.list_batches(db, user)
    if not batches:
        st.info("Nothing uploaded yet.")
        return

    for batch in batches:
        with st.container(border=True):
            head, actions = st.columns([3, 1])
            with head:
                st.markdown(f"**{batch.label}**  ·  batch #{batch.id}")
                st.caption(
                    f"{batch.orders_rows_loaded} orders · {batch.payments_rows_loaded} payments "
                    f"· uploaded {batch.created_at:%d %b %Y %H:%M}"
                )
                st.caption(f"`{batch.orders_filename}` + `{batch.payments_filename}`")
            with actions:
                if st.button("Reconcile", key=f"recon_{batch.id}", width='stretch'):
                    _run_engine(db, user, batch)
                if st.button("Delete", key=f"del_{batch.id}", width='stretch'):
                    ingest_service.delete_batch(db, user, batch.id)
                    if st.session_state.get("active_batch_id") == batch.id:
                        st.session_state["active_batch_id"] = None
                        st.session_state["active_run_id"] = None
                        _clear_dashboard_state()
                    st.cache_data.clear()
                    st.rerun()

            issues = ingest_service.get_data_quality_issues(db, user, batch.id)
            if issues:
                with st.expander(f"Data quality: {len(issues)} issue(s) recorded"):
                    st.caption(
                        "Every imperfect row is recorded rather than silently fixed, so the "
                        "cleaning is auditable. Only rows marked *dropped* were excluded."
                    )
                    # Group issues by type for better organization
                    by_type = {}
                    for issue in issues:
                        issue_type = issue.code
                        if issue_type not in by_type:
                            by_type[issue_type] = []
                        by_type[issue_type].append(issue)
                    
                    st.divider()
                    for issue_type, issue_list in by_type.items():
                        issue_label = ISSUE_LABELS.get(issue_type, issue_type)
                        dropped_count = sum(1 for i in issue_list if i.dropped)
                        
                        with st.container(border=True):
                            col1, col2, col3 = st.columns([3, 1, 1])
                            with col1:
                                st.markdown(f"**{issue_label}**")
                                st.caption(f"{len(issue_list)} occurrence(s)")
                            with col2:
                                if dropped_count > 0:
                                    st.caption(f"ℹ️ {dropped_count} dropped from reconciliation")
                            with col3:
                                if issue_type in _DATA_QUALITY_EXPLANATIONS:
                                    if st.button(
                                        "📖 Why this happens",
                                        key=f"dq_explain_{batch.id}_{issue_type}",
                                        width='stretch'
                                    ):
                                        exp = _DATA_QUALITY_EXPLANATIONS[issue_type]
                                        st.info(f"**What happened:** {exp['what_happened']}")
                                        st.warning(f"**Likely cause:** {exp['likely_cause']}")
                                        st.success(f"**Recommended action:** {exp['recommended_action']}")
                                        st.caption(f"🎯 Owner: {exp['owner_team']} · 📊 Confidence: {exp['confidence']}")
                            
                            # Show details table for this issue type
                            with st.expander(f"📋 Details ({len(issue_list)} rows)", expanded=False):
                                st.dataframe(
                                    [
                                        {
                                            "File": issue.source,
                                            "Row": issue.source_row,
                                            "Identifier": issue.identifier,
                                            "Dropped": "✓" if issue.dropped else "—",
                                            "Detail": issue.message,
                                        }
                                        for issue in issue_list
                                    ],
                                    width='stretch',
                                    hide_index=True,
                                )
