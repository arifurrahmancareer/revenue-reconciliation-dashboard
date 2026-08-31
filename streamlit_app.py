"""
streamlit_app.py -- The single entry point for the whole application.

WHY THIS FILE IS AT THE REPOSITORY ROOT AND NAMED THIS WAY
  Streamlit Community Cloud asks for a "main file path" and defaults to `streamlit_app.py`
  at the repo root. Matching that convention means the deployment form needs no editing,
  and `streamlit run streamlit_app.py` works locally with no arguments.

ARCHITECTURE IN ONE PARAGRAPH
  There is one process. This script imports the reconciliation engine directly as a Python
  package; there is no HTTP hop, no API client and no second service to deploy or keep
  awake. Secrets live in `st.secrets`, which is read SERVER-SIDE only -- the browser
  receives rendered output, never configuration. Identity lives in `st.session_state`,
  which is server-side per session and cannot be read or forged by the client.

THE THREE STREAMLIT FACTS THAT SHAPE THIS CODE
  1. The entire script re-runs top-to-bottom on every interaction. Anything expensive must
     be cached (`@st.cache_resource` for the engine, `@st.cache_data` for query results).
  2. Nothing survives a re-run except `st.session_state`. Every button result that must
     persist is written there explicitly.
  3. A database session must NOT survive a re-run, so exactly one is opened per run and
     closed in a finally block.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# set_page_config must be the first Streamlit call in the script, before any other st.*
# call anywhere in the import graph -- otherwise Streamlit raises at runtime.
st.set_page_config(
    page_title="Reconciliation Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

from app.core.config import get_settings          # noqa: E402 - must follow set_page_config
from app.db.session import init_db, session_scope  # noqa: E402
from app.services import auth_service              # noqa: E402
from app.ui import auth_view, dashboard_view, upload_view  # noqa: E402
from app.services import ingest_service, recon_service  # noqa: E402


@st.cache_resource(show_spinner=False)
def _bootstrap() -> bool:
    """Create tables once per container, not once per click.

    @st.cache_resource is the right decorator here: the result is a process-wide singleton
    shared by all sessions, unlike @st.cache_data which is for serialisable per-input
    values.
    """
    init_db()
    return True


def _init_state() -> None:
    """Seed every session key exactly once, so no view has to guess whether a key exists."""
    defaults = {
        "user_id": None,
        "user_email": None,
        "active_batch_id": None,
        "active_run_id": None,
        "explanations": {},   # discrepancy_id -> explanation payload, survives re-runs
        # Dashboard filter state (initialized to defaults)
        "filter_type": "All types",
        "filter_severity": "All",
        "filter_search": "",
        "filter_primary": True,
        "drill_page": 0,
        "_filter_signature": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def sign_out() -> None:
    for key in ("user_id", "user_email", "active_batch_id", "active_run_id"):
        st.session_state[key] = None
    # Clear all dashboard state
    st.session_state["explanations"] = {}
    st.session_state["drill_page"] = 0
    st.session_state["_filter_signature"] = None
    st.session_state["filter_type"] = "All types"
    st.session_state["filter_severity"] = "All"
    st.session_state["filter_search"] = ""
    st.session_state["filter_primary"] = True
    # Cached reads are keyed by user id; clearing them stops the next account from seeing
    # the previous account's cached rows.
    st.cache_data.clear()


def _render_sidebar(db, user) -> str:
    settings = get_settings()

    with st.sidebar:
        st.markdown(f"### {settings.app_name}")
        st.caption(f"Signed in as **{user.email}**")
        st.divider()

        # ======================================== Quick Upload Section ========================================
        st.markdown("#### 📤 Quick Upload")
        st.caption("Upload orders and payments CSVs directly from the sidebar")
        
        with st.form("sidebar_upload_form", clear_on_submit=True):
            orders_file = st.file_uploader("📋 orders.csv", type=["csv"], key="sidebar_orders_upload")
            payments_file = st.file_uploader("💳 payments.csv", type=["csv"], key="sidebar_payments_upload")
            label = st.text_input(
                "Label (optional)", placeholder="e.g. April export", max_chars=120, key="sidebar_label"
            )
            submitted = st.form_submit_button("Upload & Reconcile", width='stretch')

        if submitted:
            if orders_file is None or payments_file is None:
                st.error("Please choose both files.")
            else:
                try:
                    # Use the upload view's ingest function to process files
                    batch = upload_view._ingest(
                        db, user,
                        orders_file.getvalue(), payments_file.getvalue(),
                        orders_file.name, payments_file.name, label,
                    )
                    if batch is not None:
                        upload_view._run_engine(db, user, batch)
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

        # ======================================== Data Management ========================================
        st.divider()
        st.markdown("#### ⚙️ Data Management")
        st.caption("Manage your uploaded runs and datasets.")

        col1, col2 = st.columns(2)
        if col1.button("🔄 Reset Data", width='stretch', help="Delete all current batches and restore the clean sample dataset."):
            with st.spinner("Resetting to sample data..."):
                batches = ingest_service.list_batches(db, user)
                for b in batches:
                    ingest_service.delete_batch(db, user, b.id)
                st.session_state["active_batch_id"] = None
                st.session_state["active_run_id"] = None
                upload_view._clear_dashboard_state()
                st.cache_data.clear()

                samples_dir = Path(__file__).resolve().parent / "data" / "samples"
                orders_csv = samples_dir / "orders.csv"
                payments_csv = samples_dir / "payments.csv"
                if orders_csv.exists() and payments_csv.exists():
                    batch = upload_view._ingest(
                        db, user,
                        orders_csv.read_bytes(), payments_csv.read_bytes(),
                        "orders.csv", "payments.csv", "Sample dataset",
                    )
                    if batch is not None:
                        upload_view._run_engine(db, user, batch)
                else:
                    st.rerun()

        if col2.button("🗑️ Clear All", width='stretch', help="Delete all your data and start fresh."):
            with st.spinner("Clearing data..."):
                batches = ingest_service.list_batches(db, user)
                for b in batches:
                    ingest_service.delete_batch(db, user, b.id)
                st.session_state["active_batch_id"] = None
                st.session_state["active_run_id"] = None
                upload_view._clear_dashboard_state()
                st.cache_data.clear()
                st.rerun()

        # ======================================== Status ========================================
        with st.expander("Status", expanded=False):
            st.write(f"**Environment:** {settings.environment}")
            st.write(f"**Database:** {'SQLite' if settings.is_ephemeral_db else 'PostgreSQL'}")
            st.write(f"**AI explanations:** {'OpenAI' if settings.llm_enabled else 'Deterministic fallback'}")
            if settings.llm_enabled:
                st.write(f"**Model:** `{settings.openai_model}` @ temp {settings.llm_temperature}")

        # Configuration warnings are surfaced, not buried in a log the reviewer cannot see.
        for warning in settings.warnings:
            st.warning(warning, icon="⚠️")

        st.divider()
        if st.button("Sign out", width='stretch'):
            sign_out()
            st.rerun()

        st.caption(
            "Findings come from a deterministic rules engine. AI is used only to explain "
            "findings that already exist -- never to decide whether something matches."
        )

    return None


def main() -> None:
    _bootstrap()
    _init_state()

    # ONE session per script run, closed in a finally block inside session_scope.
    with session_scope() as db:
        user = auth_service.get_user(db, st.session_state.get("user_id"))

        if user is None:
            # Not signed in (or the account was deactivated/deleted since last click).
            if st.session_state.get("user_id"):
                sign_out()
            auth_view.render(db)
            return

        _render_sidebar(db, user)

        # Main view - we only have the dashboard now
        dashboard_view.render(db, user)


main()
