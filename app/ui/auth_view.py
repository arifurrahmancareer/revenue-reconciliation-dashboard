"""
auth_view.py -- Sign in / create account.

The view calls `auth_service` directly. There is no token to store: on success the user's
integer id goes into `st.session_state`, which lives on the SERVER for the lifetime of the
browser session. The client holds only an opaque session identifier, so there is nothing
in the browser to tamper with.
"""

from __future__ import annotations

import streamlit as st

from ..core.config import get_settings
from ..core.security import AuthError
from ..services import auth_service

# Published in the README so a reviewer can sign in without creating anything.
DEMO_EMAIL = "demo@example.com"
DEMO_PASSWORD = "ReconDemo2025!"


def _sign_in_success(user) -> None:
    st.session_state["user_id"] = user.id
    st.session_state["user_email"] = user.email
    st.session_state["active_batch_id"] = None
    st.session_state["active_run_id"] = None
    st.session_state["explanations"] = {}
    # Clear cached reads so a previous account's cached rows can never be shown.
    st.cache_data.clear()
    st.rerun()


def render(db) -> None:
    settings = get_settings()

    st.title("💳 Order / Payment Reconciliation")
    st.caption(
        "Upload an orders export and a payments export. The engine matches them, "
        "classifies every discrepancy, and prices the money at risk."
    )

    left, right = st.columns([1, 1], gap="large")

    with left:
        sign_in_tab, sign_up_tab = st.tabs(["Sign in", "Create account"])

        with sign_in_tab:
            with st.form("sign_in_form"):
                email = st.text_input("Email", value="", autocomplete="username")
                password = st.text_input(
                    "Password", type="password", autocomplete="current-password"
                )
                submitted = st.form_submit_button("Sign in", width='stretch')
            if submitted:
                try:
                    user = auth_service.sign_in(db, email, password)
                except AuthError as exc:
                    st.error(str(exc))
                else:
                    _sign_in_success(user)

            if st.button("Use the demo account", width='stretch'):
                try:
                    user = auth_service.sign_in(db, DEMO_EMAIL, DEMO_PASSWORD)
                except AuthError:
                    st.error(
                        "The demo account does not exist yet on this deployment. "
                        "Create an account, or run `python scripts/seed_demo_user.py`."
                    )
                else:
                    _sign_in_success(user)

        with sign_up_tab:
            if not settings.allow_signup:
                st.info("Sign-up is disabled on this deployment. Please use the demo account.")
            else:
                with st.form("sign_up_form"):
                    new_email = st.text_input("Email", key="su_email")
                    new_password = st.text_input(
                        "Password", type="password", key="su_password",
                        help="At least 8 characters. Length beats punctuation.",
                    )
                    confirm = st.text_input("Confirm password", type="password", key="su_confirm")
                    created = st.form_submit_button("Create account", width='stretch')
                if created:
                    if new_password != confirm:
                        st.error("The two passwords do not match.")
                    else:
                        try:
                            user = auth_service.sign_up(db, new_email, new_password)
                        except AuthError as exc:
                            st.error(str(exc))
                        else:
                            _sign_in_success(user)

    with right:
        st.markdown("#### What this does")
        st.markdown(
            "- Normalises messy references (`ord-1801 ` and `ORD-1801` are one order)\n"
            "- Parses two different date formats and tolerates missing dates\n"
            "- Classifies seven discrepancy types with a documented rule for each\n"
            "- Nets refunds against charges before judging an amount\n"
            "- Prices every finding as revenue at risk, owed to the customer, or to investigate\n"
            "- Explains any finding in plain English on demand"
        )
        st.info(
            f"**Demo account**\n\nEmail: `{DEMO_EMAIL}`\n\nPassword: `{DEMO_PASSWORD}`",
            icon="🔑",
        )
        if settings.is_ephemeral_db:
            st.caption(
                "This deployment uses SQLite on a disposable filesystem, so accounts and "
                "uploads reset when the app restarts."
            )
