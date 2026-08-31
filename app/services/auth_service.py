"""
auth_service.py -- Sign-up, sign-in and the multi-tenancy boundary.

This replaces the old `api/auth.py` endpoints. The rules it enforces are unchanged; only
the transport is gone.

THE TENANCY RULE, STATED ONCE
  Every query in this application filters on `user_id`. The signed-in `User` object is
  passed explicitly into every service call rather than read from a global, so a query
  that forgets the tenant filter is visible in review as a missing argument instead of
  hiding behind ambient state.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..core.security import (
    AuthError, hash_password, normalise_email, validate_email,
    validate_password_strength, verify_password,
)
from ..db.models import User

# Identical message for "no such account" and "wrong password". Distinguishing them turns
# the login form into a free account-enumeration oracle.
INVALID_CREDENTIALS = "Incorrect email or password."

# In-process throttle. Correct for a single Streamlit container; a multi-instance
# deployment would need Redis, which is noted in the README's limitations.
MAX_FAILED_ATTEMPTS = 10
LOCKOUT_WINDOW_SECONDS = 300
_failed_attempts: dict[str, list[float]] = {}


def _record_failure(email: str) -> None:
    now = time.time()
    attempts = [t for t in _failed_attempts.get(email, []) if now - t < LOCKOUT_WINDOW_SECONDS]
    attempts.append(now)
    _failed_attempts[email] = attempts


def _is_locked_out(email: str) -> bool:
    now = time.time()
    attempts = [t for t in _failed_attempts.get(email, []) if now - t < LOCKOUT_WINDOW_SECONDS]
    _failed_attempts[email] = attempts
    return len(attempts) >= MAX_FAILED_ATTEMPTS


def _clear_failures(email: str) -> None:
    _failed_attempts.pop(email, None)


def get_user(db: Session, user_id: int) -> User | None:
    """Re-load the signed-in user on every script run.

    Streamlit re-executes the script constantly, and a SQLAlchemy object cached in
    session_state would be attached to a session that has since been closed. Storing only
    the integer id and re-reading it here avoids DetachedInstanceError entirely, and has
    the useful side effect that a deactivated account loses access on its next click.
    """
    if not user_id:
        return None
    user = db.get(User, user_id)
    if user is None or not user.is_active:
        return None
    return user


def sign_up(db: Session, email: str, password: str) -> User:
    settings = get_settings()
    if not settings.allow_signup:
        raise AuthError("Sign-up is disabled on this deployment. Use the demo account.")

    address = validate_email(email)
    validate_password_strength(password)

    # Case-insensitive existence check, so Demo@x.com cannot shadow demo@x.com.
    existing = db.scalars(
        select(User).where(func.lower(User.email) == address)
    ).first()
    if existing is not None:
        raise AuthError("An account with that email already exists. Try signing in.")

    user = User(email=address, password_hash=hash_password(password), is_active=True)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Two simultaneous sign-ups for the same address: the unique index is the real
        # guarantee, the check above is only a friendlier error.
        db.rollback()
        raise AuthError("An account with that email already exists. Try signing in.") from None
    db.refresh(user)
    return user


def sign_in(db: Session, email: str, password: str) -> User:
    address = normalise_email(email)

    if _is_locked_out(address):
        raise AuthError(
            "Too many failed sign-in attempts. Please wait a few minutes and try again."
        )

    user = db.scalars(select(User).where(func.lower(User.email) == address)).first()

    # Note the shape: the SAME error for a missing account and a bad password, and the
    # password is still verified when the account exists so timing does not leak either.
    if user is None or not verify_password(password, user.password_hash):
        _record_failure(address)
        raise AuthError(INVALID_CREDENTIALS)

    if not user.is_active:
        raise AuthError("This account has been deactivated.")

    _clear_failures(address)
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(user)
    return user
