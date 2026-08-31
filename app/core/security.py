"""
security.py -- Password hashing and validation.

WHAT CHANGED WHEN THE HTTP LAYER WAS REMOVED
  The previous version minted JWTs, because a browser talking to a separate API needs a
  bearer token. There is no separate API any more: Streamlit runs the whole thing in one
  server-side process and identity lives in `st.session_state`, which the browser can
  neither read nor forge -- it only ever holds an opaque session id.

  So the JWT code is gone. Keeping it would have been cargo-culted security: a token the
  app mints, hands to itself, and verifies with its own key adds ceremony, not safety.
  Password hashing is the part that still matters, and it is unchanged.

WHY bcrypt DIRECTLY, NOT passlib
  passlib has been unmaintained since 2020 and its bcrypt backend breaks against modern
  bcrypt releases. The bcrypt package is small, maintained, and this is a dozen lines.
"""

from __future__ import annotations

import base64
import hashlib
import re

import bcrypt

# Cost 12: roughly 250ms per hash on typical hardware. Slow enough to make offline
# brute-forcing expensive, fast enough that a login does not feel broken.
BCRYPT_ROUNDS = 12

MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 256

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AuthError(Exception):
    """Raised for any authentication or credential-validation failure."""


def _prehash(password: str) -> bytes:
    """SHA-256 then base64, before bcrypt.

    bcrypt silently TRUNCATES at 72 bytes. Without this, 'long-passphrase...' and
    'long-passphrase...different-tail' can hash identically -- a real vulnerability that
    punishes users for choosing strong passphrases. Base64 keeps the digest in the ASCII
    range bcrypt expects, and the result is always 44 bytes.
    """
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_prehash(password), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time comparison via bcrypt.checkpw; never raises on malformed input."""
    try:
        return bcrypt.checkpw(_prehash(password), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        # A corrupt or truncated hash in the database must read as "wrong password",
        # not as a 500 that reveals something is broken with this specific account.
        return False


def validate_password_strength(password: str) -> None:
    """Length-first policy, deliberately.

    Composition rules ('one uppercase, one symbol') push people towards Passw0rd! and are
    no longer recommended by NIST. Length is what actually resists guessing.
    """
    if len(password) < MIN_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise AuthError(f"Password must be at most {MAX_PASSWORD_LENGTH} characters.")
    if password.strip() == "":
        raise AuthError("Password cannot be only whitespace.")


def normalise_email(email: str) -> str:
    """Lowercase and trim, so 'Demo@Example.com ' and 'demo@example.com' are one account."""
    return (email or "").strip().lower()


def validate_email(email: str) -> str:
    normalised = normalise_email(email)
    if not _EMAIL_RE.match(normalised):
        raise AuthError("Please enter a valid email address.")
    if len(normalised) > 320:
        raise AuthError("That email address is too long.")
    return normalised
