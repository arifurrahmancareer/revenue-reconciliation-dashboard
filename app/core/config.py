"""
config.py -- One place where every setting is read, validated and explained.

WHY NOT pydantic-settings ANY MORE
  This app is now a single Streamlit process, and Streamlit has its own secrets mechanism
  (`st.secrets`, backed by `.streamlit/secrets.toml` locally and the Secrets box on
  Streamlit Community Cloud). pydantic-settings only reads environment variables and .env
  files, so it would have ignored the very mechanism the deployment target uses. A small
  explicit resolver is easier to defend than a dependency that half-fits.

RESOLUTION ORDER (first hit wins)
  1. st.secrets      -- how secrets are supplied on Streamlit Community Cloud
  2. os.environ      -- how secrets are supplied in Docker/CI/anything else
  3. the default     -- safe local development values

WHY streamlit IS IMPORTED LAZILY AND DEFENSIVELY
  The domain layer, the tests and `scripts/run_local_recon.py` must run with no Streamlit
  installed and no script context. `import streamlit` at module scope would break all
  three, so it is wrapped and any failure degrades to environment variables.

THE SECRET NEVER REACHES THE BROWSER
  Streamlit executes this file on the SERVER. `st.secrets` is never serialised into the
  page, and no key is passed to a chart, a widget or a component. The browser only ever
  receives rendered output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from functools import lru_cache

DEV_PLACEHOLDER_SECRET = "dev-only-change-me"


def _from_streamlit(key: str) -> str | None:
    """Read one key from st.secrets, tolerating every way this can legitimately fail.

    Streamlit is optional (tests, CLI scripts), `st.secrets` raises if no secrets file
    exists, and nested tables need a flat-key lookup. All of that collapses to "no value".
    """
    try:
        import streamlit as st
    except Exception:                      # streamlit not installed -- CLI or test run
        return None
    try:
        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:                      # no secrets.toml, or no script context
        return None
    return None


def _get(key: str, default: str = "") -> str:
    """st.secrets -> environment -> default."""
    value = _from_streamlit(key)
    if value is None:
        value = os.environ.get(key)
    if value is None:
        return default
    return value.strip()


def _get_bool(key: str, default: bool) -> bool:
    raw = _get(key, "").lower()
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key, "") or default)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key, "") or default)
    except ValueError:
        return default


def _normalise_database_url(raw: str) -> str:
    """Make any Postgres URL usable by SQLAlchemy 2 + psycopg 3.

    Neon, Supabase and Render all hand out `postgres://` or `postgresql://`, and SQLAlchemy
    would then look for psycopg2, which is not installed. Rewriting the scheme here means a
    user can paste the provider's string verbatim and it just works.
    """
    if raw.startswith("postgres://"):
        return "postgresql+psycopg://" + raw[len("postgres://"):]
    if raw.startswith("postgresql://"):
        return "postgresql+psycopg://" + raw[len("postgresql://"):]
    return raw


@dataclass(frozen=True)
class Settings:
    """Immutable, fully resolved configuration."""

    environment: str = "development"
    app_name: str = "Reconciliation Dashboard"

    # --- storage -----------------------------------------------------------------
    database_url: str = "sqlite:///./recon.db"

    # --- auth --------------------------------------------------------------------
    # Salt for the signed "stay signed in" cookie value. Sessions themselves live in
    # server-side session_state, so this is not a bearer token for the API -- there is no
    # API any more. It still must not be the placeholder in production.
    session_secret: str = DEV_PLACEHOLDER_SECRET
    allow_signup: bool = True

    # --- uploads -----------------------------------------------------------------
    max_upload_bytes: int = 10 * 1024 * 1024      # 10 MB, checked before parsing

    # --- LLM ---------------------------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    openai_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2                  # see README section 5
    llm_max_output_tokens: int = 600
    llm_timeout_seconds: float = 20.0
    llm_max_retries: int = 1
    llm_cache_enabled: bool = True

    # --- derived -----------------------------------------------------------------
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in ("production", "prod")

    @property
    def llm_enabled(self) -> bool:
        """An empty key is not an error. The app runs fully without a model; explanations
        simply come back deterministic and are labelled as such in the UI."""
        return bool(self.openai_api_key)

    @property
    def is_ephemeral_db(self) -> bool:
        """True when data will not survive a restart.

        Streamlit Community Cloud gives every app a container with a disposable filesystem,
        so a SQLite file there is wiped on reboot or redeploy. That is fine for a demo and
        fatal for anything real, so the UI says so out loud instead of losing a user's
        uploads silently.
        """
        return self.database_url.startswith("sqlite")

    @property
    def temperature_decimal(self) -> Decimal:
        return Decimal(str(self.llm_temperature))


def _build() -> Settings:
    environment = _get("ENVIRONMENT", "development")
    database_url = _normalise_database_url(_get("DATABASE_URL", "sqlite:///./recon.db"))
    session_secret = _get("SESSION_SECRET", DEV_PLACEHOLDER_SECRET)
    api_key = _get("OPENAI_API_KEY", "") or None
    base_url = _get("OPENAI_BASE_URL", "") or _get("QWEN_API_BASE", "") or None

    temperature = _get_float("LLM_TEMPERATURE", 0.2)
    # Clamp rather than crash: a bad value in a secrets box should not take the app down,
    # but it must not be honoured either.
    temperature = min(max(temperature, 0.0), 1.0)

    warnings: list[str] = []
    if database_url.startswith("sqlite"):
        warnings.append(
            "Using SQLite. On Streamlit Community Cloud the filesystem is disposable, so "
            "uploads and accounts are lost when the app restarts. Set DATABASE_URL to a "
            "Postgres connection string (Neon/Supabase) to make data durable."
        )
    if session_secret == DEV_PLACEHOLDER_SECRET and environment.lower().startswith("prod"):
        warnings.append(
            "SESSION_SECRET is still the development placeholder. Set a real one in "
            "Streamlit secrets."
        )
    if not api_key:
        warnings.append(
            "No OPENAI_API_KEY configured. AI explanations will use the deterministic "
            "fallback and are labelled accordingly."
        )

    return Settings(
        environment=environment,
        database_url=database_url,
        session_secret=session_secret,
        allow_signup=_get_bool("ALLOW_SIGNUP", True),
        max_upload_bytes=_get_int("MAX_UPLOAD_BYTES", 10 * 1024 * 1024),
        openai_api_key=api_key,
        openai_base_url=base_url,
        openai_model=_get("OPENAI_MODEL", "gpt-4o-mini"),
        llm_temperature=temperature,
        llm_max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 600),
        llm_timeout_seconds=_get_float("LLM_TIMEOUT_SECONDS", 20.0),
        llm_max_retries=_get_int("LLM_MAX_RETRIES", 1),
        llm_cache_enabled=_get_bool("LLM_CACHE_ENABLED", True),
        warnings=tuple(warnings),
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached: Streamlit re-runs the whole script on every interaction, so re-reading and
    re-validating configuration on each widget click would be pure waste."""
    return _build()


def generate_secret() -> str:
    """Helper for the README/setup instructions."""
    import secrets
    return secrets.token_urlsafe(48)
