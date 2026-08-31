"""
session.py -- Engine, session factory, and the per-script-run session scope.

TWO DATABASES, ONE CODEBASE
  SQLite locally (zero setup, one file, `sqlite3 recon.db` to inspect) and Postgres in
  production (concurrent writers, survives redeploys). The only differences are the two
  SQLite-specific settings below; no application code branches on the backend.

STREAMLIT-SPECIFIC CONCERNS
  1. Streamlit re-runs the whole script on every interaction, so the ENGINE must be created
     once and reused. It lives at module scope, and Python's module cache keeps it alive
     across re-runs; creating a new engine per click would leak connection pools until the
     database refused new connections.
  2. Streamlit serves each browser session from a worker thread, hence the SQLite
     same-thread setting below.
  3. A SESSION, unlike the engine, is opened and closed per script run. Holding one open in
     st.session_state across re-runs is the classic cause of stale reads and
     DetachedInstanceError.

WHY create_all AND NOT ALEMBIC
  A deliberate scope call for a take-home. `Base.metadata.create_all()` is a single line
  and cannot corrupt anything on an empty database. For a system that will evolve after
  launch, Alembic is the correct answer -- see 'Known limitations' in the README. Calling
  it out is better than pretending create_all is a migration strategy.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ..core.config import get_settings
from .models import Base

settings = get_settings()
_is_sqlite = settings.database_url.startswith("sqlite")

engine = create_engine(
    settings.database_url,
    # SQLite + threads: Streamlit runs each user session on a worker thread, and SQLite's
    # default same-thread check would reject those connections.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # Managed Postgres closes idle connections; pre_ping replaces a dead one instead of
    # throwing on the first interaction after the app has been idle.
    pool_pre_ping=True,
    # Small pool: one Streamlit container serves a handful of concurrent sessions, and
    # free-tier Postgres plans cap connections aggressively.
    pool_size=5,
    max_overflow=5,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    # expire_on_commit=False: without it, reading an attribute after commit triggers a
    # fresh SELECT -- and after the session closes that becomes a DetachedInstanceError,
    # which in Streamlit shows up as a red traceback halfway down the page.
    expire_on_commit=False,
    class_=Session,
)


if _is_sqlite:
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - infra glue
        """SQLite needs to be told to behave.

        foreign_keys=ON  : OFF by default, so ON DELETE CASCADE would silently do nothing
                           and deleting a batch would leave orphaned discrepancies.
        journal_mode=WAL : readers no longer block the writer, which matters because one
                           browser tab can be reading while another is still ingesting.
        """
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def init_db() -> None:
    """Create tables if they do not exist. Idempotent; safe to call on every startup."""
    Base.metadata.create_all(bind=engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """One database session for one Streamlit script run.

    try/finally guarantees the connection returns to the pool even when a view raises,
    which is what stops a handful of tracebacks from exhausting the pool and bricking the
    app until it is rebooted. Commits stay explicit in the service layer, so a run that
    fails half way through does not leave partial rows behind.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
