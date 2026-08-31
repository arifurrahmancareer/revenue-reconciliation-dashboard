"""
explain_service.py -- AI explanations, with caching and a guaranteed answer.

This replaces the old `api/explain.py` endpoint. The security property it existed to
provide is unchanged and still central:

  THE MODEL KEY NEVER REACHES THE BROWSER.
  `openai` is imported only inside `services/llm.py`, which runs server-side inside the
  Streamlit process. No key is passed to a widget, a chart, a component or a download.
  Grep the `app/ui` package for 'openai' -- there are no hits.

WHAT THE MODEL IS AND IS NOT ALLOWED TO DO
  It receives a finding the deterministic engine has ALREADY made, with its evidence, and
  writes up the likely cause and the recommended action. It cannot create, suppress or
  re-price a finding. Reconciliation output has to be reproducible; prose does not.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..core.config import get_settings
from ..db.models import AiExplanation, Discrepancy, ReconRun, User
from .llm import DISCLAIMER, PROMPT_VERSION, LlmService, explanation_as_dict
from .recon_service import discrepancy_to_llm_payload, run_priorities


def _as_dict(row: AiExplanation, source: str) -> dict:
    """Shape the stored row exactly like a fresh result, so the UI has one code path."""
    return {
        "discrepancy_id": row.discrepancy_id,
        "what_happened": row.what_happened,
        "likely_cause": row.likely_cause,
        "recommended_action": row.recommended_action,
        "owner_team": row.owner_team,
        "confidence": row.confidence,
        "source": source,
        "model": row.model,
        "temperature": float(row.temperature),
        "was_repaired": row.was_repaired,
        "latency_ms": row.latency_ms,
        "disclaimer": DISCLAIMER,
    }


def get_cached_explanation(db: Session, user: User, discrepancy_id: int) -> dict | None:
    """Look up a previous explanation for this row, this user, this model, this prompt.

    `prompt_version` is part of the key on purpose: improving a prompt must invalidate old
    text rather than serving yesterday's wording forever.
    """
    settings = get_settings()
    if not settings.llm_cache_enabled:
        return None

    row = db.scalars(
        select(AiExplanation)
        .where(
            AiExplanation.discrepancy_id == discrepancy_id,
            AiExplanation.user_id == user.id,
            AiExplanation.model == settings.openai_model,
            AiExplanation.prompt_version == PROMPT_VERSION,
        )
        .order_by(AiExplanation.created_at.desc())
    ).first()
    return _as_dict(row, source="cache") if row else None


def explained_discrepancy_ids(db: Session, user: User, run_id: int) -> set[int]:
    """Which rows in this run already have an explanation, so the table can badge them."""
    settings = get_settings()
    rows = db.execute(
        select(AiExplanation.discrepancy_id)
        .join(Discrepancy, Discrepancy.id == AiExplanation.discrepancy_id)
        .where(
            AiExplanation.user_id == user.id,
            Discrepancy.run_id == run_id,
            AiExplanation.model == settings.openai_model,
            AiExplanation.prompt_version == PROMPT_VERSION,
        )
    ).all()
    return {row[0] for row in rows}


def explain_discrepancy(
    db: Session,
    user: User,
    discrepancy: Discrepancy,
    run: ReconRun,
    refresh: bool = False,
) -> dict:
    """Return an explanation for one finding. NEVER raises because of the model.

    Order of operations:
      1. cache hit (unless the user pressed Regenerate)
      2. call the model through LlmService, which itself falls back to deterministic text
         on timeout, bad key, or unrepairable JSON
      3. persist the result so the next click is free
    """
    settings = get_settings()

    if not refresh:
        cached = get_cached_explanation(db, user, discrepancy.id)
        if cached is not None:
            return cached

    payload = discrepancy_to_llm_payload(discrepancy, run)
    result = LlmService(settings).explain(payload)
    explanation = explanation_as_dict(result.explanation)

    # Persist. A cache write must never be the thing that breaks the page, so a storage
    # failure is swallowed after rollback -- the user still gets their answer.
    if settings.llm_cache_enabled:
        try:
            if refresh:
                for stale in db.scalars(
                    select(AiExplanation).where(
                        AiExplanation.discrepancy_id == discrepancy.id,
                        AiExplanation.user_id == user.id,
                        AiExplanation.model == result.model,
                        AiExplanation.prompt_version == PROMPT_VERSION,
                    )
                ):
                    db.delete(stale)
            db.add(
                AiExplanation(
                    user_id=user.id,
                    discrepancy_id=discrepancy.id,
                    model=result.model,
                    prompt_version=PROMPT_VERSION,
                    temperature=result.temperature,
                    what_happened=explanation.get("what_happened") or "",
                    likely_cause=explanation.get("likely_cause"),
                    recommended_action=explanation.get("recommended_action") or "",
                    owner_team=explanation.get("owner_team"),
                    confidence=explanation.get("confidence"),
                    source=result.source,
                    was_repaired=result.was_repaired,
                    latency_ms=result.latency_ms,
                )
            )
            db.commit()
        except Exception:                       # noqa: BLE001 - caching is best-effort
            db.rollback()

    return {
        "discrepancy_id": discrepancy.id,
        **explanation,
        "source": result.source,
        "model": result.model,
        "temperature": result.temperature,
        "was_repaired": result.was_repaired,
        "latency_ms": result.latency_ms,
        "disclaimer": DISCLAIMER,
    }


def explain_run(db: Session, user: User, run: ReconRun, limit: int = 10) -> dict:
    """A short digest of the whole run for the top of the dashboard.

    Deliberately NOT cached: it is one call per click on an explicit button, and the run's
    contents can change underneath it when tolerances are re-run.
    """
    settings = get_settings()
    priorities = run_priorities(db, user, run.id, limit=limit)
    result = LlmService(settings).summarise_run(run.summary_json, priorities)
    return {
        "run_id": run.id,
        "headline": result.headline,
        "themes": list(result.themes),
        "priorities": list(result.priorities),
        "source": result.source,
        "model": result.model,
        "temperature": result.temperature,
        "disclaimer": DISCLAIMER,
    }
