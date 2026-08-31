"""
test_llm_parsing.py -- Tests for the malformed-LLM-response ladder.

WHY THIS FILE EXISTS: "handle malformed LLM responses gracefully" is easy to claim and hard
to prove. Each test below is a real failure mode of a chat model asked for JSON -- code
fences, a chatty preamble, a trailing comma, a nested wrapper key, an empty field -- and each
one asserts the parser either recovers the content or refuses cleanly. It must never raise,
because a broken model response must not become a 500 on the dashboard.

No network access and no API key are required: `parse_model_json` is a pure function.

Run:  pytest backend/tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

# tests/ now sits at the repository root, alongside the app/ package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.domain.rules import DiscrepancyType  # noqa: E402
from app.services.llm import (  # noqa: E402
    PROMPT_VERSION, deterministic_explanation, explanation_as_dict, parse_model_json,
)

GOOD = (
    '{"what_happened": "Two identical charges were captured 29 minutes apart.",'
    ' "likely_cause": "A retry after a timeout on the first capture.",'
    ' "recommended_action": "Void the second charge and refund the customer.",'
    ' "owner_team": "Payments", "confidence": "high"}'
)


# ======================================================================================
# The happy path
# ======================================================================================
def test_clean_json_needs_no_repair():
    explanation, was_repaired = parse_model_json(GOOD)

    assert explanation is not None
    assert was_repaired is False
    assert explanation.what_happened.startswith("Two identical charges")
    assert explanation.recommended_action.startswith("Void the second charge")
    assert explanation.owner_team == "Payments"
    assert explanation.confidence == "high"


# ======================================================================================
# Recoverable malformations -- each must parse AND report was_repaired=True
# ======================================================================================
def test_code_fenced_json_is_recovered():
    """The most common failure: the model wraps JSON in a markdown block."""
    explanation, was_repaired = parse_model_json(f"```json\n{GOOD}\n```")
    assert explanation is not None
    assert was_repaired is True
    assert explanation.owner_team == "Payments"


def test_bare_fence_is_recovered():
    explanation, was_repaired = parse_model_json(f"```\n{GOOD}\n```")
    assert explanation is not None
    assert was_repaired is True


def test_prose_wrapped_json_is_recovered():
    """'Sure! Here is the JSON you asked for:' -- sliced out from first { to last }."""
    noisy = f"Sure! Here is the analysis you requested:\n\n{GOOD}\n\nLet me know if you need more."
    explanation, was_repaired = parse_model_json(noisy)
    assert explanation is not None
    assert was_repaired is True
    assert explanation.confidence == "high"


def test_trailing_comma_is_repaired():
    """Valid in JavaScript, invalid in JSON, and models emit it regularly."""
    broken = (
        '{"what_happened": "Payment failed but the order shipped.",'
        ' "recommended_action": "Contact the customer for payment.",}'
    )
    explanation, was_repaired = parse_model_json(broken)
    assert explanation is not None
    assert was_repaired is True
    assert explanation.what_happened == "Payment failed but the order shipped."


def test_wrapper_keys_are_unwrapped():
    """Models like to nest the answer under a key of their own choosing."""
    for wrapper in ("explanation", "result", "response", "data", "output"):
        explanation, was_repaired = parse_model_json(f'{{"{wrapper}": {GOOD}}}')
        assert explanation is not None, f"failed to unwrap {wrapper}"
        assert was_repaired is True
        assert explanation.owner_team == "Payments"


def test_optional_fields_may_be_absent():
    """Only what_happened and recommended_action are required -- those two are what the UI
    promises. Everything else degrades to a safe default rather than failing the response."""
    minimal = (
        '{"what_happened": "The order has no matching payment.",'
        ' "recommended_action": "Chase the customer for payment."}'
    )
    explanation, _ = parse_model_json(minimal)

    assert explanation is not None
    # The optional fields are FILLED with honest defaults rather than left empty: the UI
    # renders every field, and a blank 'Likely cause' looks like a bug to the user.
    assert explanation.likely_cause == "Not stated by the model."
    assert explanation.owner_team == "Finance"
    assert explanation.confidence in ("high", "medium", "low")


def test_confidence_is_normalised_not_trusted():
    """An out-of-vocabulary confidence must be coerced, never passed through to the UI."""
    for supplied in ("HIGH", " High ", "very high", "certain", "42", ""):
        payload = (
            '{"what_happened": "x", "recommended_action": "y", "confidence": '
            f'"{supplied}"}}'
        )
        explanation, _ = parse_model_json(payload)
        assert explanation is not None
        assert explanation.confidence in ("high", "medium", "low"), supplied


# ======================================================================================
# Unrecoverable input -- must return None, never raise
# ======================================================================================
def test_unrecoverable_input_returns_none_without_raising():
    for payload in (
        "",                                    # empty response
        "   ",
        "I'm sorry, I can't help with that.",  # a refusal, no JSON at all
        "{not json at all",                    # truncated / broken
        "[1, 2, 3]",                           # a list, not an object
        '{"unexpected": "shape"}',             # object without required fields
        '{"what_happened": ""}',               # present but empty -> treated as missing
        '{"recommended_action": "do a thing"}',  # half the required fields
        "null",
    ):
        explanation, was_repaired = parse_model_json(payload)
        assert explanation is None, f"should not have parsed: {payload!r}"
        assert isinstance(was_repaired, bool)


def test_none_input_is_handled():
    explanation, _ = parse_model_json(None)
    assert explanation is None


# ======================================================================================
# The deterministic fallback -- what the user sees when the model is unavailable
# ======================================================================================
def test_every_discrepancy_type_has_a_fallback():
    """The endpoint must be able to answer for ANY finding without the model, so there is a
    hand-written explanation for every discrepancy type the engine can emit."""
    for discrepancy_type in DiscrepancyType:
        explanation = deterministic_explanation(
            {
                "discrepancy_type": discrepancy_type.value,
                "order_key": "ORD-9999",
                "summary": "Test finding",
                "detail": "Test detail",
            }
        )
        assert explanation.what_happened
        assert explanation.recommended_action
        assert explanation.owner_team
        assert explanation.confidence in ("high", "medium", "low")


def test_fallback_mentions_the_order_it_describes():
    """Generic advice is not useful advice: the fallback is grounded in the actual finding."""
    explanation = deterministic_explanation(
        {
            "discrepancy_type": DiscrepancyType.MISSING_PAYMENT.value,
            "order_key": "ORD-1201",
            "summary": "No payment found for 94.87",
            "detail": "Order ORD-1201 has no payment rows.",
        }
    )
    assert "ORD-1201" in explanation.what_happened + explanation.recommended_action


def test_explanation_serialises_for_the_api():
    """The dict handed to the response model must contain exactly the keys the frontend reads."""
    explanation, _ = parse_model_json(GOOD)
    payload = explanation_as_dict(explanation)

    for key in ("what_happened", "likely_cause", "recommended_action", "owner_team",
                "confidence"):
        assert key in payload


def test_prompt_version_is_pinned():
    """The cache key includes PROMPT_VERSION, so changing a prompt must be a deliberate,
    visible bump rather than silently serving text generated by an older prompt."""
    assert PROMPT_VERSION
    assert isinstance(PROMPT_VERSION, str)
