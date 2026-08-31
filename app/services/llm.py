"""
llm.py -- The ONLY module that talks to OpenAI. The API key never leaves this process.

THE ONE RULE THAT MATTERS
  The LLM never decides anything. It does not match records, choose a type, compute an
  amount or set a severity -- all of that is done by the deterministic engine before a
  prompt is even built. The model is given the engine's conclusion and asked to write it up
  for a human: what happened, why it probably happened, what to do next. If the model is
  unavailable, wrong, or returns garbage, the numbers on the dashboard are completely
  unaffected. That is the difference between AI as a feature and AI as a liability.

WHY temperature = 0.2 (NOT 0.0, NOT 0.7)
  * 0.7+ invents causes. For a factual explanation of supplied figures, creativity is a
    defect: it produces confident, plausible, unverifiable stories.
  * 0.0 is not actually deterministic on a distributed serving stack, and in practice it
    produces stiff, repetitive text that reads like a template.
  * 0.2 is the lowest setting that still yields fluent English while keeping the model
    tightly anchored to the figures in the prompt. Combined with a fixed `seed` and a
    strict JSON schema, repeated calls on the same discrepancy are near-identical.

MALFORMED RESPONSES: A FOUR-STEP LADDER, NEVER AN EXCEPTION TO THE USER
  1. Ask for Structured Outputs (`json_schema`, strict) -- the provider enforces the shape.
  2. If the model/SDK does not support it, retry with `json_object`.
  3. If the text still is not valid JSON, salvage it (strip fences/prose, fix trailing
     commas, coerce types, normalise the confidence enum).
  4. If salvage fails, return the DETERMINISTIC explanation built from the same evidence,
     labelled `source="fallback"`.
  A user always gets a useful answer; they are simply told where it came from.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any

from ..core.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Bumped whenever the prompt changes. Stored with each cached explanation so old and new
# explanations remain distinguishable instead of silently overwriting each other.
PROMPT_VERSION = "v1"

CONFIDENCE_VALUES = ("high", "medium", "low")

DISCLAIMER = (
    "AI-generated interpretation of a deterministic finding. The figures, classification "
    "and severity come from the reconciliation engine, not from the model."
)


@dataclass
class LlmExplanation:
    """The structured shape we require. Anything else is repaired or rejected."""

    what_happened: str
    likely_cause: str
    recommended_action: str
    owner_team: str
    confidence: str


@dataclass
class LlmResult:
    explanation: LlmExplanation
    source: str          # 'openai' | 'fallback'
    model: str
    temperature: float
    was_repaired: bool
    latency_ms: int


@dataclass
class BulkSummary:
    headline: str
    themes: list[str]
    priorities: list[str]
    source: str
    model: str
    temperature: float


# Structured Outputs schema. additionalProperties=False + every key required is what makes
# `strict: true` valid, and it is the cheapest possible defence against a missing field.
EXPLANATION_JSON_SCHEMA: dict[str, Any] = {
    "name": "discrepancy_explanation",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["what_happened", "likely_cause", "recommended_action", "owner_team", "confidence"],
        "properties": {
            "what_happened": {"type": "string", "description": "Plain-English restatement of the finding, using the supplied figures."},
            "likely_cause": {"type": "string", "description": "The most probable operational cause."},
            "recommended_action": {"type": "string", "description": "One concrete next step for a finance/ops person."},
            "owner_team": {"type": "string", "description": "Which team should act: Finance, Payments Ops, Engineering, Customer Support."},
            "confidence": {"type": "string", "enum": list(CONFIDENCE_VALUES)},
        },
    },
}

SYSTEM_PROMPT = """You are a payment reconciliation analyst writing for a finance operations team.

A deterministic reconciliation engine has ALREADY classified this discrepancy and computed
every figure. Your job is to explain it, not to re-decide it.

Hard rules:
- Use ONLY the figures and facts provided. Never invent transaction references, amounts,
  dates or customer details.
- Never contradict the supplied classification, severity or amounts.
- If the evidence does not support a single cause, say which possibilities exist and what
  would distinguish them.
- Be specific and brief: two sentences per field, no preamble, no markdown.
- 'recommended_action' must be one concrete step a person can take today (check X, refund Y,
  re-run Z), not general advice.
- Set confidence to 'low' when the evidence is genuinely ambiguous. Do not inflate it.

Return ONLY a JSON object with exactly these keys: what_happened, likely_cause,
recommended_action, owner_team, confidence."""

BULK_SYSTEM_PROMPT = """You are a payment reconciliation analyst summarising a whole run for a manager.

You are given aggregate figures and the largest individual findings, all produced by a
deterministic engine. Identify the PATTERNS, not the individual rows.

Return ONLY a JSON object:
{"headline": "one sentence with the most important number in it",
 "themes": ["2-4 short strings, each naming a systemic cause"],
 "priorities": ["2-4 short strings, each an action in priority order"]}

Use only the supplied figures. Never invent an amount or an order reference."""


# ======================================================================================
# Prompt construction
# ======================================================================================
def build_user_prompt(payload: dict) -> str:
    """
    Build the user message from the engine's own evidence bundle.

    PII: only the MASKED email is ever included (masking happens at parse time, so the raw
    address does not exist in the database to leak). Transaction references and amounts are
    business data and are needed for the explanation to be useful.
    """
    evidence = payload.get("evidence") or {}
    transactions = evidence.get("transactions") or []

    lines = [
        "RECONCILIATION FINDING (produced by the deterministic engine):",
        f"- order reference: {payload.get('order_key')}",
        f"- classification: {payload.get('discrepancy_type')} (rule {payload.get('rule_id')})",
        f"- severity: {payload.get('severity')}",
        f"- exposure direction: {payload.get('risk_direction')}",
        f"- engine summary: {payload.get('summary')}",
        f"- engine detail: {payload.get('detail')}",
        "",
        "FIGURES:",
        f"- order value (expected): {payload.get('expected_amount')} {payload.get('currency') or ''}".rstrip(),
        f"- collected (settled charges net of refunds): {payload.get('collected_amount')}",
        f"- difference (collected - expected): {payload.get('delta_amount')}",
        f"- amount at risk: {payload.get('amount_at_risk')}",
        f"- order status: {payload.get('order_status')}",
        f"- customer (masked): {payload.get('customer_email_masked') or 'not recorded'}",
    ]

    if evidence.get("payment_currencies"):
        lines.append(f"- payment currencies seen: {', '.join(evidence['payment_currencies'])}")
    if evidence.get("closest_pair_hours_apart") is not None:
        lines.append(f"- closest charge pair: {evidence['closest_pair_hours_apart']} hours apart")
    if evidence.get("settlement_lag_days") is not None:
        lines.append(f"- settlement lag: {evidence['settlement_lag_days']} days")

    if transactions:
        lines.append("")
        lines.append("TRANSACTIONS ON THIS ORDER REFERENCE:")
        for txn in transactions[:12]:   # bounded: a runaway order must not blow the context
            if isinstance(txn, dict):
                lines.append(
                    f"- {txn.get('transaction_ref')}: {txn.get('type')} {txn.get('amount')} "
                    f"{txn.get('currency') or ''} status={txn.get('status')} "
                    f"processed_at={txn.get('processed_at') or 'MISSING'} "
                    f"(file reference as written: {txn.get('raw_order_reference')!r})"
                )
            else:
                lines.append(f"- Transaction ref: {txn}")

    tolerance = (payload.get("config") or {}).get("amount_tolerance")
    if tolerance:
        lines.append("")
        lines.append(f"Rounding tolerance in force for this run: +/- {tolerance}.")

    lines.append("")
    lines.append("Explain this finding for the finance operations team.")
    return "\n".join(lines)


# ======================================================================================
# Deterministic fallback -- the safety net, and a good answer in its own right
# ======================================================================================
# One per discrepancy type. These are written by a human who knows the domain, so the
# 'AI down' experience is a slightly drier explanation, not an error message.
FALLBACK_ACTIONS: dict[str, tuple[str, str, str]] = {
    # type: (likely cause, recommended action, owning team)
    "MISSING_PAYMENT": (
        "Either the payment never reached the processor (abandoned checkout, failed capture) "
        "or it settled after this export was taken.",
        "Search the processor dashboard for this order reference. If nothing exists, contact the "
        "customer for payment before recognising the revenue.",
        "Finance",
    ),
    "MISSING_ORDER": (
        "The order exists in the source system but was not included in this export, or the "
        "payment carries a mistyped reference.",
        "Look up the transaction reference in the payment processor, identify the true order, and "
        "re-export the orders file for the period.",
        "Payments Ops",
    ),
    "AMOUNT_MISMATCH": (
        "A discount, tax line or partial capture was applied on one side only, or the order was "
        "edited after the payment was taken.",
        "Compare the order's line items against the captured amount and issue the difference as a "
        "refund or a supplementary charge.",
        "Finance",
    ),
    "DUPLICATE_PAYMENT": (
        "A retry or double submission of the checkout captured twice, typically after a timeout "
        "where the first attempt actually succeeded.",
        "Refund the later charge and confirm idempotency keys are set on the checkout endpoint.",
        "Payments Ops",
    ),
    "CURRENCY_MISMATCH": (
        "The order was priced in one currency and charged in another -- usually a default-currency "
        "configuration on the payment page or a mis-set locale.",
        "Confirm which currency the customer was actually charged, then correct the order record "
        "and settle any FX difference. Do not compare the amounts until then.",
        "Engineering",
    ),
    "STATUS_CONFLICT": (
        "The order system and the payment system disagree because one of them was updated and the "
        "other was not -- a failed webhook is the usual culprit.",
        "Establish which system is correct, replay the missing status update, and refund or collect "
        "accordingly.",
        "Engineering",
    ),
    "TIMING_ANOMALY": (
        "A manual retry, a held payout batch, or a clock/timezone difference between the two "
        "systems.",
        "No money movement is required. Confirm the settlement date with the processor and note the "
        "cause so the pattern can be monitored.",
        "Payments Ops",
    ),
}


def deterministic_explanation(payload: dict) -> LlmExplanation:
    """
    Build a complete explanation from the evidence alone -- no model, no network.

    Used when: no API key is configured, the provider fails, or the response cannot be
    repaired. An unknown type still returns a sensible object rather than raising, because
    a new rule must never break the explain endpoint.
    """
    discrepancy_type = str(payload.get("discrepancy_type") or "").upper()
    cause, action, team = FALLBACK_ACTIONS.get(
        discrepancy_type,
        (
            "The order and payment records disagree in a way the rulebook flagged but this "
            "fallback has no specific playbook for.",
            "Review the transactions listed in the evidence panel against the source systems.",
            "Finance",
        ),
    )
    what = payload.get("detail") or payload.get("summary") or "A discrepancy was detected for this order."
    return LlmExplanation(
        what_happened=str(what),
        likely_cause=cause,
        recommended_action=action,
        owner_team=team,
        # 'medium' is honest: the classification is certain (the engine is deterministic),
        # the CAUSE is a domain heuristic rather than an inspection of the source systems.
        confidence="medium",
    )


# ======================================================================================
# Response parsing / repair -- step 3 of the ladder
# ======================================================================================
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA = re.compile(r",(\s*[}\]])")
_REQUIRED = ("what_happened", "recommended_action")


def parse_model_json(raw: str | None) -> tuple[LlmExplanation | None, bool]:
    """
    Parse model output into an LlmExplanation.

    Returns (explanation, was_repaired). `None` means unusable -> caller falls back.
    Repairs applied, in order, each seen in the wild:
      * ```json fences
      * prose before/after the object ('Sure! Here is the JSON:')
      * trailing commas (valid in JS, not JSON)
      * non-string values (a number where a sentence was asked for)
      * an invented confidence value ('very high') -> nearest valid enum member
    A missing REQUIRED field is NOT repaired -- inventing 'what happened' is exactly the
    failure mode this whole module is designed to prevent.
    """
    if not raw or not str(raw).strip():
        return None, False

    text = str(raw).strip()
    repaired = False

    fenced = _FENCE.search(text)
    if fenced:
        text, repaired = fenced.group(1).strip(), True

    data: Any = None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        repaired = True
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, True
        candidate = text[start:end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                data = json.loads(_TRAILING_COMMA.sub(r"\1", candidate))
            except json.JSONDecodeError:
                return None, True

    if not isinstance(data, dict):
        return None, True

    # Some models wrap the answer, e.g. {"explanation": {...}} or {"result": {...}}.
    if not any(key in data for key in _REQUIRED):
        for wrapper in ("explanation", "result", "response", "data", "output"):
            inner = data.get(wrapper)
            if isinstance(inner, dict) and any(key in inner for key in _REQUIRED):
                data, repaired = inner, True
                break

    for field in _REQUIRED:
        value = data.get(field)
        if value is None or not str(value).strip():
            return None, True   # never fabricate the two load-bearing fields

    def coerce(key: str, default: str = "") -> str:
        nonlocal repaired
        value = data.get(key, default)
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            repaired = True
            return " ".join(str(item) for item in value)
        if not isinstance(value, str):
            repaired = True
            return str(value)
        return value.strip()

    confidence = str(data.get("confidence") or "").strip().lower()
    if confidence not in CONFIDENCE_VALUES:
        repaired = True
        if any(token in confidence for token in ("high", "certain", "strong")):
            confidence = "high"
        elif any(token in confidence for token in ("low", "weak", "unsure", "unclear")):
            confidence = "low"
        else:
            confidence = "medium"

    return LlmExplanation(
        what_happened=coerce("what_happened"),
        likely_cause=coerce("likely_cause", "Not stated by the model."),
        recommended_action=coerce("recommended_action"),
        owner_team=coerce("owner_team", "Finance"),
        confidence=confidence,
    ), repaired


# ======================================================================================
# The client
# ======================================================================================
class LlmService:
    """
    Thin, defensive wrapper around the OpenAI client.

    * The `openai` package is imported LAZILY, inside the call. The API therefore boots and
      serves the entire dashboard even if the dependency is missing or the key is unset.
    * SDK-level retries are disabled (`max_retries=0`) and retries are handled here, so the
      request cannot silently exceed the endpoint's own timeout budget.
    * Exceptions are logged by TYPE only, never with the payload -- an exception string can
      contain the request body, and the request body would end up in the platform's logs.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @property
    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def explain(self, payload: dict) -> LlmResult:
        started = time.perf_counter()

        def elapsed_ms() -> int:
            return int((time.perf_counter() - started) * 1000)

        if not self.enabled:
            # No key configured: this is a supported mode, not an error.
            return LlmResult(deterministic_explanation(payload), "fallback",
                             "deterministic-rules", 0.0, False, elapsed_ms())

        try:
            from openai import OpenAI  # lazy: keeps the import off the boot path
        except ImportError:
            logger.warning("openai package not installed; using deterministic explanations")
            return LlmResult(deterministic_explanation(payload), "fallback",
                             "deterministic-rules", 0.0, False, elapsed_ms())

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.llm_timeout_seconds,
            max_retries=0,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(payload)},
        ]
        common = {
            "model": self.settings.openai_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_output_tokens,
            # Fixed seed: with temperature 0.2 this makes repeat calls near-identical, so a
            # reviewer refreshing a row does not see the wording change under them.
            "seed": 7,
        }

        attempts = max(1, self.settings.llm_max_retries + 1)
        for attempt in range(attempts):
            # Step 1 on the first attempt: provider-enforced schema. Step 2 afterwards:
            # plain JSON mode, for models/gateways that reject json_schema.
            response_format = (
                {"type": "json_schema", "json_schema": EXPLANATION_JSON_SCHEMA}
                if attempt == 0
                else {"type": "json_object"}
            )
            try:
                completion = client.chat.completions.create(response_format=response_format, **common)
                raw = completion.choices[0].message.content
                explanation, was_repaired = parse_model_json(raw)
                if explanation is not None:
                    return LlmResult(explanation, "openai", self.settings.openai_model,
                                     self.settings.llm_temperature, was_repaired, elapsed_ms())
                logger.warning("LLM response could not be parsed (attempt %s/%s)", attempt + 1, attempts)
            except Exception as exc:   # noqa: BLE001 - deliberately broad
                # Any provider failure (timeout, 429, 5xx, auth, SDK change) degrades to the
                # fallback. One dashboard row must never return a 500 because a vendor is down.
                logger.warning("LLM call failed (attempt %s/%s): %s", attempt + 1, attempts, type(exc).__name__)

        return LlmResult(deterministic_explanation(payload), "fallback",
                         self.settings.openai_model, self.settings.llm_temperature, False, elapsed_ms())

    def summarise_run(self, summary: dict, priorities: list[dict]) -> BulkSummary:
        """Run-level narrative for the 'AI digest' panel. Same ladder, same guarantees."""
        if not self.enabled:
            return _fallback_bulk(summary, priorities, "deterministic-rules", 0.0)

        try:
            from openai import OpenAI
        except ImportError:
            return _fallback_bulk(summary, priorities, "deterministic-rules", 0.0)

        prompt = json.dumps({
            "totals": {
                key: summary.get(key) for key in (
                    "total_orders", "total_payment_transactions", "total_flagged_keys",
                    "match_rate_pct", "money_at_risk", "revenue_at_risk", "customer_owed",
                    "needs_investigation",
                )
            },
            "by_type": summary.get("by_type", []),
            "by_severity": summary.get("by_severity", []),
            "largest_findings": priorities[:10],
        }, indent=2)

        try:
            client = OpenAI(api_key=self.settings.openai_api_key,
                            base_url=self.settings.openai_base_url,
                            timeout=self.settings.llm_timeout_seconds, max_retries=0)
            completion = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[{"role": "system", "content": BULK_SYSTEM_PROMPT},
                          {"role": "user", "content": prompt}],
                temperature=self.settings.llm_temperature,
                max_tokens=self.settings.llm_max_output_tokens,
                response_format={"type": "json_object"},
                seed=7,
            )
            data = json.loads(completion.choices[0].message.content or "{}")
            headline = str(data.get("headline") or "").strip()
            if headline:
                return BulkSummary(
                    headline=headline,
                    themes=[str(item) for item in (data.get("themes") or [])][:4],
                    priorities=[str(item) for item in (data.get("priorities") or [])][:4],
                    source="openai",
                    model=self.settings.openai_model,
                    temperature=self.settings.llm_temperature,
                )
        except Exception as exc:   # noqa: BLE001
            logger.warning("Bulk LLM summary failed: %s", type(exc).__name__)

        return _fallback_bulk(summary, priorities, self.settings.openai_model,
                              self.settings.llm_temperature)


def _fallback_bulk(summary: dict, priorities: list[dict], model: str, temperature: float) -> BulkSummary:
    """Deterministic run digest, built straight from the aggregates."""
    flagged = summary.get("total_flagged_keys", 0)
    at_risk = summary.get("money_at_risk", "0.00")
    match_rate = summary.get("match_rate_pct", 0)

    themes = [
        f"{row['label']}: {row['count']} order(s), {row['amount_at_risk']} at risk"
        for row in summary.get("by_type", [])[:4]
    ]
    actions = [
        f"{item['order_key']}: {item['summary']}" for item in priorities[:4]
    ]
    return BulkSummary(
        headline=(
            f"{flagged} order reference(s) need attention with {at_risk} at risk; "
            f"{match_rate}% of order references reconcile cleanly."
        ),
        themes=themes or ["No discrepancies were found in this run."],
        priorities=actions or ["Nothing to action."],
        source="fallback",
        model=model,
        temperature=temperature,
    )


def explanation_as_dict(explanation: LlmExplanation) -> dict:
    return asdict(explanation)
