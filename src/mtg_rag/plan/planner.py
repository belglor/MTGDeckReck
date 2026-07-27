"""The planner's entry point: theme in, a validated `list[PlannedQuery]` out.

Ties the stage together. Read the format template ([ADR 0023]), build the prompt
([ADR 0021]), call the injected `LLMClient`, and validate the raw text into a
`list[PlannedQuery]` at the boundary ([ADR 0022]). Pipeline order is Plan →
Retrieve → Curate; this is the Plan step's whole surface.

The retry policy is the trap this module exists to get right ([ADR 0022]): on
malformed output the model is re-asked exactly once — the cap in `config`, not a
literal here — and then the failure raises. Never a second retry (that hides a
genuinely unreliable model rather than surfacing it to be swapped behind the
seam) and never a degraded plan (a subtly-wrong plan flows downstream into a pool
no curation can rescue, [ADR 0004]).
"""

from __future__ import annotations

from mtg_rag.llm import LLMClient
from mtg_rag.plan.config import MAX_PLAN_RETRIES, TEMPLATE_DIR, TEMPLATE_SUFFIX
from mtg_rag.plan.parse import MalformedPlanError, parse_plan
from mtg_rag.plan.prompt import build_messages
from mtg_rag.plan.query import PlannedQuery


def plan(request: str, *, format_name: str, client: LLMClient) -> list[PlannedQuery]:
    """Plan the searches for `request` under `format_name`, or raise.

    Reads `<format_name>.md` from the template directory, prompts `client` with
    it and the request, and validates the reply into a `list[PlannedQuery]`. On
    malformed output the client is re-asked once (`MAX_PLAN_RETRIES`); a second
    malformed reply raises `MalformedPlanError` ([ADR 0022]).
    """
    template = (TEMPLATE_DIR / f"{format_name}{TEMPLATE_SUFFIX}").read_text(encoding="utf-8")
    messages = build_messages(request, template)

    # One initial attempt, then up to `MAX_PLAN_RETRIES` re-asks. The final
    # `parse_plan` returns its plan or raises the malformed failure itself, so
    # every path out of here is a validated plan or a loud raise — no degrading.
    text = client.complete(system=messages.system, user=messages.user)
    for _ in range(MAX_PLAN_RETRIES):
        try:
            return parse_plan(text)
        except MalformedPlanError:
            text = client.complete(system=messages.system, user=messages.user)
    return parse_plan(text)
