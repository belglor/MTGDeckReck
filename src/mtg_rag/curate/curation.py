"""Curation's entry point: theme and pool in, a validated recommendation out.

Ties the stage together. Read the format template ([ADR 0025]), build the prompt
from it and the retrieved candidates, call the injected `LLMClient` ([ADR 0021]),
and validate the raw text into a `list[CuratedCard]` at the boundary ([ADR
0024]). Pipeline order is Plan → Retrieve → Curate; this is the Curate step's
whole surface, mirroring `plan/planner.py`.

The retry policy is the trap this module exists to get right ([ADR 0024]): on
malformed output the model is re-asked exactly once — the cap in `config`, not a
literal here — and then the failure raises. Never a second retry (that hides a
genuinely unreliable model rather than surfacing it to be swapped behind the
seam) and never a degraded recommendation. Curation is terminal, so "never
degrade" protects the reader rather than a later stage: a partial parse would
hand back a deck quietly missing cards, shown as a finished answer with nothing
to signal the gap.
"""

from __future__ import annotations

from collections.abc import Sequence

from mtg_rag.curate.config import MAX_CURATION_RETRIES
from mtg_rag.curate.parse import MalformedRecommendationError, parse_recommendation
from mtg_rag.curate.prompt import CurationCard, build_messages
from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.llm import LLMClient
from mtg_rag.templates_config import TEMPLATE_DIR, TEMPLATE_SUFFIX


def curate(
    request: str,
    *,
    format_name: str,
    cards: Sequence[CurationCard],
    client: LLMClient,
) -> list[CuratedCard]:
    """Curate `cards` for `request` under `format_name`, or raise.

    Reads `<format_name>.md` from the template directory, prompts `client` with
    it, the request, and the candidate pool, and validates the reply into a
    `list[CuratedCard]`. On malformed output the client is re-asked once
    (`MAX_CURATION_RETRIES`); a second malformed reply raises
    `MalformedRecommendationError` ([ADR 0024]).

    `cards` is the pool already hydrated and paired with its `purpose`
    hypotheses; assembling it from the retrieved candidates is the caller's job.
    Its ids are the closed vocabulary the parser checks returned ids against, so
    a card the model invents fails validation here rather than reaching the user.
    """
    template = (TEMPLATE_DIR / f"{format_name}{TEMPLATE_SUFFIX}").read_text(encoding="utf-8")
    messages = build_messages(request, template, cards)
    pool_ids = {card.oracle_id for card in cards}

    # One initial attempt, then up to `MAX_CURATION_RETRIES` re-asks. The final
    # `parse_recommendation` returns its recommendation or raises the malformed
    # failure itself, so every path out of here is a validated recommendation or
    # a loud raise — no degrading.
    text = client.complete(system=messages.system, user=messages.user)
    for _ in range(MAX_CURATION_RETRIES):
        try:
            return parse_recommendation(text, pool_ids=pool_ids)
        except MalformedRecommendationError:
            text = client.complete(system=messages.system, user=messages.user)
    return parse_recommendation(text, pool_ids=pool_ids)
