"""What a recommendation gets wrong that the corpus can prove ([ADR 0026]).

Five checks over a `list[CuratedCard]`, three of them needing the hydrated rows
to compare a claim against the card it was made about. None of them judges
whether a rationale is a *persuasive* argument — that is the taste [ADR 0006]
leaves to human judgment, and the line [ADR 0026] draws.

`None` means undefined, never 0.0, for an empty recommendation. Curation
selecting nothing is a valid answer ([ADR 0024]) but it exhibits no defect and
measures nothing.

The two checks that compare against a card skip ids the frame does not carry,
mirroring `curate/render.py`'s guard, and divide by what could actually be
checked. A recommendation whose ids have all left the corpus reports `None`
rather than a clean zero it did not earn.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence

import polars as pl

from mtg_rag.corpus_config import ID_COLUMN
from mtg_rag.curate.recommendation import CuratedCard
from mtg_rag.defects.config import CARD_TYPES
from mtg_rag.defects.text import copied_rate, normalize, repeat_rate

#: A rationale asserting the card **is** some type: "is a creature", "is an
#: enchantment", "is a powerful and radiant creature". Up to three words of
#: modifier are allowed between the article and the type, which covers the
#: adjective pile #91 recorded without stretching to a different clause.
#:
#: Deliberately not a bare search for the type word. Measured over the corpus
#: (2026-07-22, 31,284 commander-legal cards with text), 44.8% of cards name a
#: type they are not in their own oracle text — `Sensory Deprivation` is an Aura
#: whose text reads "Enchanted creature gets -1/-1" — so a substring scan would
#: flag any faithful rationale for nearly half the corpus. This pattern's
#: false-positive rate on the same text is 0.26%.
_TYPE_CLAIM = re.compile(rf"\bis an?\s+(?:\w+\s+|and\s+){{0,3}}({'|'.join(CARD_TYPES)})s?\b")


def role_count(recommendation: Sequence[CuratedCard]) -> int | None:
    """How many distinct roles the recommendation uses.

    **Reported, never asserted.** [ADR 0006] holds that nothing is asserted
    about how curation groups, so this ADR 0026 check prints a number with no
    threshold and no target — #91 found 4 of 5 runs putting every card under
    `theme payoff`, and the count makes that visible without anyone declaring
    what a correct number of buckets would be.

    Counted on normalized role text, so `Theme Payoff` and `theme payoff` are
    one job rather than two. That can differ from the number of headings
    `curate/render.py` prints, which groups on the exact string: the question
    here is how many distinct jobs curation named, not how many headings a
    reader would see.
    """
    if not recommendation:
        return None
    return len({normalize(card.role) for card in recommendation})


def duplicate_rationale_rate(recommendation: Sequence[CuratedCard]) -> float | None:
    """Share of rationales repeating one already used in the recommendation.

    #91's angels run stamped one sentence across 20+ of its 30 cards. Exact
    after normalization, so a near-paraphrase counts as distinct — this is a
    lower bound on the repetition #91 describes, not a measure of it.
    """
    return repeat_rate([card.rationale for card in recommendation])


def parroting_rate(
    recommendation: Sequence[CuratedCard], *, examples: Collection[str]
) -> float | None:
    """Share of rationales copied verbatim from the prompt's worked example.

    `examples` is what the curation prompt actually shows
    (`curate.prompt.EXAMPLE_RATIONALES`), passed in rather than imported so this
    stays a pure function of its arguments. #91 found the example's mill
    sentence returned as the top card's rationale in 5 of 5 runs, including
    three where mill had nothing to do with the theme.
    """
    return copied_rate([card.rationale for card in recommendation], examples)


def self_quotation_rate(recommendation: Sequence[CuratedCard], rows: pl.DataFrame) -> float | None:
    """Share of rationales that are quoted from the card's own printed text.

    [ADR 0005] asks the rationale to argue theme fit in a way the player can
    check. Handing back the card's own oracle or flavor text argues nothing —
    #91 recorded `"An apocalypse in dragon form."`, a flavor line, as a
    rationale.

    The whole rationale must appear inside the card's text, not merely overlap
    it: a rationale that *paraphrases* what a card does is doing its job, and
    only wholesale copying is the defect.
    """
    checkable = _by_id(rows)
    scored = [
        _is_quoted(card, checkable[card.oracle_id])
        for card in recommendation
        if card.oracle_id in checkable
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)


def false_type_claim_rate(
    recommendation: Sequence[CuratedCard], rows: pl.DataFrame
) -> float | None:
    """Share of rationales claiming the card is a type the corpus says it is not.

    #91 recorded `Illumination` (Instant), `Holy Avenger` (Artifact — Equipment)
    and `Worn Powerstone` (Artifact) all described as "a powerful and radiant
    creature". A player can see the type on their own screen, so a false claim
    here costs trust in every other rationale in the list.

    A **lower bound**: only an explicit "is a <type>" claim is caught, because
    the looser reading is unusable (see `_TYPE_CLAIM`). A rationale that gets a
    card's type wrong without saying "is a" passes this check.
    """
    checkable = _by_id(rows)
    scored = [
        _claims_wrong_type(card, checkable[card.oracle_id])
        for card in recommendation
        if card.oracle_id in checkable
    ]
    if not scored:
        return None
    return sum(scored) / len(scored)


def _by_id(rows: pl.DataFrame) -> dict[str, dict[str, str]]:
    """The card text each check compares against, keyed by `oracle_id`."""
    return {
        row[ID_COLUMN]: {
            "type_line": row["type_line"] or "",
            "oracle_text": row["oracle_text"] or "",
            "flavor_text": row["flavor_text"] or "",
        }
        for row in rows.iter_rows(named=True)
    }


def _is_quoted(card: CuratedCard, texts: dict[str, str]) -> bool:
    rationale = normalize(card.rationale)
    if not rationale:
        return False
    printed = (normalize(texts["oracle_text"]), normalize(texts["flavor_text"]))
    return any(text and rationale in text for text in printed)


def _claims_wrong_type(card: CuratedCard, texts: dict[str, str]) -> bool:
    type_line = normalize(texts["type_line"])
    claimed = set(_TYPE_CLAIM.findall(normalize(card.rationale)))
    return any(claim not in type_line for claim in claimed)
