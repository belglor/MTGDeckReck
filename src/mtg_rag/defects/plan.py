"""What a plan gets wrong that a string comparison can prove ([ADR 0026]).

Pure functions over the query texts alone — no model, no corpus, no store — so
the instrument has no opinions and its tests need no weights. Both checks
answer a question of fact: did the planner ask the same thing twice, and did it
hand back the prompt's own worked example. Neither judges whether a query is a
*good* search, which is the line [ADR 0026] draws and [ADR 0006] drew first.

These take the query strings rather than `PlannedQuery` objects. The typed
field beside them is on its way out ([#97]), and a check about text has no
business depending on the shape of the record carrying it.

`None` means undefined, never 0.0 — an empty plan exhibited neither defect and
measured nothing, so reporting a zero would put a false point in a table meant
to be compared across runs. `evals/metrics.py` follows the same rule.

Comparison is on casefolded, outer-whitespace-stripped text, and no deeper. A
planner writing "Mill" and " mill " has asked the same question twice and
retrieval would fuse the same cards from both. Normalizing further — stemming,
collapsing inner spaces, dropping punctuation — would start deciding that two
differently-worded queries are "really" the same, which is a judgment this
instrument is not allowed to make.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def duplicate_rate(queries: Sequence[str]) -> float | None:
    """Share of a plan's queries that repeat one already in it.

    Counts every occurrence beyond the first: three copies of one query waste
    two of the plan's slots, not one (#72's aristocrats run emitted `creature
    sacrifice` three times). A duplicate costs a real retrieval slot, since the
    same text fuses to the same cards.

    `None` for an empty plan.
    """
    if not queries:
        return None
    normalized = [_normalize(query) for query in queries]
    return (len(normalized) - len(set(normalized))) / len(normalized)


def parroting_rate(queries: Sequence[str], *, examples: Collection[str]) -> float | None:
    """Share of a plan's queries copied verbatim from the prompt's example.

    `examples` is the worked example the planner's prompt actually shows —
    `plan.prompt.EXAMPLE_QUERIES`, passed in rather than imported so this stays
    a pure function of its arguments. #72 records this as the planner's most
    frequent failure: a small model anchoring on the few-shot example instead
    of the theme, inserting `mana rocks` into decks that have no use for it.

    Matching is exact after normalization, never by substring: a query that
    merely *contains* an example ("mana rocks that untap") is a more specific
    search, and flagging it would inflate the number precisely where the
    planner did better.

    `None` for an empty plan.
    """
    if not queries:
        return None
    copied = {_normalize(example) for example in examples}
    return sum(_normalize(query) in copied for query in queries) / len(queries)


def _normalize(text: str) -> str:
    """Casefold and strip, so case and outer padding do not hide a repeat."""
    return text.strip().casefold()
