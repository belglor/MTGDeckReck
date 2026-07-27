"""The two defect rates both model stages need, over plain text.

A plan repeating a query and a recommendation repeating a rationale are the
same measurement over different fields, as are a plan copying its prompt's
example query and a recommendation copying its example rationale. `plan.py` and
`curate.py` name them for their own stage and carry the evidence; the counting
lives here once.

`None` means undefined, never 0.0 — an empty sequence exhibited no defect and
measured nothing, so a zero would put a false point in a table compared across
runs. `evals/metrics.py` follows the same rule.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence


def normalize(text: str) -> str:
    """Casefold and strip, so case and outer padding do not hide a match.

    Deliberately shallow. Going further — stemming, collapsing inner
    whitespace, dropping punctuation — would start ruling that two differently
    worded strings are "really" the same, which is a judgment these checks are
    not allowed to make ([ADR 0026]).
    """
    return text.strip().casefold()


def repeat_rate(texts: Sequence[str]) -> float | None:
    """Share of `texts` that repeat one already in the sequence.

    Counts every occurrence beyond the first, so three copies of one string are
    two repeats rather than one.
    """
    if not texts:
        return None
    normalized = [normalize(text) for text in texts]
    return (len(normalized) - len(set(normalized))) / len(normalized)


def copied_rate(texts: Sequence[str], examples: Collection[str]) -> float | None:
    """Share of `texts` equal to one of `examples` after normalization.

    Exact, never by substring: a string that merely *contains* an example is
    more specific than the example, and counting it would inflate the number
    precisely where the model did better.
    """
    if not texts:
        return None
    copied = {normalize(example) for example in examples}
    return sum(normalize(text) in copied for text in texts) / len(texts)
