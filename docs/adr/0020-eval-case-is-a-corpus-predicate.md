---
status: "accepted"
date: 2026-07-24
---

# An eval case is a corpus predicate measured as lift, not a list of expected cards

## Context and Problem Statement

[ADR 0006](0006-eval-measures-retrieval-recall.md) settled *which layer* the eval measures — the candidate pool, never curation — and [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md) settled *how a number is read*, selection versus regression. Neither says what a case is.

ADR 0006 illustrates one: "urzatron lands" → Urza's Mine, Urza's Power-Plant, Urza's Tower, scored by recall@k. Taken as the only form, that shape fails silently. ADR 0006 spells the land `Urza's Power-Plant`; the corpus name is **`Urza's Power Plant`**, no hyphen, and the hyphenated form is its *type line*. A golden set typed from the ADR's own text scores 2/3 forever and sends the next reader hunting an embedding bug that does not exist. It also costs Magic knowledge per case, recurring on every corpus refresh, which puts the author's taste inside the instrument — the failure ADR 0006 excluded taste queries to avoid.

So: what is an eval case, and what number does it produce?

## Considered Options

- Hand-listed expected cards scored by recall@k — ADR 0006's illustration, kept as the only form
- A corpus predicate scored by raw precision@k
- A corpus predicate scored by lift over the predicate's base rate in the constrained corpus
- An LLM judge for the theme-driven requests a predicate cannot express
- Defer the eval until the planner exists, so cases can be written against real queries

## Decision Outcome

Chosen option: "A corpus predicate scored by lift", because it removes card curation from the instrument and produces the only number comparable across cases, constraints, and corpus refreshes.

**An eval case is `(query, predicate, one or more constraint sets)`.** The predicate *is* the ground truth, and it is evaluated twice — there is no separate list of cards that should have been retrieved:

| Evaluated over | Gives |
|---|---|
| the constrained corpus, `frame.filter(constraint_expr(constraints, frame))` | `base_rate` — how common the property is among cards retrieval was *allowed* to return |
| the returned pool | `precision@k` — how common it is among cards retrieval *chose* |

`lift = precision@k / base_rate`. The denominator is the constrained corpus and not the whole one; that is load-bearing, and the wrong version fails plausibly rather than loudly.

A case carries its own `PlannedQuery` list, so a pool is a deterministic function of (corpus, index, query, constraints). Whether the planner itself should be evaluated is a separate question this does not answer.

This **extends** ADRs 0006 and 0011 and supersedes neither; neither file is edited. 0006's decision stands, and a predicate is a stricter reading of "mechanically determined" than a typed card list. What it revises is 0006's *illustration*, and with it recall@k, which has nothing to divide by once the expectation is a predicate rather than a closed set. 0011's two modes carry over: they are about how a number is read, not which number it is.

### Two predicate kinds, and they are not equally solid

**Mechanic predicates** — `keywords ∋ "Madness"` — read Scryfall's own structured field. Whether a card has madness is a matter of fact, closed and machine-checkable, and nobody has to curate a card list or know any Magic to maintain one.

**Theme predicates** — `oracle_text` matching `(?i)graveyard` — are **a proxy, and this ADR would rather say so than imply otherwise.** The predicate also matches "exile target card from a graveyard", which is not a graveyard-theme card. It is admissible because it is a *fixed* ruler compared to itself across runs ([ADR 0011](0011-evaluation-scope-and-baseline-semantics.md)), and a biased ruler still detects movement. It would be inadmissible as an absolute score, and none is claimed.

### Lift, not precision, because precision is incomparable

Measured against `data/cards.parquet` (34,201 real cards, `corpus_updated_at` 2026-07-22) and the index built from it (Qwen3-Embedding-0.6B, dim 1024), through `retrieve()`, commander + paper, all three channels, `k=25`:

| Query | Constraints | precision@25 | base rate | lift |
|---|---|---|---|---|
| "cards that connive" | commander | 36.0% | 0.2% | **228×** |
| "graveyard recursion" | commander | 72.0% | 13.95% | **5.16×** |
| " | commander, `colors=W` | 52.0% | 10.06% | **5.17×** |
| " | commander, `colors=B` | 64.0% | 19.68% | **3.25×** |

36% reads as a failure and is a 228× enrichment. Then hold the query fixed and tighten the constraint: forcing white drops precision by twenty points and moves lift by 0.01, because the constrained corpus is simply thinner in graveyard cards. Precision is incomparable between cases, between constraint sets, and across corpus refreshes; lift is comparable on all three.

**Constraint interaction is therefore lift retention**: the tighter run's lift over the looser one's, 1.00 for white above. That is the mechanical form of "the theme survived the constraint", and it asserts nothing about which cards should have survived it. Retention below 1.0 is not automatically a regression — black scores 0.63 because its base rate is already 19.68%, so there is less headroom to enrich, not worse retrieval.

### The eval reports; it never fails

No thresholds and no gates — [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md)'s "regression signal, not a merge blocker" taken literally rather than softened into a lenient threshold.

Pool invariants — that nothing illegal, off-color, or off-platform reaches the pool — are **not eval cases.** They are deterministic filter behavior with a right answer, already asserted in `tests/test_retrieve_pool.py`. Restating them in a slower place that by construction cannot fail would weaken a check that currently blocks a merge.

### Consequences

- Good, because a golden set costs no card curation: `keywords` supplies the ground truth, and a predicate matching nothing raises where a mistyped card name silently scored zero forever
- Good, because lift is comparable on all three axes where precision is comparable on none
- Good, because "the theme survived the constraint" becomes a number without anyone deciding which cards a theme is made of
- Good, because pool invariants keep the stronger home: a failing unit test blocks a merge, and an eval by design does not
- Bad, because theme predicates are proxies, so a theme case's absolute lift means little and only its movement is informative
- Bad, because mechanic cases run saturated in the hundreds of × (figures in issue #50), so they catch a large break and miss a subtle one — and nothing here measures the gap #50 records, where a request in a player's words rather than the mechanic's name retrieves at 0×
- Bad, because the eval cannot run in CI — it needs the corpus, the index, and the model — so it is a thing a human remembers to run, and an unrun eval measures nothing
