---
status: "accepted"
date: 2026-07-24
---

# An eval case is a corpus predicate measured as lift, not a list of expected cards

## Context and Problem Statement

[ADR 0006](0006-eval-measures-retrieval-recall.md) settled *which layer* the eval measures — the candidate pool, never curation — and [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md) settled *how a number is read* — selection versus regression, with a geometry change opening a new baseline. Neither says what an eval case actually is.

ADR 0006 illustrates one: "urzatron lands" → Urza's Mine, Urza's Power-Plant, Urza's Tower, scored by recall@k. Taken as the only form, that shape has two problems. It fails silently — ADR 0006 spells the land `Urza's Power-Plant`, but the corpus name is **`Urza's Power Plant`**, no hyphen; the hyphenated form is its *type line*. A golden set typed from the ADR's own text scores 2/3 forever and sends the next reader hunting an embedding bug that does not exist. And it costs Magic knowledge per case, recurring on every corpus refresh, which puts the author's taste inside the instrument — the exact failure ADR 0006 excluded taste queries to avoid.

So: what is an eval case, and what number does it produce?

## Considered Options

- Hand-listed expected cards scored by recall@k — ADR 0006's illustration, kept as the only form
- A corpus predicate scored by raw precision@k
- A corpus predicate scored by lift over the predicate's base rate in the constrained corpus
- An LLM judge for the theme-driven requests a predicate cannot express
- Defer the eval until the planner exists, so cases can be written against real queries

## Decision Outcome

Chosen option: "A corpus predicate scored by lift", because it removes card curation from the instrument and produces the only number that is comparable across cases, constraints, and corpus refreshes.

**An eval case is `(query, predicate, one or more constraint sets)`.** The predicate *is* the ground truth, and it is evaluated twice — there is no separate list of cards that should have been retrieved:

| Evaluated over | Gives |
|---|---|
| the constrained corpus, `frame.filter(constraint_expr(constraints, frame))` | `base_rate` — how common the property is among cards retrieval was *allowed* to return |
| the returned pool | `precision@k` — how common it is among cards retrieval *chose* |

`lift = precision@k / base_rate`. The denominator is the constrained corpus and not the whole one; that is load-bearing, and the wrong version fails plausibly rather than loudly (see the second table below).

### Two predicate kinds, and they are not equally solid

**Mechanic predicates** — `keywords ∋ "Madness"` — read Scryfall's own structured field. Whether a card has madness is a matter of fact, closed and machine-checkable, and nobody has to curate a card list or know any Magic to maintain one.

**Theme predicates** — `oracle_text` matching `(?i)graveyard` — are **a proxy, and this ADR would rather say so than imply otherwise.** The predicate also matches "exile target card from a graveyard", which is not a graveyard-theme card. It is admissible anyway because it is a *fixed* ruler: the eval compares the same predicate to itself across runs ([ADR 0011](0011-evaluation-scope-and-baseline-semantics.md)), and a biased ruler still detects movement. It would be inadmissible as an absolute score, and no absolute score is claimed.

### Lift, not precision, because precision is incomparable

Measured against `data/cards.parquet` (34,201 real cards, `corpus_updated_at` 2026-07-22) and the index built from it (Qwen3-Embedding-0.6B, dim 1024), through `retrieve()`, commander + paper, all three channels, `k=25`:

| Query | Constraints | precision@25 | base rate | lift |
|---|---|---|---|---|
| "cards that connive" | commander | 36.0% | 0.2% | **228×** |
| "graveyard recursion" | commander | 72.0% | 14.0% | **5.2×** |
| "spooky graveyard theme" | commander, `colors=W` | 40.0% | 10.1% | **4.0×** |

36% reads as a failure and is a 228× enrichment. And precision fell 72% → 40% when white was forced, which reads as a large regression and is not one — the constrained corpus is simply thinner in graveyard cards, and the pool came back full of white gravedigging. Lift barely moved. Precision is incomparable on three axes at once — between cases, between constraint sets, and across corpus refreshes — and lift is comparable on all three, which is why raw precision was rejected.

**Constraint interaction is therefore lift retention.** A case carries a list of constraint sets; two or more make it a constraint-interaction case, and retention is the tighter run's lift over the looser one's — 0.77 above. That is the mechanical form of "the theme survived the constraint", and it asserts nothing about which cards should have survived it.

### What this certifies, and the shape of the floor

Oracle channel alone, `k=50`, same corpus and index:

| Query | Predicate | lift |
|---|---|---|
| `"madness"` | `keywords ∋ Madness` | 508× |
| `"if you discard this card, cast it for its madness cost"` | " | 373× |
| `"if you discard this card you may cast it from exile"` | " | 20.7× |
| `"cards that reward you for discarding them"` | " | **0×** |
| `"discard cards"` | " | **0×** |
| `"escape"` | `keywords ∋ Escape` | 514× |
| `"cast this card from your graveyard by exiling other cards"` | " | 138× |
| `"cast spells from your graveyard"` | " | **0×** |

Retrieval keys on the mechanic's *name*, not on what the mechanic means. This is not a pool-depth artifact — searched across all three channels, `"discard cards"` finds no madness cards at `k=25`, `k=100`, or `k=300` — and the cards it does return are literally correct, since they say "discard a card" while madness cards lead with the madness cost.

Mechanic cases therefore sit in the hundreds of ×, and that is what [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md)'s "floor below which the embeddings are broken" looks like when nothing is broken. They are saturated *because things work*; a drop from 500× to 50× is the signal. The gap in the lower rows is real and is recorded as its own concern (issue #50), not smuggled into this decision.

### The eval reports; it never fails

No thresholds and no gates, which is [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md)'s "regression signal, not a merge blocker" taken literally rather than softened into a lenient threshold. A number's job is to move and make someone ask why.

Pool invariants — that nothing illegal, off-color, or off-platform reaches the pool — are **not eval cases.** They are deterministic filter behavior with a right answer, and they are already asserted in `tests/test_retrieve_pool.py`. Restating them in a slower, data-dependent place that by construction cannot fail would weaken a check that currently blocks a merge.

### The planner stays out of the loop

An eval case carries its own `PlannedQuery` list, so a pool is a deterministic function of (corpus, index, query, constraints). Fixed synthetic probes are not a compromise here — they are better than realistic queries for this job, because they do not move when the planner changes, and a ruler that changes with the thing measured is the objection [ADR 0006](0006-eval-measures-retrieval-recall.md) raised against an LLM judge. That judge stays rejected for the same reason, and deferring the eval until the planner exists was rejected because it inverts the dependency: without the eval, a planner change has nothing to be measured against.

### Relation to ADRs 0006 and 0011

This ADR **extends both and supersedes neither**, and neither file is edited — they are an immutable log.

ADR 0006's decision is reaffirmed unchanged: assert on the candidate pool, never on curation; only queries whose answer is mechanically determined. A corpus predicate is a *stricter* reading of "mechanically determined" than a hand-typed card list, not a looser one. What this ADR revises is 0006's **illustration** of an expected set as a list of card names, and with it recall@k, which presupposes a closed expected set and has nothing to divide by once the expectation is a predicate.

ADR 0011 carries over whole. Selection versus regression is a statement about how a number is read, not about which number it is, so both modes apply to lift unchanged — and the baseline-reset rule binds harder here, since lift is computed from the index and a geometry change replaces every distance in it at once.

### Consequences

- Good, because a golden set costs no card curation: `keywords` supplies the ground truth, so the set is maintainable by someone who is not a Magic judge and does not rot when the corpus refreshes
- Good, because the `Urza's Power-Plant` class of failure is gone — a predicate that matches nothing is a malformed case that raises, where a mistyped card name silently scored zero forever
- Good, because lift is comparable between cases, between constraint sets, and across corpus refreshes, where precision is comparable across none of them
- Good, because "the theme survived the constraint" becomes a number without anyone deciding which cards a theme is made of, which is the property the recommender is actually judged on
- Good, because pool invariants keep the stronger home: a failing unit test blocks a merge, and an eval by design does not
- Bad, because theme predicates are proxies — `oracle_text ~ graveyard` catches incidental mentions, so a theme case's absolute lift means little and only its movement is informative
- Bad, because mechanic cases are saturated in the hundreds of ×, so they detect a large break and would not notice a subtle ranking degradation
- Bad, because nothing here measures the gap the lower table exposes: a request phrased in a player's words rather than the mechanic's name retrieves at 0×, and the eval as specified reports no number for it
- Bad, because the eval cannot run in CI — it needs the corpus, the index, and the model — so it is a thing a human remembers to run, and an unrun eval measures nothing
