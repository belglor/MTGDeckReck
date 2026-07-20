---
status: "accepted"
date: 2026-07-20
---

# Evaluation certifies retrieval recall in two modes; curation and flavor fidelity stay out of automated scope

## Context and Problem Statement

Recall@k on the golden set ([ADR 0006](0006-eval-measures-retrieval-recall.md)) is the only automated signal this system has. Two readings of it are natural and both mislead. The first treats a recall change after an embedding-model swap as a verdict on the new model — "recall went from 0.78 to 0.81, the new model is better." The second treats a green eval as evidence the recommender is good. What does recall@k actually certify, and how does a change to the embedding model, the channel set, or the vector dimension interact with the stored baseline?

## Considered Options

- One longitudinal metric: keep a single stored baseline and compare every later number against it, including across model, channel, and dimension changes
- Two modes: separate *selection* (candidate configurations compared head-to-head, now) from *regression* (a fixed configuration guarded against drift over time), and treat any change to the embedding geometry as a baseline reset
- Close the gap now: build a flavor-quality eval — an LLM judge, or a human-labeled golden set of flavor-driven themes — so the metric covers the requests the product actually exists for

## Decision Outcome

Chosen option: "Two modes", because the single-baseline reading silently compares numbers that are not comparable, and the flavor-quality eval has no defensible ground truth yet.

Recall@k certifies one thing: **mechanical retrieval recall — the floor below which the embeddings are broken.** For a mechanically-determined query ([ADR 0006](0006-eval-measures-retrieval-recall.md)), either the three Urza lands came back in the candidate pool or they did not. That is a real signal about the embedding-and-retrieval layer, and it *is* the embeddings' evaluation — not curation's. Curation is deliberately never measured, because it has no right answer; that is settled in [ADR 0006](0006-eval-measures-retrieval-recall.md) and is not reopened here.

The metric is used two ways, and conflating them is the mistake:

- **Selection.** Run candidate configurations against the same golden set, at the same `k`, under the same filters, at the same time, and compare their recall directly. This is how an embedding model is chosen or a channel weighting is tuned. It is always valid, because everything except the thing under test is held constant by construction.
- **Regression.** Store one configuration's recall as a baseline and compare later numbers against it to catch drift. This is what [ADR 0006](0006-eval-measures-retrieval-recall.md) means by "compared to itself across changes." It is valid only *within a fixed embedding geometry*.

A change to the embedding model, the channel set, or the vector dimension **resets the regression baseline** — the new number opens a new epoch and is recorded as the baseline, not diffed against the old one. This generalizes the specific case already stated in [ADR 0007](0007-multi-channel-embedding.md) (adding or re-embedding a channel invalidates the baseline) to every change that alters the vector geometry. The reason is the one [ADR 0008](0008-rrf-fusion-not-raw-scores.md) gives for refusing to average raw similarities across channels: scores — and the rankings they induce — from different geometries are not commensurable. A model swap replaces every distance in the index at once, so every ranking can reorder for reasons that have nothing to do with retrieval quality. An 0.78→0.81 move across a swap is a single aggregate over ~12 queries, well within noise at that sample size, and cannot distinguish a better model from one that happens to suit these twelve queries; read as an improvement, it invites a false conclusion, and read as a regression it invites a false rollback.

The flavor-quality eval is deferred, not rejected on principle. It is the thing worth having — the golden set is all mechanically-determined themes precisely because those have defensible answers, which means it **under-samples the flavor-driven requests the recommender exists to serve** ([ADR 0006](0006-eval-measures-retrieval-recall.md)). But an LLM judge was already rejected in [ADR 0006](0006-eval-measures-retrieval-recall.md) for making the instrument as variable as the thing measured, and a hand-labeled flavor set has no defensible ground truth — "a spooky graveyard deck" has no expected card list that is a matter of fact rather than taste, and an eval built on an indefensible expected set measures the author's preferences and calls it a metric. Until there is a ground truth that is not taste, flavor fidelity stays a matter for human judgment.

So the automated evaluation contract is: **retrieval recall is certified (mechanically, as a floor); curation quality and flavor-retrieval fidelity are not, by design.**

### Consequences

- Good, because a model swap stops being a paradox — it opens a new regression epoch, and the new model is still evaluated, in selection mode, against the same golden set
- Good, because the eval's limits are legible: a green recall number means retrieval is not broken, not that the recommender is good at the flavor requests it exists for
- Good, because the reset rule is now general — model, channel, and dimension changes are covered by one principle instead of a channel-only consequence bullet
- Bad, because the product's core competence, flavor retrieval, has no automated signal; a regression there is invisible until a human notices it
- Bad, because selection-mode comparisons are only sound if run conditions are held identical (same golden set, `k`, and filters), which is operational discipline the harness must enforce rather than a property the metric guarantees
