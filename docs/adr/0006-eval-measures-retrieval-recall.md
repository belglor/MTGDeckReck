---
status: "accepted"
date: 2026-07-19
---

# Evaluation measures retrieval recall on the candidate pool, not final recommendations

## Context and Problem Statement

Every later change to embeddings, chunking, or retrieval needs a baseline to be measured against, or the system drifts with no way to tell improvement from regression. But this is a *theme* recommender: it is supposed to give different users different cards for the same request, and its curation layer is deliberately interpretive. Asserting that a given request produces a given recommendation would test away the product's whole premise. What can be measured without pinning down what is meant to vary?

## Considered Options

- Assert expected cards appear in the final curated recommendation, end to end
- Assert expected cards appear in the retrieved candidate pool, before any curation
- Score recommendation quality with an LLM judge
- Defer evaluation until the pipeline is complete

## Decision Outcome

Chosen option: "Recall on the candidate pool", because it separates the layer that has a right answer from the layer that does not.

Retrieval is deterministic and mechanical. Given "urzatron lands", either Urza's Mine, Urza's Power-Plant, and Urza's Tower came back in the pool or they did not, and if they did not, the embeddings are broken — no interpretation required. Curation is the opposite: two defensible answers to the same request can share no cards at all, and that is the feature, not a defect.

So the assertion is: **the expected cards appear somewhere in the candidate pool handed to curation.** Nothing is asserted about what curation selects, how it orders, how many it returns, or how it groups. The eval checks that the ingredients reached the kitchen; what gets cooked is deliberately unconstrained.

This constrains the golden set. Queries must be **mechanically determined** — named mechanics and archetypes whose card list is a matter of fact rather than taste:

- "urzatron lands" → Urza's Mine, Urza's Power-Plant, Urza's Tower
- "cards that connive" → the cards printed with connive
- "landfall payoffs", "treasure token generators", "goblin kindred"

Queries whose answer is a matter of taste — "a spooky graveyard deck", "something janky and fun" — are excluded, not because they are unimportant (they are the actual product) but because there is no defensible expected set for them, and an eval built on an indefensible expected set measures the author's preferences and calls it a metric.

An LLM judge was rejected as the baseline: it makes the measuring instrument as variable as the thing measured, and a regression baseline needs to be boring.

The metric is recall@k, compared **to itself** across changes. It is a regression signal, not a quality gate and not a merge blocker — the number's job is to move when embeddings change and to make someone ask why.

### Consequences

- Good, because embedding and retrieval changes get a real before/after signal from the first change onward
- Good, because curation stays free to be personal, varied, and opinionated — the eval never touches it
- Good, because a failure localizes immediately: a recall drop is a retrieval problem, since nothing else is in scope
- Bad, because the half of the system users actually experience — curation quality, explanation quality, whether a deck feels coherent — is not measured here at all, and needs human judgment
- Bad, because the golden set can only cover mechanically-determined themes, so it under-samples the flavor-driven requests the recommender exists for
- Bad, because recall@k is sensitive to the choice of k: setting it near the real pool size makes the metric easy and uninformative, and it should be tightened if it saturates
