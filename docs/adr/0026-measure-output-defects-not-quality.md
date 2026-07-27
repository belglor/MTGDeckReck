---
status: "accepted"
date: 2026-07-27
---

# Plan and recommendation *defects* are measured mechanically; their *quality* stays unmeasured

## Context and Problem Statement

[Issue #72](https://github.com/belglor/MTGDeckReck/issues/72) and [issue #91](https://github.com/belglor/MTGDeckReck/issues/91) record that the planner and curation return structurally valid but poor output — queries copied from the prompt's own example, invented card vocabulary, rationales quoting a card's flavor text, rationales calling an Instant a creature, and role grouping collapsing to one bucket. Three hand-inspected sweeps now exist and nothing measures any of them, so every proposed fix would be judged by eye.

But [ADR 0006](0006-eval-measures-retrieval-recall.md) deliberately put curated output beyond evaluation: "Nothing is asserted about what curation selects, how it orders, how many it returns, or how it groups." Does measuring anything about that output reopen a settled decision?

## Considered Options

- Extend the golden-set eval to cover curation's output
- A separate instrument measuring mechanical defects, leaving quality unmeasured
- Score recommendation quality with an LLM judge
- Keep eyeballing sweeps

## Decision Outcome

Chosen option: **a separate instrument measuring defects only**, because [ADR 0006](0006-eval-measures-retrieval-recall.md) excluded questions of *taste*, and most of what #72 and #91 document is not taste.

ADR 0006's reasoning is that curation has no right answer — "two defensible answers to the same request can share no cards at all, and that is the feature, not a defect." That holds for *which* cards and *why they were chosen*. It does not hold for the failures actually recorded:

- A rationale describing `Illumination` as "a powerful and radiant creature" contradicts the corpus, which says Instant.
- A query term appearing nowhere in 38,328 cards' names, text, types or keywords is invented vocabulary, not an unusual search.
- Output matching the prompt's own worked example verbatim is parroting, whatever the request was.
- The same rationale stamped across 20+ of 30 cards is degenerate repetition.
- The same `query_text` repeated within one plan wastes a retrieval slot and fuses to the same results.

Each has a right answer in exactly the sense ADR 0006 requires. So this measures the layer ADR 0006 called interpretive, on the axes where it is not — which does not reopen the rejection, because the rejection was about taste.

**What stays unmeasured**, and deliberately: whether a query is a *good* search, whether a rationale is a *persuasive* argument, whether a deck *feels* coherent. ADR 0006's stated consequence — that this half "needs human judgment" — stands untouched. A plan can score clean here and still be bad.

**The one check that touches ADR 0006's text is reported, never asserted.** Counting how many distinct roles a recommendation uses is "how it groups", which ADR 0006 says nothing is asserted about. So the count is printed as a number with no threshold, no target and no pass/fail — matching [ADR 0020](0020-eval-case-is-a-corpus-predicate.md)'s existing stance that the instrument reports and never fails a run. #91 found 4 of 5 runs collapsing into a single bucket; the number makes that visible without this ADR declaring what a correct bucket count would be.

**Placement: a module of its own, not `mtg_rag.evals`.** Not to escape the path-scoped rule that governs that package — the argument above is why the rule's reasoning does not reach here — but because the two instruments answer different questions against different inputs, and a reader of `evals/` should not have to work out which one a given check belongs to. The golden set keeps measuring retrieval lift on the candidate pool; this measures output shape.

**An LLM judge is rejected for ADR 0006's own reason**: it makes the measuring instrument as variable as the thing measured, and a regression baseline needs to be boring. Every check here is a string or set operation over the corpus, so the instrument has no opinions and needs no model to run.

### Consequences

- Good, because fixes to #72 and #91 finally get a before/after signal, and a change that improves one failure mode while worsening another becomes visible instead of arguable
- Good, because ADR 0006's core survives intact: taste stays unmeasured and curation stays free to be personal, varied and opinionated
- Good, because every check is a pure function over data already in the corpus, so no test needs a model and the instrument cannot drift the way its subject does
- Good, because a defect check localizes: parroting points at the prompt's example, invented vocabulary points at missing grounding, and the two suggest different fixes
- Bad, because measuring defects is not measuring quality — every number here can improve while the recommendations get worse, and nothing in this instrument would notice
- Bad, because it is a second instrument to keep working alongside the golden set, with its own theme set to maintain and its own numbers to stamp
- Bad, because the invented-vocabulary check rests on tokenization choices that are judgment calls in disguise: too lax and every query passes, too strict and legitimate multi-word phrasing fails
- Bad, because a mechanical check invites gaming — a prompt tuned to avoid the example's exact strings scores better without reasoning better
