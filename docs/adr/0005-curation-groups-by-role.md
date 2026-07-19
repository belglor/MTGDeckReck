---
status: "accepted"
date: 2026-07-19
---

# Curation returns cards grouped by role with theme-fit rationale

## Context and Problem Statement

Retrieval hands back a deduplicated pool of candidate cards, ordered by vector similarity. Similarity order is not a useful answer: it interleaves a theme payoff, a ramp spell, and a removal spell with no indication that they do different jobs. What shape should the final recommendation take?

## Considered Options

- A flat list ranked by relevance
- Cards grouped by role — payoffs, enablers, support packages — each with an explanation of why it fits the theme
- A numeric theme-fit score per card, leaving presentation to the caller
- Return the raw retrieved pool and let the UI do the grouping

## Decision Outcome

Chosen option: "Grouped by role with rationale", because a deck is a composition, not a ranking. A user asking for a graveyard deck is not looking for the seventy-fifth most graveyard-ish card; they are looking for enough payoffs, the enablers to support them, and a functioning mana base — and a flat list cannot express whether they have that. Grouping makes the shape of the answer legible and makes a gap visible ("only two enablers came back") in a way a ranked list actively hides.

The rationale is load-bearing rather than presentational. This recommender's premise is flavor and theme over meta ([spec](../spec.md)), and theme fit is a claim that has to be argued to be worth anything — "Life from the Loam fits because it recurs lands from the graveyard" is checkable by the user in a way a relevance score is not. A numeric score was rejected for exactly this reason: it projects a judgment that is qualitative onto a scale that implies a precision the model does not have.

Grouping in the UI instead of the model was rejected because role assignment needs the reasoning the app does not have. A card can be a payoff for one theme and an enabler for another, and only something that understands the request can tell which.

### Consequences

- Good, because the answer has the shape of a deck, so a missing role is visible rather than buried in ranking order
- Good, because every recommendation carries a checkable argument, which is what the user can actually push back on
- Good, because the `purpose` field from the planner ([ADR 0004](0004-planner-typed-query-schema.md)) gives curation a starting hypothesis for each candidate's role
- Bad, because grouping and explaining every candidate costs more tokens than emitting a ranked list, growing with pool size
- Bad, because the rationale is model-generated prose and can be plausible but wrong — a confident explanation of a theme fit that isn't there is harder to spot than a bad ranking
- Bad, because role boundaries are fuzzy, so the same card may be grouped differently across two runs of the same request
