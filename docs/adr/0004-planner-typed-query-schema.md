---
status: "accepted"
date: 2026-07-19
---

# The planner emits a typed query schema; the model chooses the queries

## Context and Problem Statement

A single user request — "a spooky graveyard deck" — is not one search. It needs several: the theme payoffs, the enablers that turn the theme on, and the generic ramp, draw, and removal any deck wants. Something has to decide what those searches are. How much of that decision belongs to the application, and how much to the model?

## Considered Options

- A fixed, hardcoded list of queries derived mechanically from the user's theme string
- One query per template role, fixed count, model fills in only the text
- A planner LLM call returning free-form prose that the app parses into queries
- A planner LLM call returning a typed schema, `[{query_text, purpose}]`, with the model choosing both the queries and how many

## Decision Outcome

Chosen option: "Typed schema, model chooses the queries", because the two halves of this problem have different owners. *That* a plan must cover theme payoffs, enablers, and the support roles is stable across every request — that is structure, and the format template states it ([ADR 0003](0003-sectioned-format-templates.md)). *What* to search for is entirely request-dependent: "graveyard" wants self-mill and recursion, "connive" wants looting and +1/+1 counters, and no amount of hardcoding anticipates the third theme a user invents. Fixing the query list would cap the system's range at whatever its author thought of.

Letting the model choose the *count* follows from the same reasoning. A narrow theme genuinely needs fewer queries than a broad one, and forcing exactly one query per role produces padding for narrow themes and truncation for broad ones.

The output is a typed schema rather than prose because the app has to execute it. Parsing free-form model output means a phrasing drift becomes a silent misparse; a schema turns the same drift into a validation error at the boundary.

The `purpose` field is not decoration — it travels with each query so the curation call knows which role a candidate was retrieved *for*, which is what makes role grouping possible downstream ([ADR 0005](0005-curation-groups-by-role.md)).

### Consequences

- Good, because the system handles themes nobody anticipated, without a code change per theme
- Good, because malformed planner output fails loudly at the schema boundary instead of becoming a subtly wrong search
- Good, because queries are independent, so the app can execute the whole plan in parallel and dedupe once
- Bad, because a variable query count makes cost and latency per request variable, and a model that over-plans is a real failure mode to watch
- Bad, because plan quality is now a model-dependent surface: a planner that misreads the theme produces a pool that no amount of good curation can rescue
