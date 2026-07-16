---
status: "accepted"
date: 2026-07-16
---

# Format legality and color identity are retrieval filters, not planner/LLM concerns

## Context and Problem Statement

Every recommended card must be legal in the user's chosen format and within their color identity. Should these constraints be enforced by asking the planner/curation LLM to respect them, or by the retrieval layer before any LLM sees a candidate?

## Considered Options

- Prompt-based: tell the LLM the format and colors, trust it to filter or self-correct
- Metadata filter: apply format legality and color identity as deterministic filters at retrieval time, before candidates ever reach an LLM call

## Decision Outcome

Chosen option: "Metadata filter", because legality and color identity are objective, Scryfall-provided metadata with a single correct answer per card — there's no reasoning for an LLM to add, only an opportunity for it to hallucinate an exception. Filtering at retrieval time guarantees illegal or off-color cards structurally cannot reach the LLM, rather than relying on it to police itself.

### Consequences

- Good, because illegal/off-color recommendations become impossible by construction, not just unlikely
- Good, because it keeps planner/curation prompts focused on theme-fit reasoning, not rule-checking
- Bad, because format/color-identity data must stay in sync with Scryfall's bulk refresh — a stale local dataset could let something through in error (mitigated by keeping the ingestion refresh idempotent and running it regularly)
- Note: in v1, format and colors are supplied directly by the user via the UI; inferring them from a free-text query is deferred to v2+
