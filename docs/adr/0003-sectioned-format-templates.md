---
status: "accepted"
date: 2026-07-19
---

# Deck-building guidance lives in hand-maintained, sectioned format templates

## Context and Problem Statement

Recommendations need soft guidance that hard filters cannot express: roughly how much ramp, draw, and removal a Commander deck wants alongside its theme cards; that this is a casual, social format; how to think about a request. This guidance is format-specific — Commander's shape is not Modern's. Where should it live, and in what form?

## Considered Options

- Markdown template files per format, authored once and maintained by hand
- Deck-composition heuristics as code or structured config (a ratio table the app reads)
- Generate the guidance per request from a meta-prompt describing the format
- A single monolithic prompt blob per format, unsectioned

## Decision Outcome

Chosen option: "Markdown template files, sectioned", because the guidance is prose aimed at an LLM, not data aimed at the app. Nothing in the pipeline arithmetic-checks "10 ramp / 10 draw / 8 removal" — it is advice, and encoding advice as a config schema buys type safety over a value nobody validates while making the surrounding rationale homeless. Markdown keeps the numbers next to the reasoning that justifies them, which is the part the model actually needs.

The templates are themselves LLM-generated and then frozen into the repo. Generating them per request was rejected: it spends a model call to re-derive text that does not vary by request, and makes the system's advice non-reproducible between two identical queries.

Each template is organized into stable named sections (`## Heuristics`, `## Framing`, `## Workflow`) rather than one blob. The planner and the curation call need different parts of the file — the planner needs the roles to cover, curation needs the framing and the composition targets — and addressable sections let each call be given the relevant subset instead of the whole document.

### Consequences

- Good, because a heuristic and the reasoning behind it live in the same place, in a form both a human and a model can read
- Good, because identical requests get identical guidance; template changes are diffable and reviewable like any other change
- Good, because sectioning lets each LLM call receive only the guidance it needs, keeping prompts focused
- Bad, because the templates are frozen text with no feedback loop — changing a composition heuristic means a human editing markdown, and nothing propagates evidence from real recommendations back into the file
- Bad, because section names become an interface: renaming a heading is a breaking change for whatever call selects on it

Note that this ADR records the *intent* behind sectioning. Selective loading is not implemented, and should not be built until there is a call that needs it.
