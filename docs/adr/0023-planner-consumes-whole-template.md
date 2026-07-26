---
status: "accepted"
date: 2026-07-26
---

# The planner is handed the whole format template; selective section-loading stays deferred

## Context and Problem Statement

[ADR 0003](0003-sectioned-format-templates.md) organized each format template into stable named sections (`## Heuristics`, `## Framing`, `## Workflow`) *so that a call can be given the relevant subset instead of the whole document* — the planner needs the roles to cover, curation needs the framing and the composition targets. It then left the machinery unbuilt on purpose: *"selective loading is not implemented, and should not be built until there is a call that needs it."*

The planner ([ADR 0021](0021-planner-local-llm-client.md)) is the first real consumer of `commander.md`. So the question 0003 deferred is now live for exactly one call: does the planner get its subset (the roles), or the whole file?

## Considered Options

- Hand the planner the whole `commander.md`
- Build section-addressable selective loading now, and give the planner only its sections

## Decision Outcome

Chosen option: **hand the planner the whole `commander.md`; selective section-loading stays deferred**, because "a call needs a subset" and "whole-file feeding is a problem" are the same condition, and neither is true yet.

The planner will receive the **Framing** and **Workflow** sections that are aimed partly at curation, not just the **Heuristics** roles it strictly needs. That is the accepted cost, and it is small: `commander.md` is soft prose guidance, not a large document, so the extra sections are a handful of tokens and a slight risk that the model weights curation-oriented advice it should ignore. The day that stops being tolerable — prompt token cost climbing, or the planner getting distracted by curation-oriented prose — is exactly the trigger to build selective loading. Until then, the whole file is the simplest thing that works: the planner's prompt is assembled by reading one file, which is the easiest consumption to reason about and to fake in a test.

Deferring costs nothing structurally. 0003 already made the section names a stable interface, so when selective loading is built it will select on those existing headings — the sections are named and stable whether or not anything addresses them yet. And the second consumer that would want a genuinely *different* subset, curation, does not exist (it is out of scope for the planner work). Two calls actually wanting different subsets, with one of them hurting, is the situation that justifies the machinery; building it for a single consumer that is not hurting is exactly the speculative scaffolding CLAUDE.md forbids.

This is a "not yet", not a "no": it records that the planner — the first call that *could* have triggered selective loading — does not, and leaves 0003's intent standing for whenever a call genuinely needs a subset.

### Consequences

- Good, because no selective-loading machinery is built for a single consumer that does not need it; 0003's stable section names keep the eventual build a small, localized change
- Good, because the planner's template consumption is the simplest possible — read one file — which is the easiest thing to reason about and to fake behind the `LLMClient` seam in tests
- Bad, because the planner's prompt carries the Framing and Workflow sections meant partly for curation, spending a few irrelevant tokens and risking the model weighting advice it should ignore; accepted while the file is small, and it is the signal to revisit
- Bad, because "whole file" couples the planner's prompt to the template's total size, so a future large template would push token and cross-talk cost up before anyone decides to split — mitigated because template growth is a deliberate, hand-authored, reviewable change ([ADR 0003](0003-sectioned-format-templates.md))
