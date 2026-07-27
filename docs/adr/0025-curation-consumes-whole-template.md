---
status: "accepted"
date: 2026-07-27
---

# Curation is handed the whole format template; selective section-loading stays deferred

## Context and Problem Statement

[ADR 0003](0003-sectioned-format-templates.md) organized each format template into stable named sections (`## Heuristics`, `## Framing`, `## Workflow`) *so that a call can be given the relevant subset instead of the whole document* — the planner needs the roles to cover, curation needs the framing and the composition targets. [ADR 0023](0023-planner-consumes-whole-template.md) answered that question for the planner, the first consumer: whole file, selective loading deferred, until a second consumer wanting a genuinely different subset shows up and the whole-file cost starts hurting.

Curation is that second consumer 0023 named. It wants the **Framing** section (casual, theme-first, not competitive) and the **Heuristics** composition targets it groups picks against — not the **Workflow** steps aimed at the planner's query-drafting. So the question is live again: does curation get its subset, or the whole file?

## Considered Options

- Hand curation the whole `commander.md`, same as the planner
- Build section-addressable selective loading now, and give curation only Framing and Heuristics

## Decision Outcome

Chosen option: **hand curation the whole `commander.md`; selective section-loading stays deferred**, because building it here would be reacting to "a second consumer exists" rather than to "whole-file feeding is a problem" — and 0023 was explicit that it's the second condition, not the first, that justifies the machinery. `commander.md` has not grown; it is the same small file the planner already reads whole, and curation carrying the Workflow section it doesn't strictly need costs the same handful of tokens the planner already accepted for Framing and Workflow.

This closes the loop 0023 opened without triggering it: curation, the named future second consumer, has arrived, and it still doesn't hurt. Two consumers wanting different subsets is necessary but not sufficient — the file also has to be big enough, or the cross-talk distracting enough, that reading one file stops being the simplest thing that works. Neither is true. Building selective loading now, for a file this size, would be exactly the speculative scaffolding CLAUDE.md forbids: machinery sized for a problem that hasn't shown up yet.

### Consequences

- Good, because no selective-loading machinery is built while both consumers still tolerate the whole file; 0003's stable section names keep the eventual build a small, localized change whenever it's needed
- Good, because curation's template consumption is the simplest possible — read the same file the planner reads — with no per-consumer loading path to maintain or test
- Bad, because curation's prompt carries the Workflow section written for the planner's query-drafting, spending a few irrelevant tokens and risking the model weighting planner-oriented advice it should ignore; accepted while the file is small, same trade-off 0023 accepted in the other direction
- Bad, because the trigger condition still hasn't been named anywhere it'll be checked automatically — prompt token cost climbing, or either consumer visibly getting distracted by the other's section, is what should prompt revisiting this, and that's on whoever notices, not on a test or a metric
