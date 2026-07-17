# CLAUDE.md

MTGDeckReck: a RAG system that recommends Magic: The Gathering cards from a natural-language deck theme (e.g. "spooky graveyard shenanigans"). Optimizes for fun, casual, thematic play — never competitiveness or meta. Full spec: `docs/spec.md`. Settled architecture decisions: `docs/adr/`.

## Commands

`just` is the command surface — see `justfile`. Key recipes: `just setup`, `just lint`, `just typecheck`, `just test`, `just check` (all of the above).

## Git practices

- Branch per unit of work, named `area/short-description` (e.g. `phase-0/verification-tooling`).
- Small, logically-scoped commits — one concern per commit; message explains *why*, not just what.
- PR descriptions: Summary (bulleted, what changed), Reviewer notes (non-obvious decisions, trade-offs, deferred work), Test plan (what was actually verified).
- Never force-push a branch under review or skip hooks (`--no-verify`) to push a commit through — fix the underlying issue instead.

## CI / PR workflow

- `main` is protected: no direct pushes. Changes land via PR with a green `check` run, and a human merges — agent review is advisory, not a merge gate.
- `.github/workflows/ci.yml` mirrors `just check` (lint → typecheck → test) on every PR to `main`. Keep it that way: to change what CI does, edit the `justfile` recipe, not the workflow.
- The Claude GitHub app auto-reviews each PR (`claude-code-review.yml`) and answers `@claude` mentions on issues and PRs (`claude.yml`).
- New work starts from an issue filed with the **Agent task** template (`.github/ISSUE_TEMPLATE/agent-task.yml`), which applies the `agent-ready` label. Scope one concern per issue.

## Testing philosophy

TDD: write the test before the code it verifies. Don't chase coverage percentage — test core functionality, real logic branches, and edge cases (empty results, boundary values, malformed input). Skip tests that just restate the implementation.

## Guardrails (global)

Downprioritize ranking or filtering by play rate, win rate, or tournament meta — this is a casual/thematic recommender first, not a competitive one. Meta signal isn't forbidden outright, just never the primary driver.

Don't add fields, hooks, or abstractions for a feature that isn't in current scope (`docs/spec.md`), even if `docs/vision.md` mentions it. If it turns out to be needed, that's a small diff later. Cite this rule if asked to add a "zero-cost hook" or similar forward-compatibility scaffolding.

Guardrails that only apply to part of the tree live in `.claude/rules/` instead, paired with the ADR that explains them.

## Self-check: scoped rules

`.claude/rules/*.md` path-scoped guardrails are not always reliably enforced (frontmatter loading bugs have been reported upstream). When working under a path a rule should cover, run `/memory` (or `/context`) and confirm the rule actually loaded before relying on it — don't assume it's active just because the file exists.

## Two kinds of on-demand markdown

- `.claude/skills/<name>/SKILL.md` — dev-time skills. Claude Code loads them; the coding agent reads them; they describe how to work on *this repo*. Empty for now — first candidate is v2's "add a retrieval channel" procedure.
- `src/mtg_rag/templates/*.md` (e.g. `commander.md`) — format templates. The app loads them into planner/curation prompts; the product's own LLM call reads them; they describe how to build a deck. Shipped behavior, versioned and traced like the rest of the prompts — not dev tooling.

## Keep docs in sync

After implementing a new feature, update `README.md` and this file (`CLAUDE.md`) to reflect the new functionality.
