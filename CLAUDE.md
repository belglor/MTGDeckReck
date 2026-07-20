# CLAUDE.md

MTGDeckReck: a RAG system that recommends Magic: The Gathering cards from natural-language user requests, favoring flavor and theme over competitiveness or meta.

What it is and how it works: `docs/spec.md`. Why it works that way: `docs/adr/`.

## Commands

`just` is the command surface — see `justfile`. Key recipes: `just setup`, `just lint`, `just typecheck`, `just test`, `just check` (all of the above).

`just ingest` builds the card corpus at `data/cards.parquet` from Scryfall's bulk
snapshot; `just notebook` opens JupyterLab. Both are manual — there is no
scheduled refresh. `data/` is gitignored and fully reproducible.

## Git practices

- Branch per unit of work, named `area/short-description` (e.g. `ci/on-demand-review`).
- Small, logically-scoped commits — one concern per commit; message explains *why*, not just what.
- PR descriptions: Summary (bulleted, what changed), Reviewer notes (non-obvious decisions, trade-offs, deferred work), Test plan (what was actually verified).
- Never force-push a branch under review or skip hooks (`--no-verify`) to push a commit through — fix the underlying issue instead.

## CI / PR workflow

- `main` is protected: no direct pushes. Changes land via PR with a green `check` run, and a human merges — agent review is advisory, not a merge gate.
- `.github/workflows/ci.yml` mirrors `just check` (lint → typecheck → test) on every PR to `main`. Keep it that way: to change what CI does, edit the `justfile` recipe, not the workflow.
- Don't write `@claude` or `@autoreview` into an issue body or comment unless you mean to start a workflow run — `claude.yml` triggers on `issues: [opened]`. PR bodies are safe; there is no `pull_request` trigger. `README.md` documents both for humans.
- New work starts from an issue filed with the **Agent task** template (`.github/ISSUE_TEMPLATE/agent-task.yml`), which applies the `agent-ready` label. Scope one concern per issue.

## Testing philosophy

TDD: write the test before the code it verifies. Don't chase coverage percentage — test core functionality, real logic branches, and edge cases (empty results, boundary values, malformed input). Skip tests that just restate the implementation.

## Guardrails

- Don't add fields, hooks, or abstractions for a feature that isn't in current scope (`docs/spec.md`). If it turns out to be needed, that's a small diff later. Cite this rule if asked to add a "zero-cost hook" or similar forward-compatibility scaffolding.
- After a feature lands, update `README.md` and this file to match.

## Agent context files

| Path | Loaded | Holds |
|---|---|---|
| `CLAUDE.md` | always | repo-wide rules — this file |
| `.claude/rules/*.md` | by path glob | rules for one subtree, each paired with the ADR that justifies it |
| `.claude/skills/<name>/SKILL.md` | on demand | procedures for working on this repo; none yet |
| `src/mtg_rag/templates/*.md` | by the app, at runtime | deckbuilding guides fed to planner/curation prompts — shipped behavior, not dev tooling |
| `notebooks/*.ipynb` | never | exploration only; commit without outputs (`nbstripout` enforces) |

Path-scoped rules don't always load (upstream frontmatter bugs). Confirm with `/context` before relying on one.
