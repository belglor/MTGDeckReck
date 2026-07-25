# CLAUDE.md

MTGDeckReck: a RAG system that recommends Magic: The Gathering cards from natural-language user requests, favoring flavor and theme over competitiveness or meta.

What it is and how it works: `docs/spec.md`. Why it works that way: `docs/adr/`. Terms used without explanation: `docs/glossary.md`.

## Commands

`just` is the command surface — see `justfile`. Key recipes: `just setup`, `just lint`, `just typecheck`, `just test`, `just check` (all of the above).

`just ingest` builds the card corpus at `data/cards.parquet` from Scryfall's bulk
snapshot; `just embed` builds the vector index at `data/vectors/` (one Chroma
collection per channel) plus its `data/vectors.meta.json` sidecar; `just retrieve
"a query"` searches that index and prints a fused candidate pool; `just eval`
runs the golden set against that index and reports retrieval lift; `just
notebook` opens JupyterLab. The build steps are manual — there is no scheduled
refresh. `data/` is gitignored and fully reproducible.

`just eval` needs a built corpus and index, so it stays out of `just check` and
out of CI. Its numbers are a regression signal, never a gate, and it never fails
a run — see [ADR 0020](docs/adr/0020-eval-case-is-a-corpus-predicate.md).

The justfile sets `positional-arguments` so recipe arguments arrive as `"$@"`
rather than being re-split on whitespace — `just retrieve "graveyard recursion"`
must be one query, not three.

`just setup` installs everything, the embedding model included — retrieval
encodes queries with it, so it is a core dependency and imported normally.

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

## Documentation

Applies to everything written here — `README.md`, docstrings, code comments, ADRs, issue bodies, PR descriptions, commit messages. Write for someone who hasn't read the rest of the repo.

- Favor understandability over precision. If a plainer phrasing is nearly as accurate, use it and let the ADR carry the exact version.
- Be concise and focused: answer the question at hand, skip context the reader doesn't need in order to act, and keep examples minimal.
- Say a thing once. A decision lives in its ADR; code and `README.md` state the rule and link to it rather than retelling the reasoning.

The issue template and the PR sections named above are shapes to fill, not quotas to meet. A section with nothing to report stays short or says so.

## Guardrails

- Don't add fields, hooks, or abstractions for a feature that isn't in current scope (`docs/spec.md`). If it turns out to be needed, that's a small diff later. Cite this rule if asked to add a "zero-cost hook" or similar forward-compatibility scaffolding.
- Module-level constants live in a config module, never in the preamble of the module that reads them. Exclusion lists, tunables, file and directory names, URLs, separators, model ids, dimensions, batch sizes — all of it. Use `<package>/config.py` for a package (`ingest/config.py`) and `<module>_config.py` for a package-root module (`corpus_config.py`); the code that needs a value imports it. This keeps a module's logic and its data able to change independently, so adding a layout or bumping a dimension is a one-line diff in a file whose whole job is holding values.
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
