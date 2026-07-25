# CLAUDE.md

MTGDeckReck: a RAG system that recommends Magic: The Gathering cards from natural-language user requests, favoring flavor and theme over competitiveness or meta.

What it is and how it works: `docs/spec.md`. Why it works that way: `docs/adr/`. Terms used without explanation: `docs/glossary.md`.

## Commands

`just` is the command surface — see `justfile`. Core: `just setup`, `just lint`, `just typecheck`, `just test`, `just check` (all four).

Build steps are manual (no scheduled refresh); `data/` is gitignored and reproducible:

- `just ingest` — card corpus at `data/cards.parquet` from Scryfall's bulk snapshot.
- `just embed` — vector index at `data/vectors/`, one Chroma collection per channel, plus its `data/vectors.meta.json` sidecar.
- `just retrieve "a query"` — searches the index, prints a fused candidate pool.
- `just eval` — runs the golden set, reports retrieval lift.
- `just notebook` — opens JupyterLab.

`just eval` needs a built corpus and index, so it stays out of `just check` and CI. Its numbers are a regression signal, never a gate; it never fails a run ([ADR 0020](docs/adr/0020-eval-case-is-a-corpus-predicate.md)).

Two traps: the justfile sets `positional-arguments`, so `just retrieve "graveyard recursion"` is one query, not three; and `just setup` installs the embedding model, which retrieval encodes queries with — a core dependency, imported normally.

## Git practices

- Branch per unit of work, named `area/short-description` (e.g. `ci/on-demand-review`).
- Small, logically-scoped commits — one concern each; the message says *why*, not just what.
- PR descriptions: Summary (what changed), Reviewer notes (non-obvious decisions, trade-offs, deferred work), Test plan (what was verified).
- Never force-push a branch under review, or skip hooks (`--no-verify`) — fix the underlying issue.
- Before opening a PR, self-review the diff against this file's rules and cut what a reviewer would flag.

## CI / PR workflow

- `main` is protected: no direct pushes. Changes land via PR with a green `check`; a human merges. Agent review is advisory, not a gate.
- `.github/workflows/ci.yml` mirrors `just check`. To change what CI does, edit the `justfile` recipe, not the workflow.
- `@claude` / `@autoreview` in an issue or comment starts a workflow run — don't write them unless you mean to. PR bodies are safe (no `pull_request` trigger).
- New work starts from an issue on the **Agent task** template (`.github/ISSUE_TEMPLATE/agent-task.yml`, label `agent-ready`). One concern per issue.

## Testing philosophy

TDD: write the test first. Don't chase coverage — test real logic branches and edge cases (empty results, boundaries, malformed input), and skip tests that just restate the implementation.

Unit-test the unit; fake its collaborators. A test that stands up a real collaborator to check logic that isn't it tests the wrong thing, and is usually the flakiest. End-to-end and integration tests belong to their own ecosystem, deferred until the pieces stand alone. If a collaborator can't be faked without a seam, that's the code asking for one.

## Documentation

Applies to everything written — `README.md`, docstrings, comments, ADRs, issues, PRs, commit messages. Write for someone who hasn't read the rest of the repo.

- Plain words over jargon; a plainer phrasing over an exact one when it's nearly as accurate (let the ADR carry the exact version).
- Be concise: answer the question at hand, skip context the reader doesn't need to act, keep examples minimal.
- Say a thing once. A decision lives in its ADR; code and `README.md` state the rule and link it.
- Two signs a doc was written for the author's live context, not a fresh reader: it runs much longer than its siblings, or it only parses once you've opened another file. Define a term at first use; don't explain a branch that can't happen.
- Stamp a table or measurement with what produced it (corpus date, config, `k`). An older stamped result stands for its snapshot: record fresher numbers as new rather than reopening or "correcting" the old over drift. Fix genuine errors, not version skew.

The issue template and PR sections are shapes to fill, not quotas. A section with nothing to report stays short or says so.

## Guardrails

- Don't add fields, hooks, or abstractions for out-of-scope features (`docs/spec.md`) — if needed later, that's a small diff. Cite this if asked for a "zero-cost hook" or similar forward-compatibility scaffolding.
- Match a function's resilience to its real callers. External edges — Scryfall ingestion, CLI input, file parsing — stay strict; code reached only from inside trusts the invariants its callers guarantee rather than re-checking states they can't produce. A guard against the impossible is dead code dragging a dead test behind it — when its failing input can't arise, cut it.
- Module-level constants live in a config module, never in the preamble of the module that reads them — exclusion lists, tunables, file/dir names, URLs, separators, model ids, dimensions, batch sizes. Use `<package>/config.py` (`ingest/config.py`) or `<module>_config.py` for a package-root module (`corpus_config.py`); the reader imports what it needs, so logic and data change independently.
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
