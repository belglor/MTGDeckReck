# MTGDeckReck

[![CI](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml/badge.svg)](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml)

Agent-driven RAG project for Magic: The Gathering deckbuilding recommendations.

## Development

`just` is the command surface — see `justfile`:

- `just setup` — sync dependencies, install pre-commit hooks
- `just check` — lint, typecheck, test; exactly what CI runs

`main` is protected: changes land through a pull request with green CI. Every PR is
reviewed automatically by the Claude GitHub app; a human makes the call to merge.
