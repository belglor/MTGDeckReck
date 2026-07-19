# MTGDeckReck

[![CI](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml/badge.svg)](https://github.com/belglor/MTGDeckReck/actions/workflows/ci.yml)

Agent-driven RAG project for Magic: The Gathering deckbuilding recommendations.

## Development

`just` is the command surface — see `justfile`:

- `just setup` — sync dependencies, install pre-commit hooks
- `just check` — lint, typecheck, test; exactly what CI runs

`main` is protected: changes land through a pull request with green CI, and a human
makes the call to merge.

### Review

Reviewing your own work before you push is your own business — whatever local flow
and tooling you like. Nothing here enforces it.

What the project offers is one command: **comment `@autoreview` on a pull request**
and a multi-agent Claude review runs against the diff, posting only findings it
scores as high-confidence. It is not automatic and not a merge gate — it costs real
quota per invocation, so ask for it when a change warrants it rather than on every
push. Restricted to repo collaborators.

You can also mention `@claude` on any issue or PR to ask a question or hand off a task.
