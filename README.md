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

`@autoreview` is this repo's automated review process: **comment it on a pull
request** and a multi-agent Claude review runs against the diff, posting only
findings it scores as high-confidence. Restricted to repo collaborators.

It runs on demand rather than on every push, because each invocation costs real
quota. On demand is not the same as optional — it is the accepted way changes get
reviewed here; it simply isn't wired up as a merge gate.

You can also mention `@claude` on any issue or PR to ask a question or hand off a task.
