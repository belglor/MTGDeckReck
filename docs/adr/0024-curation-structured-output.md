---
status: "accepted"
date: 2026-07-27
---

# Curation validates a flat `[{oracle_id, role, rationale}]` schema at the boundary; one retry, then raise

## Context and Problem Statement

Curation is the final backend stage (Plan → Retrieve → Curate). It hands the retrieved candidate pool to a local instruct model and asks it to group the cards by role with a theme-fit rationale ([ADR 0005](0005-curation-groups-by-role.md)). That ADR settled *what* the answer is — grouped by role, each card carrying a checkable argument, no numeric score — but not the mechanism: what exact JSON the model must return, what typed structure it validates into, and what happens when the output does not validate. The model enforces no schema, so it is free to return malformed JSON, prose around it, an invented card id, or a role bucket with no cards. This ADR fixes the contract and the failure policy, mirroring [ADR 0022](0022-planner-structured-output.md) for the planner.

## Considered Options

- **Output shape:** a flat array `[{oracle_id, role, rationale}]` that the renderer groups by `role`, versus nested role groups `{role: [{oracle_id, rationale}]}` the model emits pre-grouped
- **Card vocabulary:** accept whatever `oracle_id`s the model returns, versus require every returned id to be a member of the retrieved pool
- **Failure policy:** best-effort parse; validate strictly then degrade (keep the entries that parsed); or validate strictly, retry once, then raise

## Decision Outcome

Chosen: **a flat `[{oracle_id, role, rationale}]` array, validated at the boundary; every `oracle_id` must come from the retrieved pool; on malformed output, retry once, then raise** — never parse prose, never degrade, never a partial answer.

**Flat array, renderer groups.** The model returns a flat list of `{oracle_id, role, rationale}` objects; the renderer groups them by `role` for display ([ADR 0005](0005-curation-groups-by-role.md)'s grouped shape). A flat schema is chosen over nested role groups for two reasons. Every entry validates identically — one object shape, the same three-key check on each — which mirrors the planner's flat `[{query_text, purpose}]` ([ADR 0004](0004-planner-typed-query-schema.md)) and keeps the parser a single loop rather than a walk over role keys each holding a list. And it puts the grouping in exactly one place: with nested output, `role` is a structural key the model authors *and* the presentation shape, so the two can disagree (a card under one key, its rationale arguing another); flat, `role` is a plain value on the card and grouping happens once, in the renderer, where a role with zero cards simply does not appear rather than needing to be emitted as an empty bucket.

**`role` is model-authored prose, not a closed enum.** Like the planner's `purpose`, `role` is a free non-empty string the model chooses, guided by the format template — not validated against a fixed vocabulary. [ADR 0005](0005-curation-groups-by-role.md) already holds that role assignment needs reasoning the app does not have and that role boundaries are fuzzy; pinning `role` to an enum would relocate that judgment into the schema, where it does not belong. The schema checks that `role` and `rationale` are present and non-empty (whitespace counts as empty, as with `PlannedQuery`), nothing more.

**`oracle_id` is closed-vocabulary: from the pool, or it raises.** `oracle_id` is different from the prose fields because it has an external referent — it is the identity key the pool is keyed on ([ADR 0010](0010-oracle-id-identity-key.md)), and the renderer hydrates each returned id back to a real card. So the one closed-vocabulary rule: **every returned `oracle_id` must be a member of the retrieved pool.** A card may be *omitted* — not every candidate makes the deck, and selecting down is curation's job — but an id may never be *invented*. An invented id hydrates to nothing or, worse, silently to a card the model never reasoned about; either way the rationale attached to it is unmoored. Omission is a legitimate answer; invention is a validation failure that raises.

**One retry, then raise — even though curation is terminal.** On malformed output — bad JSON, prose, missing or extra keys, an empty field, an out-of-pool id — the parser raises; the caller retries once, then raises for good. The cap is **one**, for [ADR 0022](0022-planner-structured-output.md)'s reasoning exactly: one retry absorbs a stochastic formatting slip, a second would start hiding a model genuinely unreliable at the task instead of surfacing it to be swapped behind the client seam. It lives as a named constant `MAX_CURATION_RETRIES` in `curate/config.py` (the config-module guardrail in CLAUDE.md), not a literal in the call, so raising it is a deliberate act.

That "never degrade" still holds here needs saying, because curation is terminal — nothing downstream consumes its output, so the usual argument (a subtly-wrong result poisons a later stage) does not apply. It holds anyway, because the *user* is downstream. A partial parse — drop the entries that failed, show the rest — would hand back a deck quietly missing cards it should contain, presented as a finished recommendation with no signal that anything was cut. That is precisely the "plausible but wrong" risk [ADR 0005](0005-curation-groups-by-role.md) flags, now dressed as a complete answer. A half-answer shown as whole is worse than a loud failure the user can retry: the failure is recoverable, the silent gap is not even visible. So a malformed structure fails loudly rather than degrading to a convincing fragment.

### Consequences

- Good, because a flat schema validates uniformly and mirrors the planner's, so curation's parser is the same shape of code as the plan's — one loop, one per-entry check
- Good, because grouping lives only in the renderer, so `role` and its bucket cannot drift apart and an empty role simply does not render
- Good, because the closed-vocabulary rule catches an invented or hallucinated card at the boundary, where an unmoored rationale would otherwise reach the user as if grounded
- Good, because keeping `role` model-authored honors [ADR 0005](0005-curation-groups-by-role.md): the fuzzy, reasoning-dependent judgment stays with the model, not the schema
- Good, because the retry cap is a named constant in `curate/config.py`, a documented decision rather than a magic number that creeps
- Bad, because validating `oracle_id` against the pool couples the parser to the pool (it needs the set of valid ids passed in), where the planner's parser validated in isolation
- Bad, because raising on a terminal stage means a formatting slip the model cannot recover in one retry surfaces to the user as an error rather than a degraded-but-shown deck — the deliberate cost of refusing to show a half-answer as whole
- Bad, because first-try validity is model-dependent, so a flaky model spends a retry's latency or raises; as with the planner, constrained decoding ([ADR 0022](0022-planner-structured-output.md)) is the recorded escalation path if that rate proves too low, deferred until measured
