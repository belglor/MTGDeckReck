---
status: "accepted"
date: 2026-07-26
---

# The planner validates structured output at the boundary; one retry, then raise

## Context and Problem Statement

The planner ([ADR 0021](0021-planner-local-llm-client.md)) prompts a local instruct model for the `[{query_text, purpose}]` JSON that becomes a `list[PlannedQuery]`. `transformers` enforces no schema, so the model is free to return malformed JSON, prose around the JSON, extra keys, or empty fields. Two questions follow, and neither is settled: what does the planner do with output that doesn't validate, and how hard does it try to get output that does?

[ADR 0004](0004-planner-typed-query-schema.md) already settled *why* the plan is a typed schema — the app executes it, so a phrasing drift must become a validation error, not a silently wrong search — and `.claude/rules/planner-typed-output.md` states the rule. This ADR records the mechanism that rule implies: where validation happens, and the retry policy around it.

## Considered Options

- Best-effort parse: accept whatever comes back, scraping queries out of prose when the JSON is absent
- Validate strictly, then **degrade** on failure — keep the entries that parsed, or fall back to a default plan
- Validate strictly, **retry once**, then **raise** on continued failure
- Guarantee validity up front with constrained decoding (`outlines` / `lm-format-enforcer`)

## Decision Outcome

Chosen option: **validate at the boundary into `list[PlannedQuery]`; on malformed output, retry once, then raise** — never parse prose, never degrade. Constrained decoding is recorded as the known escalation path and deferred past 1.0.

**Validate at the boundary.** The model's raw text is parsed as JSON and validated into `PlannedQuery` at the seam between the model and the rest of the app. The schema already does the semantic half: `PlannedQuery.__post_init__` rejects empty or whitespace-only fields, so `"  "` fails rather than passing as a query that says nothing. Anything that does not validate — bad JSON, missing keys, empty fields — is a failure to surface, not a mess to salvage. This is the rule in `.claude/rules/planner-typed-output.md`: **never** scrape queries from prose (a phrasing drift then becomes a silent misparse, the exact failure [ADR 0004](0004-planner-typed-query-schema.md) chose a schema to kill), and **never** degrade to a partial or default plan (a subtly-wrong plan flows downstream into a pool no curation can rescue — [ADR 0004](0004-planner-typed-query-schema.md)'s own words). A loud failure is recoverable; a quietly-wrong recommendation is not.

**One retry, then raise — and the cap is recorded so it cannot creep.** One retry absorbs a transient formatting slip: the model is stochastic, and the same prompt often yields valid JSON on a second attempt. A *second* retry would start papering over a model that is genuinely unreliable at the task — which is signal we want to see and act on (swap the model behind the [ADR 0021](0021-planner-local-llm-client.md) seam), not bury under brute force. The cap is therefore **one**, and it lives as a named constant in `plan/config.py` (per the config-module guardrail in CLAUDE.md), not as a literal buried in the call. Recording it there with this reasoning makes changing it a deliberate act rather than a number that drifts upward one "just to be safe" edit at a time.

**Constrained decoding is deferred, not rejected.** `outlines` and `lm-format-enforcer` mask token sampling during generation so only tokens that keep the output conforming to the schema can be emitted — malformed JSON becomes impossible by construction rather than caught after the fact, which is worth the most precisely for a small local model whose first-try validity is the risk. We do **not** adopt it now, for three reasons: it is a new dependency coupled to `transformers`' generation internals; it guarantees only *structure*, not *sense*, so the boundary validation above stays either way; and whether it earns its keep depends entirely on the chosen model's real first-try valid-JSON rate, which [ADR 0021](0021-planner-local-llm-client.md) deliberately did not measure. So: ship prompt-and-validate, record constrained decoding as the escalation path, and decide after 1.0 with real reliability numbers in hand. The `LLMClient` seam means adding it later changes the adapter, not the planner's callers.

### Consequences

- Good, because malformed planner output fails loudly at the schema boundary instead of becoming a subtly-wrong search — [ADR 0004](0004-planner-typed-query-schema.md)'s intent, realized at the mechanism level
- Good, because one retry absorbs stochastic noise while the cap of one keeps a genuinely unreliable model visible rather than hidden behind retries
- Good, because no new dependency is taken on now: prompt-and-validate is model-agnostic, and the seam keeps constrained decoding a later, caller-invisible addition
- Good, because the retry cap is a named constant in `plan/config.py`, so it is a documented decision rather than a magic number that creeps
- Bad, because prompt-and-validate's first-try success is model-dependent: a flaky model spends a retry's latency or raises, and we have chosen to feel that rather than mask it until the constrained-decoding question is settled
- Bad, because a raise on the failing path costs roughly twice the generation latency (the retry re-runs the whole call); acceptable because the planner is one call per request and a raise should be rare and loud
- Bad, because deferring constrained decoding leaves the small local model's structured-output ceiling unaddressed until after 1.0 — a known deferred risk, not a solved one
