---
status: "proposed"
date: 2026-07-26
---

# The planner runs a local instruct model behind a client seam

## Context and Problem Statement

The planner is the first LLM call in the repo: it turns a user's free-text theme into the typed `[{query_text, purpose}]` plan that `retrieve()` already consumes ([ADR 0004](0004-planner-typed-query-schema.md)). Nothing about *how that call is made* is decided — where the model runs, how the code reaches it, and which model it is. The embedder answered the same shape of question for vectors ([ADR 0012](0012-embedding-model.md)): a local, permissively-licensed model on a single developer machine (an RTX 2070), reached through a `Protocol` the pipeline codes against. Does the planner follow that pattern, and which model does it run?

## Considered Options

How the call is made:

- A hosted chat API (OpenAI, Anthropic, …) behind a client interface
- A local instruct model run in-process via `transformers`/`torch`, behind an `LLMClient` `Protocol`
- A local model via an Ollama server the client talks to over HTTP

Which model, once local is chosen:

| Model | Rough size | Pros | Cons |
|---|---|---|---|
| **Qwen3 instruct (0.6B / 1.7B / 4B)** | 0.6–4B | Same family as our embedder (`Qwen3-Embedding-0.6B`) → shared tokenizer and loading idioms, least new code; Apache 2.0, no licence fine-print; solid instruction-following and JSON for its size; size to taste | Newer, so fewer copy-paste examples; ships an optional "thinking" mode to switch off for clean JSON |
| **Llama 3.2 3B Instruct** | ~3B | Very well documented; strong quality | Custom community licence with usage terms (read before shipping); 3B is tight alongside the embedder on 8 GB |
| **Phi-3.5-mini** | ~3.8B | Punches above its size; MIT; reliable structured output | Heaviest here — needs quantizing to fit alongside the embedder |
| **Gemma 2 2B Instruct** | ~2B | Small and capable; well supported | Custom licence with usage terms; different family = more glue code |

(SmolLM2 1.7B is the fully-open lightest option, but the weakest at complex instructions and reliable JSON — a step too far down.)

## Decision Outcome

Chosen option: **a local instruct model run in-process behind an `LLMClient` `Protocol`, with Qwen3 instruct as the provisional default**, because it matches the pattern [ADR 0012](0012-embedding-model.md) already set for the embedder and keeps a third party out of the request path — while the seam makes the specific model a cheap, reversible choice.

**The seam.** `LLMClient` is a `Protocol` mirroring `Encoder` (`src/mtg_rag/embed/encoder.py`): the planner codes against the interface, and the concrete adapter — a thin wrapper over a `transformers` model — is faked in tests. This reuses the embedder's *stack* (`transformers`/`torch`, and its device/dtype selection: fp16 + sdpa on the Turing card, per [ADR 0012](0012-embedding-model.md)), not its weights — the embedding model only emits vectors and cannot generate text. No API-key or secret machinery is built: the model is local, and adding remote-provider plumbing "for later" is exactly the forward-compatibility scaffolding CLAUDE.md forbids. Ollama stays a fallback if in-process generation proves awkward on the target hardware; it is not the default, because it adds a running server to a pipeline that otherwise loads a model the same way `embed/` already does.

**The model — a provisional pick, not a bake-off.** The default is the **Qwen3 instruct family** (0.6B / 1.7B / 4B), for one dominant reason: the embedder is already `Qwen3-Embedding-0.6B`, so a Qwen3 instruct model shares the tokenizer family and loading idioms and the client is very nearly reused code. It is Apache 2.0, so there is no licence to read carefully. Default to the **1.7B** size — 0.6B if we want it tiny, 4B quantized if plan quality disappoints — which fits alongside the ~1.2 GB embedder on the RTX 2070's 8 GB.

We deliberately did **not** run a scored model bake-off. The planner sits behind the seam above, is outside automated eval ([ADR 0020](0020-eval-case-is-a-corpus-predicate.md)), and ships before 1.0 — three reasons a full leaderboard (plan quality × JSON-reliability × VRAM × latency × licence × download, per candidate, in a committed notebook) is effort spent to settle a choice the seam already reduces to a one-line change. The proportionate step is the low-hanging fruit: pick on family fit, size, and licence; confirm with a quick hands-on smoke test — a handful of theme prompts, eyeballing whether the plans are sensible and the JSON parses first-try — and revisit the model after 1.0 with real usage in hand. **Llama 3.2 3B** is the fallback if Qwen's structured-output reliability underwhelms in that smoke test; the options table above is the shortlist it would come from.

Naming the choice is still required: the point of the seam is not that the model is unimportant but that the first pick need not be the last. This ADR records *why Qwen3* (family fit) and *what else was on the table* (the options table) so a later reader can re-open the question — without pretending a measured comparison happened.

### Consequences

- Good, because the planner reuses the embedder's local, permissively-licensed pattern — no key, no per-call cost, nothing leaves the machine, and the `transformers`/`torch` stack is already a dependency of the embedding pass
- Good, because Qwen3 instruct shares the embedder's family, so the concrete client is mostly reused loading code rather than a new integration
- Good, because the `LLMClient` seam makes the model a reversible decision: swapping to Llama 3.2 or Phi-3.5 later is a config-and-adapter change, not a rewrite — which is precisely what lets this ADR pick lightly
- Bad, because the pick rests on family fit and a smoke test rather than measured evidence, so Qwen3's plan quality and first-try JSON rate are asserted-with-a-sniff-test, not proven; the revisit after 1.0 is where that gets earned
- Bad, because a small local instruct model is a real quality risk for structured output: `transformers` enforces no schema, so an unreliable model would raise the plan's retry rate (structured output is its own decision — ADR 0022, forthcoming), and the 8 GB card caps how large a model we can reach for to compensate
- Bad, because running generation in-process keeps the model resident alongside the embedder, so fitting both on 8 GB is a live constraint a future larger model could break
