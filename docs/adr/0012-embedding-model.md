---
status: "accepted"
date: 2026-07-20
---

# The embedding model is Qwen3-Embedding-0.6B, run locally

## Context and Problem Statement

The three semantic channels ([ADR 0007](0007-multi-channel-embedding.md)) need a model to turn card text into vectors, and nothing has chosen one — the spec names the channels but not the model. The choice sets the vector dimension, the index size, the dependency weight, and the recurring cost of every re-embed. This is a theme recommender that favours open tooling, and the embedding pass runs on a single developer machine (an RTX 2070, Turing), not a cluster. Which model, and hosted or local?

## Considered Options

- `Qwen/Qwen3-Embedding-0.6B`, run locally
- `BAAI/bge-base-en-v1.5` (768-dim), run locally
- `BAAI/bge-small-en-v1.5` (384-dim), run locally
- A hosted embedding API (Voyage, Cohere, OpenAI)

## Decision Outcome

Chosen option: "`Qwen/Qwen3-Embedding-0.6B`, run locally", because it is the strongest open-weight option that fits the hardware, and keeping embedding local keeps a third party out of the request path.

`Qwen3-Embedding-0.6B` is Apache 2.0, ~600M parameters, and emits **1024-dim** vectors. It is trained with Matryoshka representation learning, so the dimension is truncatable later without re-training — a `truncate_dim` lever we can reach for if the index size (below) becomes a problem, treated as a measured change against the golden set rather than a guess. On the target **RTX 2070 (Turing, sm_75)** the model runs in fp16; Turing has no bf16 and no flash-attention-2 (Ampere+), so the loader pins `torch_dtype=float16` and `attn_implementation="sdpa"`. Recording that here means the constraint is a decision, not something rediscovered by a crash.

The model is **asymmetric by design**, and this is a property `retrieve/` inherits, not an `embed/` detail: its published prompts set the document prefix to the empty string and give queries an `"Instruct: …\nQuery:"` prefix. Documents are embedded with no instruction; queries are embedded with the instruction. Both sides must honour this or the geometry silently mismatches — the encode path exposes it as distinct document/query calls rather than a flag buried in a comment.

The `bge` models were the open alternatives. `bge-base-en-v1.5` (768-dim) is lighter but older and measurably weaker on current retrieval benchmarks; `bge-small-en-v1.5` (384-dim) is faster and smaller still but weaker again, and the loss lands hardest on the flavor channel, where the signal is the most subtle and the corpus the smallest ([ADR 0007](0007-multi-channel-embedding.md)) — the last place to economise on model quality. Qwen3-0.6B fits the same local, free, permissively-licensed slot while scoring better, so it dominates them for this use.

A hosted API was rejected on the project's open-tooling preference rather than on quality. It would be a paid third party sitting in the request path, an API key to manage, and a recurring cost on every re-embed — which, given the index is rebuilt wholesale ([ADR 0015](0015-wholesale-index-rebuild.md)), is a bill that recurs on any corpus or text change. None of that buys anything a local model at this corpus size can't deliver.

### Consequences

- Good, because embedding stays local, free, and permissively licensed — no key, no per-call cost, nothing leaves the machine
- Good, because 1024-dim Matryoshka vectors leave a truncation lever for shrinking the index later without switching models
- Bad, because `sentence-transformers` + `torch` (CUDA) is a ~2.5 GB dependency; it is needed only on the machine that runs the embedding pass, so it belongs behind an optional extra rather than in the core install
- Bad, because the model weights are a ~1.2 GB download, and the 88,268-vector index ([ADR 0013](0013-structural-card-predicate.md)) is ~361 MB of raw float32 on disk — both far larger than the ~30 MB parquet
- Bad, because swapping the model resets the recall baseline — this is the general rule in [ADR 0011](0011-evaluation-scope-and-baseline-semantics.md), and a model change is exactly the geometry change it describes; the golden set is re-run and the new number recorded, not compared against the old
