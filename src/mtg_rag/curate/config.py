"""Curation configuration — constants only.

How hard curation retries on malformed output. `curation.py` holds the
orchestration ([ADR 0024]); the value it reads lives here, so retuning the
retry policy is an edit to data rather than to logic. The chat model itself is
a stage-neutral concern — its knobs live in `mtg_rag.llm_config` ([ADR 0021]),
shared with the planner.
"""

from __future__ import annotations

#: How many times curation re-asks the model after malformed output before it
#: raises ([ADR 0024]). Deliberately one, mirroring the planner's
#: `MAX_PLAN_RETRIES` ([ADR 0022]): a single retry absorbs a stochastic
#: formatting slip, while a second would start hiding a model that is
#: genuinely unreliable at the task instead of surfacing it to be swapped
#: behind the `LLMClient` seam.
MAX_CURATION_RETRIES = 1

#: How many of the pool's top candidates curation is shown, which is a hardware
#: bound rather than a judgment about how many cards make a good deck.
#: Attention cost grows with the square of the prompt, and the whole retrieved
#: pool does not fit: measured on the 8 GB RTX 2070 with `Qwen/Qwen3-1.7B`
#: (corpus 2026-07-22, `DEFAULT_POOL_SIZE` 100), the full pool is ~12.2k tokens
#: and runs out of CUDA memory outright, ~5.3k tokens (40 candidates) prefills
#: in 66 s by spilling into shared memory, and ~4.1k tokens (30) prefills in
#: 2 s. 30 is the last size on the fast side of that cliff.
#:
#: The cost is real: curation picks from the top of the pool only, so a role
#: that ranked poorly can go uncovered, and the recommendation is thinner than
#: a deck of this format wants. Lifting that means giving curation more than
#: one call, not raising this number past what the hardware measured above.
CURATION_POOL_SIZE = 30
