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

#: What share of `MAX_NEW_TOKENS` the prompt tells curation its whole reply must
#: fit inside ([#106]).
#:
#: Below 1.0 because the number is a budget the model is asked to respect, not a
#: limit anything enforces: generation stops at the real cap regardless, and a
#: reply that reaches it is cut mid-string and cannot be parsed. Leaving a
#: margin means a model that overshoots its stated budget by a little still
#: lands inside the cap and produces a readable answer.
#:
#: 0.9 is a judgement, not a measurement. A small model counts its own tokens
#: badly, so this is a nudge rather than a guarantee — the load-bearing fix for
#: [#106] is the prompt forbidding quoted card text, which is what made replies
#: long in the first place.
RATIONALE_BUDGET_FRACTION = 0.9

#: How many of the pool's top candidates curation is shown, which is a hardware
#: bound rather than a judgment about how many cards make a good deck.
#: Attention cost grows with the square of the prompt, and the whole retrieved
#: pool does not always fit.
#:
#: The cliff this constant used to sit on (30, in this file's history) was not
#: actually a hardware ceiling — it was Qwen3's grouped-query attention
#: defeating both of torch's fused SDPA kernels, leaving only the
#: quadratic-memory "math" kernel to fall back on (`llm.py`'s
#: `prefer_efficient_attention` has the mechanism; [#93] is the spike that
#: found it). Forcing the memory-efficient kernel instead, at zero new
#: dependencies, moved the cliff a long way out. Measured on the 8 GB RTX 2070
#: with `Qwen/Qwen3-1.7B` (corpus 2026-07-22): unpatched (`math`), 30
#: candidates (~4.1k tokens) prefills in 2 s at 6.20 GiB, 40 (~5.3k tokens) in
#: 47 s at 7.88 GiB by spilling into shared memory, and 100 (~12.4k tokens,
#: `DEFAULT_POOL_SIZE`, the full pool) OOMs outright. Patched
#: (`efficient_attention`), the same three sizes take 0.7 s at 4.83 GiB, 0.9 s
#: at 5.29 GiB, and 2.8 s at 8.09 GiB respectively — the full pool now fits,
#: but at 8.09 of 8 GiB there is no headroom left for anything else this
#: process holds.
#:
#: 80 candidates (~10.0k tokens) patched takes 2.1 s at 7.14 GiB — the largest
#: size measured with real margin (~1 GiB free), so that is this constant
#: rather than 100. A machine with more VRAM, or a smaller model, could
#: reasonably push it further using the same measurement recipe.
#:
#: The remaining gap to the full pool is a job for more than one curation
#: call, not a bigger number here ([#94]).
CURATION_POOL_SIZE = 80
