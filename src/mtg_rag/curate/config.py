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
