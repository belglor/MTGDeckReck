"""Planner configuration — constants only.

How hard the planner retries. `planner.py` holds the orchestration; the value it
reads lives here, so retuning the retry policy is an edit to data rather than to
logic ([ADR 0022]). Two neighbouring concerns are stage-neutral and live at the
package root, shared with curation: the chat model's knobs in
`mtg_rag.llm_config` ([ADR 0021]), and where the format templates live in
`mtg_rag.templates_config` ([ADR 0023], [ADR 0025]).
"""

from __future__ import annotations

#: How many times the planner re-asks the model after malformed output before it
#: raises ([ADR 0022]). Deliberately one: a single retry absorbs a stochastic
#: formatting slip, while a second would start hiding a model that is genuinely
#: unreliable at the task instead of surfacing it to be swapped behind the
#: `LLMClient` seam. A named constant, not a literal in the call, so raising the
#: cap is a deliberate act rather than a drift.
MAX_PLAN_RETRIES = 1
