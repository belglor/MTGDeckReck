"""Planner configuration — constants only.

Where the format templates live and how hard the planner retries. `planner.py`
holds the orchestration; the values it reads live here, so retuning the retry
policy is an edit to data rather than to logic ([ADR 0022]). The chat model
itself is a stage-neutral concern — its knobs live in `mtg_rag.llm_config`
([ADR 0021]), shared with curation.
"""

from __future__ import annotations

from pathlib import Path

#: Where the format templates live: `src/mtg_rag/templates/`, one `<format>.md`
#: per format. The planner reads the whole file for its `format_name` ([ADR
#: 0023]); resolved from this module's location so it holds wherever the package
#: is installed.
TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "templates"

#: File suffix for a format template: `<format>.md`. One fact shared by the
#: planner (which builds the path for a `format_name`) and the plan CLI (which
#: globs the directory to list the formats it will accept), so it lives here
#: rather than as a literal in each.
TEMPLATE_SUFFIX = ".md"

#: How many times the planner re-asks the model after malformed output before it
#: raises ([ADR 0022]). Deliberately one: a single retry absorbs a stochastic
#: formatting slip, while a second would start hiding a model that is genuinely
#: unreliable at the task instead of surfacing it to be swapped behind the
#: `LLMClient` seam. A named constant, not a literal in the call, so raising the
#: cap is a deliberate act rather than a drift.
MAX_PLAN_RETRIES = 1
