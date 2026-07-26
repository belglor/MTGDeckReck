"""Planner configuration — constants only.

The instruct model id and the knobs its `.generate()` call reads, plus where the
format templates live and how hard the planner retries. `client.py` holds the
loading and generation behavior and `planner.py` the orchestration; the values
they pass live here, so retuning the model or the retry policy is an edit to data
rather than to logic ([ADR 0021], [ADR 0022]).
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

#: The planner's instruct model ([ADR 0021]): Qwen3-1.7B, the post-trained
#: (chat) checkpoint — not `Qwen/Qwen3-1.7B-Base`, which is not instruction
#: tuned. Apache 2.0, same family as the embedder (`Qwen/Qwen3-Embedding-0.6B`),
#: so it shares the tokenizer idioms; ~1.7B params fits alongside the ~1.2 GB
#: embedder on the 8 GB RTX 2070. Llama 3.2 3B is the documented fallback.
MODEL_ID = "Qwen/Qwen3-1.7B"

#: Qwen3 ships an optional "thinking" mode that emits a reasoning preamble
#: before its answer. The planner wants clean JSON and nothing else, so it is
#: switched off at the chat template ([ADR 0021]); on, the preamble would have
#: to be stripped before parsing.
ENABLE_THINKING = False

#: Generation cap. A plan is a short JSON array of `{query_text, purpose}`
#: objects, so this only needs headroom for the largest sensible plan — a bound
#: against a runaway generation, not a target length. With thinking off there is
#: no reasoning preamble competing for the budget.
MAX_NEW_TOKENS = 1024

#: Sampling — Qwen3's published recommendation for non-thinking mode
#: (temperature 0.7, top-p 0.8, top-k 20, min-p 0). min-p is left at its 0
#: default, so it is not passed. Greedy decoding is deliberately avoided: Qwen
#: documents it as prone to repetition on these models. The stochastic slack
#: this leaves is what [ADR 0022]'s one retry absorbs.
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
