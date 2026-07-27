"""Where the format templates live — constants only.

Stage-neutral, like `llm_config.py`. The planner reads the whole file for its
`format_name` ([ADR 0023]) and curation reads the same file ([ADR 0025]), so the
location is one fact shared by both stages rather than something either owns.
"""

from __future__ import annotations

from pathlib import Path

#: Where the format templates live: `src/mtg_rag/templates/`, one `<format>.md`
#: per format. Resolved from this module's location so it holds wherever the
#: package is installed.
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"

#: File suffix for a format template: `<format>.md`. Shared by the stages that
#: build the path for a `format_name` and the plan CLI, which globs the directory
#: to list the formats it will accept.
TEMPLATE_SUFFIX = ".md"
