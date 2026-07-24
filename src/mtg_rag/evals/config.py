"""Eval configuration: values only.

The golden set is committed next to this module rather than written under
`data/`. `data/` is gitignored and fully reproducible from `just ingest`; the
cases are neither — they are the definition of what the eval measures, and
losing them would lose the instrument.
"""

from __future__ import annotations

from typing import Literal

#: How a case names its expected set ([ADR 0020]). `keyword` reads Scryfall's
#: structured `keywords` column, so membership is a matter of fact. `oracle_text`
#: is a regex over rules prose and is a **proxy** — it is admissible only because
#: it is a fixed ruler compared to itself across runs, never as an absolute score.
type PredicateKind = Literal["keyword", "oracle_text"]

#: Name of the golden set, resolved relative to this package.
GOLDEN_NAME = "golden.toml"

#: Pool depth every case is measured at.
#:
#: Deliberately not `retrieve`'s `DEFAULT_POOL_SIZE` of 100. Three channels at
#: `CHANNEL_TOP_K` can surface at most ~142 distinct cards for a one-query case,
#: so 100 sits close enough to that ceiling to make the metric easy — the
#: sensitivity [ADR 0006] warns about. 25 is also the depth every figure in
#: [ADR 0020] was measured at, so a run can be checked against them.
DEFAULT_EVAL_K = 25

#: Where the JSON report lands under the CLI's --data-dir.
REPORT_NAME = "eval-report.json"
