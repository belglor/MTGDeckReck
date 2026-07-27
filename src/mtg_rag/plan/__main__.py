"""Command-line entry point for the whole pipeline — Plan → Retrieve → Curate.

    just plan "a spooky graveyard deck that mills itself" --colors B
    just plan "elfball ramp into a big finisher" --plan-only

Plans the searches for a free-text theme, runs them through retrieval, and hands
the fused pool to curation, which returns the cards grouped by the job each does
([ADR 0005]). Two flags stop early: `--plan-only` after the queries — the
query-only output this CLI began as, needing neither corpus nor index — and
`--pool-only` after the candidate pool, which is what `just retrieve` prints.

The trap this CLI has to get right: the planner emits only `query_text` /
`purpose`, and the hard constraints (format legality, colour identity, platform)
come from the flags into `Constraints`, **never** from the model ([ADR 0001]).
Curation inherits that guarantee for free — it chooses from the retrieved pool
and nothing else, so a card the filters excluded cannot be recommended.

The full path loads two local models — the chat model and the embedder — so it
plans first and encodes second; the chat model is sized to sit alongside the
~1.2 GB embedder on the 8 GB target (`llm_config.py`). Curation reuses the
client the planner already loaded rather than paying for a second copy, which
is what sharing the seam across stages buys ([ADR 0021]).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from mtg_rag.cli import use_utf8_stdout
from mtg_rag.ingest.config import PLATFORMS
from mtg_rag.llm import LLMClient, QwenChatClient
from mtg_rag.plan.planner import plan
from mtg_rag.plan.query import PlannedQuery
from mtg_rag.retrieve.config import DEFAULT_PLATFORM
from mtg_rag.templates_config import TEMPLATE_DIR, TEMPLATE_SUFFIX

if TYPE_CHECKING:
    import polars as pl

    from mtg_rag.curate.prompt import CurationCard
    from mtg_rag.retrieve.filters import Constraints
    from mtg_rag.retrieve.fusion import Candidate


def _available_formats() -> list[str]:
    return sorted(path.stem for path in TEMPLATE_DIR.glob(f"*{TEMPLATE_SUFFIX}"))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m mtg_rag.plan",
        description="Recommend a deck for a request: plan the searches, retrieve, then curate.",
    )
    parser.add_argument("theme", help="a plain-English description of the deck to build")
    parser.add_argument(
        "--format",
        dest="format_name",
        default="commander",
        help="format whose template guides the plan and whose legality filters "
        "(default: commander)",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print the planned queries and stop, without searching (needs no corpus or index)",
    )
    parser.add_argument(
        "--pool-only",
        action="store_true",
        help="print the retrieved candidate pool and stop, without curating it",
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="color identity to stay within, e.g. BG. Omit for no color "
        "constraint; pass '' for colorless-only",
    )
    parser.add_argument(
        "--platform",
        default=DEFAULT_PLATFORM,
        choices=PLATFORMS,
        help=f"where the cards must be playable (default: {DEFAULT_PLATFORM})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="where the corpus and the vector index live (default: data)",
    )
    return parser.parse_args(argv)


def _print_queries(queries: list[PlannedQuery], format_name: str) -> None:
    print(f"\nPlanned {len(queries)} queries for {format_name}:\n")
    for position, query in enumerate(queries, start=1):
        print(f"{position:>3}. {query.query_text}")
        print(f"     {query.purpose}")


def _prepare_retrieval(args: argparse.Namespace) -> tuple[Constraints, pl.DataFrame, Path] | None:
    """The corpus, the vector dir, and the flag-built `Constraints` — or `None`.

    Runs before the planner model loads, so a missing build or an unknown format
    fails cheap. Mirrors the retrieve CLI's existence checks and messages; the
    `Constraints` are assembled from the flags exactly as it does, which is the
    one thing the model must never supply ([ADR 0001]).
    """
    import polars as pl

    from mtg_rag.embed.config import VECTOR_DIR_NAME
    from mtg_rag.ingest.config import CORPUS_NAME
    from mtg_rag.retrieve.filters import Constraints, available_formats, parse_color_identity

    data_dir: Path = args.data_dir
    corpus_path = data_dir / CORPUS_NAME
    vector_dir = data_dir / VECTOR_DIR_NAME

    if not corpus_path.exists():
        print(f"No corpus at {corpus_path}. Run `just ingest` first.", file=sys.stderr)
        return None
    if not vector_dir.exists():
        print(f"No vector index at {vector_dir}. Run `just embed` first.", file=sys.stderr)
        return None

    frame = pl.read_parquet(corpus_path)
    if args.format_name not in available_formats(frame):
        available = ", ".join(sorted(available_formats(frame)))
        print(f"Unknown format {args.format_name!r}. Available: {available}", file=sys.stderr)
        return None

    constraints = Constraints(
        format_name=args.format_name,
        color_identity=parse_color_identity(args.colors),
        platform=args.platform,
    )
    return constraints, frame, vector_dir


def _plan(args: argparse.Namespace) -> tuple[list[PlannedQuery], LLMClient] | None:
    """Load the chat model, plan the searches, print them — or `None` if it won't load.

    Hands the client back along with the queries: curation needs the same model,
    and re-constructing it would reload multi-GB weights the process is already
    holding ([ADR 0021]).
    """
    print("Loading the model...")
    try:
        client = QwenChatClient()
    except Exception as error:
        # A missing weights download, no network, or too little memory all surface
        # here; the user wants one clear line, not a traceback out of transformers.
        print(f"Could not load the planner model: {error}", file=sys.stderr)
        return None

    queries = plan(args.theme, format_name=args.format_name, client=client)
    _print_queries(queries, args.format_name)
    return queries, client


def _search_and_print(
    queries: list[PlannedQuery],
    *,
    constraints: Constraints,
    frame: pl.DataFrame,
    vector_dir: Path,
) -> tuple[list[Candidate], pl.DataFrame] | None:
    """Run the planned queries through retrieval, print the fused pool, return it.

    The pool and its hydrated rows go back to the caller, which curation needs
    and which saves hydrating the same ids twice. `None` means the pool came
    back empty — reported here, and nothing left to curate.

    Loads the embedder and the store here — after the plan — so `--plan-only`
    never pays for them, and so the two local models load one after the other
    rather than at once.
    """
    import time

    from mtg_rag.embed.encoder import QwenEncoder
    from mtg_rag.retrieve.pool import hydrate, retrieve
    from mtg_rag.retrieve.render import print_pool
    from mtg_rag.store.chroma import open_client

    print("\nLoading the embedder...")
    encoder = QwenEncoder()
    client = open_client(vector_dir)

    started = time.perf_counter()
    pool = retrieve(queries, constraints=constraints, frame=frame, client=client, encoder=encoder)
    elapsed = time.perf_counter() - started

    if not pool:
        print("No candidates. The constraints may be unsatisfiable for these queries.")
        return None

    rows = hydrate(frame, [candidate.oracle_id for candidate in pool])
    print(f"\n{len(pool)} candidates in {elapsed * 1000:.0f} ms\n")
    print_pool(pool, rows, explain=False)
    return pool, rows


def _curation_cards(pool: list[Candidate], rows: pl.DataFrame) -> list[CurationCard]:
    """Pair each hydrated row with the purposes of the searches that found it.

    The hand-off `curate` expects, which `curate/prompt.py` leaves to the caller
    so it stays a pure string transform. Driven by `rows` rather than by `pool`,
    because hydration drops ids the corpus no longer holds and this must drop
    them too.

    Purposes are de-duplicated but keep source order — a card found three times
    for "self-mill" says that once. They are curation's starting hypothesis for
    the card's role, not the answer ([ADR 0005]). Null corpus fields become
    empty strings, which is how the prompt knows to leave the line out.
    """
    from mtg_rag.curate.prompt import CurationCard

    purposes = {
        candidate.oracle_id: tuple(dict.fromkeys(source.purpose for source in candidate.sources))
        for candidate in pool
    }
    return [
        CurationCard(
            oracle_id=row["oracle_id"],
            name=row["name"],
            mana_cost=row["mana_cost"] or "",
            type_line=row["type_line"] or "",
            oracle_text=row["oracle_text"] or "",
            flavor_text=row["flavor_text"] or "",
            purposes=purposes[row["oracle_id"]],
        )
        for row in rows.iter_rows(named=True)
    ]


def _release_gpu_cache() -> None:
    """Hand the embedder's GPU blocks back before curation prompts the model.

    Retrieval is done by the time this runs, so the encoder and the store are
    unreachable — but torch's caching allocator keeps their VRAM reserved until
    it is asked for it back, and curation's prefill is the largest allocation
    the process makes. A no-op off CUDA.
    """
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _curate_and_print(
    theme: str,
    *,
    format_name: str,
    pool: list[Candidate],
    rows: pl.DataFrame,
    client: LLMClient,
) -> int:
    """Curate the pool and print the recommendation; the process exit code.

    Reuses the client the planner loaded. A malformed recommendation survives
    curation's one retry ([ADR 0024]) and arrives here as an exception; the user
    gets one line and a non-zero exit, not a traceback.
    """
    from mtg_rag.curate.config import CURATION_POOL_SIZE
    from mtg_rag.curate.curation import curate
    from mtg_rag.curate.parse import MalformedRecommendationError
    from mtg_rag.curate.render import print_recommendation

    _release_gpu_cache()

    # Curation sees the top of the pool, not all of it: attention cost grows
    # with the square of the prompt and the whole pool does not fit on the
    # target GPU (`curate/config.py` records the measurement). Cut on the rows
    # rather than on the pool, because `hydrate` may return fewer — every row
    # then still has a candidate behind it to read purposes from.
    shown = rows.head(CURATION_POOL_SIZE)
    if len(shown) < len(rows):
        print(f"\nCurating the top {len(shown)} of {len(rows)} candidates...\n")
    else:
        print(f"\nCurating {len(shown)} candidates...\n")

    try:
        recommendation = curate(
            theme, format_name=format_name, cards=_curation_cards(pool, shown), client=client
        )
    except MalformedRecommendationError as error:
        print(f"Could not read the model's recommendation: {error}", file=sys.stderr)
        return 1

    print_recommendation(recommendation, shown)
    return 0


def main(argv: list[str] | None = None) -> int:
    use_utf8_stdout()
    args = _parse_args(argv)

    # Validated before the model loads: an unknown format is a cheap, immediate
    # failure, not a reason to spend a weights download first. The choices are the
    # template files themselves — the planner is handed the whole template ([ADR
    # 0023]), so a format exists exactly when its `<format>.md` does.
    available = _available_formats()
    if args.format_name not in available:
        print(
            f"Unknown format {args.format_name!r}. Available: {', '.join(available)}",
            file=sys.stderr,
        )
        return 1

    if args.plan_only:
        return 0 if _plan(args) is not None else 1

    # Full path: check the corpus, index, and format before the planner loads, so
    # a missing build fails without a weights download. Constraints come from the
    # flags here, never from the plan the model returns next ([ADR 0001]).
    prepared = _prepare_retrieval(args)
    if prepared is None:
        return 1
    constraints, frame, vector_dir = prepared

    planned = _plan(args)
    if planned is None:
        return 1
    queries, client = planned

    searched = _search_and_print(
        queries, constraints=constraints, frame=frame, vector_dir=vector_dir
    )
    # An empty pool is a valid answer, already reported — an unsatisfiable
    # request is honest output rather than a failure, so this exits clean.
    if searched is None or args.pool_only:
        return 0
    pool, rows = searched

    return _curate_and_print(
        args.theme, format_name=args.format_name, pool=pool, rows=rows, client=client
    )


if __name__ == "__main__":
    raise SystemExit(main())
