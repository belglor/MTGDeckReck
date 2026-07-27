"""Mechanical defect checks over planner and curation output ([ADR 0026]).

Separate from `mtg_rag.evals` on purpose. The golden set measures *retrieval
lift* on the candidate pool and deliberately asserts nothing about what
curation does with it ([ADR 0006]); this package measures the *shape* of what
the two model stages emit — repeats, copied examples, claims that contradict
the corpus. Two instruments, two questions, so a reader of either does not have
to work out which one a check belongs to.

What it does not measure: whether a query is a good search or a rationale a
persuasive argument. Those need human judgment, and [ADR 0006]'s consequence
saying so still stands.
"""

from __future__ import annotations
