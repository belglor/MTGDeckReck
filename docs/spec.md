# MTG Theme-Deck RAG Recommender — Project Spec

## What this is
MTGDeckReck: a RAG system that recommends Magic: The Gathering cards from natural-language user requests. Here are some example questions the chatbot should be able to answer to
- I want to make a commander deck with a spooky graveyard theme
- I am looking to build a deck that heavily plays with connive
- Here is a list of cards for my next commander deck. I am missing some cards, what would you recommend to add?
- This is my current deck, give me some suggestions on how to improve (do I have to much ramp? Should I add removal?)

Recommendations are following the user prompts and preferences, favoring flavor and theme rather than competitiveness or meta.

Downprioritize ranking or filtering by play rate, win rate, or tournament meta — this is a casual/thematic recommender first, not a competitive one. Meta signal isn't forbidden outright, just never the primary driver.

## Architecture

How the system works today. Where a decision has an ADR, the entry links it — the
ADR carries the reasoning and the alternatives; this section describes the result.

- **Data source:** Scryfall bulk JSON dump (free, refreshed daily). Provides oracle text, type line, creature types, color identity, per-format legalities, prices (USD/EUR/MTGO tix), rulings, set, flavor text, and art image URLs (`art_crop`).
- **Retrieval unit:** one card = one record. No document chunking. One record carries several vectors, which is not the same thing. See [ADR 0002](adr/0002-one-card-one-record.md).
- **Multi-channel embedding:** semantic text properties are embedded as separate channels — oracle text (card name folded in), flavor text, and type line incl. creature types — so a flavor-phrased request has a surface built for it. Structured facts (legality, color identity, price, rarity, mana value, set) are never embedded; they stay filters and sort keys. See [ADR 0007](adr/0007-multi-channel-embedding.md).
- **Rank fusion, never raw scores:** each channel returns its own ranking, and the planner issues several queries; these are combined by Reciprocal Rank Fusion over ordinal positions, `1 / (60 + rank)`, weighted uniformly. Raw similarity scores are never averaged across channels — they are not commensurable. See [ADR 0008](adr/0008-rrf-fusion-not-raw-scores.md).
- **Hard constraints are filters, not prompts:** format legality and color identity are deterministic metadata filters applied at retrieval time, so illegal or off-color cards never reach the LLM. Both are specified by the user in the UI — inferring them from the query is out of scope. See [ADR 0001](adr/0001-legality-color-as-filters-not-prompts.md).
- **Soft guidance via format templates:** skill-style markdown files (e.g. `commander.md`) loaded on demand. They carry deck-composition heuristics (roughly 10 ramp / 10 draw / 8 removal alongside theme cards), the casual/social framing, and workflow guidance for the LLM. The templates are LLM-generated and then maintained by hand — changing a heuristic means editing the markdown. Each is organized into stable named sections so a call can be given the relevant subset rather than the whole file. See [ADR 0003](adr/0003-sectioned-format-templates.md).
- **Query planning — dynamic with hardcoded structure:** a single planner LLM call must output a typed schema, `[{query_text, purpose}]`. The format template tells it which roles to cover (theme payoffs, enablers, ramp, draw, removal…); the model decides the actual queries and how many. The app executes them in parallel, dedupes, and hands candidates to the curation call. See [ADR 0004](adr/0004-planner-typed-query-schema.md).
- **Curation layer:** the LLM receives the retrieved pool and groups cards by role (payoffs, enablers, support packages), explaining why each fits the theme, guided by the format template. See [ADR 0005](adr/0005-curation-groups-by-role.md).
- **Evaluation from day one:** a golden set of ~12 mechanically-determined queries with known-good expected cards, plus a hit-rate script. Recall is measured on the retrieved candidate pool, not on the curated recommendation — curation is meant to vary. Every later embedding or retrieval change is measured against this baseline. See [ADR 0006](adr/0006-eval-measures-retrieval-recall.md).

## Current scope
Focus: overall code structure; the querying system is the main feature.

Components to implement:
- **Ingestion:** Scryfall bulk download, normalization to card records, nightly refresh job (prices come along for free).
- **Embeddings:** three channels — oracle text (card name folded in), flavor text, type line (incl. creature types).
- **Vector store:** local (Chroma or FAISS); ~30k unique cards, no infrastructure needed.
- **Retrieval:** metadata filters (format legality, color identity) + per-channel semantic search + RRF fusion into one candidate pool.
- **Planner call:** structured output, schema `[{query_text, purpose}]`.
- **Format template:** `commander.md`.
- **Curation call:** role grouping + theme-fit explanations.
- **UI:** format picker, color-identity picker, free-text theme input.
- **Evals:** golden query set + hit-rate script.

Deliverables: working end-to-end recommendation flow for Commander; baseline retrieval metrics on the golden set.
