# MTG Theme-Deck RAG Recommender — Project Spec

## What this is
A retrieval-augmented recommendation system that suggests Magic: The Gathering cards from a natural-language description of a deck's theme or main mechanic (e.g. "spooky graveyard shenanigans", "everything is cats").

**Design stance:** optimize for fun, casual, and thematic play. Competitiveness, play rate, and tournament data are downprioritized — not the focus of this project, though not categorically off-limits.

**Primary format:** Commander (current scope).

## Core architecture decisions
- **Data source:** Scryfall bulk JSON dump (free, refreshed daily). Provides oracle text, type line, creature types, color identity, per-format legalities, prices (USD/EUR/MTGO tix), rulings, set, flavor text, and art image URLs (`art_crop`).
- **Retrieval unit:** one card = one record. No document chunking.
- **Hard constraints are filters, not prompts:** format legality and color identity are deterministic metadata filters applied at retrieval time — illegal or off-color cards never reach the LLM. Both are specified by the user in the UI; inferring them from the query is out of scope for now.
- **Soft guidance via format templates:** skill-style markdown files (e.g. `commander.md`) loaded on demand. They carry deck-composition heuristics (roughly 10 ramp / 10 draw / 8 removal alongside theme cards), the casual/social framing, and workflow guidance for the LLM.
- **Query planning — dynamic with hardcoded structure:** a single planner LLM call must output a typed schema, `[{query_text, purpose}]`. The format template tells it which roles to cover (theme payoffs, enablers, ramp, draw, removal…); the model decides the actual queries and how many. The app executes them in parallel, dedupes, and hands candidates to the curation call.
- **Cross-channel fusion:** if and when multiple embedding channels exist, they are combined via weighted reciprocal rank fusion (RRF) — never by comparing raw similarity scores across different embedding spaces.
- **Curation layer:** the LLM receives the retrieved pool and groups cards by role (payoffs, enablers, support packages), explaining why each fits the theme, guided by the format template.
- **Evaluation from day one:** a golden set of ~12 theme queries with known-good expected cards, plus a hit-rate script. Every later embedding or retrieval change is measured against this baseline.

## Current scope
Focus: overall code structure; the querying system is the main feature.

Components to implement:
- **Ingestion:** Scryfall bulk download, normalization to card records, nightly refresh job (prices come along for free).
- **Embeddings:** single text channel — card name + type line (incl. creature types) + oracle text.
- **Vector store:** local (Chroma or FAISS); ~30k unique cards, no infrastructure needed.
- **Retrieval:** metadata filters (format legality, color identity) + semantic search.
- **Planner call:** structured output, schema `[{query_text, purpose}]`.
- **Format template:** `commander.md`.
- **Curation call:** role grouping + theme-fit explanations.
- **UI:** format picker, color-identity picker, free-text theme input.
- **Evals:** golden query set + hit-rate script.

Deliverables: working end-to-end recommendation flow for Commander; baseline retrieval metrics on the golden set.
