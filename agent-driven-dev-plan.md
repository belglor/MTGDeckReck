# Agent-Driven Development Plan — TODOs & Learning Milestones

Companion to `mtg-rag-project-spec.md`. Each item is both a task and a skill: **Do** (the todo), **Learn** (the concept it teaches), **Done when** (a checkable outcome proving both).

Guiding principle for everything here: make the repo **legible** (agents find context) and **verifiable** (agents get fast, deterministic feedback).

## Toolbox (what gets installed, and why)
| Tool | Role |
|---|---|
| uv | Python env + dependency management with lockfile |
| ruff | Linting + formatting (one tool, fast) |
| pyright | Static type checking |
| pytest | Test runner |
| pre-commit | Local hooks: ruff, format, gitleaks |
| gitleaks | Secret scanning |
| just | Canonical command surface (`just test`, `just evals`) |
| GitHub Actions | CI: lint → typecheck → tests → evals |
| Claude Code + GitHub app | Coding agent + PR review agent |
| pydantic | Typed schemas (planner output, config) |
| Chroma or FAISS | Local vector store |
| Langfuse | LLM tracing/observability |
| MCP Python SDK | Internal dev-tool server |

---

## Phase 0 — Scaffolding & verification loop (before any RAG code, ~1 week)

### 0.1 Repo + environment
- [ ] Create GitHub repo; `uv init` with Python 3.12; commit `pyproject.toml` + `uv.lock`
- [ ] `src/mtg_rag/` layout with empty modules mirroring the architecture: `ingest/`, `embed/`, `store/`, `retrieve/`, `plan/`, `curate/`, `evals/`, plus `templates/commander.md`
- [ ] `.gitignore` (venv, `.env`, `data/`), minimal `README.md`
- **Learn:** modern Python project layout; why lockfiles matter for agents (reproducible = debuggable)
- **Done when:** fresh clone → `uv sync` → working env, no manual steps

### 0.2 Verification tooling
- [ ] ruff (lint + format) configured in `pyproject.toml`
- [ ] pyright in strict-ish mode; fix or explicitly ignore every finding
- [ ] pytest + one trivial test so the harness is proven
- [ ] pre-commit with ruff, ruff-format, gitleaks; `.env.example` committed, real `.env` ignored
- [ ] `justfile`: `setup`, `lint`, `typecheck`, `test`, `check` (all of the above), later `evals`, `ingest`
- **Learn:** the fast-feedback loop as *the* enabler of agent autonomy; secrets hygiene
- **Done when:** `just check` passes; a commit containing a fake API key is blocked by the hook (test this deliberately)

### 0.3 Agent context
- [ ] `CLAUDE.md` / `AGENTS.md` at root: what the project is, how to run everything (point at justfile), conventions, guardrails from the spec
- [ ] `docs/adr/` with a template + first ADRs: 0001 legality/color as filters not prompts; 0002 RRF fusion, never raw scores; 0003 planner structured-output schema; 0004 local vector store choice
- [ ] Copy the project spec into `docs/spec.md`
- **Learn:** context engineering; Architecture Decision Records as institutional memory for agents *and* future-you
- **Done when:** in a fresh Claude Code session you ask "how do I run the tests?" and "why RRF instead of score averaging?" — both answered correctly from repo files alone

### 0.4 GitHub spine
- [ ] Branch protection on `main`: PRs only, CI must pass
- [ ] Actions workflow: lint → typecheck → tests (with uv caching); evals job added in Phase 1
- [ ] Install the Claude Code GitHub app; enable agent PR review, you stay final human gate
- [ ] Issue template with an **acceptance criteria** field; labels: `v1`/`v2`/`v3`, `agent-ready`
- **Learn:** CI/CD fundamentals; PR discipline as a solo dev; agent-in-the-loop code review
- **Done when:** one trivial PR completes the full loop: agent opens it → CI green → agent review comment → you merge

### 0.5 First agent-driven task
- [ ] Write one tightly scoped issue (good candidate: "Scryfall bulk download module") with machine-checkable acceptance criteria; hand it to the coding agent end-to-end
- **Learn:** task scoping — the core skill of agent-driven development; most agent failures are scoping failures
- **Done when:** the agent completes it without mid-task clarification — or you can articulate exactly what was under-specified, and the next issue lands

---

## Phase 1 — During v1 (MVP pipeline)

### 1.1 Eval harness FIRST (before retrieval code exists)
- [ ] `evals/golden.yaml`: ~12 theme queries → expected card names
- [ ] Hit-rate script; `just evals`; wire into CI (report-only at first, threshold-gated later)
- **Learn:** eval-driven development — the RAG equivalent of test-first; defining "done" for fuzzy systems
- **Done when:** every PR shows an eval report; you've watched a change move the number

### 1.2 Ingestion & data hygiene
- [ ] Download Scryfall bulk `oracle_cards` (httpx), normalize to card records, store locally (parquet or sqlite)
- [ ] Idempotent refresh script; schedule it (cron or Actions `schedule:`)
- **Learn:** data pipeline basics: idempotency, caching, raw-vs-processed separation
- **Done when:** running refresh twice changes nothing; deleting `data/` and re-running restores everything

### 1.3 Core pipeline (the v1 feature work, agent-driven via issues)
- [ ] Embedding job → local vector store; metadata filters (format, color identity)
- [ ] Planner call with pydantic-validated structured output (`channel`/`weight` fields present, oracle-only default)
- [ ] Curation call + minimal UI
- **Learn:** structured outputs as the contract between LLM calls and code
- **Done when:** end-to-end theme query returns curated, legal, on-color recommendations; golden-set baseline recorded

### 1.4 Observability
- [ ] Langfuse (free tier or self-hosted): trace planner → retrieval → curation; log planner outputs and channel weights
- [ ] Prompts live as versioned files in the repo, not inline strings
- **Learn:** LLM tracing; prompt versioning as code
- **Done when:** given one bad recommendation, you open its trace and point at the failing stage in under a minute

### 1.5 Internal MCP server
- [ ] Small server (MCP Python SDK) exposing: `get_card(name)`, `search_cards(filters)`, `run_retrieval(query, weights)` against your local store; register it in Claude Code
- **Learn:** MCP protocol; tool design for agents (small, typed, composable)
- **Done when:** while debugging retrieval, the coding agent calls your MCP tools to inspect real data instead of guessing

---

## Phase 2 — During v2 (richer embeddings)

- [ ] Enrichment batch job (LLM tags per card) with caching so reruns cost ~nothing; tag set versioned
- [ ] SigLIP/CLIP embedding job over `art_crop`s; art channel behind a flag
- [ ] Experiment discipline: every channel/weight change is a PR whose description includes a baseline-vs-new golden-set table
- [ ] v2.5: cross-encoder reranker on top ~100, behind a flag
- **Learn:** batch LLM jobs and cost control; feature flags; A/B experiment hygiene on a fixed eval set
- **Done when:** you can name, with numbers, which channel earned its place — and at least one experiment got *rejected* by the evals

---

## Phase 3 — v3 (agents & parallelism)

- [ ] git worktrees + two background agents on independent, `agent-ready` issues simultaneously
- [ ] Build the first *product* agent (cross-vendor price comparison): explicit tool definitions, loop, stop conditions
- [ ] Then community-signal agents (Reddit primers, EDHREC theme pages — discovery, not meta-ranking)
- **Learn:** parallel agent orchestration; building agents (not just using them) — tool schemas, loops, termination
- **Done when:** two agent PRs merge in one day without stepping on each other; the price agent survives a flaky-source failure gracefully

---

## Sequencing rule of thumb
Phase 0 fully done before writing RAG code. 1.1 (evals) before 1.3 (pipeline). Observability and MCP can land mid-v1. Nothing in Phase 2 merges without an eval delta. Phase 3 waits until issues naturally decouple.
