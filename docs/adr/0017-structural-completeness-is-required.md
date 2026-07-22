---
status: "accepted"
date: 2026-07-22
---

# A printing missing `layout` or `set_type` is refused at ingestion

## Context and Problem Statement

[ADR 0013](0013-structural-card-predicate.md) classifies an object as a card or a non-card from two fields, `layout` and `set_type`. It also decided what to do when one of them is absent: treat it as not-excluded, so the object is kept. The reasoning was that a card should be dropped only when a value actively matches an exclusion list, never merely because a field is missing.

Ingestion went further and *fabricated* the missing value — `layout=_lower_or_none(raw, "layout") or "unknown"`. A printing with no layout became `layout: "unknown"`, which matches no exclusion list, passes the predicate, and enters the index indistinguishable from a well-recorded card.

That is the wrong default for this system. The output is a recommendation shown to a person. An object whose own record is too incomplete to classify is exactly the kind of thing that should not be recommended, and there are ~34,000 real cards — losing a malformed handful costs nothing measurable, while surfacing an artifact costs trust.

## Considered Options

- Keep the permissive default: absent means not-excluded, and fabricate `"unknown"` for a missing layout
- Keep the permissive default, but stop fabricating — carry the null through
- Refuse the printing at ingestion, in `normalize_card`, the way a missing `oracle_id` is already refused
- Exclude it in the structural predicate instead, leaving ingestion permissive

## Decision Outcome

Chosen option: "Refuse the printing at ingestion", because the question being asked is *"is this record complete enough to use?"*, which is a boundary question, and the boundary already has a mechanism for it. `normalize_card` raises `MalformedCardError` for a missing `oracle_id`, and `just ingest` reports every rejection with its reason. Reusing that keeps one place deciding what a usable record is, and — importantly — makes the drop **visible** rather than silent, which was the strongest argument for the permissive default in the first place.

Excluding in the predicate instead was rejected because it widens [ADR 0013](0013-structural-card-predicate.md)'s scope from "is this object card-shaped" to "is this record trustworthy". Those are different questions, and spreading the second one across every consumer of the predicate is how they drift apart.

The required set is deliberately just `oracle_id`, `layout` and `set_type`. It is not "every field": dropping a real card because Scryfall omitted its `rarity` would discard a usable record over a field that decides nothing. These three are required because the pipeline cannot proceed without them — the first is the join key for every vector ([ADR 0010](0010-oracle-id-identity-key.md)), and the other two are the entire basis for the card/non-card judgement.

The predicate is aligned to match, in both its shapes: an absent `layout` or `set_type` now means *not a real card*, reversing [ADR 0013](0013-structural-card-predicate.md) on that one point. It is spelled out as `fill_null(True)` rather than left to polars' three-valued logic — which would drop the row anyway, but by accident rather than on purpose, and could be silently reversed by anyone rearranging the expression.

**This costs nothing today, and that is measured, not assumed.** Across all 116,138 printing objects in the `default_cards` snapshot — 113,494 of them English — **zero** are missing `layout`, `set_type`, `set`, `set_name`, `released_at` or `rarity`. Ingestion still rejects exactly the 81 printings it rejected before, all for a missing `oracle_id`.

It should also be clear about what this does *not* address. The failure that actually occurred during [ADR 0016](0016-ingest-every-printing.md)'s implementation was a `set_type` that was present, valid, and wrong for the card — a memorabilia reprint representing Tundra. Completeness checking is no defence against wrong-but-present values. This is hardening against a hypothetical, adopted because it is free and because it removes a fabricated value that was real.

### Consequences

- Good, because a missing structural field is now reported by name at ingestion instead of being invented and forgotten
- Good, because `CardRecord.set_type` and `layout` are non-optional, so the guarantee is in the type rather than in a convention
- Good, because both shapes of the predicate now agree on absent values, removing the branch most likely to drift between them
- Good, because it costs nothing today, verified against every printing in the snapshot
- Bad, because it reverses part of [ADR 0013](0013-structural-card-predicate.md) barely a week after it was written, on a case neither version has ever encountered
- Bad, because a future Scryfall object type that legitimately omits `set_type` would be rejected wholesale rather than degraded, and would need this decision revisited — the rejection is at least loud
- Bad, because it is defensive code for an unreachable case, and defensive code that never runs is never known to work beyond its tests
