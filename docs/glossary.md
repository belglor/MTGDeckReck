# Glossary

Terms this project's docs, code and issues use without further explanation. Magic
terms are here only where the project treats them in a particular way.

### card vs printing

A **card** is the game object — "Sol Ring". A **printing** is one physical release
of it; Sol Ring has over a hundred. They differ in set, rarity, artwork, price,
flavor text and which platforms carry them, but never in rules text or legality.
The corpus stores one row per card.

### bulk snapshot

Scryfall publishes the whole card database as downloadable files, rebuilt daily,
listed at `api.scryfall.com/bulk-data`. Two matter here: `oracle_cards` has one
object per card, and `default_cards` has one per printing. We ingest
`default_cards` and collapse it ourselves — see [ADR 0016](adr/0016-ingest-every-printing.md).

### representative printing

The one printing chosen to supply a card's single-valued fields — set, rarity,
release date, prices. It is the most recent printing that is itself a real card.

### oracle_id

Scryfall's identifier for a *card*, stable across printings and reprints. Every
corpus row and every vector is keyed by it. Scryfall's other identifier, `id`,
names a printing and shifts between snapshots, so it is not used here.

### corpus

The local card table, `data/cards.parquet`. One row per card, and the single
source of truth for everything a filter reads.

### sidecar

A small JSON file recording what a build step produced and what it was built from
— `cards.meta.json`, `vectors.meta.json`. Comparing it against the current inputs
is how a re-run knows it has nothing to do.

### channel

One kind of card text, embedded separately: oracle text, flavor text, or type
line. Each gets its own vectors and its own search results, because the three are
different registers of language and averaging them blurs all three. See
[ADR 0007](adr/0007-multi-channel-embedding.md).

### allowlist

The set of card ids a search is permitted to return. Filters run over the corpus
first and produce this list; the vector search is then constrained to it, so an
illegal or off-color card cannot come back at all.

### candidate pool

The deduplicated set of cards retrieval hands to the LLM for curation. Recall is
measured here, not on the final recommendation.

### RRF (Reciprocal Rank Fusion)

How several ranked lists become one. Each card scores `1 / (60 + rank)` in every
list it appears in, and the scores are summed. It uses only positions, never
similarity scores, because scores from different channels are not comparable. See
[ADR 0008](adr/0008-rrf-fusion-not-raw-scores.md).

### format legality

Whether a card may be played in a given format — Commander, Modern, Vintage and so
on. Scryfall reports one value per format per card: `legal`, `not_legal`, `banned`,
or `restricted` (playable, but limited to one copy).

### color identity

Every color of mana in a card's cost *and* its rules text. Commander decks may
only contain cards whose color identity fits within the commander's, which makes
this a filter rather than a preference. A colorless card fits in any deck.

### structural non-card

An object in Scryfall's data that is not a playable card — a token, emblem, plane,
scheme, art-series print, or memorabilia item. They are ingested along with real
cards and excluded at query time. See
[ADR 0013](adr/0013-structural-card-predicate.md).
