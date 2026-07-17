from mtg_rag.retrieve.rank import Candidate, rank_candidates


def test_rank_candidates_orders_best_first() -> None:
    candidates = [
        Candidate(name="Sol Ring", theme_score=0.1, win_rate=0.9),
        Candidate(name="Gravecrawler", theme_score=0.9, win_rate=0.4),
        Candidate(name="Blood Artist", theme_score=0.8, win_rate=0.6),
    ]

    ranked = rank_candidates(candidates, limit=3)

    assert ranked[0].name == "Sol Ring"
