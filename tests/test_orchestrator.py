import pytest

from mtg_rag.plan.orchestrator import PlannedQuery, QueryOrchestrator


class StubClient:
    """Returns a canned completion; the planning path never calls it."""

    def __init__(self, response: str = "{}") -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.fixture
def orchestrator() -> QueryOrchestrator:
    return QueryOrchestrator(StubClient())


def test_plans_one_query_per_role(orchestrator: QueryOrchestrator) -> None:
    queries = orchestrator.plan_queries("graveyard shenanigans", roles=["ramp", "removal"])

    assert queries == [
        PlannedQuery(query_text="graveyard shenanigans ramp", purpose="ramp"),
        PlannedQuery(query_text="graveyard shenanigans removal", purpose="removal"),
    ]


def test_falls_back_to_the_default_roles(orchestrator: QueryOrchestrator) -> None:
    queries = orchestrator.plan_queries("artifact tinkering")

    assert [q.purpose for q in queries] == [
        "theme payoff",
        "enabler",
        "ramp",
        "card draw",
        "removal",
    ]


def test_blank_theme_plans_nothing(orchestrator: QueryOrchestrator) -> None:
    assert orchestrator.plan_queries("   ") == []


def test_theme_is_trimmed_before_expansion(orchestrator: QueryOrchestrator) -> None:
    queries = orchestrator.plan_queries("  lifegain  ", roles=["ramp"])

    assert queries[0].query_text == "lifegain ramp"
