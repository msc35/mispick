"""Query generation, neighbour detection, and the cache."""

from __future__ import annotations

from pathlib import Path

import pytest

from mispick.generate import (
    GenerationPlan,
    QueryCache,
    build_no_tool_prompt,
    build_prompt,
    build_query_set,
    generate_for_tool,
    nearest_neighbour,
)
from mispick.models.mock import MockBackend
from mispick.types import Tool, ToolSet


class TestPlan:
    def test_default_is_the_spec_split(self) -> None:
        plan = GenerationPlan.for_n(8)
        assert (plan.straightforward, plan.paraphrased, plan.hard_negative) == (4, 2, 2)
        assert plan.total == 8

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 12, 20])
    def test_scales_to_any_n_without_losing_the_count(self, n: int) -> None:
        assert GenerationPlan.for_n(n).total == n

    def test_tiny_n_drops_the_extras_rather_than_going_negative(self) -> None:
        plan = GenerationPlan.for_n(1)
        assert plan.straightforward == 1
        assert plan.hard_negative == 0

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            GenerationPlan.for_n(0)


class TestNeighbour:
    def test_identical_descriptions_are_nearest(self) -> None:
        tools = [
            Tool(name="search_docs", description="Search for information."),
            Tool(name="search_issues", description="Search for information."),
            Tool(name="refund_order", description="Refund a customer's order in full."),
        ]
        assert nearest_neighbour(tools[0], tools).name == "search_issues"

    def test_picks_the_closest_by_wording(self) -> None:
        tools = [
            Tool(name="a", description="Send an email to a customer about an invoice."),
            Tool(name="b", description="Send an email to a customer about a receipt."),
            Tool(name="c", description="Restart the database cluster."),
        ]
        assert nearest_neighbour(tools[0], tools).name == "b"

    def test_is_none_when_alone(self) -> None:
        only = Tool(name="a", description="x")
        assert nearest_neighbour(only, [only]) is None

    def test_is_deterministic_on_ties(self) -> None:
        tools = [Tool(name=n, description="same") for n in ("c", "b", "a")]
        first = nearest_neighbour(tools[0], tools)
        assert first is not None
        assert nearest_neighbour(tools[0], list(reversed(tools))).name == first.name


class TestPrompt:
    def test_shows_the_target_description_and_only_the_neighbour_name(self) -> None:
        """The anti-leak rule: a neighbour's wording must never reach the generator."""
        target = Tool(name="search_docs", description="Search the handbook for a policy.")
        neighbour = Tool(name="search_issues", description="SECRETWORDING about bug reports.")
        prompt = build_prompt(target, neighbour, GenerationPlan())
        assert "Search the handbook for a policy." in prompt
        assert "search_issues" in prompt
        assert "SECRETWORDING" not in prompt

    def test_asks_for_the_right_counts(self) -> None:
        prompt = build_prompt(
            Tool(name="a", description="d"), Tool(name="b", description="e"), GenerationPlan()
        )
        assert "4 straightforward" in prompt
        assert "2 indirect" in prompt
        assert "2 requests that actually need b" in prompt

    def test_omits_hard_negatives_when_there_is_no_neighbour(self) -> None:
        prompt = build_prompt(Tool(name="a", description="d"), None, GenerationPlan())
        assert "requests that actually need" not in prompt
        assert "nearest other tool" not in prompt

    def test_no_tool_prompt_lists_every_tool(self) -> None:
        prompt = build_no_tool_prompt([Tool(name="a"), Tool(name="b")], 3)
        assert "a, b" in prompt
        assert "3 realistic user requests" in prompt


class TestGeneration:
    async def test_labels_hard_negatives_with_the_neighbour_as_expected(self) -> None:
        """A hard negative for T belongs to T's neighbour, so that is its expected answer."""
        tools = [
            Tool(name="search_docs", description="Search for information."),
            Tool(name="search_issues", description="Search for information."),
        ]
        queries = await generate_for_tool(
            MockBackend(), tools[0], tools, plan=GenerationPlan()
        )
        negatives = [q for q in queries if q.kind == "hard_negative"]
        assert negatives, "the plan asked for hard negatives"
        for q in negatives:
            assert q.expected == "search_issues"
            assert q.neighbour == "search_issues"
        for q in queries:
            if q.kind != "hard_negative":
                assert q.expected == "search_docs"

    async def test_never_returns_more_than_the_plan_asked_for(self) -> None:
        tools = [Tool(name="a", description="Do a thing."), Tool(name="b", description="Do b.")]
        plan = GenerationPlan(straightforward=1, paraphrased=1, hard_negative=1)
        queries = await generate_for_tool(MockBackend(), tools[0], tools, plan=plan)
        assert len(queries) == 3

    async def test_query_ids_are_unique(self, tool_set: ToolSet) -> None:
        queries = await build_query_set(MockBackend(), tool_set, n=8, cache=None)
        ids = [q.id for q in queries]
        assert len(ids) == len(set(ids))

    async def test_includes_no_tool_queries(self, tool_set: ToolSet) -> None:
        queries = await build_query_set(MockBackend(), tool_set, n=4, no_tool_count=3, cache=None)
        assert len([q for q in queries if q.expected is None]) == 3


class _CountingBackend(MockBackend):
    """Counts generate() calls, which is what the cache is meant to avoid."""

    def __init__(self) -> None:
        super().__init__()
        self.generate_calls = 0

    async def generate(self, prompt: str, **kwargs: object) -> str:  # type: ignore[override]
        self.generate_calls += 1
        return await super().generate(prompt)


class TestCache:
    async def test_second_run_makes_no_model_calls(self, tool_set: ToolSet, tmp_path: Path) -> None:
        cache = QueryCache.open(tmp_path)
        backend = MockBackend()
        await build_query_set(backend, tool_set, n=4, cache=cache)
        cache.save(generator=backend.name)
        generated = len(cache.data["tools"])
        assert generated == len(tool_set.tools)

        reloaded = QueryCache.open(tmp_path)
        cold = _CountingBackend()
        await build_query_set(cold, tool_set, n=4, cache=reloaded)
        assert cold.generate_calls == 0, "a warm cache must not hit the model"

    async def test_changing_one_description_regenerates_only_that_tool(
        self, tool_set: ToolSet, tmp_path: Path
    ) -> None:
        cache = QueryCache.open(tmp_path)
        await build_query_set(MockBackend(), tool_set, n=4, cache=cache)
        cache.save(generator="mock")

        target = tool_set.tools[0]
        edited_before = cache.fresh_for(target)
        assert edited_before is not None

        other = tool_set.tools[1]
        other_before = cache.fresh_for(other)

        target.description = "A completely different description now."
        assert cache.fresh_for(target) is None, "the edited tool must be regenerated"
        assert cache.fresh_for(other) == other_before, "its neighbours must be left alone"

    async def test_user_edits_survive(self, tool_set: ToolSet, tmp_path: Path) -> None:
        cache = QueryCache.open(tmp_path)
        await build_query_set(MockBackend(), tool_set, n=4, cache=cache)
        path = cache.save(generator="mock")

        text = path.read_text().replace("Sort out", "HAND EDITED")
        path.write_text(text)

        reloaded = QueryCache.open(tmp_path)
        queries = await build_query_set(MockBackend(), tool_set, n=4, cache=reloaded)
        assert any("HAND EDITED" in q.text for q in queries)

    async def test_regenerate_flag_ignores_the_cache(
        self, tool_set: ToolSet, tmp_path: Path
    ) -> None:
        cache = QueryCache.open(tmp_path)
        await build_query_set(MockBackend(), tool_set, n=4, cache=cache)
        cache.save(generator="mock")

        backend = _CountingBackend()
        await build_query_set(backend, tool_set, n=4, cache=cache, regenerate=True)
        assert backend.generate_calls > 0, "--regenerate must go back to the model"

    async def test_prunes_tools_that_no_longer_exist(
        self, tool_set: ToolSet, tmp_path: Path
    ) -> None:
        cache = QueryCache.open(tmp_path)
        await build_query_set(MockBackend(), tool_set, n=4, cache=cache)
        assert len(cache.data["tools"]) == 6

        smaller = ToolSet(tools=tool_set.tools[:2], servers=tool_set.servers)
        await build_query_set(MockBackend(), smaller, n=4, cache=cache)
        assert set(cache.data["tools"]) == {t.qualified_name for t in smaller.tools}

    def test_a_corrupt_cache_file_is_ignored_not_fatal(self, tmp_path: Path) -> None:
        (tmp_path / "queries.yaml").write_text("this: [is, not, the, right, shape")
        cache = QueryCache.open(tmp_path)
        assert cache.fresh_for(Tool(name="anything")) is None

    def test_saved_file_explains_itself(self, tmp_path: Path) -> None:
        cache = QueryCache.open(tmp_path)
        path = cache.save(generator="ollama/qwen3.5:4b")
        text = path.read_text()
        assert "ollama/qwen3.5:4b" in text
        assert "your test set" in text
