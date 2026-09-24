"""Selection: wire names, phantoms, K runs, and backend errors."""

from __future__ import annotations

from mispick.models.base import Selection, build_tool_payload, from_wire_name, wire_name
from mispick.models.mock import MockBackend, ScriptedBackend
from mispick.select import RunConfig, run_selection
from mispick.types import Query, Tool, ToolSet


class TestWireNames:
    def test_single_server_uses_the_bare_name(self) -> None:
        assert wire_name(Tool(name="search")) == "search"

    def test_multi_server_qualifies_and_escapes_the_colon(self) -> None:
        tool = Tool(name="search", server="docs")
        assert tool.qualified_name == "docs:search"
        assert wire_name(tool) == "docs__search"

    def test_round_trips(self) -> None:
        tools = [Tool(name="search", server="docs"), Tool(name="search", server="wiki")]
        for tool in tools:
            assert from_wire_name(wire_name(tool), tools) == tool.qualified_name

    def test_a_bare_name_resolves_when_unambiguous(self) -> None:
        tools = [Tool(name="search", server="docs"), Tool(name="publish", server="docs")]
        assert from_wire_name("search", tools) == "docs:search"

    def test_a_bare_name_stays_unresolved_when_ambiguous(self) -> None:
        """Two servers with the same tool name: guessing would fake a correct answer."""
        tools = [Tool(name="search", server="docs"), Tool(name="search", server="wiki")]
        assert from_wire_name("search", tools) is None

    def test_an_invented_name_does_not_resolve(self) -> None:
        assert from_wire_name("search_everything", [Tool(name="search")]) is None


class TestToolPayload:
    def test_renders_openai_function_definitions(self, tool_set: ToolSet) -> None:
        payload = build_tool_payload(tool_set.sorted_tools())
        assert len(payload) == 6
        entry = payload[0]
        assert entry["type"] == "function"
        assert set(entry["function"]) == {"name", "description", "parameters"}

    def test_a_missing_schema_still_declares_an_object(self) -> None:
        payload = build_tool_payload([Tool(name="a", input_schema={})])
        assert payload[0]["function"]["parameters"]["type"] == "object"

    def test_a_missing_description_becomes_empty_not_none(self) -> None:
        payload = build_tool_payload([Tool(name="a")])
        assert payload[0]["function"]["description"] == ""


class TestRunSelection:
    async def test_runs_every_query_k_times(self, tool_set: ToolSet) -> None:
        queries = [Query(id="q1", text="close ticket 5", expected="close_ticket")]
        result = await run_selection(
            MockBackend(), tool_set, queries, config=RunConfig(k=4, concurrency=2)
        )
        assert len(result.choices) == 4
        assert sorted(c.run for c in result.choices) == [0, 1, 2, 3]

    async def test_records_the_model_name_and_seed_support(self, tool_set: ToolSet) -> None:
        result = await run_selection(
            MockBackend(), tool_set, [Query(id="q", text="hi", expected=None)]
        )
        assert result.config.model == "mock"
        assert result.config.supports_seed is True

    async def test_validates_arguments_against_the_schema(self, tool_set: ToolSet) -> None:
        queries = [Query(id="q", text="close the ticket", expected="close_ticket")]
        result = await run_selection(MockBackend(), tool_set, queries, config=RunConfig(k=1))
        choice = result.choices[0]
        assert choice.chosen == "close_ticket"
        assert choice.args_valid is True

    async def test_a_phantom_tool_is_recorded_as_such(self, tool_set: ToolSet) -> None:
        backend = ScriptedBackend(["search_everything"])
        queries = [Query(id="q", text="find it", expected="search_docs")]
        result = await run_selection(backend, tool_set, queries, config=RunConfig(k=1))
        choice = result.choices[0]
        assert choice.chosen is None
        assert choice.phantom == "search_everything"

    async def test_choosing_nothing_is_recorded_as_none(self, tool_set: ToolSet) -> None:
        backend = ScriptedBackend([None])
        queries = [Query(id="q", text="what is the weather", expected=None)]
        result = await run_selection(backend, tool_set, queries, config=RunConfig(k=1))
        assert result.choices[0].chosen is None
        assert result.choices[0].phantom is None

    async def test_a_backend_error_is_kept_not_swallowed(self, tool_set: ToolSet) -> None:
        class Broken(MockBackend):
            async def choose(self, *args: object, **kwargs: object) -> Selection:
                return Selection(error="connection refused")

        result = await run_selection(
            Broken(), tool_set, [Query(id="q", text="x", expected="search_docs")],
            config=RunConfig(k=2),
        )
        assert all(c.error for c in result.choices)
        assert result.errors == ["connection refused"]

    async def test_seed_varies_per_run_so_k_runs_are_not_identical_by_construction(
        self, tool_set: ToolSet
    ) -> None:
        """If every run used the same seed, stability would be meaningless."""
        seen: list[int | None] = []

        class Recording(MockBackend):
            async def choose(
                self, tools, query, *, temperature=0.0, seed=None
            ):  # type: ignore[no-untyped-def]
                seen.append(seed)
                return await super().choose(tools, query, temperature=temperature, seed=seed)

        await run_selection(
            Recording(),
            tool_set,
            [Query(id="q", text="x", expected="search_docs")],
            config=RunConfig(k=3, seed=100, concurrency=1),
        )
        assert sorted(s for s in seen if s is not None) == [100, 101, 102]


class TestDeterminism:
    def test_temperature_zero_with_a_seed_is_reproducible(self) -> None:
        config = RunConfig(temperature=0.0, seed=1, supports_seed=True)
        assert config.deterministic

    def test_a_hot_run_is_not_reproducible(self) -> None:
        assert not RunConfig(temperature=0.7, seed=1, supports_seed=True).deterministic

    def test_a_backend_without_seed_support_is_not_reproducible(self) -> None:
        """Anthropic takes no seed, so we must not claim determinism."""
        assert not RunConfig(temperature=0.0, seed=5, supports_seed=False).deterministic
