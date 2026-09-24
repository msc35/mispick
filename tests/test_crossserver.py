"""Cross-server analysis - the differentiator no competitor ships."""

from __future__ import annotations

import pytest

from mispick.crossserver import analyse, find_collisions
from mispick.metrics import compute
from mispick.select import RunConfig, RunResult
from mispick.types import Choice, Query, Tool, ToolSet


def two_server_set() -> ToolSet:
    return ToolSet(
        tools=[
            Tool(name="search_docs", description="Search for information.", server="wiki"),
            Tool(name="publish_page", description="Publish a draft page.", server="wiki"),
            Tool(name="search_docs", description="Search for information.", server="workspace"),
            Tool(name="close_ticket", description="Close a support ticket.", server="workspace"),
        ]
    )


def result_from(pairs: list[tuple[str, str]], tool_set: ToolSet) -> RunResult:
    queries = [
        Query(id=f"q{i}", text=f"query {i}", expected=exp) for i, (exp, _) in enumerate(pairs)
    ]
    choices = [
        Choice(query_id=f"q{i}", chosen=got, args_valid=True) for i, (_, got) in enumerate(pairs)
    ]
    return RunResult(
        tool_set=tool_set, queries=queries, choices=choices, config=RunConfig(model="mock", k=1)
    )


class TestCollisions:
    def test_finds_a_name_on_two_servers(self) -> None:
        found = find_collisions(two_server_set())
        assert len(found) == 1
        assert found[0].name == "search_docs"
        assert found[0].servers == ["wiki", "workspace"]

    def test_flags_identical_descriptions_as_worse(self) -> None:
        found = find_collisions(two_server_set())
        assert found[0].identical_descriptions is True

    def test_different_descriptions_are_a_milder_collision(self) -> None:
        ts = ToolSet(
            tools=[
                Tool(name="search", description="Search the wiki.", server="a"),
                Tool(name="search", description="Search the issue tracker.", server="b"),
            ]
        )
        assert find_collisions(ts)[0].identical_descriptions is False

    def test_a_single_server_has_no_collisions(self, tool_set: ToolSet) -> None:
        assert find_collisions(tool_set) == []

    def test_qualified_names_use_the_config_label(self) -> None:
        """The spec warns serverInfo.name is not unique, so we key on the config label."""
        collision = find_collisions(two_server_set())[0]
        assert collision.qualified == ["wiki:search_docs", "workspace:search_docs"]

    def test_three_servers_are_all_listed(self) -> None:
        ts = ToolSet(tools=[Tool(name="search", server=s) for s in ("a", "b", "c")])
        assert find_collisions(ts)[0].servers == ["a", "b", "c"]


class TestLeakage:
    def test_counts_a_trial_that_crossed_servers(self) -> None:
        ts = two_server_set()
        result = result_from([("workspace:search_docs", "wiki:search_docs")], ts)
        cross = analyse(result, compute(result))
        assert cross.cross_server_rate.hits == 1
        assert cross.cross_server_rate.total == 1
        assert cross.leaks == [("workspace:search_docs", "wiki:search_docs", 1)]

    def test_a_within_server_mistake_is_not_a_leak(self) -> None:
        ts = two_server_set()
        result = result_from([("wiki:search_docs", "wiki:publish_page")], ts)
        cross = analyse(result, compute(result))
        assert cross.cross_server_rate.hits == 0
        assert cross.leaks == []

    def test_tracks_lost_and_stolen_separately(self) -> None:
        ts = two_server_set()
        result = result_from(
            [
                ("workspace:search_docs", "wiki:search_docs"),
                ("workspace:search_docs", "wiki:search_docs"),
                ("wiki:publish_page", "workspace:close_ticket"),
            ],
            ts,
        )
        cross = analyse(result, compute(result))
        by_label = {s.label: s for s in cross.servers}
        assert by_label["workspace"].lost == 2
        assert by_label["workspace"].stolen == 1
        assert by_label["wiki"].stolen == 2
        assert by_label["wiki"].lost == 1

    def test_per_server_accuracy(self) -> None:
        ts = two_server_set()
        result = result_from(
            [
                ("wiki:search_docs", "wiki:search_docs"),
                ("wiki:publish_page", "wiki:publish_page"),
                ("workspace:search_docs", "wiki:search_docs"),
            ],
            ts,
        )
        cross = analyse(result, compute(result))
        by_label = {s.label: s for s in cross.servers}
        assert by_label["wiki"].accuracy.value == 1.0
        assert by_label["workspace"].accuracy.value == 0.0

    def test_single_server_runs_are_not_multi_server(self, tool_set: ToolSet) -> None:
        result = result_from([("search_docs", "search_docs")], tool_set)
        assert analyse(result, compute(result)).is_multi_server is False

    def test_choosing_nothing_is_not_a_cross_server_leak(self) -> None:
        ts = two_server_set()
        result = result_from([("wiki:search_docs", None)], ts)  # type: ignore[list-item]
        cross = analyse(result, compute(result))
        assert cross.cross_server_rate.hits == 0

    def test_worst_collision_prefers_identical_descriptions(self) -> None:
        ts = ToolSet(
            tools=[
                Tool(name="mild", description="Search the wiki.", server="a"),
                Tool(name="mild", description="Search the tracker.", server="b"),
                Tool(name="severe", description="Same words.", server="a"),
                Tool(name="severe", description="Same words.", server="b"),
            ]
        )
        result = result_from([], ts)
        worst = analyse(result, compute(result)).worst_collision
        assert worst is not None and worst.name == "severe"


class TestReportsIncludeIt:
    def test_json_carries_the_cross_server_block(self) -> None:
        from mispick.report import json as json_report

        ts = two_server_set()
        result = result_from([("workspace:search_docs", "wiki:search_docs")], ts)
        payload = json_report.build(result, compute(result))
        block = payload["crossServer"]
        assert block["isMultiServer"] is True
        assert block["nameCollisions"][0]["name"] == "search_docs"
        assert block["leaks"][0]["count"] == 1

    @pytest.mark.parametrize("renderer", ["markdown", "html"])
    def test_named_collision_appears_in_prose_reports(self, renderer: str) -> None:
        import importlib

        module = importlib.import_module(f"mispick.report.{renderer}")
        ts = two_server_set()
        result = result_from([("workspace:search_docs", "wiki:search_docs")], ts)
        text = module.render(result, compute(result))
        assert "Across servers" in text
        assert "search_docs" in text
        assert "descriptions are identical" in text

    def test_terminal_reports_the_leak(self) -> None:
        from rich.console import Console

        from mispick.report import terminal

        ts = two_server_set()
        result = result_from([("workspace:search_docs", "wiki:search_docs")], ts)
        console = Console(width=110, record=True, no_color=True)
        terminal.render(result, compute(result), console)
        out = console.export_text()
        assert "Across servers" in out
        assert "Cross-server confusions" in out
        assert "Name collisions" in out

    def test_single_server_report_omits_the_section(self, tool_set: ToolSet) -> None:
        from rich.console import Console

        from mispick.report import terminal

        result = result_from([("search_docs", "search_docs")], tool_set)
        console = Console(width=110, record=True, no_color=True)
        terminal.render(result, compute(result), console)
        assert "Across servers" not in console.export_text()
