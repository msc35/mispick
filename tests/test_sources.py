"""M1: loading tools from snapshots, live servers, and multi-server configs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mispick.sources.config import ConfigError, collisions, parse_config
from mispick.sources.snapshot import SnapshotError, load_snapshot, parse_snapshot
from mispick.types import ToolSet


class TestSnapshot:
    def test_loads_the_fixture(self, tool_set: ToolSet) -> None:
        assert len(tool_set.tools) == 6
        names = [t.name for t in tool_set.sorted_tools()]
        assert names == sorted(names), "sorted_tools must be deterministic"
        assert "search_docs" in names and "search_issues" in names

    def test_records_server_provenance(self, tool_set: ToolSet) -> None:
        info = tool_set.servers[0]
        assert info.name == "acme-workspace"
        assert info.version == "1.4.2"
        assert info.protocol_version == "2026-07-28"
        assert info.source == "snapshot"

    def test_reads_annotations_and_schema(self, tool_set: ToolSet) -> None:
        by_name = {t.name: t for t in tool_set.tools}
        assert by_name["search_docs"].annotations["readOnlyHint"] is True
        assert by_name["refund_order"].annotations["destructiveHint"] is True
        assert by_name["close_ticket"].input_schema["required"] == ["ticket_id"]

    def test_display_name_precedence(self, tool_set: ToolSet) -> None:
        by_name = {t.name: t for t in tool_set.tools}
        # title wins
        assert by_name["create_ticket"].display_name == "Create Support Ticket"
        # falls back to name
        assert by_name["search_docs"].display_name == "search_docs"

    @pytest.mark.parametrize(
        "payload",
        [
            {"tools": [{"name": "a"}]},
            {"result": {"tools": [{"name": "a"}]}},
            [{"name": "a"}],
        ],
        ids=["result-object", "jsonrpc-envelope", "bare-array"],
    )
    def test_accepts_all_three_shapes(self, payload: object) -> None:
        assert [t.name for t in parse_snapshot(payload).tools] == ["a"]

    def test_rejects_nonsense_with_a_helpful_message(self) -> None:
        with pytest.raises(SnapshotError, match="Could not find a tool list"):
            parse_snapshot({"nope": 1})

    def test_rejects_a_tool_with_no_name(self) -> None:
        with pytest.raises(SnapshotError, match="missing a string 'name'"):
            parse_snapshot([{"description": "no name here"}])

    def test_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(SnapshotError, match="No such snapshot file"):
            load_snapshot(tmp_path / "absent.json")

    def test_invalid_json_says_so(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        with pytest.raises(SnapshotError, match="not valid JSON"):
            load_snapshot(bad)

    def test_accepts_snake_case_input_schema(self) -> None:
        ts = parse_snapshot([{"name": "a", "input_schema": {"type": "object"}}])
        assert ts.tools[0].input_schema == {"type": "object"}


class TestFingerprint:
    def test_changes_when_a_description_changes(self, tool_set: ToolSet) -> None:
        before = tool_set.fingerprint()
        tool_set.tools[0].description = "something else entirely"
        assert tool_set.fingerprint() != before

    def test_is_stable_across_tool_reordering(self, tool_set: ToolSet) -> None:
        before = tool_set.fingerprint()
        tool_set.tools.reverse()
        assert tool_set.fingerprint() == before, "cache key must not depend on server order"

    def test_per_tool_fingerprint_ignores_annotations(self, tool_set: ToolSet) -> None:
        tool = tool_set.tools[0]
        before = tool.fingerprint()
        tool.annotations["readOnlyHint"] = False
        assert tool.fingerprint() == before


class TestLiveServer:
    async def test_stdio_against_the_fixture_server(self) -> None:
        from mispick.sources.server import load_stdio

        ts = await load_stdio("python -m tests.fixtures.fixture_server", label="fixture")
        assert len(ts.tools) == 6
        assert ts.servers[0].name == "acme-workspace"
        assert ts.servers[0].source == "stdio"
        assert ts.servers[0].protocol_version == "2026-07-28"

    async def test_in_process(self, fixture_server) -> None:
        from mispick.sources.server import load_in_process

        ts = await load_in_process(fixture_server)
        assert {t.name for t in ts.tools} == {
            "search_docs",
            "search_issues",
            "create_ticket",
            "close_ticket",
            "send_invoice",
            "refund_order",
        }

    async def test_empty_command_is_rejected(self) -> None:
        from mispick.sources.server import ServerError, load_stdio

        with pytest.raises(ServerError, match="Empty --cmd"):
            await load_stdio("   ")

    async def test_failure_to_start_is_explained(self) -> None:
        from mispick.sources.server import ServerError, load_stdio

        with pytest.raises(ServerError, match="Could not start or query"):
            await load_stdio("this-command-does-not-exist-12345")

    async def test_a_crashing_server_has_its_stderr_quoted(self) -> None:
        """A server that dies should tell the user why, not just 'failed'."""
        from mispick.sources.server import ServerError, load_stdio

        with pytest.raises(ServerError) as excinfo:
            await load_stdio("python -m tests.fixtures.crashing_server", label="crasher")
        message = str(excinfo.value)
        assert "The server's last output was:" in message
        assert "missing GITHUB_TOKEN" in message


class TestPagination:
    """The empty-string cursor is the bug this guards. See docs/research.md section 3."""

    async def test_follows_pages_and_stops_only_on_none(self) -> None:
        from mispick.sources.server import _collect

        pages = [
            _FakePage([{"name": "a"}], next_cursor=""),  # empty string: VALID, keep going
            _FakePage([{"name": "b"}], next_cursor="p2"),
            _FakePage([{"name": "c"}], next_cursor=None),  # only None ends it
        ]
        client = _FakeClient(pages)
        ts = await _collect(client, "fake", "snapshot")
        assert [t.name for t in ts.tools] == ["a", "b", "c"]
        assert client.cursors == [None, "", "p2"]

    async def test_refuses_a_cursor_loop(self) -> None:
        from mispick.sources.server import ServerError, _collect

        pages = [
            _FakePage([{"name": "a"}], next_cursor="same"),
            _FakePage([{"name": "b"}], next_cursor="same"),
        ]
        with pytest.raises(ServerError, match="repeating pagination cursor"):
            await _collect(_FakeClient(pages), "fake", "snapshot")


class _FakeTool:
    def __init__(self, raw: dict) -> None:
        self._raw = raw

    def model_dump(self, **_: object) -> dict:
        return dict(self._raw)


class _FakePage:
    def __init__(self, tools: list[dict], next_cursor: str | None) -> None:
        self.tools = [_FakeTool(t) for t in tools]
        self.next_cursor = next_cursor


class _FakeClient:
    server_info = None
    protocol_version = "2026-07-28"

    def __init__(self, pages: list[_FakePage]) -> None:
        self._pages = pages
        self.cursors: list[str | None] = []

    async def list_tools(self, cursor: str | None = None) -> _FakePage:
        self.cursors.append(cursor)
        return self._pages.pop(0)


class TestConfig:
    def test_parses_claude_desktop_shape(self) -> None:
        specs = parse_config(
            {
                "mcpServers": {
                    "docs": {"command": "python", "args": ["-m", "docs_server"]},
                    "remote": {"url": "https://example.com/mcp"},
                }
            }
        )
        assert {s.label for s in specs} == {"docs", "remote"}
        by_label = {s.label: s for s in specs}
        assert by_label["docs"].cmd == "python -m docs_server"
        assert by_label["remote"].url == "https://example.com/mcp"

    def test_skips_disabled_servers(self) -> None:
        specs = parse_config(
            {"mcpServers": {"on": {"command": "a"}, "off": {"command": "b", "disabled": True}}}
        )
        assert [s.label for s in specs] == ["on"]

    def test_rejects_a_config_with_no_servers(self) -> None:
        with pytest.raises(ConfigError, match="No 'mcpServers' key"):
            parse_config({"somethingElse": 3})

    def test_empty_server_map_is_an_error(self) -> None:
        with pytest.raises(ConfigError, match="no usable server entries"):
            parse_config({"mcpServers": {}})

    async def test_reports_a_broken_server_without_dying(self, tmp_path: Path) -> None:
        from mispick.sources.config import load_config

        cfg = tmp_path / "mcp.json"
        cfg.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        "good": {
                            "command": "python",
                            "args": ["-m", "tests.fixtures.fixture_server"],
                        },
                        "broken": {"command": "definitely-not-a-real-binary-99"},
                    }
                }
            )
        )
        ts, failures = await load_config(cfg)
        assert len(ts.tools) == 6, "the working server must still be measured"
        assert len(failures) == 1
        assert "broken" in failures[0]

    def test_finds_cross_server_collisions(self) -> None:
        ts = parse_snapshot([{"name": "search"}], server="docs")
        ts.tools += parse_snapshot([{"name": "search"}], server="issues").tools
        ts.tools += parse_snapshot([{"name": "unique"}], server="issues").tools
        found = collisions(ts)
        assert found == {"search": ["docs", "issues"]}

    def test_qualified_names_key_on_config_label(self) -> None:
        ts = parse_snapshot([{"name": "search"}], server="docs")
        assert ts.tools[0].qualified_name == "docs:search"
        assert parse_snapshot([{"name": "search"}]).tools[0].qualified_name == "search"
