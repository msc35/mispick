"""Report renderers: every format must be valid, self-contained, and honest."""

from __future__ import annotations

import datetime as dt
import json as jsonlib
import re
import xml.etree.ElementTree as ET

import pytest

from mispick.metrics import Metrics, compute
from mispick.models.mock import MockBackend
from mispick.report import badge, html, markdown
from mispick.report import json as json_report
from mispick.report.provenance import provenance_of
from mispick.select import RunConfig, RunResult, run_selection
from mispick.types import Query, Tool, ToolSet


@pytest.fixture
async def measured(tool_set: ToolSet) -> tuple[RunResult, Metrics]:
    """A real small run through the mock backend."""
    from mispick.generate import build_query_set

    backend = MockBackend()
    queries = await build_query_set(backend, tool_set, n=4, no_tool_count=2, cache=None)
    result = await run_selection(
        backend, tool_set, queries, config=RunConfig(n=4, k=3, seed=5, temperature=0.7)
    )
    return result, compute(result)


class TestProvenanceEverywhere:
    """SPEC section 7: every report states the model, N, K and the date."""

    def test_json(self, measured: tuple[RunResult, Metrics]) -> None:
        payload = json_report.build(*measured)
        run = payload["run"]
        assert run["model"] == "mock"
        assert run["n"] == 4
        assert run["k"] == 3
        assert run["date"] == dt.date.today().isoformat()
        assert "depend on the model" in run["caveat"]

    def test_markdown(self, measured: tuple[RunResult, Metrics]) -> None:
        text = markdown.render(*measured)
        assert "mock" in text
        assert "N=4" in text and "K=3" in text
        assert dt.date.today().isoformat() in text
        assert "depend on the model" in text

    def test_html(self, measured: tuple[RunResult, Metrics]) -> None:
        text = html.render(*measured)
        assert "mock" in text
        assert "N=4" in text and "K=3" in text
        assert dt.date.today().isoformat() in text
        assert "depend on the model" in text


class TestJson:
    def test_is_valid_json_and_round_trips(self, measured: tuple[RunResult, Metrics]) -> None:
        payload = jsonlib.loads(json_report.render(*measured))
        assert payload["tool"] == "mispick"
        assert payload["schemaVersion"] == json_report.SCHEMA_VERSION

    def test_rates_carry_hits_total_and_interval(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        acc = json_report.build(*measured)["summary"]["accuracy"]
        assert acc["total"] > 0
        assert 0.0 <= acc["value"] <= 1.0
        low, high = acc["ci95"]
        assert low <= acc["value"] <= high

    def test_an_unmeasured_rate_is_null_not_zero(self, tool_set: ToolSet) -> None:
        """K=1 means no stability. Reporting 0 would be a lie; null is the truth."""
        result = RunResult(tool_set=tool_set, queries=[], choices=[], config=RunConfig(k=1))
        payload = json_report.build(result, compute(result))
        assert payload["summary"]["stability"]["value"] is None
        assert payload["summary"]["stability"]["ci95"] is None

    def test_includes_every_trial_with_its_expectation(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        result, _ = measured
        payload = json_report.build(*measured)
        assert len(payload["trials"]) == len(result.choices)
        assert all("expected" in t for t in payload["trials"])

    def test_matrix_rows_and_columns_are_declared(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        matrix = json_report.build(*measured)["confusionMatrix"]
        assert set(matrix["counts"]) <= set(matrix["rows"])
        for row in matrix["counts"].values():
            assert set(row) <= set(matrix["columns"])


class TestMarkdown:
    def test_fits_a_pr_comment(self, measured: tuple[RunResult, Metrics]) -> None:
        text = markdown.render(*measured)
        assert len(text) < 65536, "GitHub rejects comments above 65536 characters"

    def test_uses_collapsible_sections_for_the_bulk(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        text = markdown.render(*measured)
        assert text.count("<details>") >= 2

    def test_tables_are_rectangular(self, measured: tuple[RunResult, Metrics]) -> None:
        """A ragged markdown table renders as garbage on GitHub."""
        text = markdown.render(*measured)
        for block in re.findall(r"(?:^\|.*\|$\n?)+", text, re.MULTILINE):
            rows = [r for r in block.strip().splitlines() if r.startswith("|")]
            widths = {r.count("|") for r in rows}
            assert len(widths) == 1, f"ragged table: {rows[:3]}"

    def test_comparison_shows_the_delta(self, measured: tuple[RunResult, Metrics]) -> None:
        result, metrics = measured
        text = markdown.render_comparison((result, metrics), (result, metrics))
        assert f"{metrics.score} → {metrics.score}" in text

    def test_comparison_refuses_different_models(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        """A delta between two models is not a regression."""
        result, metrics = measured
        other = RunResult(
            tool_set=result.tool_set,
            queries=result.queries,
            choices=result.choices,
            config=RunConfig(model="something-else", k=3),
        )
        text = markdown.render_comparison((result, metrics), (other, compute(other)))
        assert "Cannot compare" in text
        assert "not a regression" in text

    def test_comparison_names_a_new_confusion(self, tool_set: ToolSet) -> None:
        queries = [
            Query(id="q1", text="find the policy", expected="search_docs"),
        ]
        clean = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[],
            config=RunConfig(model="mock", k=1),
        )
        from mispick.types import Choice

        clean.choices = [Choice(query_id="q1", chosen="search_docs", args_valid=True)]
        broken = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q1", chosen="search_issues", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        text = markdown.render_comparison(
            (broken, compute(broken)), (clean, compute(clean))
        )
        assert "New confusions" in text
        assert "search_issues" in text


class TestHtml:
    def test_is_entirely_self_contained(self, measured: tuple[RunResult, Metrics]) -> None:
        """SPEC section 7: must open from file:// with no network."""
        text = html.render(*measured)
        assert "http://" not in text
        assert "https://" not in text.replace('xmlns="http://www.w3.org/', "")
        assert "<link" not in text
        assert "src=" not in text

    def test_embeds_the_raw_data(self, measured: tuple[RunResult, Metrics]) -> None:
        text = html.render(*measured)
        match = re.search(
            r'<script type="application/json" id="mispick-data">(.+?)</script>', text, re.DOTALL
        )
        assert match, "the html report should carry its own data"
        assert jsonlib.loads(match.group(1))["tool"] == "mispick"

    def test_supports_dark_mode(self, measured: tuple[RunResult, Metrics]) -> None:
        assert "prefers-color-scheme: dark" in html.render(*measured)

    def test_escapes_a_malicious_tool_description(self) -> None:
        """A tool description is untrusted input; it must not become markup.

        The embedded JSON block is the dangerous part: HTML escaping does not apply inside
        `<script>`, but `</script>` still closes the element, so a description containing
        one would break out and the remainder would render as live markup.
        """
        nasty = '</script><img src=x onerror=alert(1)><script>alert("xss")</script>'
        tool_set = ToolSet(tools=[Tool(name="evil", description=nasty)])
        result = RunResult(tool_set=tool_set, queries=[], choices=[], config=RunConfig())
        text = html.render(result, compute(result))
        assert "<script>alert" not in text
        assert "</script><img" not in text

        # The invariant that actually matters: inside the data block no raw "<" survives,
        # so no tag can be formed however the description is crafted. The payload text may
        # still appear as inert JSON data, which is fine.
        match = re.search(r'id="mispick-data">(.+?)</script>', text, re.DOTALL)
        assert match
        block = match.group(1)
        assert "<" not in block

        # ...and the data survives intact for whoever reads the JSON back out.
        assert jsonlib.loads(block)["tools"][0]["description"] == nasty

    def test_has_one_row_per_matrix_row(self, measured: tuple[RunResult, Metrics]) -> None:
        _, metrics = measured
        text = html.render(*measured)
        for row in metrics.row_labels:
            assert f'<th class="row">{row}</th>' in text


class TestBadge:
    def test_is_valid_xml(self, measured: tuple[RunResult, Metrics]) -> None:
        _, metrics = measured
        svg = badge.render(metrics, model="ollama/qwen3.5:4b")
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_states_the_score_and_the_model(self, measured: tuple[RunResult, Metrics]) -> None:
        """A score without a model is not a fact about the server."""
        _, metrics = measured
        svg = badge.render(metrics, model="ollama/qwen3.5:4b")
        assert f"{metrics.score}/100" in svg
        assert "ollama/qwen3.5:4b" in svg

    @pytest.mark.parametrize(
        ("score", "expected"),
        [(100, badge.GREEN), (90, badge.YELLOW_GREEN), (75, badge.YELLOW),
         (60, badge.ORANGE), (10, badge.RED)],
    )
    def test_colour_tracks_the_score(self, score: int, expected: str) -> None:
        assert badge.colour_for(score) == expected

    def test_width_grows_with_the_model_name(self, measured: tuple[RunResult, Metrics]) -> None:
        _, metrics = measured
        short = ET.fromstring(badge.render(metrics, model="a"))
        long = ET.fromstring(badge.render(metrics, model="anthropic/claude-haiku-4-5-20251001"))
        assert int(long.get("width") or 0) > int(short.get("width") or 0)


class TestProvenanceObject:
    def test_one_line_omits_seed_when_absent(self, tool_set: ToolSet) -> None:
        result = RunResult(tool_set=tool_set, queries=[], choices=[], config=RunConfig(seed=None))
        assert "seed=" not in provenance_of(result).one_line

    def test_one_line_includes_seed_when_the_backend_honours_it(self, tool_set: ToolSet) -> None:
        result = RunResult(
            tool_set=tool_set,
            queries=[],
            choices=[],
            config=RunConfig(seed=42, supports_seed=True),
        )
        assert "seed=42" in provenance_of(result).one_line

    def test_one_line_marks_a_seed_the_backend_ignores(self, tool_set: ToolSet) -> None:
        """Anthropic takes no seed. Printing a bare "seed=42" would be a false claim."""
        result = RunResult(
            tool_set=tool_set,
            queries=[],
            choices=[],
            config=RunConfig(seed=42, supports_seed=False),
        )
        assert "seed=42-ignored" in provenance_of(result).one_line

    def test_one_line_marks_temperature_unsupported(self, tool_set: ToolSet) -> None:
        """The Messages API accepts no temperature, so the report must not imply one."""
        result = RunResult(
            tool_set=tool_set,
            queries=[],
            choices=[],
            config=RunConfig(temperature=0.7, supports_temperature=False),
        )
        line = provenance_of(result).one_line
        assert "temperature=n/a" in line
        assert "temperature=0.7" not in line


class TestTerminal:
    def test_provenance_survives_a_narrow_terminal(
        self, measured: tuple[RunResult, Metrics]
    ) -> None:
        """The date is not optional, so it must not be truncated by a panel subtitle."""
        from rich.console import Console

        from mispick.report import terminal

        result, metrics = measured
        console = Console(width=52, record=True, no_color=True)
        terminal.render(result, metrics, console)
        out = console.export_text()
        assert dt.date.today().isoformat() in out
        assert "N=4" in out and "K=3" in out
        assert "mock" in out

    def test_reports_no_confusion_when_there_is_none(self, tool_set: ToolSet) -> None:
        from rich.console import Console

        from mispick.report import terminal
        from mispick.types import Choice

        queries = [Query(id="q", text="find the policy", expected="search_docs")]
        result = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q", chosen="search_docs", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        console = Console(width=100, record=True, no_color=True)
        terminal.render(result, compute(result), console)
        assert "No tool was mistaken" in console.export_text()
