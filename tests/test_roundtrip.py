"""A saved report must re-render identically, without re-measuring."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mispick.generate import build_query_set
from mispick.metrics import Metrics, compute
from mispick.models.mock import MockBackend
from mispick.report import json as json_report
from mispick.report.load import ReportError, from_payload, load
from mispick.select import RunConfig, RunResult, run_selection
from mispick.types import ToolSet


@pytest.fixture
async def measured(tool_set: ToolSet) -> tuple[RunResult, Metrics]:
    backend = MockBackend()
    queries = await build_query_set(backend, tool_set, n=4, no_tool_count=2, cache=None)
    result = await run_selection(
        backend, tool_set, queries, config=RunConfig(n=4, k=3, seed=5, temperature=0.7)
    )
    return result, compute(result)


class TestRoundTrip:
    def test_metrics_survive_a_round_trip(self, measured: tuple[RunResult, Metrics]) -> None:
        original_result, original = measured
        payload = json_report.build(original_result, original)
        _, reloaded = from_payload(payload)

        assert reloaded.score == original.score
        assert reloaded.accuracy.hits == original.accuracy.hits
        assert reloaded.accuracy.total == original.accuracy.total
        assert reloaded.stability.hits == original.stability.hits
        assert reloaded.arg_validity.hits == original.arg_validity.hits
        assert reloaded.matrix == original.matrix
        assert reloaded.row_labels == original.row_labels
        assert reloaded.column_labels == original.column_labels
        assert reloaded.unstable_queries == original.unstable_queries

    def test_the_token_estimate_survives(self, measured: tuple[RunResult, Metrics]) -> None:
        """It is computed from inputSchema, so the schema has to be in the report."""
        result, metrics = measured
        _, reloaded = from_payload(json_report.build(result, metrics))
        assert reloaded.token_estimate == metrics.token_estimate
        assert reloaded.token_estimate > 0

    def test_provenance_survives(self, measured: tuple[RunResult, Metrics]) -> None:
        result, metrics = measured
        reloaded_result, _ = from_payload(json_report.build(result, metrics))
        assert reloaded_result.config.model == result.config.model
        assert reloaded_result.config.n == result.config.n
        assert reloaded_result.config.k == result.config.k
        assert reloaded_result.config.seed == result.config.seed

    def test_cross_server_survives(self) -> None:
        from mispick.crossserver import analyse
        from mispick.types import Choice, Query, Tool

        tool_set = ToolSet(
            tools=[
                Tool(name="search", description="Search.", server="a"),
                Tool(name="search", description="Search.", server="b"),
            ]
        )
        queries = [Query(id="q", text="find it", expected="a:search")]
        result = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q", chosen="b:search", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        metrics = compute(result)
        reloaded_result, reloaded_metrics = from_payload(json_report.build(result, metrics))
        cross = analyse(reloaded_result, reloaded_metrics)
        assert cross.is_multi_server
        assert cross.cross_server_rate.hits == 1
        assert cross.collisions[0].name == "search"

    def test_reading_from_disk(
        self, measured: tuple[RunResult, Metrics], tmp_path: Path
    ) -> None:
        result, metrics = measured
        path = tmp_path / "r.json"
        path.write_text(json_report.render(result, metrics))
        _, reloaded = load(path)
        assert reloaded.score == metrics.score


class TestErrors:
    def test_a_missing_file_names_the_path(self, tmp_path: Path) -> None:
        with pytest.raises(ReportError, match="No such report file"):
            load(tmp_path / "nope.json")

    def test_invalid_json_says_so(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text("{oops")
        with pytest.raises(ReportError, match="not valid JSON"):
            load(bad)

    def test_someone_elses_json_is_rejected_with_advice(self, tmp_path: Path) -> None:
        other = tmp_path / "other.json"
        other.write_text(json.dumps({"tool": "something-else"}))
        with pytest.raises(ReportError, match="does not look like a mispick"):
            load(other)

    def test_a_future_schema_is_refused_rather_than_misread(self) -> None:
        with pytest.raises(ReportError, match="different version"):
            from_payload({"tool": "mispick", "schemaVersion": 999})


class TestCli:
    def test_report_command_rerenders(
        self, measured: tuple[RunResult, Metrics], tmp_path: Path
    ) -> None:
        from typer.testing import CliRunner

        from mispick.cli import EXIT_OK, app

        result, metrics = measured
        path = tmp_path / "r.json"
        path.write_text(json_report.render(result, metrics))

        runner = CliRunner()
        out = runner.invoke(app, ["report", str(path), "--format", "md"])
        assert out.exit_code == EXIT_OK, out.output
        assert f"{metrics.score}/100" in out.output

    def test_compare_command_flags_a_drop(self, tool_set: ToolSet, tmp_path: Path) -> None:
        from typer.testing import CliRunner

        from mispick.cli import EXIT_BELOW_THRESHOLD, app
        from mispick.types import Choice, Query

        queries = [Query(id="q", text="find the policy", expected="search_docs")]
        good = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q", chosen="search_docs", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        bad = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q", chosen="search_issues", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        base = tmp_path / "base.json"
        head = tmp_path / "head.json"
        base.write_text(json_report.render(good, compute(good)))
        head.write_text(json_report.render(bad, compute(bad)))

        runner = CliRunner()
        out = runner.invoke(app, ["compare", str(base), str(head), "--max-drop", "5"])
        assert out.exit_code == EXIT_BELOW_THRESHOLD
        assert "New confusions" in out.output
        assert "search_issues" in out.output
