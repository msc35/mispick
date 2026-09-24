"""CLI surface: exit codes and error messages that tell you what to do next."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

from typer.testing import CliRunner

from mispick.cli import EXIT_BELOW_THRESHOLD, EXIT_ERROR, EXIT_OK, app

runner = CliRunner()


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == EXIT_OK
    assert "mispick" in result.output


def test_tools_from_snapshot(snapshot_path: Path) -> None:
    result = runner.invoke(app, ["tools", "--snapshot", str(snapshot_path)])
    assert result.exit_code == EXIT_OK
    assert "search_docs" in result.output
    assert "search_issues" in result.output
    assert "acme-workspace" in result.output


def test_no_target_explains_the_options() -> None:
    result = runner.invoke(app, ["tools"])
    assert result.exit_code == EXIT_ERROR
    assert "No target given" in result.output
    assert "--snapshot" in result.output


def test_two_targets_is_an_error(snapshot_path: Path) -> None:
    result = runner.invoke(
        app, ["tools", "--snapshot", str(snapshot_path), "--cmd", "python -m whatever"]
    )
    assert result.exit_code == EXIT_ERROR
    assert "only one target" in result.output


def test_missing_snapshot_names_the_path(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tools", "--snapshot", str(tmp_path / "nope.json")])
    assert result.exit_code == EXIT_ERROR
    assert "No such snapshot file" in result.output


def test_empty_tool_list_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"tools": []}')
    result = runner.invoke(app, ["tools", "--snapshot", str(empty)])
    assert result.exit_code == EXIT_ERROR
    assert "no tools" in result.output


def test_snapshot_round_trip(tmp_path: Path) -> None:
    out = tmp_path / "captured.json"
    result = runner.invoke(
        app,
        ["snapshot", "--cmd", "python -m tests.fixtures.fixture_server", "-o", str(out)],
    )
    assert result.exit_code == EXIT_OK, result.output
    payload = json.loads(out.read_text())
    assert len(payload["tools"]) == 6
    assert payload["serverInfo"]["name"] == "acme-workspace"
    assert payload["protocolVersion"] == "2026-07-28"
    # and the file we just wrote loads back
    again = runner.invoke(app, ["tools", "--snapshot", str(out)])
    assert again.exit_code == EXIT_OK
    assert "refund_order" in again.output


def test_config_mode_reports_collisions() -> None:
    cfg = Path("tests/fixtures/two_servers.json")
    result = runner.invoke(app, ["tools", "--config", str(cfg)])
    assert result.exit_code == EXIT_OK, result.output
    assert "Cross-server name collisions" in result.output
    assert "search_docs" in result.output
    assert "wiki" in result.output and "workspace" in result.output


def test_only_limits_to_one_server() -> None:
    cfg = Path("tests/fixtures/two_servers.json")
    result = runner.invoke(app, ["tools", "--config", str(cfg), "--only", "wiki"])
    assert result.exit_code == EXIT_OK, result.output
    assert "publish_page" in result.output
    assert "refund_order" not in result.output


class TestRun:
    """The MVP command, driven with the offline mock backend."""

    def test_reports_the_planted_confusion(self, snapshot_path: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "run",
                "--snapshot",
                str(snapshot_path),
                "--model",
                "mock",
                "--cache-dir",
                str(tmp_path),
                "-n",
                "8",
                "-k",
                "3",
                "--seed",
                "7",
            ],
        )
        assert result.exit_code == EXIT_OK, result.output
        assert "Confusion matrix" in result.output
        assert "score" in result.output
        # the two tools that share a description must show up as the worst pair
        assert "search_docs" in result.output and "search_issues" in result.output

    def test_every_report_states_model_n_k_and_date(
        self, snapshot_path: Path, tmp_path: Path
    ) -> None:
        """SPEC section 7: non-negotiable provenance."""
        result = runner.invoke(
            app,
            ["run", "--snapshot", str(snapshot_path), "--model", "mock",
             "--cache-dir", str(tmp_path), "-n", "4", "-k", "2"],
        )
        assert result.exit_code == EXIT_OK, result.output
        assert "mock" in result.output
        assert "N=4" in result.output
        assert "K=2" in result.output
        assert dt.date.today().isoformat() in result.output
        assert "Results depend on the model" in result.output

    def test_fail_under_returns_exit_one(self, snapshot_path: Path, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            ["run", "--snapshot", str(snapshot_path), "--model", "mock",
             "--cache-dir", str(tmp_path), "-n", "4", "-k", "1", "--fail-under", "100"],
        )
        assert result.exit_code == EXIT_BELOW_THRESHOLD
        assert "below --fail-under" in result.output

    def test_fail_under_passes_when_the_score_clears_it(
        self, snapshot_path: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["run", "--snapshot", str(snapshot_path), "--model", "mock",
             "--cache-dir", str(tmp_path), "-n", "4", "-k", "1", "--fail-under", "1"],
        )
        assert result.exit_code == EXIT_OK, result.output

    def test_an_unknown_provider_lists_the_real_ones(
        self, snapshot_path: Path, tmp_path: Path
    ) -> None:
        result = runner.invoke(
            app,
            ["run", "--snapshot", str(snapshot_path), "--model", "hotdog/wat",
             "--cache-dir", str(tmp_path)],
        )
        assert result.exit_code == EXIT_ERROR
        assert "Unknown model provider" in result.output
        assert "ollama" in result.output

    def test_a_dead_backend_explains_itself_rather_than_printing_zeros(
        self, snapshot_path: Path, tmp_path: Path
    ) -> None:
        """Pointed at an Ollama that is not running, it must say so, not report 0%."""
        result = runner.invoke(
            app,
            ["run", "--snapshot", str(snapshot_path), "--model", "ollama/nope",
             "--cache-dir", str(tmp_path), "-n", "1", "-k", "1"],
        )
        assert result.exit_code == EXIT_ERROR
        assert "ollama" in result.output.lower()
