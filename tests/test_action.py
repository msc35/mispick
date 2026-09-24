"""The GitHub Action definition. A broken action.yml fails only on someone else's PR."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "action.yml"
SCRIPTS = ROOT / "scripts"


@pytest.fixture(scope="module")
def action() -> dict:
    return yaml.safe_load(ACTION.read_text())


class TestActionDefinition:
    def test_is_a_composite_action(self, action: dict) -> None:
        assert action["runs"]["using"] == "composite"

    def test_every_target_is_an_input(self, action: dict) -> None:
        for name in ("cmd", "url", "config", "snapshot"):
            assert name in action["inputs"]

    def test_exposes_the_score_as_an_output(self, action: dict) -> None:
        assert set(action["outputs"]) >= {"score", "accuracy", "report", "skipped"}

    def test_defaults_to_deterministic(self, action: dict) -> None:
        """A check that moves with sampling noise is a flaky check."""
        assert action["inputs"]["temperature"]["default"] == "0"
        assert action["inputs"]["seed"]["default"] == "1"

    def test_has_no_default_model(self, action: dict) -> None:
        """Defaulting to Ollama in CI would time out; the skip path explains instead."""
        assert action["inputs"]["model"]["default"] == ""

    def test_every_referenced_script_exists(self, action: dict) -> None:
        for step in action["runs"]["steps"]:
            run = step.get("run", "")
            for token in run.split():
                if token.endswith(".sh"):
                    name = token.rsplit("/", 1)[-1]
                    assert (SCRIPTS / name).is_file(), f"missing script: {name}"

    def test_scripts_are_executable(self) -> None:
        for script in SCRIPTS.glob("*.sh"):
            assert script.stat().st_mode & 0o111, f"{script.name} is not executable"


class TestScripts:
    @pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
    def test_is_valid_bash(self, script: str) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPTS / script)], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("script", sorted(p.name for p in SCRIPTS.glob("*.sh")))
    def test_uses_strict_mode(self, script: str) -> None:
        assert "set -euo pipefail" in (SCRIPTS / script).read_text()

    def test_skips_cleanly_with_no_model(self, tmp_path: Path) -> None:
        """No key must mean "skipped with an explanation", never a red X."""
        (tmp_path / "tools.json").write_text(
            (ROOT / "tests/fixtures/confusing_tools.json").read_text()
        )
        outputs = tmp_path / "gh_output"
        outputs.touch()
        result = subprocess.run(
            ["bash", str(SCRIPTS / "measure.sh")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "GITHUB_OUTPUT": str(outputs),
                "MISPICK_SNAPSHOT": "tools.json",
                "MISPICK_MODEL": "",
                "OPENAI_API_KEY": "",
                "ANTHROPIC_API_KEY": "",
            },
        )
        assert result.returncode == 0, result.stderr
        assert "skipped=true" in outputs.read_text()
        comment = (tmp_path / "mispick-comment.md").read_text()
        assert "skipped" in comment
        assert "ANTHROPIC_API_KEY" in comment, "the skip must say how to enable the check"

    def test_no_target_is_an_error(self, tmp_path: Path) -> None:
        outputs = tmp_path / "gh_output"
        outputs.touch()
        result = subprocess.run(
            ["bash", str(SCRIPTS / "measure.sh")],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            env={
                "PATH": "/usr/bin:/bin:/usr/local/bin",
                "GITHUB_OUTPUT": str(outputs),
                "MISPICK_MODEL": "mock",
            },
        )
        assert result.returncode == 1
        assert "No target" in result.stdout + result.stderr


class TestCiWorkflow:
    def test_parses(self) -> None:
        workflow = yaml.safe_load((ROOT / ".github/workflows/ci.yml").read_text())
        # PyYAML reads a bare `on:` key as the boolean True.
        assert workflow.get("on") or workflow.get(True)
        assert "test" in workflow["jobs"]

    def test_runs_lint_types_and_tests(self) -> None:
        text = (ROOT / ".github/workflows/ci.yml").read_text()
        assert "ruff check" in text
        assert "mypy" in text
        assert "pytest" in text
