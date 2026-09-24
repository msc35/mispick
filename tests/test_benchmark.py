"""The benchmark scripts. They point at other people's servers, so the rules matter."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmark"
SCRIPTS = sorted(BENCH.glob("*.py"))


class TestScriptsAreSound:
    @pytest.mark.parametrize("script", [p.name for p in SCRIPTS])
    def test_parses(self, script: str) -> None:
        ast.parse((BENCH / script).read_text())

    @pytest.mark.parametrize("script", [p.name for p in SCRIPTS])
    def test_imports_cleanly(self, script: str) -> None:
        """A NameError at the top of a 30-server sweep is an expensive way to find a typo."""
        result = subprocess.run(
            [sys.executable, "-c", f"import runpy,sys; sys.argv=['x','--help']; "
             f"runpy.run_path({str(BENCH / script)!r}, run_name='__main__')"],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        # --help exits 0 via SystemExit
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("script", [p.name for p in SCRIPTS])
    def test_never_calls_a_tool(self, script: str) -> None:
        """The benchmark's first rule. Enforced here as well as in the library."""
        tree = ast.parse((BENCH / script).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr != "call_tool", f"{script} references tool invocation"
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"call_tool", "tools/call"}


class TestPick:
    def test_flags_descriptions_that_imply_a_credential(self) -> None:
        sys.path.insert(0, str(BENCH))
        try:
            import importlib

            pick = importlib.import_module("pick")
        finally:
            sys.path.pop(0)

        assert not pick.looks_key_free({"description": "Needs a GitHub API key."})
        assert not pick.looks_key_free({"description": "Sign in with OAuth to continue."})
        assert pick.looks_key_free({"description": "Read and write local files."})

    def test_builds_a_runnable_command_per_ecosystem(self) -> None:
        sys.path.insert(0, str(BENCH))
        try:
            import importlib

            pick = importlib.import_module("pick")
        finally:
            sys.path.pop(0)

        npm = {"registryType": "npm", "identifier": "some-mcp", "version": "1.2.3"}
        assert pick.install_command(npm) == "npx -y some-mcp@1.2.3"
        pypi = {"registryType": "pypi", "identifier": "some_mcp", "version": "0.1.0"}
        assert pick.install_command(pypi) == "uvx --from some_mcp==0.1.0 some_mcp"

    def test_respects_the_registry_page_cap(self) -> None:
        """limit=101 is a 422 from the registry, so paging must stay at 100."""
        sys.path.insert(0, str(BENCH))
        try:
            import importlib

            pick = importlib.import_module("pick")
        finally:
            sys.path.pop(0)
        import inspect

        assert pick.PAGE_SIZE <= 100
        assert "version=latest" in inspect.getsource(pick.fetch_page), (
            "without version=latest the registry returns every version and double-counts"
        )


@pytest.fixture
def measured_site(tmp_path: Path) -> Path:
    """Run measure.py and build_site.py over two local snapshots with the mock backend."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    fixture = json.loads((ROOT / "tests/fixtures/confusing_tools.json").read_text())
    (snapshots / "confusable.json").write_text(json.dumps(fixture))

    clear = json.loads(json.dumps(fixture))
    for tool in clear["tools"]:
        if tool["name"] == "search_docs":
            tool["description"] = "Search the written product handbook for documented policy."
        if tool["name"] == "search_issues":
            tool["description"] = "Search reported bug tickets and their comment threads."
    (snapshots / "clear.json").write_text(json.dumps(clear))

    (snapshots / "index.json").write_text(
        json.dumps(
            {
                "servers": [
                    {
                        "name": "io.github.acme/confusable",
                        "title": "Confusable Server",
                        "repository": "https://github.com/acme/confusable",
                        "status": "captured",
                        "snapshot": "confusable.json",
                    },
                    {
                        "name": "io.github.acme/clear",
                        "title": "Clear Server",
                        "status": "captured",
                        "snapshot": "clear.json",
                    },
                    {
                        "name": "io.github.acme/locked",
                        "title": "Locked Server",
                        "status": "needs_credentials",
                        "reason": "ACME_API_KEY must be set",
                    },
                ]
            }
        )
    )

    results = tmp_path / "results"
    subprocess.run(
        [sys.executable, str(BENCH / "measure.py"), "--snapshots", str(snapshots),
         "--out", str(results), "--model", "mock", "--queries", "4", "--runs", "2",
         "--seed", "1"],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    site = tmp_path / "site"
    subprocess.run(
        [sys.executable, str(BENCH / "build_site.py"), "--results", str(results),
         "--out", str(site)],
        check=True,
        cwd=ROOT,
        capture_output=True,
    )
    return site


class TestSite:
    def test_builds_a_page_per_server_plus_an_index(self, measured_site: Path) -> None:
        assert (measured_site / "index.html").is_file()
        assert (measured_site / "servers" / "confusable.html").is_file()
        assert (measured_site / "servers" / "clear.html").is_file()
        assert (measured_site / ".nojekyll").is_file(), "Pages would otherwise run Jekyll"

    def test_states_the_model_n_k_and_date(self, measured_site: Path) -> None:
        import datetime as dt

        text = (measured_site / "index.html").read_text()
        assert "mock" in text
        assert "N=4" in text and "K=2" in text
        assert dt.date.today().isoformat() in text

    def test_wording_is_neutral(self, measured_site: Path) -> None:
        """"Most confused pair", never "worst server"."""
        text = (measured_site / "index.html").read_text().lower()
        assert "most confused pair" in text
        for word in ("worst", "bad server", "broken server", "failing server"):
            assert word not in text

    def test_links_each_server_to_its_own_repository(self, measured_site: Path) -> None:
        text = (measured_site / "index.html").read_text()
        assert "https://github.com/acme/confusable" in text

    def test_lists_what_it_could_not_measure(self, measured_site: Path) -> None:
        """Dropping skipped servers silently would overstate coverage."""
        text = (measured_site / "index.html").read_text()
        assert "Not measured" in text
        assert "Locked Server" in text
        assert "ACME_API_KEY" in text

    def test_carries_the_caveat(self, measured_site: Path) -> None:
        text = (measured_site / "index.html").read_text()
        assert "not a judgement of the servers" in text

    def test_server_pages_are_self_contained_apart_from_repo_links(
        self, measured_site: Path
    ) -> None:
        text = (measured_site / "servers" / "clear.html").read_text()
        assert "<link" not in text
        assert "src=" not in text

    def test_server_pages_link_back(self, measured_site: Path) -> None:
        text = (measured_site / "servers" / "clear.html").read_text()
        assert "back to the leaderboard" in text

    def test_the_clearer_server_scores_higher(self, measured_site: Path) -> None:
        """A sanity check on the whole pipeline: distinct descriptions should win."""
        import re

        text = (measured_site / "index.html").read_text()
        rows = re.findall(r"servers/(\w+)\.html'>(\d+)</a>", text)
        scores = {slug: int(score) for slug, score in rows}
        assert scores["clear"] > scores["confusable"], scores


class TestWorkflows:
    def test_pages_deploy_is_manual(self) -> None:
        """The leaderboard names other people's projects; a bad run must not auto-publish."""
        import yaml

        workflow = yaml.safe_load((ROOT / ".github/workflows/pages.yml").read_text())
        triggers = workflow.get("on") or workflow.get(True)
        assert set(triggers) == {"workflow_dispatch"}

    def test_benchmark_does_not_recapture_by_default(self) -> None:
        import yaml

        workflow = yaml.safe_load((ROOT / ".github/workflows/benchmark.yml").read_text())
        inputs = (workflow.get("on") or workflow.get(True))["workflow_dispatch"]["inputs"]
        assert inputs["refresh-snapshots"]["default"] is False

    def test_a_rerun_template_exists(self) -> None:
        template = ROOT / ".github/ISSUE_TEMPLATE/benchmark-rerun.md"
        assert template.is_file()
        assert "not a judgement of the server" in template.read_text()
