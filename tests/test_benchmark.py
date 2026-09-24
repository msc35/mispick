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
    """Server selection. The registry has no popularity order, so this does the ranking."""

    @staticmethod
    def _pick():
        sys.path.insert(0, str(BENCH))
        try:
            import importlib

            return importlib.import_module("pick")
        finally:
            sys.path.pop(0)

    def test_skips_entries_that_imply_a_credential(self) -> None:
        pick = self._pick()
        entries = [
            {
                "server": {
                    "name": "x/keyed",
                    "description": "Needs a GitHub API key.",
                    "repository": {"url": "https://github.com/o/keyed"},
                    "packages": [
                        {"registryType": "npm", "identifier": "k", "transport": {"type": "stdio"}}
                    ],
                }
            },
            {
                "server": {
                    "name": "x/open",
                    "description": "Read and write local files.",
                    "repository": {"url": "https://github.com/o/open"},
                    "packages": [
                        {"registryType": "npm", "identifier": "o", "transport": {"type": "stdio"}}
                    ],
                }
            },
        ]
        assert [c["name"] for c in pick.candidates(entries)] == ["x/open"]

    def test_requires_a_github_repo_and_a_stdio_package(self) -> None:
        pick = self._pick()
        no_repo = {"server": {"name": "a", "packages": [
            {"registryType": "npm", "identifier": "a", "transport": {"type": "stdio"}}]}}
        no_pkg = {"server": {"name": "b", "repository": {"url": "https://github.com/o/b"}}}
        remote_only = {"server": {"name": "c", "repository": {"url": "https://github.com/o/c"},
                                  "remotes": [{"type": "streamable-http", "url": "https://x"}]}}
        assert pick.candidates([no_repo, no_pkg, remote_only]) == []

    def test_parses_owner_and_repo_from_the_url(self) -> None:
        pick = self._pick()
        entry = {"server": {"name": "n", "description": "fine",
                            "repository": {"url": "https://github.com/Some-Owner/the.repo.git"},
                            "packages": [{"registryType": "pypi", "identifier": "p",
                                          "transport": {"type": "stdio"}}]}}
        c = pick.candidates([entry])[0]
        assert c["owner"] == "Some-Owner"
        assert c["repo"] == "the.repo"

    def test_builds_a_runnable_command_per_ecosystem(self) -> None:
        pick = self._pick()
        npm = {"registryType": "npm", "identifier": "some-mcp", "version": "1.2.3"}
        assert pick.install_command(npm) == "npx -y some-mcp@1.2.3"
        pypi = {"registryType": "pypi", "identifier": "some_mcp", "version": "0.1.0"}
        assert pick.install_command(pypi) == "uvx --from some_mcp==0.1.0 some_mcp"

    def test_ranks_by_stars(self) -> None:
        pick = self._pick()
        items = [
            {"name": "low", "owner": "a", "repo": "low", "stars": 3},
            {"name": "high", "owner": "b", "repo": "high", "stars": 900},
            {"name": "mid", "owner": "c", "repo": "mid", "stars": 50},
        ]
        assert [c["name"] for c in pick.rank(items, 3, 2)] == ["high", "mid", "low"]

    def test_skips_archived_repositories(self) -> None:
        pick = self._pick()
        items = [
            {"name": "dead", "owner": "a", "repo": "dead", "stars": 9999, "archived": True},
            {"name": "alive", "owner": "b", "repo": "alive", "stars": 1},
        ]
        assert [c["name"] for c in pick.rank(items, 5, 2)] == ["alive"]

    def test_caps_one_owner_so_a_publisher_cannot_dominate(self) -> None:
        """The failure this exists for: 13 of the first 40 entries came from one publisher."""
        pick = self._pick()
        items = [
            {"name": f"spam{i}", "owner": "flood", "repo": f"r{i}", "stars": 100 - i}
            for i in range(10)
        ] + [{"name": "other", "owner": "someone", "repo": "x", "stars": 1}]
        picked = pick.rank(items, 10, 2)
        assert sum(1 for c in picked if c["owner"] == "flood") == 2
        assert "other" in [c["name"] for c in picked]

    def test_measures_one_repository_once(self) -> None:
        """Several registry names can point at the same repo."""
        pick = self._pick()
        items = [
            {"name": "x/thing", "owner": "o", "repo": "thing", "stars": 10},
            {"name": "x/thing-mcp", "owner": "o", "repo": "Thing", "stars": 10},
        ]
        assert len(pick.rank(items, 5, 5)) == 1

    def test_reference_servers_are_included_and_launchable(self) -> None:
        """The registry carries none of these, which is why they are hand-listed."""
        pick = self._pick()
        assert len(pick.REFERENCE_SERVERS) >= 4
        for server in pick.REFERENCE_SERVERS:
            assert server["cmd"].startswith("npx -y @modelcontextprotocol/")
            assert server["source"] == "reference"
            assert server["repository"]

    def test_respects_the_registry_page_cap(self) -> None:
        """limit=101 is a 422 from the registry, so paging must stay at 100."""
        import inspect

        pick = self._pick()
        assert pick.PAGE_SIZE <= 100
        assert "version=latest" in inspect.getsource(pick.fetch_registry), (
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


class TestOversizedServers:
    """One registry server ships 682 tools - 76% of the corpus by itself."""

    def test_a_huge_surface_is_reported_but_not_measured(self, tmp_path: Path) -> None:
        snapshots = tmp_path / "snapshots"
        snapshots.mkdir()
        base = json.loads((ROOT / "tests/fixtures/confusing_tools.json").read_text())
        (snapshots / "normal.json").write_text(json.dumps(base))

        huge = dict(base)
        huge["serverInfo"] = {"name": "huge", "version": "1.0"}
        huge["tools"] = [
            {
                "name": f"tool_{i}",
                "description": f"Does thing {i}.",
                "inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}},
            }
            for i in range(200)
        ]
        (snapshots / "huge.json").write_text(json.dumps(huge))
        (snapshots / "index.json").write_text(
            json.dumps(
                {
                    "servers": [
                        {"name": "x/normal", "title": "Normal", "status": "captured",
                         "snapshot": "normal.json"},
                        {"name": "x/huge", "title": "Huge Surface", "status": "captured",
                         "snapshot": "huge.json"},
                    ]
                }
            )
        )

        results = tmp_path / "results"
        subprocess.run(
            [sys.executable, str(BENCH / "measure.py"), "--snapshots", str(snapshots),
             "--out", str(results), "--model", "mock", "--queries", "2", "--runs", "1",
             "--seed", "1", "--max-tools", "120"],
            check=True, cwd=ROOT, capture_output=True,
        )
        index = json.loads((results / "index.json").read_text())

        assert [s["slug"] for s in index["servers"]] == ["normal"], "huge must not be measured"
        skipped = {s["registry_name"]: s for s in index["skipped"]}
        assert "x/huge" in skipped
        entry = skipped["x/huge"]
        assert entry["status"] == "too_many_tools"
        assert entry["tool_count"] == 200
        # The token cost is the finding, and it needs no model call.
        assert entry["tokens"] > 0
        assert "tokens on every request" in entry["reason"]

        site = tmp_path / "site"
        subprocess.run(
            [sys.executable, str(BENCH / "build_site.py"), "--results", str(results),
             "--out", str(site)],
            check=True, cwd=ROOT, capture_output=True,
        )
        page = (site / "index.html").read_text()
        assert "Huge Surface" in page
        assert "200" in page
        assert "too many tools" in page

    def test_the_limit_can_be_disabled(self, tmp_path: Path) -> None:
        """--max-tools 0 means measure everything, however long it takes."""
        source = (BENCH / "measure.py").read_text()
        assert "if args.max_tools:" in source, "0 must fall through to measuring"


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
