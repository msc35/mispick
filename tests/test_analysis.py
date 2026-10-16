"""The study's analysis scripts: feature definitions, exclusions, and the full pipeline offline.

The feature helpers need nothing beyond mispick. The model fit needs the `analysis` dependency
group (`uv sync --all-extras --group analysis`); without it those tests skip.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from analysis import features

ROOT = Path(__file__).resolve().parent.parent


class TestFeatureDefinitions:
    """Each one exactly as registered in docs/analysis-plan.md, section 5."""

    def test_name_tokens_split_on_separators_and_camel_case(self) -> None:
        assert features.name_tokens("get_file-info.v2") == ["get", "file", "info", "v2"]
        assert features.name_tokens("readMultipleFiles") == ["read", "multiple", "files"]

    def test_a_shared_prefix_is_not_overlap(self) -> None:
        names = ["workix_role_get", "workix_role_set", "workix_digest"]
        common = features.common_tokens(names)
        assert common == {"workix", "role"}
        assert features.name_jaccard("workix_role_get", "workix_digest", common) == 0.0

    def test_name_jaccard(self) -> None:
        assert features.name_jaccard("read_file", "read_text_file", set()) == pytest.approx(2 / 3)
        assert features.name_jaccard("echo", "echo", set()) == 1.0

    def test_param_doc_share(self) -> None:
        schema = {"properties": {"a": {"description": "x"}, "b": {}, "c": {"description": " "}}}
        assert features.param_doc_share(schema) == pytest.approx(1 / 3)
        assert features.param_doc_share({"type": "object"}) == 1.0
        assert features.param_doc_share(None) == 1.0

    @pytest.mark.parametrize(
        "text",
        [
            "Use this tool when the user wants a summary.",
            "Use it to fetch a page.",
            "Call when you need the raw bytes.",
            "Only for PDF files.",
            "Do not use for images.",
            "Don't use this for folders.",
            "Use instead of read_file for large files.",
            "Prefer this over search.",
        ],
    )
    def test_states_when_matches_the_registered_patterns(self, text: str) -> None:
        assert features.states_when(text) == 1

    def test_states_when_ignores_a_plain_description(self) -> None:
        assert features.states_when("Reads a file from disk and returns its contents.") == 0
        assert features.states_when("") == 0

    def test_cosine_is_undefined_for_an_empty_vector(self) -> None:
        assert features.cosine([0.0, 0.0], [1.0, 0.0]) is None
        assert features.cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_the_test_embedder_says_it_is_not_the_registered_model(self) -> None:
        assert features.HashEmbedder().describe()["kind"] == "hash"

    def test_pairs_are_ordered_and_skip_one_tool_servers(self, tmp_path: Path) -> None:
        snapshots = tmp_path / "snap"
        snapshots.mkdir()
        shutil.copy(ROOT / "tests/fixtures/confusing_tools.json", snapshots / "a.json")
        (snapshots / "solo.json").write_text(json.dumps({"tools": [{"name": "only"}]}))
        _, meta = features.build(snapshots, tmp_path / "out", features.HashEmbedder(), 120)
        assert meta["pairs"] == 6 * 5
        assert "skipped" in meta["servers"]["solo"]
        assert meta["embedder"]["kind"] == "hash"


def _analyze() -> Any:
    """The model-fitting script, or a skip when the analysis group is not installed."""
    for module in ("numpy", "pandas", "statsmodels", "matplotlib"):
        pytest.importorskip(module)
    return importlib.import_module("analysis.analyze")


def _measure_module() -> Any:
    sys.path.insert(0, str(ROOT / "benchmark"))
    try:
        return importlib.import_module("measure")
    finally:
        sys.path.pop(0)


@pytest.fixture
def mock_benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """Two fixture servers measured with the offline mock model, as the benchmark would."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    source = json.loads((ROOT / "tests/fixtures/confusing_tools.json").read_text())
    (snapshots / "alpha.json").write_text(json.dumps(source))
    for tool in source["tools"][:2]:
        tool["description"] = f"{tool.get('description') or ''} Use this when asked by name."
    (snapshots / "beta.json").write_text(json.dumps(source))

    results = tmp_path / "results"
    args = SimpleNamespace(
        snapshots=snapshots, out=results, model="mock", queries=8, runs=3, temperature=0.0,
        seed=1, concurrency=2, max_tools=120, only=None, rerun=False, rpm=0, tpm=0,
        budget_calls=0,
    )
    assert asyncio.run(_measure_module().main_async(args)) == 0
    return snapshots, results


class TestPipeline:
    def test_features_then_analysis_end_to_end(
        self, mock_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pytest.importorskip("statsmodels")
        pytest.importorskip("matplotlib")
        analyze = _analyze()

        snapshots, results = mock_benchmark
        out = tmp_path / "out"
        assert features.main(
            ["--snapshots", str(snapshots), "--out", str(out), "--embedder", "hash"]
        ) == 0
        assert analyze.main(
            ["--features", str(out / "features.csv"), "--results", str(results),
             "--out", str(out), "--bootstrap", "20"]
        ) == 0

        report = json.loads((out / "analysis.json").read_text())
        assert report["benchmark"]["model"] == "mock"
        assert report["benchmark"]["consistent"] is True
        assert report["counts"]["servers"] == 2
        effects = {e["predictor"]: e for e in report["primary"]["effects"]}
        assert set(effects) | set(report["primary"]["not_estimable"]) == set(analyze.PREDICTORS)
        for effect in effects.values():
            assert effect["lo"] <= effect["or"] <= effect["hi"]
            assert 0.0 <= effect["q"] <= 1.0
        assert report["verdicts"][0]["verdict"] in {"effect", "null", "not estimable"}
        assert set(report["sensitivity"]) == {
            "S1_cluster_bootstrap", "S2_with_hard_negatives", "S3_without_largest_server"
        }
        for chart in ("effects.png", "confusion_by_similarity.png", "server_accuracy.png"):
            assert chart in report["charts"]
            assert (out / chart).stat().st_size > 1000

    def test_primary_set_has_no_hard_negatives(
        self, mock_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        pytest.importorskip("pandas")
        analyze = _analyze()

        _, results = mock_benchmark
        trials = analyze.trial_frame(analyze.load_results(results))
        assert "hard_negative" in set(trials["kind"])
        snapshots, _ = mock_benchmark
        features.build(snapshots, tmp_path, features.HashEmbedder(), 120)
        import pandas as pd

        table = pd.read_csv(tmp_path / "features.csv")
        _, kept, _ = analyze.apply_exclusions(table, trials, analyze.PRIMARY_KINDS)
        assert set(kept["kind"]) <= analyze.PRIMARY_KINDS


class TestExclusions:
    """Section 4 of the plan, on hand-built data."""

    @staticmethod
    def _frames(errors: int = 0, per_tool: int = 18) -> tuple[Any, Any]:
        pd = pytest.importorskip("pandas")
        tools = ["a", "b", "c"]
        pairs = pd.DataFrame(
            [
                {"server": "s", "tool_a": x, "tool_b": y, "same_server": 1, "empty_a": 0,
                 "empty_b": 0}
                for x in tools for y in tools if x != y
            ]
        )
        rows = []
        for tool in tools:
            count = per_tool if tool != "c" else 11
            for i in range(count):
                rows.append({"server": "s", "query_id": f"{tool}{i}", "run": 0,
                             "kind": "straightforward", "expected": tool, "chosen": tool,
                             "error": i < errors})
        return pairs, pd.DataFrame(rows)

    def test_a_tool_under_twelve_valid_trials_is_excluded(self) -> None:
        analyze = _analyze()

        pairs, trials = self._frames()
        kept_pairs, _, report = analyze.apply_exclusions(pairs, trials, analyze.PRIMARY_KINDS)
        assert "s::c" in report["tools"]
        assert "c" not in set(kept_pairs["tool_a"]) | set(kept_pairs["tool_b"])

    def test_errored_trials_never_count_as_valid(self) -> None:
        analyze = _analyze()

        # 6 of 47 trials errored: 13%, under the 20% server limit.
        pairs, trials = self._frames(errors=2)
        _, kept, report = analyze.apply_exclusions(pairs, trials, analyze.PRIMARY_KINDS)
        assert report["servers"] == {}
        assert report["trials"]["errored"] == 6
        assert not kept["error"].any()
        assert set(kept["expected"]) == {"a", "b"}

    def test_a_server_with_too_many_errors_is_excluded(self) -> None:
        analyze = _analyze()

        # 30 of 47 trials errored.
        pairs, trials = self._frames(errors=10)
        _, kept, report = analyze.apply_exclusions(pairs, trials, analyze.PRIMARY_KINDS)
        assert "s" in report["servers"]
        assert kept.empty


@pytest.fixture
def frozen_benchmark(tmp_path: Path) -> tuple[Path, Path]:
    """Four fixture servers measured with the mock model, laid out like a frozen artifact."""
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    source = json.loads((ROOT / "tests/fixtures/confusing_tools.json").read_text())
    for i, name in enumerate(["alpha", "beta", "gamma", "delta"]):
        variant = json.loads(json.dumps(source))
        for tool in variant["tools"][:i]:
            tool["description"] = f"{tool.get('description') or ''} Variant {name}."
        (snapshots / f"{name}.json").write_text(json.dumps(variant))
    data = tmp_path / "data"
    args = SimpleNamespace(
        snapshots=snapshots, out=data, model="mock", queries=8, runs=3, temperature=0.0,
        seed=1, concurrency=2, max_tools=120, only=None, rerun=False, rpm=0, tpm=0,
        budget_calls=0,
    )
    assert asyncio.run(_measure_module().main_async(args)) == 0
    return snapshots, data


class TestTransfer:
    def _transfer(self) -> Any:
        for module in ("numpy", "scipy", "statsmodels"):
            pytest.importorskip(module)
        return importlib.import_module("analysis.transfer")

    def test_plan_makes_no_calls_and_estimates_tokens(
        self, frozen_benchmark: tuple[Path, Path], tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        transfer = self._transfer()
        snapshots, data = frozen_benchmark
        out = tmp_path / "out"
        assert transfer.main(["--data", str(data), "--snapshots", str(snapshots), "--out",
                              str(out), "plan", "--model", "mock"]) == 0
        plan = json.loads((out / "transfer_plan.json").read_text())
        assert len(plan["subset"]["tertiles"]) == 3
        assert len(plan["subset"]["servers"]) == 4  # fewer than 4 per tertile: take them all
        assert plan["per_model"]["mock"]["input_tokens"] > 0
        assert "Nothing was run" in capsys.readouterr().out
        assert not (out / "transfer").exists()

    def test_the_subset_is_deterministic(self, frozen_benchmark: tuple[Path, Path]) -> None:
        transfer = self._transfer()
        snapshots, data = frozen_benchmark
        index = json.loads((data / "index.json").read_text())
        assert transfer.select_subset(index, snapshots) == transfer.select_subset(index, snapshots)

    def test_run_refuses_without_approval(
        self, frozen_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transfer = self._transfer()
        snapshots, data = frozen_benchmark
        assert transfer.main(["--data", str(data), "--snapshots", str(snapshots), "--out",
                              str(tmp_path), "run", "--model", "mock"]) == 2

    def test_missing_frozen_queries_are_refused_not_regenerated(
        self, frozen_benchmark: tuple[Path, Path]
    ) -> None:
        transfer = self._transfer()
        snapshots, data = frozen_benchmark
        shutil.rmtree(data / ".queries" / "alpha")
        with pytest.raises(SystemExit, match="no frozen queries"):
            transfer.frozen_queries(data, "alpha", snapshots)

    def test_the_same_model_transfers_perfectly(
        self, frozen_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        transfer = self._transfer()
        snapshots, data = frozen_benchmark
        common = ["--data", str(data), "--snapshots", str(snapshots), "--out", str(tmp_path)]
        assert transfer.main([*common, "plan", "--model", "mock"]) == 0
        assert transfer.main([*common, "run", "--model", "mock", "--yes"]) == 0
        assert transfer.main([*common, "analyze"]) == 0

        report = json.loads((tmp_path / "transfer.json").read_text())
        entry = report["models"]["mock"]
        assert entry["complete"] is True
        assert entry["spearman_all_pairs"]["rho"] == pytest.approx(1.0)
        assert entry["top_k"]["top1_match_share"] == 1.0
        assert all(t["other_better"] == 0 and t["other_worse"] == 0 for t in entry["mcnemar_f2"])


class TestHumanVsGenerated:
    def _module(self) -> Any:
        return importlib.import_module("analysis.human_vs_generated")

    def _fill(self, path: Path) -> None:
        """Stand in for the person: write requests that name each tool's purpose."""
        import yaml

        data = yaml.safe_load(path.read_text())
        for name, entry in data["tools"].items():
            words = name.replace("_", " ")
            entry["queries"] = [
                {"text": f"please {words} for account {i}", "kind": "straightforward"}
                for i in range(4)
            ]
        data["no_tool"] = [{"text": "what is the capital of France"}]
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    def test_the_template_holds_descriptions_but_no_generated_queries(
        self, frozen_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        module = self._module()
        snapshots, data = frozen_benchmark
        human = tmp_path / "human"
        common = ["--human", str(human), "--snapshots", str(snapshots), "--servers", "alpha"]
        assert module.main([*common, "template"]) == 0
        text = (human / "alpha.yaml").read_text()
        generated = (data / ".queries" / "alpha" / "queries.yaml").read_text()
        import yaml

        for tool in yaml.safe_load(generated)["tools"].values():
            for query in tool["queries"]:
                assert query["text"] not in text
        assert "search_docs" in text
        assert module.main([*common, "check"]) == 1

    def test_check_catches_the_rules(self, frozen_benchmark: tuple[Path, Path]) -> None:
        module = self._module()
        snapshots, _ = frozen_benchmark
        from mispick.sources.snapshot import load_snapshot

        tool_set = load_snapshot(snapshots / "alpha.json")
        path = snapshots.parent / "bad.yaml"
        path.write_text(module.render_template("alpha", tool_set))
        import yaml

        doc = yaml.safe_load(path.read_text())
        first = next(iter(doc["tools"]))
        doc["tools"][first]["queries"] = [
            {"text": "same thing"}, {"text": "Same thing"}, {"text": "x", "kind": "hard_negative"}
        ]
        path.write_text(yaml.safe_dump(doc))
        _, problems = module.read_human(path, tool_set)
        joined = "\n".join(problems)
        assert "duplicate request" in joined
        assert "needs `neighbour:`" in joined
        assert "(need 20)" in joined

    def test_end_to_end_with_the_mock_model(
        self, frozen_benchmark: tuple[Path, Path], tmp_path: Path
    ) -> None:
        for module_name in ("numpy",):
            pytest.importorskip(module_name)
        module = self._module()
        snapshots, data = frozen_benchmark
        human = tmp_path / "human"
        out = tmp_path / "out"
        common = ["--human", str(human), "--snapshots", str(snapshots), "--data", str(data),
                  "--out", str(out), "--servers", "alpha,beta"]
        assert module.main([*common, "template"]) == 0
        for slug in ("alpha", "beta"):
            self._fill(human / f"{slug}.yaml")
        assert module.main([*common, "check"]) == 0
        assert module.main([*common, "run", "--model", "mock"]) == 2
        assert module.main([*common, "run", "--model", "mock", "--yes", "--rpm", "0",
                            "--tpm", "0"]) == 0
        assert module.main([*common, "analyze"]) == 0

        report = json.loads((out / "human_vs_generated.json").read_text())
        assert report["complete"] is True
        assert report["servers"] == ["alpha", "beta"]
        difference = report["difference"]
        assert difference["lo"] <= difference["difference_pp"] <= difference["hi"]
        assert difference["verdict"].split()[0] in {"equivalent", "different", "inconclusive"}
