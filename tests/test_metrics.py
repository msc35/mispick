"""Metrics: the numbers must be right, and honest about uncertainty."""

from __future__ import annotations

import pytest

from mispick.metrics import (
    NONE,
    PHANTOM,
    Rate,
    accuracy_for_tools,
    compute,
    top_confused_pairs,
    wilson,
)
from mispick.select import RunConfig, RunResult, validate_arguments
from mispick.types import Choice, Query, Tool, ToolSet


def make_result(
    pairs: list[tuple[str | None, str | None]],
    *,
    k: int = 1,
    tools: list[str] | None = None,
    args_valid: bool | None = True,
) -> RunResult:
    """Build a RunResult from (expected, chosen) pairs, repeated K times."""
    names = tools or sorted({p for pair in pairs for p in pair if p})
    tool_set = ToolSet(tools=[Tool(name=n) for n in names])
    queries = [
        Query(
            id=f"q{i}",
            text=f"query {i}",
            expected=exp,
            kind="no_tool" if exp is None else "straightforward",
        )
        for i, (exp, _) in enumerate(pairs)
    ]
    choices = []
    for i, (_, got) in enumerate(pairs):
        for run in range(k):
            choices.append(
                Choice(
                    query_id=f"q{i}",
                    chosen=got,
                    run=run,
                    args_valid=args_valid if got else None,
                )
            )
    return RunResult(
        tool_set=tool_set, queries=queries, choices=choices, config=RunConfig(k=k)
    )


class TestWilson:
    def test_zero_of_ten_does_not_claim_certainty(self) -> None:
        low, high = wilson(0, 10)
        assert low == 0.0
        assert 0.25 < high < 0.35, "0/10 must still admit a real failure rate"

    def test_ten_of_ten_does_not_claim_certainty(self) -> None:
        low, high = wilson(10, 10)
        assert high == 1.0
        assert 0.65 < low < 0.75

    def test_interval_narrows_with_more_data(self) -> None:
        small = wilson(5, 10)
        large = wilson(500, 1000)
        assert (large[1] - large[0]) < (small[1] - small[0])

    def test_empty_is_not_a_crash(self) -> None:
        assert wilson(0, 0) == (0.0, 0.0)

    def test_rate_renders_the_interval(self) -> None:
        assert str(Rate(1, 2)) == "50% [9-91]", "1 of 2 tells you almost nothing"
        assert str(Rate(0, 0)) == "n/a"


class TestAccuracy:
    def test_all_correct(self) -> None:
        m = compute(make_result([("a", "a"), ("b", "b")]))
        assert m.accuracy.hits == 2
        assert m.accuracy.value == 1.0

    def test_counts_a_wrong_pick(self) -> None:
        m = compute(make_result([("a", "b"), ("a", "a")]))
        assert m.accuracy.value == 0.5

    def test_none_expected_and_none_chosen_is_correct(self) -> None:
        m = compute(make_result([(None, None)], tools=["a"]))
        assert m.accuracy.value == 1.0

    def test_errored_trials_are_excluded_not_counted_wrong(self) -> None:
        result = make_result([("a", "a")])
        result.choices.append(Choice(query_id="q0", chosen=None, error="backend exploded"))
        m = compute(result)
        assert m.accuracy.total == 1, "errored trials must not dilute accuracy"
        assert m.errored_trials == 1
        assert m.trials == 2


class TestConfusionMatrix:
    def test_records_expected_by_chosen(self) -> None:
        m = compute(make_result([("a", "b"), ("a", "b"), ("a", "a")]))
        assert m.matrix["a"] == {"b": 2, "a": 1}

    def test_has_a_none_row_and_column(self) -> None:
        m = compute(make_result([(None, "a"), ("a", None)]))
        assert m.matrix[NONE] == {"a": 1}
        assert m.matrix["a"] == {NONE: 1}
        assert NONE in m.column_labels

    def test_phantom_gets_its_own_column(self) -> None:
        result = make_result([("a", None)])
        result.choices[0].phantom = "search_everything"
        m = compute(result)
        assert m.matrix["a"] == {PHANTOM: 1}
        assert m.phantom_rate.hits == 1

    def test_pairs_ranked_worst_first(self) -> None:
        m = compute(make_result([("a", "b"), ("a", "b"), ("a", "b"), ("c", "d")]))
        assert m.confused_pairs[0].expected == "a"
        assert m.confused_pairs[0].chosen == "b"
        assert m.confused_pairs[0].count == 3
        assert m.confused_pairs[0].share == pytest.approx(1.0)

    def test_top_pairs_deduplicates_both_directions(self) -> None:
        """A→B and B→A are one pair to fix, not two."""
        m = compute(make_result([("a", "b"), ("b", "a")]))
        pairs = top_confused_pairs(m, limit=3)
        assert len(pairs) == 1

    def test_top_pairs_ignores_none_and_phantom(self) -> None:
        m = compute(make_result([("a", None), (None, "a")]))
        assert top_confused_pairs(m) == []


class TestPerTool:
    def test_precision_and_recall(self) -> None:
        # b is chosen 3 times: twice wrongly (for a), once rightly.
        m = compute(make_result([("a", "b"), ("a", "b"), ("b", "b")]))
        a, b = m.per_tool["a"], m.per_tool["b"]
        assert a.recall.hits == 0 and a.recall.total == 2
        assert b.recall.value == 1.0
        assert b.precision.hits == 1 and b.precision.total == 3

    def test_f1_is_zero_when_never_correct(self) -> None:
        m = compute(make_result([("a", "b")]))
        assert m.per_tool["a"].f1 == 0.0

    def test_every_live_tool_appears_even_if_never_chosen(self) -> None:
        m = compute(make_result([("a", "a")], tools=["a", "unused"]))
        assert "unused" in m.per_tool


class TestStability:
    def test_agreeing_runs_are_stable(self) -> None:
        m = compute(make_result([("a", "a")], k=3))
        assert m.stability.value == 1.0
        assert m.unstable_queries == []

    def test_disagreeing_runs_are_unstable(self) -> None:
        result = make_result([("a", "a")], k=3)
        result.choices[1].chosen = "b"
        m = compute(result)
        assert m.stability.value == 0.0
        assert m.unstable_queries == ["q0"]

    def test_k_of_one_reports_no_stability_rather_than_perfect(self) -> None:
        m = compute(make_result([("a", "a")], k=1))
        assert m.stability.total == 0, "K=1 measures nothing; it must not claim 100%"
        assert str(m.stability) == "n/a"


class TestOverTriggering:
    def test_calling_a_tool_on_a_no_tool_query_counts(self) -> None:
        m = compute(make_result([(None, "a"), (None, None)], tools=["a"]))
        assert m.over_trigger.hits == 1
        assert m.over_trigger.total == 2


class TestScore:
    def test_perfect_run_scores_100(self) -> None:
        m = compute(make_result([("a", "a"), ("b", "b")], k=3))
        assert m.score == 100

    def test_consistently_wrong_still_earns_the_stability_points(self) -> None:
        """Picking the wrong tool every single time is stable, which is the truth.

        Accuracy 0 and invalid args, but the three runs agree, so stability is 100% and the
        score is the 20 stability points. A run that is reliably wrong is a different
        finding from one that is randomly wrong, and the score should say so.
        """
        m = compute(make_result([("a", "b")], k=3, args_valid=False))
        assert m.accuracy.value == 0.0
        assert m.stability.value == 1.0
        assert m.score == 20

    def test_randomly_wrong_scores_below_consistently_wrong(self) -> None:
        result = make_result([("a", "b")], k=3, args_valid=False)
        result.choices[1].chosen = "c"
        m = compute(result)
        assert m.score == 0

    def test_weights_renormalise_when_stability_is_unmeasured(self) -> None:
        """With K=1 there is no stability, so the score must not be capped at 80."""
        m = compute(make_result([("a", "a")], k=1))
        assert m.score == 100

    def test_bad_arguments_cost_at_most_ten_points(self) -> None:
        m = compute(make_result([("a", "a")], k=3, args_valid=False))
        assert m.score == 90


class TestArgumentValidation:
    def test_valid_arguments_pass(self) -> None:
        tool = Tool(
            name="t",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        ok, errors = validate_arguments(tool, {"q": "hello"})
        assert ok and errors == []

    def test_missing_required_property_fails_with_a_reason(self) -> None:
        tool = Tool(
            name="t",
            input_schema={
                "type": "object",
                "properties": {"q": {"type": "string"}},
                "required": ["q"],
            },
        )
        ok, errors = validate_arguments(tool, {})
        assert not ok
        assert "q" in errors[0]

    def test_wrong_type_fails(self) -> None:
        tool = Tool(
            name="t",
            input_schema={"type": "object", "properties": {"n": {"type": "integer"}}},
        )
        ok, _ = validate_arguments(tool, {"n": "not a number"})
        assert not ok

    def test_resolves_defs_and_refs(self) -> None:
        """MCP allows $ref/$defs in inputSchema, so a naive validator would break."""
        tool = Tool(
            name="t",
            input_schema={
                "type": "object",
                "properties": {"who": {"$ref": "#/$defs/person"}},
                "required": ["who"],
                "$defs": {
                    "person": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                },
            },
        )
        ok, _ = validate_arguments(tool, {"who": {"name": "Ada"}})
        assert ok
        bad, errors = validate_arguments(tool, {"who": {}})
        assert not bad and errors

    def test_a_broken_schema_is_not_blamed_on_the_model(self) -> None:
        tool = Tool(name="t", input_schema={"type": "not-a-real-type"})
        ok, errors = validate_arguments(tool, {"anything": 1})
        assert ok and errors == []


class TestSubsetAccuracy:
    def test_restricts_to_the_named_tools(self) -> None:
        result = make_result([("a", "b"), ("c", "c")])
        rate = accuracy_for_tools(result, {"a", "b"})
        assert rate.total == 1
        assert rate.hits == 0
