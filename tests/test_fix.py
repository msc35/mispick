"""Fix mode. The point is that a rewrite has to earn its place."""

from __future__ import annotations

import json

import pytest

from mispick.fix import (
    ALPHA,
    MAX_GROWTH,
    MIN_ALLOWANCE,
    FixOutcome,
    FixReport,
    Rewrite,
    apply_rewrite,
    as_patch,
    build_rewrite_prompt,
    check_length,
    mcnemar_exact,
    measure,
    propose,
    run_fix_mode,
)
from mispick.metrics import ConfusedPair, Rate, compute
from mispick.models.base import Backend, BackendError, Selection
from mispick.models.mock import MockBackend
from mispick.select import RunConfig, RunResult, run_selection
from mispick.types import Choice, Query, Tool, ToolSet


class TestMcNemar:
    def test_no_change_is_no_evidence(self) -> None:
        assert mcnemar_exact(0, 0) == 1.0

    def test_one_improvement_is_not_evidence(self) -> None:
        """A single changed trial can never clear p <= 0.05, and must not pretend to."""
        assert mcnemar_exact(1, 0) > ALPHA

    def test_five_improvements_is_still_not_quite_evidence(self) -> None:
        """2 * (1/32) = 0.0625, just over the line. The test should not round in our favour."""
        assert mcnemar_exact(5, 0) == pytest.approx(0.0625)
        assert mcnemar_exact(5, 0) > ALPHA

    def test_six_improvements_and_none_broken_is_evidence(self) -> None:
        assert mcnemar_exact(6, 0) <= ALPHA

    def test_is_symmetric(self) -> None:
        assert mcnemar_exact(7, 2) == mcnemar_exact(2, 7)

    def test_a_wash_is_not_evidence(self) -> None:
        assert mcnemar_exact(10, 10) > ALPHA

    def test_never_exceeds_one(self) -> None:
        for i in range(8):
            assert 0.0 <= mcnemar_exact(i, i) <= 1.0


class TestLengthGuard:
    def _set(self, description: str) -> ToolSet:
        return ToolSet(tools=[Tool(name="a", description=description)])

    def test_a_modest_rewrite_passes(self) -> None:
        tool_set = self._set("Search docs.")
        rewrite = Rewrite(
            pair=ConfusedPair("a", "b", 1, 1.0),
            descriptions={"a": "Search the handbook for a written policy."},
        )
        assert check_length(rewrite, tool_set) is None

    def test_a_bloated_rewrite_is_rejected(self) -> None:
        """A rewrite must not win by crowding out its neighbours."""
        tool_set = self._set("Search docs.")
        rewrite = Rewrite(
            pair=ConfusedPair("a", "b", 1, 1.0),
            descriptions={"a": "x" * (MIN_ALLOWANCE + 1)},
        )
        reason = check_length(rewrite, tool_set)
        assert reason is not None and "budget" in reason

    def test_a_terse_description_gets_room_to_explain_itself(self) -> None:
        """The whole point of a rewrite is to say more than "Search for information."

        A pure ratio cap would allow 2.5 x 23 = 57 characters here, which is not enough to
        name the corpus and say when to prefer this tool. The floor exists for this case.
        """
        tool_set = self._set("Search for information.")
        good = (
            "Searches the written product handbook for policies and how-to guidance. Use this "
            "for documented procedure; use search_issues when the question is about a specific "
            "reported bug or ticket."
        )
        assert len(good) > MAX_GROWTH * len("Search for information.")
        assert check_length(
            Rewrite(pair=ConfusedPair("a", "b", 1, 1.0), descriptions={"a": good}), tool_set
        ) is None

    def test_a_long_original_is_capped_by_ratio_not_the_floor(self) -> None:
        long_original = "A very detailed description. " * 20
        tool_set = self._set(long_original)
        rewrite = Rewrite(
            pair=ConfusedPair("a", "b", 1, 1.0),
            descriptions={"a": "x" * int(len(long_original) * MAX_GROWTH) + "y"},
        )
        assert check_length(rewrite, tool_set) is not None

    def test_a_tool_with_no_description_gets_a_free_pass(self) -> None:
        tool_set = ToolSet(tools=[Tool(name="a", description=None)])
        rewrite = Rewrite(
            pair=ConfusedPair("a", "b", 1, 1.0),
            descriptions={"a": "A long and genuinely useful description of what this does."},
        )
        assert check_length(rewrite, tool_set) is None


class TestApplyRewrite:
    def test_leaves_the_original_untouched(self, tool_set: ToolSet) -> None:
        """SPEC section 3: never modify the user's tools, not even in memory."""
        original = tool_set.by_qualified_name()["search_docs"].description
        rewrite = Rewrite(
            pair=ConfusedPair("search_docs", "search_issues", 5, 0.5),
            descriptions={"search_docs": "Brand new wording."},
        )
        patched = apply_rewrite(tool_set, rewrite)
        assert patched.by_qualified_name()["search_docs"].description == "Brand new wording."
        assert tool_set.by_qualified_name()["search_docs"].description == original

    def test_only_touches_the_named_tools(self, tool_set: ToolSet) -> None:
        rewrite = Rewrite(
            pair=ConfusedPair("search_docs", "search_issues", 5, 0.5),
            descriptions={"search_docs": "New."},
        )
        patched = apply_rewrite(tool_set, rewrite)
        assert patched.by_qualified_name()["refund_order"].description == (
            tool_set.by_qualified_name()["refund_order"].description
        )


class TestPrompt:
    def test_shows_both_descriptions_and_the_confusion_rate(self, tool_set: ToolSet) -> None:
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        prompt = build_rewrite_prompt(pair, tool_set.by_qualified_name())
        assert "REWRITE" in prompt
        assert "search_docs" in prompt and "search_issues" in prompt
        assert "42%" in prompt

    def test_asks_for_a_cause_from_the_taxonomy(self, tool_set: ToolSet) -> None:
        """The labels come from Hasan et al., per SPEC section 4.7."""
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        prompt = build_rewrite_prompt(pair, tool_set.by_qualified_name())
        assert "unclear_purpose" in prompt
        assert "opaque_parameters" in prompt

    def test_caps_the_length_in_the_instructions(self, tool_set: ToolSet) -> None:
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        assert "2.5x" in build_rewrite_prompt(pair, tool_set.by_qualified_name())


class _RewriteBackend(Backend):
    """Returns a fixed rewrite, and selects perfectly once it is applied.

    Stands in for a model whose rewrite genuinely works, so the accept path is exercised.
    """

    name = "rewriter"
    supports_seed = True

    def __init__(self, descriptions: dict[str, str], cause: str = "unclear_purpose") -> None:
        self._descriptions = descriptions
        self._cause = cause

    async def choose(
        self, tools, query, *, temperature=0.0, seed=None
    ):  # type: ignore[no-untyped-def]
        # With the rewrite applied, pick correctly; otherwise always pick the first tool.
        new_text = set(self._descriptions.values())
        rewritten = {t.qualified_name for t in tools if t.description in new_text}
        if len(rewritten) == len(self._descriptions):
            want = query.split("|")[-1].strip()
            match = next((t for t in tools if t.qualified_name == want), None)
            if match:
                return Selection(chosen=match.name, arguments={})
        return Selection(chosen=tools[0].name, arguments={})

    async def generate(self, prompt, *, temperature=0.7, seed=None, max_tokens=8192):  # type: ignore[no-untyped-def]
        return json.dumps(
            {"descriptions": self._descriptions, "cause": self._cause, "note": "Disambiguated."}
        )


def _paired_setup() -> tuple[ToolSet, list[Query]]:
    tool_set = ToolSet(
        tools=[
            Tool(name="alpha", description="Do a thing."),
            Tool(name="beta", description="Do a thing."),
        ]
    )
    queries = [
        Query(id=f"a{i}", text="request | alpha", expected="alpha") for i in range(6)
    ] + [Query(id=f"b{i}", text="request | beta", expected="beta") for i in range(6)]
    return tool_set, queries


class TestMeasure:
    async def test_a_working_rewrite_is_accepted(self) -> None:
        tool_set, queries = _paired_setup()
        backend = _RewriteBackend(
            {
                "alpha": "Handle alpha work only, never beta work.",
                "beta": "Handle beta work only, never alpha work.",
            }
        )
        config = RunConfig(k=1)
        baseline = await run_selection(backend, tool_set, queries, config=config)
        pair = ConfusedPair("beta", "alpha", 6, 1.0)
        rewrite = await propose(backend, pair, tool_set)
        outcome = await measure(backend, tool_set, queries, baseline, rewrite, config=config)

        assert outcome.after.value > outcome.before.value
        assert outcome.improved > 0
        assert outcome.regressed == 0
        assert outcome.accepted, outcome.explanation
        assert outcome.verdict == "ACCEPTED"

    async def test_a_harmful_rewrite_is_rejected(self, tool_set: ToolSet) -> None:
        backend = MockBackend()
        queries = [
            Query(id="q1", text="close the ticket numbered 5", expected="close_ticket"),
        ]
        config = RunConfig(k=1)
        baseline = await run_selection(backend, tool_set, queries, config=config)
        rewrite = Rewrite(
            pair=ConfusedPair("close_ticket", "create_ticket", 1, 1.0),
            # deliberately makes close_ticket unfindable
            descriptions={"close_ticket": "Zzzz."},
        )
        outcome = await measure(backend, tool_set, queries, baseline, rewrite, config=config)
        assert not outcome.accepted

    async def test_an_improvement_with_too_little_evidence_is_inconclusive(self) -> None:
        """One fixed trial is an anecdote, not a result."""
        outcome = FixOutcome(
            rewrite=Rewrite(pair=ConfusedPair("a", "b", 1, 1.0), descriptions={}),
            before=Rate(0, 1),
            after=Rate(1, 1),
            p_value=mcnemar_exact(1, 0),
            improved=1,
            regressed=0,
        )
        assert outcome.verdict == "INCONCLUSIVE"
        assert not outcome.accepted
        assert "not enough to call it" in outcome.explanation

    async def test_an_oversized_rewrite_is_never_even_run(self) -> None:
        tool_set, queries = _paired_setup()
        backend = _RewriteBackend({"alpha": "x" * 500, "beta": "y" * 500})
        config = RunConfig(k=1)
        baseline = await run_selection(backend, tool_set, queries, config=config)
        pair = ConfusedPair("beta", "alpha", 6, 1.0)
        rewrite = await propose(backend, pair, tool_set)
        outcome = await measure(backend, tool_set, queries, baseline, rewrite, config=config)
        assert outcome.rejected_reason is not None
        assert outcome.after.value == outcome.before.value


class TestPropose:
    async def test_parses_the_cause_label(self, tool_set: ToolSet) -> None:
        backend = _RewriteBackend({"search_docs": "New.", "search_issues": "Other."},
                                  cause="opaque_parameters")
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        rewrite = await propose(backend, pair, tool_set)
        assert rewrite.cause == "opaque_parameters"

    async def test_an_unknown_cause_falls_back(self, tool_set: ToolSet) -> None:
        backend = _RewriteBackend({"search_docs": "New.", "search_issues": "Other."},
                                  cause="vibes")
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        assert (await propose(backend, pair, tool_set)).cause == "unclear_purpose"

    async def test_strips_markdown_from_a_description(self, tool_set: ToolSet) -> None:
        backend = _RewriteBackend({"search_docs": "**Search** the `docs`.",
                                   "search_issues": "Other."})
        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        rewrite = await propose(backend, pair, tool_set)
        assert rewrite.descriptions["search_docs"] == "Search the docs."

    async def test_a_reply_with_no_descriptions_is_an_error(self, tool_set: ToolSet) -> None:
        class Empty(MockBackend):
            async def generate(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
                return '{"cause": "unclear_purpose"}'

        pair = ConfusedPair("search_docs", "search_issues", 9, 0.42)
        with pytest.raises(BackendError, match="no 'descriptions'"):
            await propose(Empty(), pair, tool_set)

    async def test_an_unknown_tool_is_an_error(self, tool_set: ToolSet) -> None:
        pair = ConfusedPair("nope", "also_nope", 1, 1.0)
        with pytest.raises(BackendError, match="unknown tools"):
            await propose(MockBackend(), pair, tool_set)


class TestPatchOutput:
    def test_says_so_when_nothing_was_accepted(self, tool_set: ToolSet) -> None:
        report = FixReport(model="mock")
        assert "No rewrite improved" in as_patch(report, tool_set)

    def test_shows_a_diff_for_an_accepted_rewrite(self, tool_set: ToolSet) -> None:
        outcome = FixOutcome(
            rewrite=Rewrite(
                pair=ConfusedPair("search_docs", "search_issues", 9, 0.42),
                descriptions={"search_docs": "Search the written handbook for a policy."},
                cause="unclear_purpose",
                note="Named the corpus.",
            ),
            before=Rate(2, 10),
            after=Rate(9, 10),
            p_value=0.008,
            improved=7,
            regressed=0,
        )
        assert outcome.accepted
        text = as_patch(FixReport(model="mock", outcomes=[outcome]), tool_set)
        assert "--- a/search_docs" in text
        assert "+++ b/search_docs" in text
        assert "-Search for information." in text
        assert "+Search the written handbook for a policy." in text
        assert "p=0.008" in text
        assert "does not modify your source" in text


class TestRunFixMode:
    async def test_does_nothing_when_there_is_no_confusion(self, tool_set: ToolSet) -> None:
        queries = [Query(id="q", text="x", expected="search_docs")]
        result = RunResult(
            tool_set=tool_set,
            queries=queries,
            choices=[Choice(query_id="q", chosen="search_docs", args_valid=True)],
            config=RunConfig(model="mock", k=1),
        )
        report = await run_fix_mode(MockBackend(), tool_set, result, compute(result))
        assert report.outcomes == []

    async def test_tries_at_most_the_requested_number_of_pairs(self, tool_set: ToolSet) -> None:
        backend = MockBackend()
        queries = await _queries_for(backend, tool_set)
        baseline = await run_selection(backend, tool_set, queries, config=RunConfig(k=3, seed=1))
        report = await run_fix_mode(
            backend, tool_set, baseline, compute(baseline), limit=2
        )
        assert len(report.outcomes) <= 2

    async def test_a_backend_error_is_collected_not_fatal(self, tool_set: ToolSet) -> None:
        class Broken(MockBackend):
            async def generate(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
                raise BackendError("model went away")

        backend = Broken()
        queries = await _queries_for(MockBackend(), tool_set)
        baseline = await run_selection(backend, tool_set, queries, config=RunConfig(k=3, seed=1))
        report = await run_fix_mode(backend, tool_set, baseline, compute(baseline))
        assert report.outcomes == []
        assert any("model went away" in e for e in report.errors)


async def _queries_for(backend: MockBackend, tool_set: ToolSet) -> list[Query]:
    from mispick.generate import build_query_set

    return await build_query_set(backend, tool_set, n=4, no_tool_count=0, cache=None)
