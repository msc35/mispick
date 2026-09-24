"""Fix mode: propose description rewrites, then prove they help.

**Why the re-test is the whole point.** Hasan et al. (arXiv 2602.14878) rewrote tool
descriptions with an LLM across 231 tasks and found task success rose by a median 5.85
points - but **regressed in 16.67% of cases**. A tool that hands you an LLM's rewrite
without measuring it is handing you a one-in-six chance of making things worse. So every
rewrite here is re-run against the same queries and reported with a paired significance
test, and the ones that fail are shown too. The rejected rewrites are what make the
accepted ones believable.

This answers `whichtool`'s documented objection - "a generated rewrite would read as
authoritative while knowing nothing about it" - which is correct about *unmeasured*
rewrites. Ours is not authoritative because it is generated; it is credible because it
was measured.

**We never touch the user's source.** Rewrites are applied to in-memory copies and printed
as a suggestion.
"""

from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass, field

from mispick.generate import _extract_json
from mispick.metrics import ConfusedPair, Metrics, Rate, accuracy_for_tools, top_confused_pairs
from mispick.models.base import Backend, BackendError
from mispick.select import RunConfig, RunResult, run_selection
from mispick.types import SMELLS, Query, Tool, ToolSet

#: Accept a rewrite only when the paired test is at least this convincing.
ALPHA = 0.05

#: Cap a rewrite's length relative to the original, so a rewrite cannot "win" by being
#: three paragraphs long. The tool list is charged to every request.
MAX_GROWTH = 2.5

#: ...but a ratio alone is the wrong rule for a terse description. "Search for information."
#: is 23 characters, and 2.5x of that is 57 - not enough room to say which corpus it searches
#: and when to prefer it over its neighbour, which is exactly the fix we are asking for. So a
#: rewrite may always grow to this many characters regardless of ratio. It is roughly one
#: full sentence of purpose plus one of "use this when", and still cheap on every request.
MIN_ALLOWANCE = 240


@dataclass
class Rewrite:
    """A proposed pair of descriptions, plus why the model thinks they collide."""

    pair: ConfusedPair
    descriptions: dict[str, str]
    cause: str = "unclear_purpose"
    note: str = ""

    @property
    def names(self) -> set[str]:
        return {self.pair.expected, self.pair.chosen}


@dataclass
class FixOutcome:
    """One rewrite, measured."""

    rewrite: Rewrite
    before: Rate
    after: Rate
    p_value: float
    improved: int = 0
    regressed: int = 0
    rejected_reason: str | None = None

    @property
    def delta(self) -> float:
        return self.after.value - self.before.value

    @property
    def significant(self) -> bool:
        return self.p_value <= ALPHA

    @property
    def accepted(self) -> bool:
        return self.rejected_reason is None and self.delta > 0 and self.significant

    @property
    def verdict(self) -> str:
        if self.rejected_reason:
            return "REJECTED"
        if self.delta <= 0:
            return "REJECTED"
        return "ACCEPTED" if self.significant else "INCONCLUSIVE"

    @property
    def explanation(self) -> str:
        if self.rejected_reason:
            return self.rejected_reason
        if self.delta < 0:
            return "the rewrite made selection worse"
        if self.delta == 0:
            return "the rewrite changed nothing"
        if not self.significant:
            return (
                f"improved, but p={self.p_value:.2f} on {self.improved + self.regressed} "
                "changed trials is not enough to call it. Raise --runs and try again."
            )
        return f"p={self.p_value:.3f}"


@dataclass
class FixReport:
    outcomes: list[FixOutcome] = field(default_factory=list)
    model: str = "unknown"
    errors: list[str] = field(default_factory=list)

    @property
    def accepted(self) -> list[FixOutcome]:
        return [o for o in self.outcomes if o.accepted]

    @property
    def rejected(self) -> list[FixOutcome]:
        return [o for o in self.outcomes if not o.accepted]


def mcnemar_exact(improved: int, regressed: int) -> float:
    """Two-sided exact McNemar (a binomial sign test on the discordant trials).

    The before and after runs use the *same* queries, so the trials are paired and an
    unpaired comparison would be the wrong test - it would ignore that most trials did not
    change at all and dilute the evidence. Only the trials that changed carry information.

    Returns 1.0 when nothing changed, which correctly means "no evidence either way".
    """
    n = improved + regressed
    if n == 0:
        return 1.0
    k = min(improved, regressed)
    # P(X <= k) + P(X >= n-k) for X ~ Binomial(n, 0.5), which for the two-sided symmetric
    # case is 2 * P(X <= k), capped at 1.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def build_rewrite_prompt(pair: ConfusedPair, tools: dict[str, Tool]) -> str:
    """Ask for both descriptions at once, so they can be made to contrast."""
    a, b = tools[pair.expected], tools[pair.chosen]
    lines = [
        "REWRITE TASK.",
        "",
        "A language model is choosing between the tools of an API and keeps confusing these",
        f"two. In {pair.share * 100:.0f}% of requests meant for the first, it called the",
        "second instead.",
        "",
        f"  tool: {pair.expected}",
        f"  description: {a.description or '(none)'}",
        f"  parameters: {_params(a)}",
        "",
        f"  tool: {pair.chosen}",
        f"  description: {b.description or '(none)'}",
        f"  parameters: {_params(b)}",
        "",
        "Rewrite BOTH descriptions so a model can tell them apart.",
        "",
        "Rules:",
        "  - Keep the meaning exactly. Do not invent behaviour, parameters or limits that",
        "    are not already implied by the name and parameters above.",
        "  - Lead with what the tool does, in its own terms.",
        "  - Say when to use it rather than the other one. Naming the other tool is allowed.",
        f"  - Stay under {MAX_GROWTH:g}x the original length. The whole tool list is sent on",
        "    every request, so words cost money.",
        "  - Plain prose, no markdown, no examples.",
        "",
        f"Also label the likeliest cause, one of: {', '.join(SMELLS)}",
        "",
        "Reply with JSON only:",
        '{"descriptions": {"<tool name>": "<new description>", ...},'
        ' "cause": "<label>", "note": "<one sentence on what you changed>"}',
    ]
    return "\n".join(lines)


def _params(tool: Tool) -> str:
    props = (tool.input_schema or {}).get("properties") or {}
    if not props:
        return "(none)"
    return ", ".join(props)


async def propose(
    backend: Backend, pair: ConfusedPair, tool_set: ToolSet, *, seed: int | None = None
) -> Rewrite:
    """Ask the model for a rewrite of both descriptions in a confused pair."""
    tools = tool_set.by_qualified_name()
    missing = [n for n in (pair.expected, pair.chosen) if n not in tools]
    if missing:
        raise BackendError(f"cannot rewrite unknown tools: {', '.join(missing)}")

    reply = await backend.generate(
        build_rewrite_prompt(pair, tools), temperature=0.4, seed=seed
    )
    payload = _extract_json(reply)
    raw = payload.get("descriptions")
    if not isinstance(raw, dict):
        raise BackendError("the model's rewrite had no 'descriptions' object")

    descriptions: dict[str, str] = {}
    for name in (pair.expected, pair.chosen):
        value = raw.get(name) or raw.get(tools[name].name)
        if isinstance(value, str) and value.strip():
            descriptions[name] = _tidy(value)
    if not descriptions:
        raise BackendError("the model's rewrite named neither tool")

    cause = str(payload.get("cause") or "").strip().lower().replace(" ", "_")
    if cause not in SMELLS:
        cause = "unclear_purpose"
    return Rewrite(
        pair=pair,
        descriptions=descriptions,
        cause=cause,
        note=str(payload.get("note") or "").strip(),
    )


def _tidy(text: str) -> str:
    """Strip markdown and collapse whitespace - a description is plain text."""
    text = re.sub(r"[*_`#]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def budget_for(old: str) -> int:
    """How many characters a rewrite of `old` is allowed."""
    return max(int(MAX_GROWTH * len(old)), MIN_ALLOWANCE)


def check_length(rewrite: Rewrite, tool_set: ToolSet) -> str | None:
    """Reject a rewrite that wins by being longer. Returns a reason, or None if fine."""
    tools = tool_set.by_qualified_name()
    for name, new in rewrite.descriptions.items():
        old = tools[name].description or ""
        # A tool with no description at all gets a free pass - anything is an improvement.
        if not old:
            continue
        budget = budget_for(old)
        if len(new) > budget:
            return (
                f"the rewrite of {name} is {len(new)} characters, over its {budget}-character "
                f"budget ({MAX_GROWTH:g}x the original, or {MIN_ALLOWANCE} for a terse one); "
                "a longer description can win by crowding out its neighbours rather than by "
                "being clearer"
            )
    return None


def apply_rewrite(tool_set: ToolSet, rewrite: Rewrite) -> ToolSet:
    """A copy of the tool set with the new descriptions. The original is untouched."""
    patched = copy.deepcopy(tool_set)
    for tool in patched.tools:
        if tool.qualified_name in rewrite.descriptions:
            tool.description = rewrite.descriptions[tool.qualified_name]
    return patched


def _paired_counts(
    before: RunResult, after: RunResult, names: set[str]
) -> tuple[int, int]:
    """(improved, regressed) over trials that wanted one of `names`.

    Trials are matched on (query id, run index), which is what makes this paired.
    """
    wanted = {q.id for q in before.queries if q.expected in names}
    expected_of = {q.id: q.expected for q in before.queries}

    def outcomes(result: RunResult) -> dict[tuple[str, int], bool]:
        return {
            (c.query_id, c.run): (c.chosen == expected_of.get(c.query_id) and not c.phantom)
            for c in result.choices
            if c.query_id in wanted and not c.error
        }

    a, b = outcomes(before), outcomes(after)
    improved = regressed = 0
    for key, was_right in a.items():
        if key not in b:
            continue
        now_right = b[key]
        if now_right and not was_right:
            improved += 1
        elif was_right and not now_right:
            regressed += 1
    return improved, regressed


async def measure(
    backend: Backend,
    tool_set: ToolSet,
    queries: list[Query],
    baseline: RunResult,
    rewrite: Rewrite,
    *,
    config: RunConfig,
) -> FixOutcome:
    """Re-run the same queries against the rewritten descriptions and compare."""
    names = rewrite.names
    before = accuracy_for_tools(baseline, names)

    too_long = check_length(rewrite, tool_set)
    if too_long:
        return FixOutcome(
            rewrite=rewrite,
            before=before,
            after=before,
            p_value=1.0,
            rejected_reason=too_long,
        )

    patched = apply_rewrite(tool_set, rewrite)
    # The same queries, deliberately: a rewrite must be judged on the test set it was meant
    # to fix, not on a fresh one that might simply be easier.
    after_result = await run_selection(backend, patched, queries, config=config)
    after = accuracy_for_tools(after_result, names)
    improved, regressed = _paired_counts(baseline, after_result, names)
    return FixOutcome(
        rewrite=rewrite,
        before=before,
        after=after,
        p_value=mcnemar_exact(improved, regressed),
        improved=improved,
        regressed=regressed,
    )


async def run_fix_mode(
    backend: Backend,
    tool_set: ToolSet,
    baseline: RunResult,
    metrics: Metrics,
    *,
    limit: int = 3,
    on_progress: object = None,
) -> FixReport:
    """Propose and measure a rewrite for each of the worst confused pairs."""
    report = FixReport(model=backend.name)
    pairs = top_confused_pairs(metrics, limit=limit)
    if not pairs:
        return report

    config = RunConfig(
        n=baseline.config.n,
        k=baseline.config.k,
        temperature=baseline.config.temperature,
        seed=baseline.config.seed,
        concurrency=baseline.config.concurrency,
    )

    for pair in pairs:
        if callable(on_progress):
            on_progress(pair)
        try:
            rewrite = await propose(backend, pair, tool_set, seed=baseline.config.seed)
        except BackendError as exc:
            report.errors.append(f"{pair.label}: {exc}")
            continue
        try:
            outcome = await measure(
                backend, tool_set, baseline.queries, baseline, rewrite, config=config
            )
        except BackendError as exc:
            report.errors.append(f"{pair.label}: {exc}")
            continue
        report.outcomes.append(outcome)
    return report


def as_patch(report: FixReport, tool_set: ToolSet) -> str:
    """The accepted rewrites as a diff-style suggestion.

    A suggestion, not a patch to apply: tool descriptions live in the server's source in a
    form we cannot see - a decorator, a docstring, a JSON file - so mispick prints and never
    writes. See SPEC section 3.
    """
    accepted = report.accepted
    if not accepted:
        return "# No rewrite improved selection measurably. Nothing to suggest.\n"

    tools = tool_set.by_qualified_name()
    lines = [
        "# mispick suggested description rewrites",
        f"# Model: {report.model}. Only rewrites that measurably improved selection are here.",
        "# These are suggestions - mispick does not know where your descriptions live, and",
        "# does not modify your source.",
        "",
    ]
    for outcome in accepted:
        r = outcome.rewrite
        lines.append(f"## {r.pair.label}")
        lines.append(
            f"#  accuracy on this pair: {outcome.before} -> {outcome.after} "
            f"(p={outcome.p_value:.3f}, {outcome.improved} trials fixed, "
            f"{outcome.regressed} broken)"
        )
        lines.append(f"#  likely cause: {r.cause}")
        if r.note:
            lines.append(f"#  model's note: {r.note}")
        lines.append("")
        for name, new in sorted(r.descriptions.items()):
            old = tools[name].description or ""
            lines.append(f"--- a/{name}")
            lines.append(f"+++ b/{name}")
            for chunk in _wrap(old):
                lines.append(f"-{chunk}")
            for chunk in _wrap(new):
                lines.append(f"+{chunk}")
            lines.append("")
    return "\n".join(lines) + "\n"


def _wrap(text: str, width: int = 88) -> list[str]:
    """Wrap for diff display, never breaking a word."""
    if not text:
        return [""]
    words, out, line = text.split(), [], ""
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out
