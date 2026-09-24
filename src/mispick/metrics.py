"""Turn raw choices into the numbers the report prints.

Everything here is derived from `RunResult` and nothing here calls a model, so it is cheap
to test and cheap to re-run over a rewritten description.

On uncertainty: with K=3 a per-tool rate is computed over a handful of trials, so bare
percentages would overstate what we know. Every rate carries a **Wilson 95% interval**
(SPEC section 6.4). Wilson rather than the normal approximation because it behaves at 0%
and 100%, which is exactly where a confused tool lands.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

from mispick.select import RunResult
from mispick.types import Choice, Query, Tool

#: The label used for "the model chose nothing" in the matrix, as a row and a column.
NONE = "(none)"
#: The label for a tool name the model invented.
PHANTOM = "(phantom)"

#: z for a 95% two-sided interval.
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Rate:
    """A proportion with a Wilson 95% interval."""

    hits: int
    total: int

    @property
    def value(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.hits, self.total)

    @property
    def pct(self) -> float:
        return 100.0 * self.value

    def __str__(self) -> str:
        if not self.total:
            return "n/a"
        low, high = self.interval
        return f"{self.pct:.0f}% [{low * 100:.0f}-{high * 100:.0f}]"


def wilson(hits: int, total: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval. Returns (low, high), clamped to [0, 1]."""
    if total <= 0:
        return (0.0, 0.0)
    p = hits / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z / denom) * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    # Round before clamping: floating point otherwise leaves 0.9999999999999999 where the
    # interval genuinely reaches 1.
    low = max(0.0, round(centre - margin, 12))
    high = min(1.0, round(centre + margin, 12))
    return (low, high)


@dataclass
class ToolMetrics:
    """Per-tool precision and recall, in the confusion-matrix sense."""

    name: str
    #: Chosen when it should have been (true positives).
    correct: int = 0
    #: Times this tool was the right answer.
    expected: int = 0
    #: Times this tool was chosen, right or wrong.
    chosen: int = 0

    @property
    def recall(self) -> Rate:
        """Of the trials that wanted this tool, how many got it."""
        return Rate(self.correct, self.expected)

    @property
    def precision(self) -> Rate:
        """Of the trials that picked this tool, how many should have."""
        return Rate(self.correct, self.chosen)

    @property
    def f1(self) -> float:
        p, r = self.precision.value, self.recall.value
        return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


@dataclass
class ConfusedPair:
    """Expected one tool, got another."""

    expected: str
    chosen: str
    count: int
    #: Of all trials that wanted `expected`, the share that went to `chosen`.
    share: float

    @property
    def label(self) -> str:
        return f"{self.expected} → {self.chosen}"


@dataclass
class Metrics:
    """Everything the reports need."""

    accuracy: Rate
    per_tool: dict[str, ToolMetrics]
    matrix: dict[str, dict[str, int]]
    row_labels: list[str]
    column_labels: list[str]
    confused_pairs: list[ConfusedPair]
    arg_validity: Rate
    stability: Rate
    over_trigger: Rate
    phantom_rate: Rate
    #: Queries whose K runs did not agree.
    unstable_queries: list[str] = field(default_factory=list)
    token_estimate: int = 0
    trials: int = 0
    errored_trials: int = 0

    @property
    def score(self) -> int:
        """0-100, documented in SPEC section 6.6.

        score = 70*accuracy + 20*stability + 10*argument validity

        Accuracy dominates because picking the wrong tool is the failure this tool exists to
        find. Stability is weighted next: a result that changes between runs is not a result.
        Argument validity is last - a right tool with a bad argument is recoverable in a way
        that the wrong tool is not. Missing components (no args to validate, K=1 so no
        stability) are dropped and the remaining weights renormalised, so a run is never
        punished for a measurement we did not take.
        """
        parts = [(70.0, self.accuracy)]
        if self.stability.total:
            parts.append((20.0, self.stability))
        if self.arg_validity.total:
            parts.append((10.0, self.arg_validity))
        weight = sum(w for w, _ in parts)
        return round(sum(w * r.value for w, r in parts) / weight * 100)


def estimate_tool_tokens(tools: Sequence[Tool]) -> int:
    """Rough token cost of putting this tool list in front of a model.

    Deliberately crude: characters over four. It is labelled an estimate everywhere it is
    shown, and it is a lower bound - a real tokenizer and the provider's own framing will
    both add to it. Being a lower bound is what makes it usable for truncation detection: a
    provider reporting fewer prompt tokens than this dropped content.

    Takes tools rather than a RunResult so it can be called before any run exists.
    """
    chars = 0
    for tool in tools:
        chars += len(tool.qualified_name) + len(tool.description or "")
        chars += len(str(tool.input_schema))
    return chars // 4


def compute(result: RunResult) -> Metrics:
    """Everything, from the raw choices."""
    by_query: dict[str, Query] = {q.id: q for q in result.queries}
    live = {t.qualified_name for t in result.tools}

    scored = [c for c in result.choices if not c.error]
    errored = len(result.choices) - len(scored)

    def outcome(choice: Choice) -> str:
        """What the matrix column should say for this choice."""
        if choice.phantom:
            return PHANTOM
        return choice.chosen or NONE

    correct = 0
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    per_tool: dict[str, ToolMetrics] = {name: ToolMetrics(name=name) for name in sorted(live)}

    for choice in scored:
        query = by_query.get(choice.query_id)
        if query is None:
            continue
        expected = query.expected or NONE
        got = outcome(choice)
        matrix[expected][got] += 1

        if expected in per_tool:
            per_tool[expected].expected += 1
        if got in per_tool:
            per_tool[got].chosen += 1
        if got == expected:
            correct += 1
            if expected in per_tool:
                per_tool[expected].correct += 1

    # Argument validity, over the trials where a real tool was chosen.
    arg_total = sum(1 for c in scored if c.chosen and c.args_valid is not None)
    arg_ok = sum(1 for c in scored if c.chosen and c.args_valid)

    # Stability: a query is stable when its K runs all landed on the same outcome.
    grouped: dict[str, list[Choice]] = defaultdict(list)
    for choice in scored:
        grouped[choice.query_id].append(choice)
    repeated = {qid: cs for qid, cs in grouped.items() if len(cs) > 1}
    stable = 0
    unstable: list[str] = []
    for qid, cs in sorted(repeated.items()):
        if len({outcome(c) for c in cs}) == 1:
            stable += 1
        else:
            unstable.append(qid)

    # Over-triggering: on "no tool fits" queries, did it call something anyway?
    no_tool_ids = {q.id for q in result.queries if q.expected is None}
    no_tool_trials = [c for c in scored if c.query_id in no_tool_ids]
    over = sum(1 for c in no_tool_trials if c.chosen or c.phantom)

    phantoms = sum(1 for c in scored if c.phantom)

    # Confused pairs, worst first.
    pairs: list[ConfusedPair] = []
    for expected, row in matrix.items():
        row_total = sum(row.values())
        for got, count in row.items():
            if got == expected or count == 0:
                continue
            pairs.append(
                ConfusedPair(
                    expected=expected,
                    chosen=got,
                    count=count,
                    share=count / row_total if row_total else 0.0,
                )
            )
    pairs.sort(key=lambda p: (-p.count, -p.share, p.expected, p.chosen))

    rows = sorted(matrix.keys(), key=lambda r: (r == NONE, r))
    cols_seen = {got for row in matrix.values() for got in row}
    columns = sorted(cols_seen | live, key=lambda c: (c in (NONE, PHANTOM), c))

    return Metrics(
        accuracy=Rate(correct, len(scored)),
        per_tool=per_tool,
        matrix={r: dict(matrix[r]) for r in rows},
        row_labels=rows,
        column_labels=columns,
        confused_pairs=pairs,
        arg_validity=Rate(arg_ok, arg_total),
        stability=Rate(stable, len(repeated)),
        over_trigger=Rate(over, len(no_tool_trials)),
        phantom_rate=Rate(phantoms, len(scored)),
        unstable_queries=unstable,
        token_estimate=estimate_tool_tokens(result.tools),
        trials=len(result.choices),
        errored_trials=errored,
    )


def accuracy_for_tools(result: RunResult, names: set[str]) -> Rate:
    """Accuracy restricted to the queries that expected one of `names`.

    Fix mode uses this to judge a rewrite on just the pair it was meant to fix.
    """
    wanted = {q.id for q in result.queries if q.expected in names}
    scored = [c for c in result.choices if not c.error and c.query_id in wanted]
    by_query = {q.id: q for q in result.queries}
    hits = sum(
        1
        for c in scored
        if (c.chosen or NONE) == (by_query[c.query_id].expected or NONE) and not c.phantom
    )
    return Rate(hits, len(scored))


def top_confused_pairs(metrics: Metrics, limit: int = 3) -> list[ConfusedPair]:
    """The worst pairs, deduplicated so A→B and B→A count once.

    Fix mode rewrites both descriptions in a pair, so it should not be handed the same
    pair twice.
    """
    seen: set[frozenset[str]] = set()
    out: list[ConfusedPair] = []
    for pair in metrics.confused_pairs:
        if pair.chosen in (NONE, PHANTOM) or pair.expected == NONE:
            continue
        key = frozenset({pair.expected, pair.chosen})
        if key in seen:
            continue
        seen.add(key)
        out.append(pair)
        if len(out) >= limit:
            break
    return out


def summarise_counter(items: list[str]) -> list[tuple[str, int]]:
    return Counter(items).most_common()
