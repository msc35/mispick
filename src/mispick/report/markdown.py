"""Markdown, sized for a PR comment."""

from __future__ import annotations

from mispick.crossserver import analyse
from mispick.metrics import NONE, PHANTOM, Metrics, top_confused_pairs
from mispick.report.provenance import provenance_of
from mispick.select import RunResult


def render(result: RunResult, metrics: Metrics, *, title: str = "mispick") -> str:
    prov = provenance_of(result, metrics.token_estimate)
    out: list[str] = []
    a = out.append

    verdict = "✅" if metrics.score >= 90 else "⚠️" if metrics.score >= 75 else "❌"
    a(f"## {verdict} {title}: {metrics.score}/100")
    a("")
    a(f"**{prov.servers}** · {prov.tool_count} tools · `{prov.one_line}`")
    a("")
    a("| | |")
    a("|---|---|")
    a(f"| Selection accuracy | {metrics.accuracy} |")
    a(f"| Stability across {prov.k} runs | {metrics.stability} |")
    a(f"| Argument validity | {metrics.arg_validity} |")
    if metrics.over_trigger.total:
        a(f"| Over-triggering on \"no tool fits\" | {metrics.over_trigger} |")
    if metrics.phantom_rate.hits:
        a(f"| Invented a non-existent tool | {metrics.phantom_rate} |")
    a(f"| Tool list cost | ~{metrics.token_estimate} tokens (estimate, lower bound) |")
    a("")

    pairs = top_confused_pairs(metrics, limit=5)
    if pairs:
        a("### Most confused pairs")
        a("")
        a("| Intended | Chosen instead | Trials | Share |")
        a("|---|---|---|---|")
        for p in pairs:
            a(f"| `{p.expected}` | `{p.chosen}` | {p.count} | {p.share * 100:.0f}% |")
        a("")
    else:
        a("No tool was mistaken for another tool.")
        a("")

    cross = analyse(result, metrics)
    if cross.is_multi_server:
        a("### Across servers")
        a("")
        a("| Server | Tools | Accuracy on its own tools | Work lost | Work stolen |")
        a("|---|---|---|---|---|")
        for score in sorted(cross.servers, key=lambda s: s.accuracy.value):
            a(
                f"| `{score.label}` | {score.tool_count} | {score.accuracy} | "
                f"{score.lost or '·'} | {score.stolen or '·'} |"
            )
        a("")
        if cross.collisions:
            a("**Name collisions.** The same bare tool name on more than one server:")
            a("")
            for c in cross.collisions:
                note = " — and their descriptions are identical" if c.identical_descriptions else ""
                a(f"- `{c.name}` on {', '.join(f'`{s}`' for s in c.servers)}{note}")
            a("")
        if cross.leaks:
            a(
                f"**{cross.cross_server_rate} of trials went to the wrong server.** "
                "Worst crossings:"
            )
            a("")
            for expected, chosen, count in cross.leaks[:8]:
                a(f"- `{expected}` → `{chosen}` ({count} trials)")
            a("")
        else:
            a("No trial crossed from one server to another.")
            a("")

    a("<details>")
    a("<summary>Confusion matrix</summary>")
    a("")
    header = "| intended ↓ / chosen → | " + " | ".join(
        f"`{c}`" if c not in (NONE, PHANTOM) else c for c in metrics.column_labels
    ) + " |"
    a(header)
    a("|" + "---|" * (len(metrics.column_labels) + 1))
    for row in metrics.row_labels:
        cells = metrics.matrix.get(row, {})
        label = f"`{row}`" if row != NONE else row
        values = []
        for col in metrics.column_labels:
            count = cells.get(col, 0)
            if count == 0:
                values.append("·")
            elif col == row:
                values.append(f"**{count}**")
            else:
                values.append(str(count))
        a(f"| {label} | " + " | ".join(values) + " |")
    a("")
    a("</details>")
    a("")

    a("<details>")
    a("<summary>Per tool</summary>")
    a("")
    a("| Tool | Recall | Precision |")
    a("|---|---|---|")
    for name, tool in sorted(metrics.per_tool.items(), key=lambda kv: kv[1].recall.value):
        a(f"| `{name}` | {tool.recall} | {tool.precision} |")
    a("")
    a("</details>")
    a("")
    a(f"<sub>{prov.caveat} Intervals are Wilson 95%.</sub>")
    return "\n".join(out) + "\n"


def render_comparison(
    head: tuple[RunResult, Metrics],
    base: tuple[RunResult, Metrics],
    *,
    title: str = "mispick",
) -> str:
    """Head vs base, for the GitHub Action's PR comment.

    Refuses to subtract runs made with different models: a delta between two different
    models is not a regression, it is a different question.
    """
    head_result, head_metrics = head
    base_result, base_metrics = base

    if head_result.config.model != base_result.config.model:
        return (
            f"## {title}\n\n"
            f"Cannot compare these runs: head used `{head_result.config.model}` and base used "
            f"`{base_result.config.model}`. A difference between two models is not a "
            "regression. Re-run both with the same model.\n"
        )

    delta = head_metrics.score - base_metrics.score
    arrow = "▲" if delta > 0 else "▼" if delta < 0 else "="
    verdict = "✅" if delta >= 0 else "⚠️"

    out: list[str] = []
    a = out.append
    a(f"## {verdict} {title}: {base_metrics.score} → {head_metrics.score} ({arrow}{abs(delta)})")
    a("")
    acc_delta = head_metrics.accuracy.pct - base_metrics.accuracy.pct
    a(
        f"Accuracy {base_metrics.accuracy.pct:.0f}% → {head_metrics.accuracy.pct:.0f}% "
        f"({acc_delta:+.0f} points)."
    )
    a("")

    base_pairs = {(p.expected, p.chosen) for p in top_confused_pairs(base_metrics, limit=10)}
    head_pairs = top_confused_pairs(head_metrics, limit=10)
    new_pairs = [p for p in head_pairs if (p.expected, p.chosen) not in base_pairs]
    if new_pairs:
        a("**New confusions on this branch:**")
        a("")
        for p in new_pairs:
            a(f"- `{p.expected}` is now confused with `{p.chosen}` ({p.count} trials)")
        a("")

    fixed = base_pairs - {(p.expected, p.chosen) for p in head_pairs}
    if fixed:
        a("**No longer confused:**")
        a("")
        for expected, chosen in sorted(fixed):
            a(f"- `{expected}` vs `{chosen}`")
        a("")

    a(render(head_result, head_metrics, title="Full report"))
    return "\n".join(out)
