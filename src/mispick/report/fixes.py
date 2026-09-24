"""Rendering for fix mode. Shows rejected rewrites too, on purpose."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mispick.fix import FixReport
from mispick.types import ToolSet


def render_terminal(report: FixReport, tool_set: ToolSet, console: Console) -> None:
    if not report.outcomes:
        console.print("[green]Nothing to fix: no tool was mistaken for another tool.[/green]")
        for error in report.errors:
            console.print(f"[yellow]warning:[/yellow] {error}")
        return

    console.print()
    console.print(
        Panel(
            Text.from_markup(
                f"[bold]{len(report.accepted)} of {len(report.outcomes)}[/bold] rewrites "
                "measurably improved selection and are shown below.\n"
                "[dim]Rewrites are re-tested on the same queries. Ones that did not help are "
                "listed too - a rewrite nobody checked is a guess.[/dim]"
            ),
            title="fix mode",
            border_style="green" if report.accepted else "yellow",
        )
    )

    tools = tool_set.by_qualified_name()
    for outcome in report.outcomes:
        rewrite = outcome.rewrite
        colour = {
            "ACCEPTED": "green",
            "INCONCLUSIVE": "yellow",
            "REJECTED": "red",
        }[outcome.verdict]

        console.print()
        console.print(
            f"[bold]{rewrite.pair.label}[/bold]  [{colour}]{outcome.verdict}[/{colour}]"
        )

        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_column(style="dim", width=18)
        table.add_column()
        table.add_row("accuracy", f"{outcome.before}  →  {outcome.after}")
        table.add_row("trials changed", f"{outcome.improved} fixed, {outcome.regressed} broken")
        table.add_row("why", outcome.explanation)
        table.add_row("likely cause", rewrite.cause.replace("_", " "))
        if rewrite.note:
            table.add_row("model's note", rewrite.note)
        console.print(table)

        if outcome.accepted:
            for name, new in sorted(rewrite.descriptions.items()):
                old = tools[name].description or "(none)"
                console.print(f"  [bold]{name}[/bold]")
                console.print(f"    [red]- {old}[/red]")
                console.print(f"    [green]+ {new}[/green]")

    console.print()
    if report.accepted:
        console.print(
            "[dim]Run with `--patch out.diff` to write these out. mispick never edits your "
            "source.[/dim]"
        )
    else:
        console.print(
            "[yellow]No rewrite earned its place.[/yellow] [dim]That is a real result: the "
            "descriptions may not be the problem, or the tools may genuinely overlap. "
            "Hasan et al. found LLM rewrites regress about one time in six, which is why "
            "mispick re-tests instead of trusting them.[/dim]"
        )
    for error in report.errors:
        console.print(f"[yellow]warning:[/yellow] {error}")


def render_markdown(report: FixReport, tool_set: ToolSet) -> str:
    if not report.outcomes:
        return "## mispick fix\n\nNo tool was mistaken for another tool. Nothing to fix.\n"

    tools = tool_set.by_qualified_name()
    out: list[str] = []
    a = out.append
    a(f"## mispick fix: {len(report.accepted)} of {len(report.outcomes)} rewrites accepted")
    a("")
    a(
        f"Rewrites proposed by `{report.model}` and re-tested on the same queries. "
        "Only the accepted ones improved selection with a paired test at p ≤ 0.05."
    )
    a("")
    a("| Pair | Verdict | Accuracy | p | Fixed | Broken |")
    a("|---|---|---|---|---|---|")
    for o in report.outcomes:
        a(
            f"| `{o.rewrite.pair.expected}` vs `{o.rewrite.pair.chosen}` | {o.verdict} | "
            f"{o.before} → {o.after} | {o.p_value:.3f} | {o.improved} | {o.regressed} |"
        )
    a("")

    for o in report.accepted:
        a(f"### `{o.rewrite.pair.expected}` vs `{o.rewrite.pair.chosen}`")
        a("")
        a(f"Likely cause: **{o.rewrite.cause.replace('_', ' ')}**. {o.rewrite.note}")
        a("")
        for name, new in sorted(o.rewrite.descriptions.items()):
            old = tools[name].description or "(none)"
            a(f"**`{name}`**")
            a("")
            a("```diff")
            a(f"- {old}")
            a(f"+ {new}")
            a("```")
            a("")

    rejected = report.rejected
    if rejected:
        a("<details>")
        a(f"<summary>{len(rejected)} rewrites did not earn their place</summary>")
        a("")
        for o in rejected:
            a(f"- `{o.rewrite.pair.label}`: {o.explanation}")
        a("")
        a("</details>")
        a("")
    return "\n".join(out) + "\n"
