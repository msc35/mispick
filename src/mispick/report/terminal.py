"""The terminal report: score, confusion heatmap, worst pairs."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from mispick.metrics import NONE, PHANTOM, Metrics, top_confused_pairs
from mispick.report.provenance import provenance_of
from mispick.select import RunResult


def _score_colour(score: int) -> str:
    if score >= 90:
        return "green"
    if score >= 75:
        return "yellow"
    return "red"


def _heat(count: int, row_total: int, correct: bool) -> Text:
    """One matrix cell, shaded by how much of the row it took."""
    if count == 0:
        return Text("·", style="dim")
    share = count / row_total if row_total else 0.0
    if correct:
        style = "bold green" if share >= 0.8 else "green" if share >= 0.5 else "yellow"
    elif share >= 0.5:
        style = "bold red"
    elif share >= 0.25:
        style = "red"
    else:
        style = "yellow"
    return Text(str(count), style=style)


def _short(name: str, width: int = 14) -> str:
    """Shorten a qualified name for a column header."""
    if len(name) <= width:
        return name
    return name[: width - 1] + "…"


def render(result: RunResult, metrics: Metrics, console: Console | None = None) -> None:
    """Print the whole report."""
    console = console or Console()
    prov = provenance_of(result, metrics.token_estimate)

    score = metrics.score
    headline = Text()
    headline.append("score ", style="dim")
    headline.append(f"{score}/100", style=f"bold {_score_colour(score)}")
    headline.append("    accuracy ", style="dim")
    headline.append(str(metrics.accuracy), style="bold")
    headline.append("    stability ", style="dim")
    headline.append(str(metrics.stability), style="bold")
    headline.append("    args valid ", style="dim")
    headline.append(str(metrics.arg_validity), style="bold")
    console.print(
        Panel(
            headline,
            title=f"mispick · {prov.servers} · {prov.tool_count} tools",
            subtitle=prov.one_line,
            border_style=_score_colour(score),
        )
    )

    _render_matrix(console, metrics)
    _render_pairs(console, metrics)
    _render_per_tool(console, metrics)
    _render_notes(console, result, metrics)


def _render_matrix(console: Console, metrics: Metrics) -> None:
    console.print()
    console.print("[bold]Confusion matrix[/bold]  [dim]rows = intended, columns = chosen[/dim]")

    table = Table(show_lines=False, pad_edge=False, box=None)
    table.add_column("intended ↓ / chosen →", style="bold", no_wrap=True)
    for col in metrics.column_labels:
        style = "dim" if col in (NONE, PHANTOM) else ""
        table.add_column(_short(col), justify="right", style=style, no_wrap=True)

    for row in metrics.row_labels:
        cells = metrics.matrix.get(row, {})
        row_total = sum(cells.values())
        rendered = [
            _heat(cells.get(col, 0), row_total, correct=(col == row))
            for col in metrics.column_labels
        ]
        label = Text(_short(row, 22), style="dim" if row == NONE else "bold")
        table.add_row(label, *rendered)
    console.print(table)
    console.print(
        "[dim]Green on the diagonal is correct. Red off it is a confusion. "
        "Wide columns are tools that attract other tools' work.[/dim]"
    )


def _render_pairs(console: Console, metrics: Metrics) -> None:
    pairs = top_confused_pairs(metrics, limit=5)
    console.print()
    if not pairs:
        console.print("[green]No tool was mistaken for another tool.[/green]")
        return
    console.print("[bold]Most confused pairs[/bold]")
    table = Table(box=None, pad_edge=False)
    table.add_column("pair", style="bold")
    table.add_column("wrong picks", justify="right")
    table.add_column("share of that tool's trials", justify="right")
    for pair in pairs:
        table.add_row(pair.label, str(pair.count), f"{pair.share * 100:.0f}%")
    console.print(table)
    console.print(
        "[dim]Run `mispick fix` to get a re-tested description rewrite for these.[/dim]"
    )


def _render_per_tool(console: Console, metrics: Metrics) -> None:
    console.print()
    console.print("[bold]Per tool[/bold]  [dim]95% Wilson intervals in brackets[/dim]")
    table = Table(box=None, pad_edge=False)
    table.add_column("tool", style="bold")
    table.add_column("recall", justify="right")
    table.add_column("precision", justify="right")
    table.add_column("", width=12)

    worst = sorted(metrics.per_tool.values(), key=lambda t: (t.recall.value, t.name))
    for tool in worst:
        bar_len = round(tool.recall.value * 10)
        bar = Text("█" * bar_len + "░" * (10 - bar_len))
        value = tool.recall.value
        bar.stylize("green" if value >= 0.8 else "yellow" if value >= 0.5 else "red")
        table.add_row(tool.name, str(tool.recall), str(tool.precision), bar)
    console.print(table)


def _render_notes(console: Console, result: RunResult, metrics: Metrics) -> None:
    prov = provenance_of(result, metrics.token_estimate)
    console.print()
    notes: list[str] = []
    if metrics.over_trigger.total:
        notes.append(
            f"Over-triggering: called a tool on {metrics.over_trigger} of the "
            '"no tool fits" queries.'
        )
    if metrics.phantom_rate.hits:
        notes.append(
            f"Phantom tools: the model named a tool that does not exist in "
            f"{metrics.phantom_rate} of trials."
        )
    if metrics.unstable_queries:
        notes.append(
            f"{len(metrics.unstable_queries)} queries changed their answer across "
            f"{prov.k} runs."
        )
    if metrics.errored_trials:
        notes.append(
            f"[yellow]{metrics.errored_trials} of {metrics.trials} trials errored and were "
            "excluded.[/yellow]"
        )
    notes.append(
        f"Tool list costs roughly {metrics.token_estimate} tokens per request "
        "(estimate, and a lower bound)."
    )
    if not prov.deterministic:
        notes.append(
            f"Temperature is {prov.temperature:g}, so this run is not reproducible. "
            "Use --temperature 0 --seed N for a deterministic run."
        )
    for note in notes:
        console.print(f"  [dim]•[/dim] {note}")

    console.print()
    console.print(f"[dim]{prov.caveat}[/dim]")
    for error in result.errors[:3]:
        console.print(f"[yellow]backend error:[/yellow] {error}")
