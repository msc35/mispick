"""The `mispick` command line."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mispick import __version__
from mispick.fix import as_patch, run_fix_mode
from mispick.generate import DEFAULT_N, DEFAULT_NO_TOOL, QueryCache, build_query_set
from mispick.loader import Target, TargetError, load, write_snapshot
from mispick.metrics import Metrics, compute
from mispick.models.base import DEFAULT_MAX_TOKENS, BackendError
from mispick.models.registry import DEFAULT_MODEL, get_backend
from mispick.report import badge as badge_report
from mispick.report import fixes as fixes_report
from mispick.report import html as html_report
from mispick.report import json as json_report
from mispick.report import markdown as markdown_report
from mispick.report import terminal as terminal_report
from mispick.select import DEFAULT_CONCURRENCY, DEFAULT_K, RunConfig, RunResult, run_selection
from mispick.sources.config import ConfigError, collisions
from mispick.sources.server import ServerError
from mispick.sources.snapshot import SnapshotError
from mispick.types import ToolSet

app = typer.Typer(
    name="mispick",
    help="Find out which of your MCP tools the model mixes up - measured, not guessed.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err_console = Console(stderr=True)

#: Exit codes, per SPEC section 6.6.
EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_ERROR = 2

CmdOpt = Annotated[str | None, typer.Option("--cmd", help="Local stdio server command.")]
UrlOpt = Annotated[str | None, typer.Option("--url", help="Streamable HTTP server URL.")]
ConfigOpt = Annotated[
    str | None,
    typer.Option("--config", help="A claude_desktop_config.json / .mcp.json; loads every server."),
]
SnapshotOpt = Annotated[
    str | None, typer.Option("--snapshot", help="A captured tools/list JSON file (offline).")
]
TimeoutOpt = Annotated[float, typer.Option("--timeout", help="Seconds to wait per server.")]
OnlyOpt = Annotated[
    list[str] | None, typer.Option("--only", help="With --config, limit to these server labels.")
]


def _target(
    cmd: str | None,
    url: str | None,
    config: str | None,
    snapshot: str | None,
    timeout: float,
    only: list[str] | None = None,
) -> Target:
    return Target(
        cmd=cmd, url=url, config=config, snapshot=snapshot, timeout=timeout, only=only or []
    )


def _load_or_exit(target: Target) -> tuple[ToolSet, list[str]]:
    """Load tools, turning our exceptions into a clear message and exit code 2."""
    try:
        tool_set, failures = asyncio.run(load(target))
    except (TargetError, ConfigError, SnapshotError, ServerError) as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc
    for failure in failures:
        err_console.print(f"[yellow]warning:[/yellow] could not load {failure}")
    if not tool_set.tools:
        err_console.print("[red]error:[/red] that target exposed no tools; nothing to measure.")
        raise typer.Exit(EXIT_ERROR)
    return tool_set, failures


def _servers_line(tool_set: ToolSet) -> str:
    parts = []
    for info in tool_set.servers:
        name = info.name or info.label
        version = f" {info.version}" if info.version else ""
        parts.append(f"{name}{version}")
    return ", ".join(parts) if parts else "unknown"


@app.command()
def tools(
    cmd: CmdOpt = None,
    url: UrlOpt = None,
    config: ConfigOpt = None,
    snapshot: SnapshotOpt = None,
    timeout: TimeoutOpt = 20.0,
    only: OnlyOpt = None,
) -> None:
    """List the tools a target exposes, and any cross-server name collisions."""
    target = _target(cmd, url, config, snapshot, timeout, only)
    tool_set, _ = _load_or_exit(target)

    table = Table(title=f"{len(tool_set.tools)} tools from {_servers_line(tool_set)}")
    if target.is_multi_server:
        table.add_column("server", style="cyan")
    table.add_column("tool", style="bold")
    table.add_column("description")
    for tool in tool_set.sorted_tools():
        desc = (tool.description or "[dim](none)[/dim]").replace("\n", " ")
        if len(desc) > 70:
            desc = desc[:69] + "…"
        row = [tool.name, desc]
        if target.is_multi_server:
            row.insert(0, tool.server or "-")
        table.add_row(*row)
    console.print(table)

    found = collisions(tool_set)
    if found:
        console.print()
        console.print("[bold yellow]Cross-server name collisions[/bold yellow]")
        for name, servers in found.items():
            where = ", ".join(str(s) for s in servers)
            console.print(f"  [bold]{name}[/bold] is exposed by {where}")
        console.print(
            "[dim]The MCP spec warns clients to disambiguate these. Run "
            "`mispick run --config …` to measure whether the model mixes them up.[/dim]"
        )


# Registered as `snapshot` rather than by its function name: `snapshot` is also an option
# name on the other commands, so the function cannot be called that without shadowing it.
@app.command(name="snapshot")
def snapshot_cmd(
    output: Annotated[str, typer.Option("--output", "-o", help="Where to write the JSON.")],
    cmd: CmdOpt = None,
    url: UrlOpt = None,
    config: ConfigOpt = None,
    timeout: TimeoutOpt = 20.0,
) -> None:
    """Capture a server's tools/list to a file, so later runs work offline."""
    target = _target(cmd, url, config, None, timeout)
    tool_set, _ = _load_or_exit(target)
    path = write_snapshot(tool_set, output)
    console.print(
        f"Wrote {len(tool_set.tools)} tools to [bold]{path}[/bold]. "
        f"Re-run offline with: [dim]mispick tools --snapshot {path}[/dim]"
    )



class Format(StrEnum):
    terminal = "terminal"
    json = "json"
    md = "md"
    html = "html"


FormatOpt = Annotated[Format, typer.Option("--format", help="Report format.")]
OutOpt = Annotated[
    str | None, typer.Option("--out", "-o", help="Write the report here instead of stdout.")
]
BadgeOpt = Annotated[
    str | None, typer.Option("--badge", help="Also write an SVG badge to this path.")
]

ModelOpt = Annotated[
    str, typer.Option("--model", help="provider/model. Default is a local Ollama model.")
]
NOpt = Annotated[int, typer.Option("--queries", "-n", help="Queries generated per tool (N).")]
KOpt = Annotated[int, typer.Option("--runs", "-k", help="Times each query is re-run (K).")]
TempOpt = Annotated[float, typer.Option("--temperature", help="0 for a deterministic run.")]
SeedOpt = Annotated[int | None, typer.Option("--seed", help="Seed, if the backend honours one.")]
FailUnderOpt = Annotated[
    int | None,
    typer.Option("--fail-under", help="Exit 1 if the score is below this. For CI."),
]
RegenOpt = Annotated[
    bool, typer.Option("--regenerate", help="Ignore the query cache and generate afresh.")
]
CacheDirOpt = Annotated[str, typer.Option("--cache-dir", help="Where queries.yaml lives.")]
ConcurrencyOpt = Annotated[int, typer.Option("--concurrency", help="Selection calls in flight.")]
ThinkOpt = Annotated[
    bool,
    typer.Option(
        "--think/--no-think",
        help="Let a reasoning model think first. Off by default: much slower, same answers.",
    ),
]
MaxTokensOpt = Annotated[
    int,
    typer.Option(
        "--max-tokens",
        help="Output budget for query generation. Small reasoning models need a lot of it.",
    ),
]


async def _measure(
    target: Target,
    tool_set: ToolSet,
    *,
    model: str,
    n: int,
    k: int,
    temperature: float,
    seed: int | None,
    regenerate: bool,
    cache_dir: str,
    concurrency: int,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    think: bool = False,
    no_tool: int = DEFAULT_NO_TOOL,
) -> tuple[RunResult, Metrics]:
    """Generate queries if needed, run selection, compute metrics."""
    backend = get_backend(model, think=think)
    cache = QueryCache.open(cache_dir)
    try:
        with console.status("[dim]generating test queries…[/dim]") as status:

            def on_tool(name: str) -> None:
                status.update(f"[dim]generating test queries for {name}…[/dim]")

            queries = await build_query_set(
                backend,
                tool_set,
                n=n,
                no_tool_count=no_tool,
                cache=cache,
                regenerate=regenerate,
                seed=seed,
                max_tokens=max_tokens,
                on_progress=on_tool,
            )
        path = cache.save(generator=backend.name)
        console.print(
            f"[dim]{len(queries)} queries · cached in {path} · edit it and they stick[/dim]"
        )

        total = len(queries) * k
        done = 0
        with console.status(f"[dim]0/{total} selections…[/dim]") as status:

            def on_choice(_: object) -> None:
                nonlocal done
                done += 1
                status.update(f"[dim]{done}/{total} selections…[/dim]")

            result = await run_selection(
                backend,
                tool_set,
                queries,
                config=RunConfig(
                    n=n, k=k, temperature=temperature, seed=seed, concurrency=concurrency
                ),
                on_progress=on_choice,
            )
    finally:
        await backend.aclose()
    return result, compute(result)


def _emit(
    result: RunResult,
    metrics: Metrics,
    fmt: Format,
    out: str | None,
    badge: str | None,
) -> None:
    """Render in the requested format, to a file or to stdout."""
    if fmt is Format.terminal and out is None:
        terminal_report.render(result, metrics, console)
    else:
        if fmt is Format.json:
            text = json_report.render(result, metrics)
        elif fmt is Format.md:
            text = markdown_report.render(result, metrics)
        elif fmt is Format.html:
            text = html_report.render(result, metrics)
        else:
            # terminal format with --out: render without colour codes.
            file_console = Console(width=100, no_color=True, record=True)
            terminal_report.render(result, metrics, file_console)
            text = file_console.export_text()
        if out:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            err_console.print(f"[dim]wrote {path}[/dim]")
        else:
            # stdout must stay clean so it can be piped.
            print(text, end="")

    if badge:
        svg = badge_report.render(metrics, model=result.config.model)
        badge_path = Path(badge)
        badge_path.parent.mkdir(parents=True, exist_ok=True)
        badge_path.write_text(svg)
        err_console.print(f"[dim]wrote {badge_path}[/dim]")


@app.command()
def run(
    cmd: CmdOpt = None,
    url: UrlOpt = None,
    config: ConfigOpt = None,
    snapshot: SnapshotOpt = None,
    timeout: TimeoutOpt = 20.0,
    only: OnlyOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    queries: NOpt = DEFAULT_N,
    runs: KOpt = DEFAULT_K,
    temperature: TempOpt = 0.7,
    seed: SeedOpt = None,
    regenerate: RegenOpt = False,
    cache_dir: CacheDirOpt = ".mispick",
    concurrency: ConcurrencyOpt = DEFAULT_CONCURRENCY,
    max_tokens: MaxTokensOpt = DEFAULT_MAX_TOKENS,
    think: ThinkOpt = False,
    fail_under: FailUnderOpt = None,
    fmt: FormatOpt = Format.terminal,
    out: OutOpt = None,
    badge: BadgeOpt = None,
) -> None:
    """Measure which tools the model mixes up."""
    target = _target(cmd, url, config, snapshot, timeout, only)
    tool_set, _ = _load_or_exit(target)

    try:
        result, metrics = asyncio.run(
            _measure(
                target,
                tool_set,
                model=model,
                n=queries,
                k=runs,
                temperature=temperature,
                seed=seed,
                regenerate=regenerate,
                cache_dir=cache_dir,
                concurrency=concurrency,
                max_tokens=max_tokens,
                think=think,
            )
        )
    except BackendError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc

    if not [c for c in result.choices if not c.error]:
        err_console.print(
            "[red]error:[/red] every selection call failed, so there is nothing to report."
        )
        for error in result.errors[:3]:
            err_console.print(f"  {error}")
        raise typer.Exit(EXIT_ERROR)

    _emit(result, metrics, fmt, out, badge)

    if fail_under is not None and metrics.score < fail_under:
        err_console.print(
            f"\n[red]score {metrics.score} is below --fail-under {fail_under}[/red]"
        )
        raise typer.Exit(EXIT_BELOW_THRESHOLD)


PatchOpt = Annotated[
    str | None, typer.Option("--patch", help="Write the accepted rewrites here as a diff.")
]
PairsOpt = Annotated[int, typer.Option("--pairs", help="How many confused pairs to try to fix.")]


@app.command()
def fix(
    cmd: CmdOpt = None,
    url: UrlOpt = None,
    config: ConfigOpt = None,
    snapshot: SnapshotOpt = None,
    timeout: TimeoutOpt = 20.0,
    only: OnlyOpt = None,
    model: ModelOpt = DEFAULT_MODEL,
    queries: NOpt = DEFAULT_N,
    runs: KOpt = DEFAULT_K,
    temperature: TempOpt = 0.7,
    seed: SeedOpt = None,
    regenerate: RegenOpt = False,
    cache_dir: CacheDirOpt = ".mispick",
    concurrency: ConcurrencyOpt = DEFAULT_CONCURRENCY,
    max_tokens: MaxTokensOpt = DEFAULT_MAX_TOKENS,
    think: ThinkOpt = False,
    pairs: PairsOpt = 3,
    patch: PatchOpt = None,
    fmt: FormatOpt = Format.terminal,
    out: OutOpt = None,
) -> None:
    """Propose description rewrites for the worst confused pairs, and prove they help.

    Every rewrite is re-tested against the same queries. Rewrites that do not measurably
    improve selection are reported as rejected rather than hidden. Your source is never
    modified.
    """
    target = _target(cmd, url, config, snapshot, timeout, only)
    tool_set, _ = _load_or_exit(target)

    async def go() -> tuple[RunResult, Metrics, object]:
        backend = get_backend(model, think=think)
        cache = QueryCache.open(cache_dir)
        try:
            with console.status("[dim]generating test queries…[/dim]"):
                query_list = await build_query_set(
                    backend,
                    tool_set,
                    n=queries,
                    cache=cache,
                    regenerate=regenerate,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            cache.save(generator=backend.name)

            with console.status("[dim]measuring the baseline…[/dim]"):
                baseline = await run_selection(
                    backend,
                    tool_set,
                    query_list,
                    config=RunConfig(
                        n=queries,
                        k=runs,
                        temperature=temperature,
                        seed=seed,
                        concurrency=concurrency,
                    ),
                )
            base_metrics = compute(baseline)

            with console.status("[dim]proposing and re-testing rewrites…[/dim]") as status:

                def on_pair(pair: object) -> None:
                    status.update(f"[dim]re-testing a rewrite for {pair}…[/dim]")

                report = await run_fix_mode(
                    backend,
                    tool_set,
                    baseline,
                    base_metrics,
                    limit=pairs,
                    on_progress=on_pair,
                )
        finally:
            await backend.aclose()
        return baseline, base_metrics, report

    try:
        baseline, _base_metrics, report = asyncio.run(go())
    except BackendError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc

    if not [c for c in baseline.choices if not c.error]:
        err_console.print("[red]error:[/red] every selection call failed; nothing to fix.")
        for error in baseline.errors[:3]:
            err_console.print(f"  {error}")
        raise typer.Exit(EXIT_ERROR)

    from mispick.fix import FixReport

    assert isinstance(report, FixReport)

    if fmt is Format.md or out:
        text = fixes_report.render_markdown(report, tool_set)
        if out:
            path = Path(out)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            err_console.print(f"[dim]wrote {path}[/dim]")
        else:
            print(text, end="")
    else:
        fixes_report.render_terminal(report, tool_set, console)

    if patch:
        patch_path = Path(patch)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(as_patch(report, tool_set))
        err_console.print(f"[dim]wrote {patch_path}[/dim]")


@app.command(name="report")
def report_cmd(
    path: Annotated[str, typer.Argument(help="A JSON report from `mispick run --format json`.")],
    fmt: FormatOpt = Format.terminal,
    out: OutOpt = None,
    badge: BadgeOpt = None,
) -> None:
    """Re-render a saved run in another format, without measuring again."""
    from mispick.report.load import ReportError
    from mispick.report.load import load as load_report

    try:
        result, metrics = load_report(path)
    except ReportError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc
    _emit(result, metrics, fmt, out, badge)


@app.command(name="compare")
def compare_cmd(
    base: Annotated[str, typer.Argument(help="The base branch's JSON report.")],
    head: Annotated[str, typer.Argument(help="This branch's JSON report.")],
    out: OutOpt = None,
    max_drop: Annotated[
        int | None,
        typer.Option("--max-drop", help="Exit 1 if the score fell by more than this."),
    ] = None,
) -> None:
    """Compare two saved runs and report the delta. For CI."""
    from mispick.report.load import ReportError
    from mispick.report.load import load as load_report

    try:
        base_result, base_metrics = load_report(base)
        head_result, head_metrics = load_report(head)
    except ReportError as exc:
        err_console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(EXIT_ERROR) from exc

    text = markdown_report.render_comparison(
        (head_result, head_metrics), (base_result, base_metrics)
    )
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        err_console.print(f"[dim]wrote {path}[/dim]")
    else:
        print(text, end="")

    drop = base_metrics.score - head_metrics.score
    if max_drop is not None and drop > max_drop:
        err_console.print(
            f"[red]score fell by {drop} points "
            f"({base_metrics.score} → {head_metrics.score}), above --max-drop {max_drop}[/red]"
        )
        raise typer.Exit(EXIT_BELOW_THRESHOLD)


@app.command()
def version() -> None:
    """Print the mispick version."""
    console.print(f"mispick {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
