"""The `mispick` command line."""

from __future__ import annotations

import asyncio
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from mispick import __version__
from mispick.loader import Target, TargetError, load, write_snapshot
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


@app.command()
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


# `snapshot` is both an option name and a command name; register the command under the
# name we want without shadowing the option.
app.command(name="snapshot")(snapshot_cmd)


@app.command()
def version() -> None:
    """Print the mispick version."""
    console.print(f"mispick {__version__}")


if __name__ == "__main__":  # pragma: no cover
    app()
