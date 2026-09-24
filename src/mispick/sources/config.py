"""Load a whole `claude_desktop_config.json` / `.mcp.json` and fan out to every server.

This is what makes cross-server collision detection possible (SPEC differentiator 3):
tools from every server are loaded into one ToolSet, each tagged with its config label.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, NamedTuple

from mispick.sources.server import DEFAULT_TIMEOUT, load_http, load_stdio
from mispick.types import ServerInfo, ToolSet


class ConfigError(ValueError):
    """A config file we could not make sense of."""


class ServerSpec(NamedTuple):
    """One server entry from a config file."""

    label: str
    cmd: str | None = None
    url: str | None = None
    env: dict[str, str] | None = None


def parse_config(payload: Any) -> list[ServerSpec]:
    """Extract server specs from a parsed config file.

    Handles the `mcpServers` key used by Claude Desktop and `.mcp.json`, and a bare
    mapping of label to entry.
    """
    if not isinstance(payload, dict):
        raise ConfigError("Config file must be a JSON object.")
    servers = payload.get("mcpServers")
    if not isinstance(servers, dict):
        servers = payload.get("servers")
    if not isinstance(servers, dict):
        # Tolerate a bare mapping, as long as the values look like server entries.
        if payload and all(isinstance(v, dict) for v in payload.values()):
            servers = payload
        else:
            raise ConfigError(
                "No 'mcpServers' key found. Expected a Claude Desktop or .mcp.json config:\n"
                '  {"mcpServers": {"my-server": {"command": "python", "args": ["-m", "srv"]}}}'
            )

    specs: list[ServerSpec] = []
    for label, entry in servers.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("disabled") is True:
            continue
        env = entry.get("env") if isinstance(entry.get("env"), dict) else None
        url = entry.get("url") or entry.get("serverUrl") or entry.get("httpUrl")
        command = entry.get("command")
        if command:
            args = entry.get("args") or []
            if not isinstance(args, list):
                raise ConfigError(f"Server {label!r}: 'args' must be a list.")
            cmd = " ".join([str(command), *(str(a) for a in args)])
            specs.append(ServerSpec(label=label, cmd=cmd, env=env))
        elif isinstance(url, str):
            specs.append(ServerSpec(label=label, url=url, env=env))
        # Entries with neither are skipped rather than fatal: a config may hold
        # comments or future keys we do not understand.
    if not specs:
        raise ConfigError("Config file contained no usable server entries.")
    return specs


def _resolve_env(env: dict[str, str] | None) -> dict[str, str] | None:
    """Expand ${VAR} references against the real environment."""
    if not env:
        return None
    return {k: os.path.expandvars(str(v)) for k, v in env.items()}


async def _load_one(spec: ServerSpec, timeout: float) -> tuple[ToolSet | None, str | None]:
    """Load one server, returning either its tools or an error message."""
    try:
        if spec.cmd:
            ts = await load_stdio(
                spec.cmd,
                label=spec.label,
                server=spec.label,
                env=_resolve_env(spec.env),
                timeout=timeout,
            )
        elif spec.url:
            ts = await load_http(
                spec.url,
                label=spec.label,
                server=spec.label,
                bearer_token=os.environ.get("MISPICK_BEARER_TOKEN"),
                timeout=timeout,
            )
        else:  # pragma: no cover - parse_config guarantees one of the two
            return None, f"{spec.label}: no command or url"
        return ts, None
    # Deliberately broad: one unreachable server must not kill a ten-server run.
    # (`Exception` already covers ServerError; naming both was misleading.)
    except Exception as exc:
        return None, f"{spec.label}: {exc}"


async def load_config(
    path: str | Path,
    *,
    timeout: float = DEFAULT_TIMEOUT,
    only: list[str] | None = None,
) -> tuple[ToolSet, list[str]]:
    """Load every server in a config file concurrently.

    Returns the merged ToolSet and a list of human-readable failures. A server that will
    not start is reported, not fatal - a config with ten servers should still measure the
    nine that work.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"No such config file: {p}")
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{p} is not valid JSON: {exc}") from exc

    specs = parse_config(payload)
    if only:
        wanted = set(only)
        specs = [s for s in specs if s.label in wanted]
        if not specs:
            raise ConfigError(f"No servers in {p.name} matched: {', '.join(sorted(wanted))}")

    results = await asyncio.gather(*(_load_one(s, timeout) for s in specs))

    merged = ToolSet()
    failures: list[str] = []
    for spec, (ts, err) in zip(specs, results, strict=True):
        if err or ts is None:
            failures.append(err or f"{spec.label}: unknown error")
            continue
        merged.tools.extend(ts.tools)
        for info in ts.servers:
            info.label = spec.label
            merged.servers.append(info)
    if not merged.tools and failures:
        raise ConfigError(
            "Could not load any server from this config:\n  " + "\n  ".join(failures)
        )
    return merged, failures


def collisions(tool_set: ToolSet) -> dict[str, list[ServerInfo | str]]:
    """Bare tool names exposed by more than one server.

    This is the static half of cross-server analysis; the measured half is the confusion
    matrix over qualified names.
    """
    by_name: dict[str, list[str]] = {}
    for tool in tool_set.tools:
        by_name.setdefault(tool.name, []).append(tool.server or "(unlabelled)")
    return {
        name: sorted(set(servers))  # type: ignore[misc]
        for name, servers in by_name.items()
        if len(set(servers)) > 1
    }
