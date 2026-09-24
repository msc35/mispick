"""One place that turns CLI target options into a ToolSet."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from mispick.sources.config import load_config
from mispick.sources.server import DEFAULT_TIMEOUT, load_http, load_stdio
from mispick.sources.snapshot import load_snapshot
from mispick.types import ToolSet


class TargetError(ValueError):
    """The user gave us no target, or more than one."""


@dataclass
class Target:
    """Exactly one way of getting tools."""

    cmd: str | None = None
    url: str | None = None
    config: str | None = None
    snapshot: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    only: list[str] = field(default_factory=list)

    def describe(self) -> str:
        for label, value in (
            ("--cmd", self.cmd),
            ("--url", self.url),
            ("--config", self.config),
            ("--snapshot", self.snapshot),
        ):
            if value:
                return f"{label} {value}"
        return "(no target)"

    @property
    def is_multi_server(self) -> bool:
        return self.config is not None


def validate(target: Target) -> None:
    given = [k for k, v in
             (("--cmd", target.cmd), ("--url", target.url),
              ("--config", target.config), ("--snapshot", target.snapshot)) if v]
    if not given:
        raise TargetError(
            "No target given. Pass exactly one of:\n"
            "  --cmd 'python -m my_server'      a local stdio server\n"
            "  --url https://example.com/mcp    a Streamable HTTP server\n"
            "  --config ~/.../claude_desktop_config.json   every server in a config\n"
            "  --snapshot tools.json            a captured tools/list, offline"
        )
    if len(given) > 1:
        raise TargetError(f"Pass only one target, got {' and '.join(given)}.")


async def load(target: Target) -> tuple[ToolSet, list[str]]:
    """Load tools for a target. Returns the ToolSet and any non-fatal failures."""
    validate(target)
    if target.snapshot:
        return load_snapshot(target.snapshot), []
    if target.cmd:
        return await load_stdio(target.cmd, timeout=target.timeout), []
    if target.url:
        return (
            await load_http(
                target.url,
                bearer_token=os.environ.get("MISPICK_BEARER_TOKEN"),
                timeout=target.timeout,
            ),
            [],
        )
    assert target.config is not None
    return await load_config(target.config, timeout=target.timeout, only=target.only or None)


def write_snapshot(tool_set: ToolSet, path: str | Path) -> Path:
    """Write a ToolSet back out as a tools/list snapshot."""
    info = tool_set.servers[0] if tool_set.servers else None
    payload = {
        "protocolVersion": info.protocol_version if info else None,
        "serverInfo": {"name": info.name, "version": info.version} if info else None,
        "tools": [
            {
                "name": t.name,
                **({"title": t.title} if t.title else {}),
                **({"description": t.description} if t.description else {}),
                "inputSchema": t.input_schema,
                **({"annotations": t.annotations} if t.annotations else {}),
            }
            for t in tool_set.sorted_tools()
        ],
    }
    p = Path(path)
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p
