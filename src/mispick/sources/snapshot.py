"""Load a `tools/list` JSON file directly. No network, no subprocess."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mispick.types import ServerInfo, Tool, ToolSet


class SnapshotError(ValueError):
    """A snapshot file we could not make sense of."""


def _as_tool_dicts(payload: Any) -> list[dict[str, Any]]:
    """Accept the three shapes a `tools/list` capture comes in."""
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, dict):
        tools = payload.get("tools")
        if isinstance(tools, list):
            return list(tools)
        result = payload.get("result")
        if isinstance(result, dict) and isinstance(result.get("tools"), list):
            return list(result["tools"])
    raise SnapshotError(
        "Could not find a tool list in this file. Expected one of:\n"
        '  {"tools": [...]}            a tools/list result\n'
        '  {"result": {"tools": [...]}} a full JSON-RPC envelope\n'
        "  [...]                        a bare array of tools\n"
        "Capture one with: mispick snapshot --cmd '<your server>' -o tools.json"
    )


def tool_from_dict(raw: dict[str, Any], server: str | None = None) -> Tool:
    """Build a Tool from a raw spec dict, tolerating camelCase and snake_case."""
    if not isinstance(raw, dict):
        raise SnapshotError(f"Expected a tool object, got {type(raw).__name__}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise SnapshotError(f"Tool is missing a string 'name': {raw!r}")
    schema = raw.get("inputSchema") or raw.get("input_schema") or {"type": "object"}
    if not isinstance(schema, dict):
        raise SnapshotError(f"Tool {name!r} has a non-object inputSchema")
    annotations = raw.get("annotations") or {}
    if not isinstance(annotations, dict):
        annotations = {}
    return Tool(
        name=name,
        title=raw.get("title"),
        description=raw.get("description"),
        input_schema=schema,
        annotations=annotations,
        server=server,
    )


def parse_snapshot(payload: Any, label: str = "snapshot", server: str | None = None) -> ToolSet:
    """Turn already-loaded JSON into a ToolSet."""
    tools = [tool_from_dict(raw, server=server) for raw in _as_tool_dicts(payload)]
    info = ServerInfo(label=label, source="snapshot")
    if isinstance(payload, dict):
        meta = payload.get("serverInfo") or payload.get("server_info") or {}
        if isinstance(meta, dict):
            info.name = meta.get("name")
            info.version = meta.get("version")
        info.protocol_version = payload.get("protocolVersion") or payload.get("protocol_version")
    return ToolSet(tools=tools, servers=[info])


def load_snapshot(path: str | Path, server: str | None = None) -> ToolSet:
    """Read a snapshot file from disk."""
    p = Path(path)
    if not p.is_file():
        raise SnapshotError(f"No such snapshot file: {p}")
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"{p} is not valid JSON: {exc}") from exc
    return parse_snapshot(payload, label=p.stem, server=server)
