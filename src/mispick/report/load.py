"""Rebuild a run from a saved JSON report, so reports can be re-rendered without re-measuring.

A measurement costs real time and, on a cloud model, real money. Re-rendering the same run in
another format must never re-run it. The JSON report carries everything needed, which is also
why it stores every trial.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mispick.metrics import Metrics, compute
from mispick.select import RunConfig, RunResult
from mispick.types import Choice, Query, ServerInfo, Tool, ToolSet


class ReportError(ValueError):
    """A saved report we could not read."""


def load(path: str | Path) -> tuple[RunResult, Metrics]:
    """Read a JSON report back into a RunResult and its metrics."""
    p = Path(path)
    if not p.is_file():
        raise ReportError(f"No such report file: {p}")
    try:
        payload = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ReportError(f"{p} is not valid JSON: {exc}") from exc
    return from_payload(payload, source=str(p))


def from_payload(payload: Any, *, source: str = "report") -> tuple[RunResult, Metrics]:
    if not isinstance(payload, dict) or payload.get("tool") != "mispick":
        raise ReportError(
            f"{source} does not look like a mispick JSON report. "
            "Produce one with: mispick run … --format json --out report.json"
        )
    version = payload.get("schemaVersion")
    from mispick.report.json import SCHEMA_VERSION

    if version != SCHEMA_VERSION:
        raise ReportError(
            f"{source} was written by a different version of mispick "
            f"(schema {version}, this build reads {SCHEMA_VERSION})."
        )

    run = payload.get("run") or {}
    tools = [
        Tool(
            name=_bare(entry.get("name", "")),
            title=entry.get("title"),
            description=entry.get("description"),
            input_schema=entry.get("inputSchema") or {"type": "object"},
            annotations=entry.get("annotations") or {},
            server=entry.get("server"),
        )
        for entry in payload.get("tools") or []
    ]
    servers = [ServerInfo(**info) for info in payload.get("servers") or []]
    tool_set = ToolSet(tools=tools, servers=servers)

    queries = [Query(**q) for q in payload.get("queries") or []]
    choices = []
    for raw in payload.get("trials") or []:
        fields = {k: v for k, v in raw.items() if k != "expected"}
        choices.append(Choice(**fields))

    config = RunConfig(
        model=run.get("model", "unknown"),
        n=int(run.get("n") or 0),
        k=int(run.get("k") or 0),
        temperature=float(run.get("temperature") or 0.0),
        seed=run.get("seed"),
    )
    result = RunResult(
        tool_set=tool_set,
        queries=queries,
        choices=choices,
        config=config,
        errors=payload.get("backendErrors") or [],
    )
    return result, compute(result)


def _bare(qualified: str) -> str:
    """Strip a `server:` prefix, since Tool stores the two separately."""
    return qualified.split(":", 1)[1] if ":" in qualified else qualified
