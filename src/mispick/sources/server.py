"""Connect to a live MCP server and read its tool list.

**Safety rule (SPEC section 3): this module never calls a tool.**

`mcp.Client` exposes `call_tool`, so the client is created, used and closed entirely inside
these functions. Nothing that escapes this module can reach it: callers get a `ToolSet` of
plain pydantic models. The enforcement test is `tests/test_never_calls_tools.py`.
"""

from __future__ import annotations

import shlex
import tempfile
from typing import IO, Any

from mispick.sources.snapshot import tool_from_dict
from mispick.types import ServerInfo, Tool, ToolSet

#: Seconds to wait for a server to start and answer. SPEC section 6.1.
DEFAULT_TIMEOUT = 20.0

#: Refuse to follow pagination forever if a server returns a cursor loop.
MAX_PAGES = 100


class ServerError(RuntimeError):
    """We could not get a tool list from this server."""


def _tool_to_dict(tool: Any) -> dict[str, Any]:
    """Normalise an SDK Tool into a plain dict, without importing its type."""
    if hasattr(tool, "model_dump"):
        raw: dict[str, Any] = dict(tool.model_dump(by_alias=True, exclude_none=True))
    else:  # pragma: no cover - defensive
        raw = dict(tool)
    if "inputSchema" not in raw and "input_schema" in raw:
        raw["inputSchema"] = raw.pop("input_schema")
    return raw


async def _collect(client: Any, label: str, source: str) -> ToolSet:
    """Run tools/list to exhaustion. Only `list_tools` is ever called on `client`."""
    tools: list[Tool] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()

    for _ in range(MAX_PAGES):
        page = await client.list_tools(cursor=cursor)
        for raw in page.tools:
            tools.append(tool_from_dict(_tool_to_dict(raw)))
        cursor = getattr(page, "next_cursor", None)
        # An empty string is a VALID cursor; only None ends pagination.
        # See docs/research.md section 3.
        if cursor is None:
            break
        if cursor in seen_cursors:
            raise ServerError(
                f"{label} returned a repeating pagination cursor; refusing to loop. "
                "This is a bug in the server."
            )
        seen_cursors.add(cursor)
    else:
        raise ServerError(f"{label} returned more than {MAX_PAGES} pages of tools; giving up.")

    server_info = getattr(client, "server_info", None)
    info = ServerInfo(
        label=label,
        name=getattr(server_info, "name", None),
        version=getattr(server_info, "version", None),
        protocol_version=getattr(client, "protocol_version", None),
        source=source,  # type: ignore[arg-type]
    )
    return ToolSet(tools=tools, servers=[info])


def _stderr_tail(errlog: IO[str], lines: int = 8) -> str:
    """The last few lines the server wrote to stderr, for an error message."""
    try:
        errlog.flush()
        errlog.seek(0)
        text = errlog.read().strip()
    except Exception:  # pragma: no cover - defensive
        return ""
    if not text:
        return ""
    tail = text.splitlines()[-lines:]
    return "\n\nThe server's last output was:\n  " + "\n  ".join(tail)


def _attach_server(tool_set: ToolSet, server: str | None) -> ToolSet:
    if server is None:
        return tool_set
    for tool in tool_set.tools:
        tool.server = server
    return tool_set


async def load_stdio(
    cmd: str,
    *,
    label: str | None = None,
    server: str | None = None,
    env: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ToolSet:
    """Launch a local server as a subprocess and read its tools.

    `cmd` is a shell-style command line, e.g. ``python -m my_server``.

    The child's stderr goes to a temporary file rather than ours: a chatty server must not
    corrupt the terminal report, and if the server dies we can quote its last words back to
    the user. (The SDK would otherwise inherit `sys.stderr`, which is not a real file
    descriptor under pytest or any captured runner.)
    """
    from mcp import Client, StdioServerParameters
    from mcp.client.stdio import stdio_client

    parts = shlex.split(cmd)
    if not parts:
        raise ServerError("Empty --cmd. Pass something like: --cmd 'python -m my_server'")
    name = label or parts[0]
    params = StdioServerParameters(command=parts[0], args=parts[1:], env=env)

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as errlog:
        try:
            transport = stdio_client(params, errlog=errlog)
            async with Client(transport, read_timeout_seconds=timeout) as client:
                return _attach_server(await _collect(client, name, "stdio"), server)
        except ServerError:
            raise
        except Exception as exc:
            raise ServerError(
                f"Could not start or query {name!r} over stdio: {exc}\n"
                "Check the command runs on its own, and that it speaks MCP over stdio."
                + _stderr_tail(errlog)
            ) from exc


async def load_http(
    url: str,
    *,
    label: str | None = None,
    server: str | None = None,
    bearer_token: str | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> ToolSet:
    """Read tools from a Streamable HTTP server."""
    from mcp import Client

    name = label or url
    kwargs: dict[str, Any] = {"read_timeout_seconds": timeout}
    if bearer_token:
        kwargs["headers"] = {"Authorization": f"Bearer {bearer_token}"}
    try:
        async with Client(url, **kwargs) as client:
            return _attach_server(await _collect(client, name, "http"), server)
    except ServerError:
        raise
    except Exception as exc:
        raise ServerError(
            f"Could not query {name!r} over Streamable HTTP: {exc}\n"
            "Check the URL, and set MISPICK_BEARER_TOKEN if the server needs auth."
        ) from exc


async def load_in_process(server_obj: Any, *, label: str = "in-process") -> ToolSet:
    """Read tools from a `Server`/`FastMCP` instance without a subprocess.

    For fixtures and tests only - the SDK's Client accepts a server object directly.
    """
    from mcp import Client

    async with Client(server_obj) as client:
        return await _collect(client, label, "stdio")
