"""Enforcement of SPEC section 3: we never call a tool.

Three independent checks, because this is the promise that makes mispick safe to point
at a destructive server:

1. Static: no module under `src/mispick/` may *invoke* `call_tool` or mention `tools/call`
   as a string literal.
2. Reachability: nothing a source function returns may expose a `call_tool` attribute.
3. Runtime: with `Client.call_tool` replaced by a bomb, a full tool load still succeeds.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src" / "mispick"


def _python_files() -> list[Path]:
    files = sorted(SRC.rglob("*.py"))
    assert files, "found no source files to scan - has the package moved?"
    return files


def test_no_source_file_invokes_call_tool() -> None:
    """Static scan of the AST, so comments and docstrings are exempt but code is not."""
    offenders: list[str] = []
    for path in _python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            # `something.call_tool(...)` or a bare reference to `.call_tool`
            if isinstance(node, ast.Attribute) and node.attr == "call_tool":
                offenders.append(f"{path.relative_to(SRC.parent.parent)}:{node.lineno} .call_tool")
            # getattr(client, "call_tool") and the wire method name
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and node.value in {"call_tool", "tools/call"}
            ):
                offenders.append(
                    f"{path.relative_to(SRC.parent.parent)}:{node.lineno} {node.value!r}"
                )
    assert not offenders, (
        "SPEC section 3 violated - these reference tool invocation:\n" + "\n".join(offenders)
    )


def test_tool_set_does_not_expose_a_client(tool_set) -> None:
    """Whatever escapes `sources/` must be inert data."""
    assert not hasattr(tool_set, "call_tool")
    assert not hasattr(tool_set, "session")
    assert not hasattr(tool_set, "client")
    for tool in tool_set.tools:
        assert not hasattr(tool, "call_tool")
        # A Tool is a pydantic model of plain JSON, nothing more.
        assert set(tool.model_dump()) == {
            "name",
            "title",
            "description",
            "input_schema",
            "annotations",
            "server",
        }


async def test_loading_tools_never_touches_call_tool(fixture_server, monkeypatch) -> None:
    """Replace call_tool with a bomb and load tools for real."""
    from mcp.client.client import Client

    async def bomb(*args: object, **kwargs: object) -> None:
        raise AssertionError("mispick called a tool - SPEC section 3 violated")

    monkeypatch.setattr(Client, "call_tool", bomb)

    from mispick.sources.server import load_in_process

    tools = await load_in_process(fixture_server, label="fixture")
    assert len(tools.tools) == 6


def test_public_api_exports_no_invocation_helper() -> None:
    """`import mispick` must not hand anyone a way to call a tool."""
    import mispick

    for name in dir(mispick):
        assert "call_tool" not in name
