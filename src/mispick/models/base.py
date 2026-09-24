"""The backend interface.

A backend does exactly two things:

- `choose(tools, query)` - present the tool list the way a real MCP client would (as
  function/tool definitions on a chat request) and report which one the model picked.
- `generate(prompt)` - free-form text, used for query generation and fix proposals.

Selection and generation are deliberately separate calls. See `generate.py` for why.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from mispick.types import Tool

#: Generation needs a generous budget. Small reasoning models - which is most of what runs
#: locally - spend hundreds or thousands of hidden tokens before emitting a single visible
#: character, and qwen3.5:4b measurably needs ~3.5k completion tokens to answer the
#: generation prompt at all. A tight budget does not truncate the answer, it produces an
#: empty one. See docs/research.md section 6.
DEFAULT_MAX_TOKENS = 8192


class BackendError(RuntimeError):
    """The model backend could not be reached, or gave us something unusable."""


@dataclass
class Selection:
    """One model decision, before we validate it against the schema."""

    #: The tool name the model asked for, exactly as it said it. None means "no tool".
    chosen: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    #: Raw text, kept for debugging a confusing run.
    raw: str = ""
    error: str | None = None


def build_tool_payload(tools: list[Tool]) -> list[dict[str, Any]]:
    """Render tools as OpenAI-style function definitions.

    This is the fidelity that matters: the model should see what a real client would send,
    so we pass real tool definitions rather than describing the tools in a prompt.

    Multi-server tools are presented under their qualified name, because that is what a
    client aggregating several servers has to do - the MCP spec recommends exactly this
    prefixing strategy. Dots and colons are not allowed in some providers' function names,
    so `:` becomes `__`.
    """
    payload: list[dict[str, Any]] = []
    for tool in tools:
        schema = dict(tool.input_schema or {"type": "object"})
        schema.setdefault("type", "object")
        payload.append(
            {
                "type": "function",
                "function": {
                    "name": wire_name(tool),
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return payload


def wire_name(tool: Tool) -> str:
    """The function name we put on the wire for this tool."""
    return tool.qualified_name.replace(":", "__")


def from_wire_name(name: str, tools: list[Tool]) -> str | None:
    """Map a name the model returned back to a qualified tool name.

    Returns None if no tool matches, which is how we detect a phantom tool.
    """
    by_wire = {wire_name(t): t.qualified_name for t in tools}
    if name in by_wire:
        return by_wire[name]
    # Be forgiving about a model that echoes the bare name in a multi-server run, but only
    # when it is unambiguous.
    matches = {t.qualified_name for t in tools if t.name == name}
    if len(matches) == 1:
        return matches.pop()
    return None


class Backend(abc.ABC):
    """What every model backend must provide."""

    #: Human-readable identifier that goes into every report.
    name: str

    #: Whether this backend honours a seed. Reports should not promise determinism we
    #: cannot deliver.
    supports_seed: bool = False

    #: Whether this backend accepts a sampling temperature at all. Anthropic's Messages API
    #: does not - it takes neither `temperature` nor `top_p` - so a report must not print a
    #: temperature that was never sent.
    supports_temperature: bool = True

    @abc.abstractmethod
    async def choose(
        self,
        tools: list[Tool],
        query: str,
        *,
        temperature: float = 0.0,
        seed: int | None = None,
    ) -> Selection:
        """Ask the model to pick one tool for this query."""

    @abc.abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        temperature: float = 0.7,
        seed: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """Ask the model for free-form text."""

    async def aclose(self) -> None:
        """Release any client resources."""
        return None
