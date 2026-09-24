"""Shared data types.

Deliberately separate from `mispick.models`, which holds model *backends*.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, Field

QueryKind = Literal["straightforward", "paraphrased", "hard_negative", "no_tool"]

#: Confusion-cause labels, from Hasan et al. (arXiv 2602.14878). See SPEC section 4.7.
SMELLS = (
    "unclear_purpose",
    "missing_usage_guidelines",
    "unstated_limitations",
    "opaque_parameters",
    "underspecified",
    "exemplar_issues",
)


class Tool(BaseModel):
    """One tool as a model would see it.

    Our own type rather than `mcp.types.Tool`, so nothing downstream of
    `sources/` can reach a live client. See SPEC section 3.
    """

    name: str
    title: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    annotations: dict[str, Any] = Field(default_factory=dict)
    #: Config label of the server this came from. None for single-server runs.
    server: str | None = None

    @property
    def qualified_name(self) -> str:
        """`server:name`, or just `name` when single-server.

        The MCP spec warns that `serverInfo.name` is not unique across servers, so we key
        on the user's own config label instead. See docs/research.md section 3.
        """
        return f"{self.server}:{self.name}" if self.server else self.name

    @property
    def display_name(self) -> str:
        """Spec precedence: `title`, then `annotations.title`, then `name`."""
        return self.title or self.annotations.get("title") or self.name

    def fingerprint(self) -> str:
        """Cache key for generated queries: changes only when the tool changes.

        Covers name + description + schema, per SPEC section 6.2.
        """
        payload = json.dumps(
            {
                "name": self.name,
                "description": self.description,
                "input_schema": self.input_schema,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ServerInfo(BaseModel):
    """Where a tool list came from. Reports must state this."""

    label: str
    name: str | None = None
    version: str | None = None
    protocol_version: str | None = None
    source: Literal["stdio", "http", "snapshot"] = "snapshot"


class ToolSet(BaseModel):
    """Tools plus their provenance."""

    tools: list[Tool] = Field(default_factory=list)
    servers: list[ServerInfo] = Field(default_factory=list)

    def sorted_tools(self) -> list[Tool]:
        """Tool order is only a SHOULD in the spec, so sort before hashing.

        See SPEC section 6.1.
        """
        return sorted(self.tools, key=lambda t: t.qualified_name)

    def fingerprint(self) -> str:
        joined = "|".join(f"{t.qualified_name}:{t.fingerprint()}" for t in self.sorted_tools())
        return hashlib.sha256(joined.encode()).hexdigest()[:16]

    def by_qualified_name(self) -> dict[str, Tool]:
        return {t.qualified_name: t for t in self.tools}


class Query(BaseModel):
    """One generated test request."""

    id: str
    text: str
    #: Qualified name of the tool that should be chosen, or None for "no tool fits".
    expected: str | None
    kind: QueryKind = "straightforward"
    #: For hard negatives: the neighbour this query was written to sit close to.
    neighbour: str | None = None


class Choice(BaseModel):
    """What the model picked for one query on one run."""

    query_id: str
    chosen: str | None
    arguments: dict[str, Any] = Field(default_factory=dict)
    #: Set when the model named a tool that does not exist.
    phantom: str | None = None
    args_valid: bool | None = None
    args_errors: list[str] = Field(default_factory=list)
    run: int = 0
    error: str | None = None
