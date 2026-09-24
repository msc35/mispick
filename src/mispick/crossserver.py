"""Cross-server analysis: what happens when one client loads several MCP servers at once.

Nobody else measures this, and the MCP spec itself flags it as a real problem:

    "Clients or proxies that aggregate tools from multiple servers MAY encounter naming
    collisions (for example, two servers each exposing a `search` tool) and SHOULD implement
    a disambiguation strategy such as prefixing tool names with a server identifier. The
    server `name` (from `serverInfo`) is not guaranteed to be unique across servers and
    SHOULD NOT be relied upon for disambiguation."
    - MCP 2026-07-28, server/tools

Two halves:

- **Static collisions**: bare tool names exposed by more than one server. Cheap, no model.
- **Measured leakage**: trials where the model picked a tool from the *wrong server*. This is
  the part that matters, and it happens even between tools whose names do not collide - a
  `publish_page` on one server will happily swallow work meant for `create_ticket` on
  another if their descriptions read alike.

Servers are keyed on the user's own config label throughout, never on `serverInfo.name`,
exactly as the spec warns.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from mispick.metrics import NONE, PHANTOM, Metrics, Rate
from mispick.select import RunResult
from mispick.types import ToolSet


@dataclass
class NameCollision:
    """One bare tool name offered by several servers."""

    name: str
    servers: list[str]
    #: True when the colliding tools also describe themselves identically.
    identical_descriptions: bool = False

    @property
    def qualified(self) -> list[str]:
        return [f"{s}:{self.name}" for s in self.servers]


@dataclass
class ServerScore:
    """How one server fared on its own tools."""

    label: str
    tool_count: int
    accuracy: Rate
    #: Trials that wanted this server's tool but went to another server.
    lost: int = 0
    #: Trials that wanted another server's tool but landed here.
    stolen: int = 0


@dataclass
class CrossServerReport:
    collisions: list[NameCollision] = field(default_factory=list)
    servers: list[ServerScore] = field(default_factory=list)
    #: (expected, chosen, count) where the two live on different servers.
    leaks: list[tuple[str, str, int]] = field(default_factory=list)
    cross_server_rate: Rate = field(default_factory=lambda: Rate(0, 0))

    @property
    def is_multi_server(self) -> bool:
        return len(self.servers) > 1

    @property
    def worst_collision(self) -> NameCollision | None:
        if not self.collisions:
            return None
        return sorted(
            self.collisions,
            key=lambda c: (not c.identical_descriptions, -len(c.servers), c.name),
        )[0]


def find_collisions(tool_set: ToolSet) -> list[NameCollision]:
    """Bare names offered by more than one server."""
    by_name: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for tool in tool_set.tools:
        by_name[tool.name].append((tool.server or "(unlabelled)", tool.description))

    out: list[NameCollision] = []
    for name, entries in sorted(by_name.items()):
        servers = sorted({label for label, _ in entries})
        if len(servers) < 2:
            continue
        descriptions = {d or "" for _, d in entries}
        out.append(
            NameCollision(
                name=name,
                servers=servers,
                identical_descriptions=len(descriptions) == 1 and bool(descriptions.pop()),
            )
        )
    return out


def analyse(result: RunResult, metrics: Metrics) -> CrossServerReport:
    """Compute the cross-server view of a run."""
    tool_set = result.tool_set
    by_qualified = tool_set.by_qualified_name()

    def server_of(qualified: str) -> str | None:
        tool = by_qualified.get(qualified)
        return tool.server if tool else None

    report = CrossServerReport(collisions=find_collisions(tool_set))

    # Per-server accuracy and the two directions of leakage.
    tool_counts: dict[str, int] = defaultdict(int)
    for tool in tool_set.tools:
        tool_counts[tool.server or "(unlabelled)"] += 1

    hits: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    lost: dict[str, int] = defaultdict(int)
    stolen: dict[str, int] = defaultdict(int)
    leaks: dict[tuple[str, str], int] = defaultdict(int)

    cross = 0
    considered = 0

    by_query = {q.id: q for q in result.queries}
    for choice in result.choices:
        if choice.error:
            continue
        query = by_query.get(choice.query_id)
        if query is None or query.expected is None:
            continue
        expected_server = server_of(query.expected)
        if expected_server is None:
            continue
        totals[expected_server] += 1
        considered += 1

        got = PHANTOM if choice.phantom else (choice.chosen or NONE)
        if got == query.expected:
            hits[expected_server] += 1
            continue
        chosen_server = server_of(got)
        if chosen_server is not None and chosen_server != expected_server:
            cross += 1
            lost[expected_server] += 1
            stolen[chosen_server] += 1
            leaks[(query.expected, got)] += 1

    report.servers = [
        ServerScore(
            label=label,
            tool_count=tool_counts[label],
            accuracy=Rate(hits[label], totals[label]),
            lost=lost[label],
            stolen=stolen[label],
        )
        for label in sorted(tool_counts)
    ]
    report.leaks = sorted(
        ((e, c, n) for (e, c), n in leaks.items()), key=lambda t: (-t[2], t[0], t[1])
    )
    report.cross_server_rate = Rate(cross, considered)
    return report
