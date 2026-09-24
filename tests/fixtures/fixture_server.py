"""A real MCP server with six tools, two of them deliberately confusable.

`search_docs` and `search_issues` share the description "Search for information." -
a model has nothing to go on. The other four are described properly.

Runnable as a stdio server (`python -m tests.fixtures.fixture_server`) and importable
as `server` for in-process tests.
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp: MCPServer = MCPServer("acme-workspace", version="1.4.2")


@mcp.tool(description="Search for information.")
def search_docs(query: str) -> str:
    return f"docs matching {query}"


@mcp.tool(description="Search for information.")
def search_issues(query: str) -> str:
    return f"issues matching {query}"


@mcp.tool(
    description=(
        "Open a new support ticket for a customer, with a subject and a priority level."
    )
)
def create_ticket(subject: str, priority: str = "normal") -> str:
    return f"opened {subject} ({priority})"


@mcp.tool(
    description=(
        "Close an existing support ticket by its numeric id, with an optional resolution note."
    )
)
def close_ticket(ticket_id: int, resolution: str = "") -> str:
    return f"closed {ticket_id}: {resolution}"


@mcp.tool(description="Email an invoice PDF to a customer for a given order.")
def send_invoice(order_id: str, email: str) -> str:
    return f"invoice for {order_id} sent to {email}"


@mcp.tool(description="Refund a customer's order in full. This cannot be undone.")
def refund_order(order_id: str, reason: str = "") -> str:
    return f"refunded {order_id}: {reason}"


if __name__ == "__main__":
    mcp.run()
