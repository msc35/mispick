"""A second server that collides with the first: it also exposes `search_docs`.

Used to exercise cross-server collision detection (SPEC differentiator 3).
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

mcp: MCPServer = MCPServer("acme-wiki", version="0.9.0")


@mcp.tool(description="Search for information.")
def search_docs(query: str) -> str:
    return f"wiki pages matching {query}"


@mcp.tool(description="Publish a draft wiki page so it becomes visible to everyone.")
def publish_page(page_id: str) -> str:
    return f"published {page_id}"


if __name__ == "__main__":
    mcp.run()
