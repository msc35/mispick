#!/usr/bin/env python3
"""Pick servers from the official MCP registry that should start without credentials.

The registry does not expose an "API keys required" flag, so this is a heuristic over each
package's own metadata plus a keyword filter. Everything it picks is verified for real in
`capture.py`, which records what actually happened - a server that turns out to need a key is
recorded as skipped, not guessed at.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers"
#: The registry caps `limit` at 100; anything larger is a 422.
PAGE_SIZE = 100

#: Words in a description or name that almost always mean "bring your own credential".
NEEDS_KEY = re.compile(
    r"\b(api[- ]?key|token|oauth|credential|bearer|secret|login|sign[- ]?in|account|"
    r"workspace id|personal access)\b",
    re.IGNORECASE,
)


def fetch_page(cursor: str | None) -> dict[str, Any]:
    url = f"{REGISTRY}?limit={PAGE_SIZE}&version=latest"
    if cursor:
        url += f"&cursor={urllib.parse.quote(cursor)}"
    request = urllib.request.Request(url, headers={"User-Agent": "mispick-benchmark"})
    with urllib.request.urlopen(request, timeout=30) as response:
        payload: dict[str, Any] = json.load(response)
    return payload


def iter_servers(max_pages: int = 20) -> list[dict[str, Any]]:
    """Every latest-version server entry, following the registry's cursor."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        page = fetch_page(cursor)
        out.extend(page.get("servers") or [])
        cursor = (page.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
    return out


def runnable_package(server: dict[str, Any]) -> dict[str, Any] | None:
    """The first stdio package we know how to launch without cloning anything."""
    for package in server.get("packages") or []:
        if (package.get("transport") or {}).get("type") != "stdio":
            continue
        if package.get("registryType") in {"npm", "pypi"}:
            return package
    return None


def install_command(package: dict[str, Any]) -> str:
    """How to launch this package, using the ecosystem's own runner."""
    identifier = package["identifier"]
    version = package.get("version")
    if package["registryType"] == "npm":
        spec = f"{identifier}@{version}" if version else identifier
        return f"npx -y {spec}"
    spec = f"{identifier}=={version}" if version else identifier
    return f"uvx --from {spec} {identifier}"


def looks_key_free(server: dict[str, Any]) -> bool:
    haystack = " ".join(
        str(server.get(field) or "") for field in ("name", "title", "description")
    )
    return not NEEDS_KEY.search(haystack)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("benchmark/servers.json"))
    args = parser.parse_args()

    try:
        entries = iter_servers()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"error: could not reach the registry: {exc}")
        return 2

    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in entries:
        server = entry.get("server") or {}
        name = server.get("name")
        if not name or name in seen:
            continue
        package = runnable_package(server)
        if not package or not looks_key_free(server):
            continue
        seen.add(name)
        picked.append(
            {
                "name": name,
                "title": server.get("title") or name,
                "description": server.get("description"),
                "version": server.get("version"),
                "repository": (server.get("repository") or {}).get("url"),
                "website": server.get("websiteUrl"),
                "registry_type": package["registryType"],
                "cmd": install_command(package),
            }
        )
        if len(picked) >= args.limit:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": REGISTRY,
                "note": (
                    "Heuristically filtered for servers that start without credentials. "
                    "capture.py records what actually happened."
                ),
                "servers": picked,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"scanned {len(entries)} registry entries, picked {len(picked)} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
