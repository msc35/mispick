#!/usr/bin/env python3
"""Choose which MCP servers the benchmark measures.

**Why this is not just "the top N from the registry".** The official registry has no popularity
ordering - its only query parameters are `limit`, `cursor`, `search`, `updated_since` and
`version` - and it returns entries in reverse-DNS name order. Taking the first 30 gives you 30
servers whose names begin with "a", 13 of them from one publisher. Worse, the official reference
servers (`filesystem`, `memory`, `sequential-thinking`, `everything`) are **not in the registry
at all**, and those are the ones people have actually heard of and run.

So servers come from two places:

1. **Reference servers**, listed here by hand because the registry does not carry them. These
   are the surfaces most MCP users have really installed.
2. **Registry entries ranked by GitHub stars**, which is the only popularity signal available.
   Stars are a rough proxy and are stated as such on the published leaderboard.

Filters applied to the registry set: must have a GitHub repository and a launchable stdio npm or
pypi package, must not be archived, and at most `--per-owner` servers from any one owner - one
publisher with twenty near-identical packages should not crowd out everyone else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REGISTRY = "https://registry.modelcontextprotocol.io/v0/servers"
#: The registry caps `limit` at 100; 101 is a 422.
PAGE_SIZE = 100

#: The official reference servers, which the registry does not list. Only the ones actually
#: published to npm are here - `server-git`, `server-fetch` and `server-time` are not on npm.
REFERENCE_SERVERS = [
    {
        "name": "io.github.modelcontextprotocol/server-filesystem",
        "title": "Filesystem (reference)",
        "description": "Read and write local files.",
        "repository": "https://github.com/modelcontextprotocol/servers",
        "registry_type": "npm",
        "cmd": "npx -y @modelcontextprotocol/server-filesystem .",
        "source": "reference",
    },
    {
        "name": "io.github.modelcontextprotocol/server-memory",
        "title": "Memory (reference)",
        "description": "A knowledge graph the model can read and write.",
        "repository": "https://github.com/modelcontextprotocol/servers",
        "registry_type": "npm",
        "cmd": "npx -y @modelcontextprotocol/server-memory",
        "source": "reference",
    },
    {
        "name": "io.github.modelcontextprotocol/server-sequential-thinking",
        "title": "Sequential Thinking (reference)",
        "description": "A scratchpad for step-by-step reasoning.",
        "repository": "https://github.com/modelcontextprotocol/servers",
        "registry_type": "npm",
        "cmd": "npx -y @modelcontextprotocol/server-sequential-thinking",
        "source": "reference",
    },
    {
        "name": "io.github.modelcontextprotocol/server-everything",
        "title": "Everything (reference)",
        "description": "Exercises every MCP feature; a deliberately broad tool surface.",
        "repository": "https://github.com/modelcontextprotocol/servers",
        "registry_type": "npm",
        "cmd": "npx -y @modelcontextprotocol/server-everything",
        "source": "reference",
    },
]

#: Words that almost always mean "bring your own credential".
NEEDS_KEY = re.compile(
    r"\b(api[- ]?key|token|oauth|credential|bearer|secret|login|sign[- ]?in|"
    r"workspace id|personal access)\b",
    re.IGNORECASE,
)

GITHUB_REPO = re.compile(r"github\.com/([^/\s]+)/([^/\s#?]+?)(?:\.git)?/?$")


def _token() -> str | None:
    """A GitHub token, for the star lookup. Unauthenticated is 60 requests an hour."""
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        if os.environ.get(var):
            return os.environ[var]
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=15)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return None


def fetch_registry(max_pages: int = 30) -> list[dict[str, Any]]:
    """Every latest-version entry, following the cursor."""
    out: list[dict[str, Any]] = []
    cursor: str | None = None
    for _ in range(max_pages):
        url = f"{REGISTRY}?limit={PAGE_SIZE}&version=latest"
        if cursor:
            url += f"&cursor={urllib.parse.quote(cursor)}"
        request = urllib.request.Request(url, headers={"User-Agent": "mispick-benchmark"})
        with urllib.request.urlopen(request, timeout=30) as response:
            page = json.load(response)
        out.extend(page.get("servers") or [])
        cursor = (page.get("metadata") or {}).get("nextCursor")
        if not cursor:
            break
    return out


def install_command(package: dict[str, Any]) -> str:
    identifier = package["identifier"]
    version = package.get("version")
    if package["registryType"] == "npm":
        spec = f"{identifier}@{version}" if version else identifier
        return f"npx -y {spec}"
    spec = f"{identifier}=={version}" if version else identifier
    return f"uvx --from {spec} {identifier}"


def candidates(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Registry entries we could actually launch, with their GitHub coordinates."""
    out: list[dict[str, Any]] = []
    for entry in entries:
        server = entry.get("server") or {}
        url = (server.get("repository") or {}).get("url") or ""
        match = GITHUB_REPO.search(url)
        if not match:
            continue
        package = next(
            (
                p
                for p in (server.get("packages") or [])
                if (p.get("transport") or {}).get("type") == "stdio"
                and p.get("registryType") in {"npm", "pypi"}
            ),
            None,
        )
        if not package:
            continue
        haystack = " ".join(
            str(server.get(f) or "") for f in ("name", "title", "description")
        )
        if NEEDS_KEY.search(haystack):
            continue
        out.append(
            {
                "name": server["name"],
                "title": server.get("title") or server["name"],
                "description": server.get("description"),
                "repository": url,
                "owner": match.group(1),
                "repo": match.group(2),
                "registry_type": package["registryType"],
                "cmd": install_command(package),
                "source": "registry",
            }
        )
    return out


def add_stars(items: list[dict[str, Any]], token: str | None, batch: int = 25) -> None:
    """Annotate each candidate with its repository's star count, in place."""
    if not token:
        print("warning: no GitHub token, so stars cannot be read; ranking will be arbitrary")
        return
    for start in range(0, len(items), batch):
        chunk = items[start : start + batch]
        parts = [
            f'r{i}: repository(owner:"{c["owner"]}", name:"{c["repo"]}") '
            "{ nameWithOwner stargazerCount pushedAt isArchived }"
            for i, c in enumerate(chunk)
        ]
        body = json.dumps({"query": "query{" + " ".join(parts) + "}"}).encode()
        request = urllib.request.Request(
            "https://api.github.com/graphql",
            data=body,
            headers={
                "Authorization": f"Bearer {token}",
                "User-Agent": "mispick-benchmark",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # One deleted or renamed repository makes the whole query "fail" while still
            # returning data for every other alias, so read the body either way.
            try:
                payload = json.load(exc)
            except Exception:
                continue
        except Exception:
            continue
        data = payload.get("data") or {}
        for i, candidate in enumerate(chunk):
            node = data.get(f"r{i}")
            if not node:
                continue
            candidate["stars"] = node["stargazerCount"]
            candidate["pushed_at"] = node["pushedAt"][:10]
            candidate["archived"] = node["isArchived"]


def rank(items: list[dict[str, Any]], limit: int, per_owner: int) -> list[dict[str, Any]]:
    """Most-starred first, skipping archived repos and capping any one owner."""
    ranked = sorted(items, key=lambda c: (-int(c.get("stars") or 0), c["name"]))
    picked: list[dict[str, Any]] = []
    seen_repos: set[tuple[str, str]] = set()
    per: dict[str, int] = {}
    for candidate in ranked:
        if candidate.get("archived"):
            continue
        key = (candidate["owner"].lower(), candidate["repo"].lower())
        if key in seen_repos:
            # Several registry names can point at one repository; measure it once.
            continue
        owner = candidate["owner"].lower()
        if per.get(owner, 0) >= per_owner:
            continue
        seen_repos.add(key)
        per[owner] = per.get(owner, 0) + 1
        picked.append(candidate)
        if len(picked) >= limit:
            break
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=26, help="How many registry servers to add.")
    parser.add_argument("--per-owner", type=int, default=2, help="Cap per GitHub owner.")
    parser.add_argument(
        "--no-reference", action="store_true", help="Skip the hand-listed reference servers."
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark/servers.json"))
    args = parser.parse_args()

    try:
        entries = fetch_registry()
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"error: could not reach the registry: {exc}")
        return 2
    print(f"registry: {len(entries)} latest-version entries")

    pool = candidates(entries)
    print(f"  of those, {len(pool)} have a GitHub repo, a stdio package, and no obvious key need")

    add_stars(pool, _token())
    with_stars = [c for c in pool if "stars" in c]
    print(f"  resolved stars for {len(with_stars)}")

    chosen = rank(pool, args.limit, args.per_owner)
    servers = ([] if args.no_reference else list(REFERENCE_SERVERS)) + chosen

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": REGISTRY,
                "selection": (
                    "Official reference servers, which the registry does not list, plus the "
                    "most-starred registry entries that expose a launchable stdio package. "
                    "Stars are a rough popularity proxy; the registry offers no ordering. "
                    "Archived repositories are skipped and each GitHub owner is capped at "
                    f"{args.per_owner}."
                ),
                "servers": servers,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\npicked {len(servers)} servers -> {args.out}")
    for s in servers:
        stars = f"{s['stars']:>6}★" if s.get("stars") is not None else "   ref "
        print(f"  {stars}  {s['name'][:44]:<44} {s['cmd'][:40]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
