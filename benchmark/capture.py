#!/usr/bin/env python3
"""Capture each server's tools/list into a committed snapshot.

**Only `initialize` and `tools/list`.** This script imports mispick's own loader, which cannot
call a tool - see SPEC section 3 and tests/test_never_calls_tools.py.

Every server gets a verdict: captured, or skipped with the reason. A server that will not start
is a fact about running it, worth recording rather than hiding.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path
from typing import Any

from mispick.loader import write_snapshot
from mispick.sources.server import ServerError, load_stdio

#: Per-server budget. npx/uvx may need to download the package first.
TIMEOUT = 120.0

#: Signals in a failure that mean "this needs a credential", so it is skipped rather than failed.
KEY_HINTS = re.compile(
    r"(api[_ -]?key|token|unauthorized|401|403|credential|not authenticated|"
    r"missing environment|must be set)",
    re.IGNORECASE,
)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


async def capture_one(server: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": server["name"],
        "title": server.get("title"),
        "repository": server.get("repository"),
        "cmd": server["cmd"],
    }
    try:
        tool_set = await load_stdio(server["cmd"], label=server["name"], timeout=TIMEOUT)
    except ServerError as exc:
        message = str(exc)
        record["status"] = "needs_credentials" if KEY_HINTS.search(message) else "failed"
        # Keep the whole message, not just its first line: the diagnostic value is in the
        # unwrapped cause and the server's own stderr, both of which come after it.
        record["reason"] = " | ".join(
            line.strip() for line in message.splitlines() if line.strip()
        )[:600]
        return record
    # Deliberately broad: one bad server must not stop the sweep.
    except Exception as exc:
        record["status"] = "failed"
        record["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        return record

    if not tool_set.tools:
        record["status"] = "no_tools"
        record["reason"] = "the server started but exposed no tools"
        return record

    path = out_dir / f"{slug(server['name'])}.json"
    write_snapshot(tool_set, path)
    info = tool_set.servers[0] if tool_set.servers else None
    record.update(
        {
            "status": "captured",
            "snapshot": path.name,
            "tool_count": len(tool_set.tools),
            "server_name": info.name if info else None,
            "server_version": info.version if info else None,
            "protocol_version": info.protocol_version if info else None,
        }
    )
    return record


async def main_async(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.servers).read_text())
    servers = payload.get("servers") or []
    if args.only:
        servers = [s for s in servers if s["name"] in set(args.only)]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for i, server in enumerate(servers, 1):
        print(f"[{i}/{len(servers)}] {server['name']} … ", end="", flush=True)
        record = await capture_one(server, out_dir)
        print(record["status"] + (f" ({record.get('tool_count')} tools)"
                                  if record["status"] == "captured" else ""))
        records.append(record)

    index = out_dir / "index.json"
    index.write_text(json.dumps({"servers": records}, indent=2) + "\n")

    captured = sum(1 for r in records if r["status"] == "captured")
    print(f"\ncaptured {captured}/{len(records)} -> {out_dir}")
    for status in ("needs_credentials", "no_tools", "failed"):
        count = sum(1 for r in records if r["status"] == status)
        if count:
            print(f"  {status}: {count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servers", type=Path, default=Path("benchmark/servers.json"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/snapshots"))
    parser.add_argument("--only", nargs="*", help="Capture only these server names.")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
