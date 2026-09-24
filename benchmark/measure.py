#!/usr/bin/env python3
"""Measure every captured snapshot with one fixed model and seed.

Results go to one JSON per server plus an index, which build_site.py turns into the leaderboard.
Reads snapshots only - nothing here touches a live server.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path
from typing import Any

from mispick.generate import QueryCache, build_query_set
from mispick.metrics import compute, top_confused_pairs
from mispick.models.base import BackendError
from mispick.models.registry import get_backend
from mispick.report import json as json_report
from mispick.select import RunConfig, run_selection
from mispick.sources.snapshot import load_snapshot


async def measure_one(
    snapshot: Path, args: argparse.Namespace, cache_root: Path
) -> dict[str, Any]:
    tool_set = load_snapshot(snapshot)
    backend = get_backend(args.model)
    try:
        cache = QueryCache.open(cache_root / snapshot.stem)
        queries = await build_query_set(
            backend, tool_set, n=args.queries, cache=cache, seed=args.seed
        )
        cache.save(generator=backend.name)
        result = await run_selection(
            backend,
            tool_set,
            queries,
            config=RunConfig(
                n=args.queries,
                k=args.runs,
                temperature=args.temperature,
                seed=args.seed,
                concurrency=args.concurrency,
            ),
        )
    finally:
        await backend.aclose()

    metrics = compute(result)
    payload = json_report.build(result, metrics)
    payload["benchmark"] = {
        "snapshot": snapshot.name,
        "slug": snapshot.stem,
    }
    return payload


async def main_async(args: argparse.Namespace) -> int:
    snapshots = sorted(p for p in Path(args.snapshots).glob("*.json") if p.name != "index.json")
    if not snapshots:
        print(f"error: no snapshots in {args.snapshots}. Run capture.py first.")
        return 2

    index_path = Path(args.snapshots) / "index.json"
    catalogue: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    if index_path.is_file():
        for record in (json.loads(index_path.read_text()).get("servers") or []):
            if record.get("snapshot"):
                catalogue[Path(record["snapshot"]).stem] = record
            else:
                # Carried into the results so the leaderboard can say what it did not cover.
                # A benchmark that silently drops the servers it could not start overstates
                # its own coverage.
                skipped.append(
                    {
                        "registry_name": record.get("name"),
                        "title": record.get("title") or record.get("name"),
                        "repository": record.get("repository"),
                        "status": record.get("status", "failed"),
                        "reason": record.get("reason"),
                    }
                )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root = out_dir / ".queries"

    summaries: list[dict[str, Any]] = []
    for i, snapshot in enumerate(snapshots, 1):
        print(f"[{i}/{len(snapshots)}] {snapshot.stem} … ", end="", flush=True)
        try:
            payload = await measure_one(snapshot, args, cache_root)
        except BackendError as exc:
            print(f"model error: {exc}")
            return 2
        # Deliberately broad: one bad snapshot must not stop the sweep.
        except Exception as exc:
            print(f"failed: {type(exc).__name__}: {exc}")
            continue

        meta = catalogue.get(snapshot.stem, {})
        payload["benchmark"].update(
            {
                "registry_name": meta.get("name", snapshot.stem),
                "title": meta.get("title") or meta.get("name") or snapshot.stem,
                "repository": meta.get("repository"),
            }
        )
        (out_dir / f"{snapshot.stem}.json").write_text(json.dumps(payload, indent=2) + "\n")

        from mispick.report.load import from_payload

        _, reloaded = from_payload(payload)
        pairs = top_confused_pairs(reloaded, limit=1)
        summaries.append(
            {
                "slug": snapshot.stem,
                "title": payload["benchmark"]["title"],
                "registry_name": payload["benchmark"]["registry_name"],
                "repository": payload["benchmark"]["repository"],
                "tool_count": len(payload["tools"]),
                "score": payload["score"],
                "accuracy": payload["summary"]["accuracy"],
                "stability": payload["summary"]["stability"],
                "tokens": payload["summary"]["toolListTokensEstimate"],
                "top_pair": (
                    {"expected": pairs[0].expected, "chosen": pairs[0].chosen,
                     "count": pairs[0].count}
                    if pairs
                    else None
                ),
            }
        )
        print(f"score {payload['score']}")

    summaries.sort(key=lambda s: (-s["score"], s["title"]))
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "generated": dt.date.today().isoformat(),
                "model": args.model,
                "n": args.queries,
                "k": args.runs,
                "temperature": args.temperature,
                "seed": args.seed,
                "servers": summaries,
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nmeasured {len(summaries)} servers -> {out_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path, default=Path("benchmark/snapshots"))
    parser.add_argument("--out", type=Path, default=Path("benchmark/results"))
    parser.add_argument("--model", default="ollama/qwen3.5:4b")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
