#!/usr/bin/env python3
"""Measure every captured snapshot with one fixed model and seed.

Results go to one JSON per server plus an index, which build_site.py turns into the leaderboard.
Reads snapshots only - nothing here touches a live server.

The run is resumable, because a free API tier cannot finish it in one day. A server with a
result file is skipped; every finished trial is appended to `.trials/<server>.jsonl` as it
lands, and replayed on the next run. When the provider's daily quota or `--budget-calls` runs
out, progress is saved and the script exits 0, so running the same command tomorrow continues.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

from mispick.generate import QueryCache, build_query_set
from mispick.metrics import compute, top_confused_pairs
from mispick.models.base import Backend, QuotaExhaustedError
from mispick.models.pacing import PacedBackend
from mispick.models.registry import get_backend
from mispick.report import json as json_report
from mispick.select import RunConfig, run_selection
from mispick.sources.snapshot import load_snapshot
from mispick.types import Choice


async def measure_one(
    snapshot: Path, args: argparse.Namespace, backend: Backend, out_dir: Path
) -> dict[str, Any]:
    tool_set = load_snapshot(snapshot)
    cache = QueryCache.open(out_dir / ".queries" / snapshot.stem)
    try:
        queries = await build_query_set(
            backend, tool_set, n=args.queries, cache=cache, seed=args.seed
        )
    finally:
        # Queries generated before a quota stop are kept, so tomorrow does not pay for them.
        cache.save(generator=backend.name)

    trials_path = out_dir / ".trials" / f"{snapshot.stem}.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    # Trials are keyed by query id, and ids survive regeneration while the text does not, so
    # saved trials only count against the exact query set they were measured on.
    stamp_path = trials_path.with_suffix(".queries")
    stamp = query_stamp(queries)
    if not stamp_path.is_file() or stamp_path.read_text() != stamp:
        trials_path.unlink(missing_ok=True)
        stamp_path.write_text(stamp)
    completed = load_trials(trials_path)

    with trials_path.open("a") as log:

        def record(choice: Choice) -> None:
            if not choice.error:
                log.write(choice.model_dump_json() + "\n")
                log.flush()

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
            completed=completed,
            on_progress=record,
        )

    metrics = compute(result)
    payload = json_report.build(result, metrics)
    payload["benchmark"] = {
        "snapshot": snapshot.name,
        "slug": snapshot.stem,
    }
    return payload


def query_stamp(queries: list[Any]) -> str:
    """A fingerprint of the query set that saved trials were measured against."""
    material = json.dumps([[q.id, q.text, q.expected] for q in queries], sort_keys=True)
    return hashlib.sha256(material.encode()).hexdigest()


def load_trials(path: Path) -> list[Choice]:
    """Trials an interrupted run already paid for. A torn last line is dropped."""
    if not path.is_file():
        return []
    trials = []
    for line in path.read_text().splitlines():
        try:
            trials.append(Choice.model_validate_json(line))
        except ValueError:
            continue
    return trials


def summarise(payload: dict[str, Any]) -> dict[str, Any]:
    """One leaderboard row from a server's result file."""
    from mispick.report.load import from_payload

    _, reloaded = from_payload(payload)
    pairs = top_confused_pairs(reloaded, limit=1)
    bench = payload["benchmark"]
    return {
        "slug": bench["slug"],
        "title": bench["title"],
        "registry_name": bench["registry_name"],
        "repository": bench["repository"],
        "tool_count": len(payload["tools"]),
        "score": payload["score"],
        "accuracy": payload["summary"]["accuracy"],
        "stability": payload["summary"]["stability"],
        "tokens": payload["summary"]["toolListTokensEstimate"],
        "errors": len(payload.get("errors") or []),
        "top_pair": (
            {"expected": pairs[0].expected, "chosen": pairs[0].chosen, "count": pairs[0].count}
            if pairs
            else None
        ),
    }


def token_estimate(tool_set: Any) -> int:
    """The tool-list token cost, which needs no model call at all."""
    from mispick.metrics import estimate_tool_tokens

    return estimate_tool_tokens(tool_set.sorted_tools())


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

    if args.only:
        unknown = set(args.only) - {p.stem for p in snapshots}
        if unknown:
            print(f"error: no snapshot named {', '.join(sorted(unknown))} in {args.snapshots}")
            return 2
        snapshots = [p for p in snapshots if p.stem in args.only]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    backend = PacedBackend(
        get_backend(args.model),
        rpm=args.rpm,
        tpm=args.tpm,
        max_calls=args.budget_calls,
    )
    summaries: list[dict[str, Any]] = []
    pending: list[str] = []
    stopped: str | None = None
    try:
        for i, snapshot in enumerate(snapshots, 1):
            print(f"[{i}/{len(snapshots)}] {snapshot.stem} … ", end="", flush=True)
            result_path = out_dir / f"{snapshot.stem}.json"

            # A server with an enormous tool surface can cost more model time than every
            # other server combined - one registry server ships 682 tools, which is 76% of
            # this corpus. Skip the measurement but keep the server on the leaderboard with
            # its tool count and token cost, both computed without calling a model. That
            # number is the interesting finding about such a server anyway.
            if args.max_tools:
                probe = load_snapshot(snapshot)
                if len(probe.tools) > args.max_tools:
                    meta = catalogue.get(snapshot.stem, {})
                    skipped.append(
                        {
                            "registry_name": meta.get("name", snapshot.stem),
                            "title": meta.get("title") or meta.get("name") or snapshot.stem,
                            "repository": meta.get("repository"),
                            "status": "too_many_tools",
                            "reason": (
                                f"{len(probe.tools)} tools, above the --max-tools limit of "
                                f"{args.max_tools}. Its tool list alone costs roughly "
                                f"{token_estimate(probe)} tokens on every request (estimate, "
                                "and a lower bound)."
                            ),
                            "tool_count": len(probe.tools),
                            "tokens": token_estimate(probe),
                        }
                    )
                    print(f"skipped ({len(probe.tools)} tools, over --max-tools)")
                    continue

            if result_path.is_file() and not args.rerun:
                payload = json.loads(result_path.read_text())
                summaries.append(summarise(payload))
                print(f"already measured, score {payload['score']}")
                continue

            if stopped:
                pending.append(snapshot.stem)
                print("pending")
                continue

            try:
                payload = await measure_one(snapshot, args, backend, out_dir)
            except QuotaExhaustedError as exc:
                stopped = str(exc)
                pending.append(snapshot.stem)
                print("paused (quota)")
                continue
            # Deliberately broad: one bad snapshot must not stop the sweep, and on a daily
            # schedule it must not block every server after it either.
            except Exception as exc:
                pending.append(snapshot.stem)
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
            result_path.write_text(json.dumps(payload, indent=2) + "\n")
            summaries.append(summarise(payload))
            print(f"score {payload['score']}")
    finally:
        await backend.aclose()

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
                "complete": not pending,
                "pending": pending,
                "servers": summaries,
                "skipped": skipped,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nmodel calls this run: {backend.calls}")
    print(f"measured {len(summaries)} servers -> {out_dir}")
    if stopped:
        print(f"\nStopped early: {stopped}")
        print(f"{len(pending)} server(s) still to measure. Run the same command again later.")
    elif pending:
        print(f"\n{len(pending)} server(s) failed; see above. Run again to retry them.")
    else:
        print("\nAll servers measured.")
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
    parser.add_argument(
        "--only",
        action="append",
        metavar="SLUG",
        help="Measure only this snapshot (file name without .json). Repeatable.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Measure servers again even if they have a result file. Saved trials still count.",
    )
    parser.add_argument("--rpm", type=int, default=0, help="Requests per minute cap. 0 = none.")
    parser.add_argument("--tpm", type=int, default=0, help="Tokens per minute cap. 0 = none.")
    parser.add_argument(
        "--budget-calls",
        type=int,
        default=0,
        help="Stop cleanly after this many model calls, e.g. to stay under a daily quota.",
    )
    parser.add_argument(
        "--max-tools",
        type=int,
        default=120,
        help=(
            "Skip measuring a server with more tools than this, reporting its tool count and "
            "token cost instead. 0 disables the limit."
        ),
    )
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
