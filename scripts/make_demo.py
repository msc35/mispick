#!/usr/bin/env python3
"""Render the terminal report to an SVG for the README.

A committed SVG rather than a GIF: it stays crisp at any width, needs no recorder in the
toolchain, and is generated from a real run rather than reconstructed by hand. Regenerate with:

    uv run python scripts/make_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from rich.console import Console

from mispick.generate import build_query_set
from mispick.metrics import compute
from mispick.models.registry import get_backend
from mispick.report import terminal
from mispick.select import RunConfig, run_selection
from mispick.sources.snapshot import load_snapshot


async def build(args: argparse.Namespace) -> None:
    tool_set = load_snapshot(args.snapshot)
    backend = get_backend(args.model)
    try:
        queries = await build_query_set(
            backend, tool_set, n=args.queries, cache=None, seed=args.seed
        )
        result = await run_selection(
            backend,
            tool_set,
            queries,
            config=RunConfig(
                n=args.queries, k=args.runs, temperature=args.temperature, seed=args.seed
            ),
        )
    finally:
        await backend.aclose()

    console = Console(record=True, width=100)
    terminal.render(result, compute(result), console)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    title = f"mispick run --cmd '...'   ({backend.name})"
    out.write_text(console.export_svg(title=title, font_aspect_ratio=0.61))
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="tests/fixtures/confusing_tools.json")
    parser.add_argument("--model", default="mock")
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="docs/media/demo.svg")
    asyncio.run(build(parser.parse_args()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
