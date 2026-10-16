#!/usr/bin/env python3
"""RQ-b of docs/analysis-plan.md: does confusion on the benchmark model transfer to others?

Three steps, each its own command:

    plan     pick the stratified server subset (section 6) and print the input-token estimate
             per model. Makes no model call. Run this first and get approval.
    run      re-run the subset on one model, with the same snapshots and the frozen queries.
             Refuses without --yes. Never generates a query: a server whose cached queries are
             incomplete is refused instead, so the only thing that changes is the model.
    analyze  Spearman correlation of pairwise confusion rates with a bootstrap CI over
             servers, top-k agreement, and per-server McNemar tests with BH (family F2).

    uv run python analysis/transfer.py plan --model gemini/gemini-3.5-flash
    uv run python analysis/transfer.py run --model gemini/gemini-3.5-flash --yes
    uv run python analysis/transfer.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mispick.generate import QueryCache
from mispick.metrics import compute, estimate_tool_tokens
from mispick.models.base import QuotaExhaustedError
from mispick.models.ollama import SYSTEM_PROMPT
from mispick.models.pacing import PacedBackend
from mispick.models.registry import get_backend
from mispick.report import json as json_report
from mispick.select import RunConfig, run_selection
from mispick.sources.snapshot import load_snapshot
from mispick.types import Choice, Query

ROOT = Path(__file__).resolve().parent.parent

#: Section 6 of the plan.
PER_TERTILE = 4
MAX_TOOLS = 40
SEED = 20261016
TOP_K = 3
BOOTSTRAP = 2000
PRIMARY_KINDS = {"straightforward", "paraphrased"}

#: Same settings as the benchmark run.
K = 3
TEMPERATURE = 0.0
RUN_SEED = 1

#: Per-call overhead beyond the tool list and the query, matching PacedBackend's estimate.
CALL_OVERHEAD = 100


def model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", model).strip("-")


# --- plan ---------------------------------------------------------------------------------


def select_subset(index: dict[str, Any], snapshots: Path) -> dict[str, Any]:
    """Tertiles of benchmark score among eligible servers, then PER_TERTILE drawn per tertile."""
    import numpy as np

    eligible = []
    for row in index.get("servers") or []:
        path = snapshots / f"{row['slug']}.json"
        if not path.is_file():
            continue
        tools = len(load_snapshot(path).tools)
        if 2 <= tools <= MAX_TOOLS:
            eligible.append({"slug": row["slug"], "score": row["score"], "tools": tools})
    if len(eligible) < 3:
        raise SystemExit(f"error: only {len(eligible)} eligible servers; need at least 3")

    ordered = sorted(eligible, key=lambda r: (r["score"], r["slug"]))
    rng = np.random.default_rng(SEED)
    tertiles = []
    for number, group in enumerate(np.array_split(np.arange(len(ordered)), 3), 1):
        members = [ordered[i] for i in group]
        take = min(PER_TERTILE, len(members))
        picked = sorted(rng.choice(len(members), size=take, replace=False).tolist())
        tertiles.append(
            {
                "tertile": number,
                "scores": [members[0]["score"], members[-1]["score"]],
                "eligible": [m["slug"] for m in members],
                "picked": [members[i]["slug"] for i in picked],
            }
        )
    return {
        "rule": f"score tertiles, {PER_TERTILE} per tertile, <= {MAX_TOOLS} tools, seed {SEED}",
        "eligible": len(eligible),
        "tertiles": tertiles,
        "servers": [s for t in tertiles for s in t["picked"]],
    }


def frozen_queries(data: Path, slug: str, snapshots: Path) -> list[Query]:
    """The exact queries the benchmark used. Refuses if any tool lacks them."""
    tool_set = load_snapshot(snapshots / f"{slug}.json")
    cache = QueryCache.open(data / ".queries" / slug)
    queries: list[Query] = []
    missing = []
    for tool in tool_set.sorted_tools():
        cached = cache.fresh_for(tool)
        if not cached:
            missing.append(tool.qualified_name)
        else:
            queries.extend(cached)
    if missing:
        raise SystemExit(
            f"error: {slug} has no frozen queries for {', '.join(missing)}. Re-running it would "
            "generate new queries with a different model, so it cannot be compared."
        )
    return queries + cache.no_tool()


def estimate(data: Path, snapshots: Path, servers: list[str]) -> dict[str, Any]:
    """Input tokens for one model over the subset, using mispick's own (lower-bound) estimate."""
    rows = []
    for slug in servers:
        tools = load_snapshot(snapshots / f"{slug}.json").sorted_tools()
        queries = frozen_queries(data, slug, snapshots)
        per_tools = estimate_tool_tokens(tools)
        prompt = len(SYSTEM_PROMPT) // 4 + CALL_OVERHEAD
        tokens = sum(K * (per_tools + len(q.text) // 4 + prompt) for q in queries)
        rows.append(
            {"server": slug, "tools": len(tools), "queries": len(queries),
             "calls": len(queries) * K, "tool_list_tokens": per_tools, "input_tokens": tokens}
        )
    return {
        "servers": rows,
        "calls": sum(r["calls"] for r in rows),
        "input_tokens": sum(r["input_tokens"] for r in rows),
        "note": "characters/4 estimate; a lower bound. Output is ~10 tokens per call.",
    }


def cmd_plan(args: argparse.Namespace) -> int:
    index = json.loads((args.data / "index.json").read_text())
    if not index.get("complete"):
        print("warning: the frozen benchmark is not complete; the subset may change.")
    subset = select_subset(index, args.snapshots)
    cost = estimate(args.data, args.snapshots, subset["servers"])
    plan = {
        "generated": dt.date.today().isoformat(),
        "benchmark_model": index.get("model"),
        "subset": subset,
        "per_model": {m: cost for m in args.model},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transfer_plan.json").write_text(json.dumps(plan, indent=2) + "\n")

    for tertile in subset["tertiles"]:
        print(f"tertile {tertile['tertile']} (scores {tertile['scores'][0]}-"
              f"{tertile['scores'][1]}): {', '.join(tertile['picked'])}")
    print(f"\n{cost['calls']:,} calls and ~{cost['input_tokens']:,} input tokens per model "
          f"({cost['note']})")
    for model in args.model:
        print(f"  {model}: {cost['calls']:,} calls, ~{cost['input_tokens']:,} input tokens")
    print(f"\nwrote {args.out / 'transfer_plan.json'}. Nothing was run.")
    return 0


# --- run ----------------------------------------------------------------------------------


def load_trials(path: Path) -> list[Choice]:
    if not path.is_file():
        return []
    trials = []
    for line in path.read_text().splitlines():
        try:
            trials.append(Choice.model_validate_json(line))
        except ValueError:
            continue
    return trials


async def run_model(args: argparse.Namespace) -> int:
    plan = json.loads((args.out / "transfer_plan.json").read_text())
    servers = plan["subset"]["servers"]
    out = args.out / "transfer" / model_slug(args.model)
    out.mkdir(parents=True, exist_ok=True)
    backend = PacedBackend(
        get_backend(args.model), rpm=args.rpm, tpm=args.tpm, max_calls=args.budget_calls
    )
    stopped = None
    try:
        for slug in servers:
            result_path = out / f"{slug}.json"
            if result_path.is_file():
                print(f"{slug}: done")
                continue
            if stopped:
                print(f"{slug}: pending")
                continue
            tool_set = load_snapshot(args.snapshots / f"{slug}.json")
            queries = frozen_queries(args.data, slug, args.snapshots)
            log_path = out / f".{slug}.jsonl"
            with log_path.open("a") as log:

                def record(choice: Choice, log: Any = log) -> None:
                    if not choice.error:
                        log.write(choice.model_dump_json() + "\n")
                        log.flush()

                try:
                    result = await run_selection(
                        backend, tool_set, queries,
                        config=RunConfig(n=8, k=K, temperature=TEMPERATURE, seed=RUN_SEED,
                                         concurrency=args.concurrency),
                        completed=load_trials(log_path),
                        on_progress=record,
                    )
                except QuotaExhaustedError as exc:
                    stopped = str(exc)
                    print(f"{slug}: paused ({exc})")
                    continue
            payload = json_report.build(result, compute(result))
            payload["benchmark"] = {"slug": slug}
            result_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"{slug}: score {payload['score']}")
    finally:
        await backend.aclose()
    print(f"\nmodel calls this run: {backend.calls}")
    if stopped:
        print("Stopped early. Run the same command again to continue.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.yes:
        print("error: re-running costs model calls. Check `transfer.py plan` first, then pass "
              "--yes.")
        return 2
    return asyncio.run(run_model(args))


# --- analyze ------------------------------------------------------------------------------


def pair_rates(payload: dict[str, Any]) -> dict[tuple[str, str], float]:
    """Pairwise confusion rate for every ordered pair of distinct tools, primary trials only."""
    kinds = {q["id"]: q.get("kind") for q in payload["queries"]}
    tools = [t["name"] for t in payload["tools"]]
    totals: dict[str, int] = dict.fromkeys(tools, 0)
    hits: dict[tuple[str, str], int] = {}
    for trial in payload["trials"]:
        expected = trial.get("expected")
        if trial.get("error") or kinds.get(trial["query_id"]) not in PRIMARY_KINDS:
            continue
        if expected not in totals:
            continue
        totals[expected] += 1
        chosen = trial.get("chosen")
        if chosen in totals and chosen != expected:
            hits[(expected, chosen)] = hits.get((expected, chosen), 0) + 1
    return {
        (a, b): hits.get((a, b), 0) / totals[a]
        for a in tools for b in tools
        if a != b and totals[a] > 0
    }


def paired_accuracy(base: dict[str, Any], other: dict[str, Any]) -> tuple[int, int, int]:
    """(both-trial count, improved, regressed) over trials present in both, by (query, run)."""
    def outcomes(payload: dict[str, Any]) -> dict[tuple[str, int], bool]:
        return {
            (t["query_id"], t.get("run", 0)): t.get("chosen") == t.get("expected")
            for t in payload["trials"]
            if not t.get("error")
        }

    left, right = outcomes(base), outcomes(other)
    shared = set(left) & set(right)
    improved = sum(1 for k in shared if not left[k] and right[k])
    regressed = sum(1 for k in shared if left[k] and not right[k])
    return len(shared), improved, regressed


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from scipy.stats import spearmanr

    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def correlate(
    base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]], nonzero: bool
) -> dict[str, Any]:
    """Spearman over pooled pairs, with a percentile bootstrap over servers."""
    import numpy as np

    per_server: dict[str, list[tuple[float, float]]] = {}
    for slug in sorted(set(base) & set(other)):
        left, right = pair_rates(base[slug]), pair_rates(other[slug])
        points = [(left[p], right[p]) for p in sorted(set(left) & set(right))]
        if nonzero:
            points = [pt for pt in points if pt[0] > 0 or pt[1] > 0]
        per_server[slug] = points

    def rho(slugs: Sequence[str]) -> float:
        pts = [pt for s in slugs for pt in per_server[s]]
        return spearman([p[0] for p in pts], [p[1] for p in pts])

    slugs = sorted(per_server)
    estimate = rho(slugs)
    rng = np.random.default_rng(SEED)
    draws = [rho(list(rng.choice(slugs, size=len(slugs), replace=True))) for _ in range(BOOTSTRAP)]
    finite = [d for d in draws if math.isfinite(d)]
    lo, hi = (
        (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))
        if finite else (float("nan"), float("nan"))
    )
    return {
        "rho": estimate, "lo": lo, "hi": hi,
        "pairs": sum(len(v) for v in per_server.values()),
        "servers": len(slugs),
        "resamples": BOOTSTRAP,
        "undefined_resamples": len(draws) - len(finite),
    }


def top_k_agreement(base: dict[str, dict[str, Any]], other: dict[str, dict[str, Any]]) -> dict:
    overlaps, top1 = [], []
    for slug in sorted(set(base) & set(other)):
        left, right = pair_rates(base[slug]), pair_rates(other[slug])

        def top(rates: dict[tuple[str, str], float]) -> list[tuple[str, str]]:
            ranked = sorted(
                ((r, p) for p, r in rates.items() if r > 0), key=lambda x: (-x[0], x[1])
            )
            return [p for _, p in ranked[:TOP_K]]

        a, b = top(left), top(right)
        if not a and not b:
            continue
        overlaps.append(len(set(a) & set(b)) / TOP_K)
        top1.append(bool(a and b and a[0] == b[0]))
    return {
        "k": TOP_K,
        "servers_with_confusion": len(overlaps),
        "mean_overlap": sum(overlaps) / len(overlaps) if overlaps else None,
        "top1_match_share": sum(top1) / len(top1) if top1 else None,
    }


def interpret(result: dict[str, Any]) -> str:
    """Section 9 of the plan, applied mechanically."""
    lo, hi, rho = result["lo"], result["hi"], result["rho"]
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "not estimable"
    if lo <= 0 <= hi:
        return "no evidence of transfer (95% CI includes 0)"
    band = "weak" if hi < 0.4 else "strong" if lo > 0.7 else "moderate" if lo >= 0.4 and hi <= 0.7 \
        else "spans bands; read the CI"
    return f"transfer, rho {rho:.2f} [{lo:.2f}, {hi:.2f}]: {band}"


def load_dir(directory: Path, servers: list[str]) -> dict[str, dict[str, Any]]:
    return {
        s: json.loads((directory / f"{s}.json").read_text())
        for s in servers
        if (directory / f"{s}.json").is_file()
    }


def cmd_analyze(args: argparse.Namespace) -> int:
    from statsmodels.stats.multitest import multipletests

    from mispick.fix import mcnemar_exact

    plan = json.loads((args.out / "transfer_plan.json").read_text())
    servers = plan["subset"]["servers"]
    base = load_dir(args.data, servers)
    report: dict[str, Any] = {
        "generated": dt.date.today().isoformat(),
        "plan": "docs/analysis-plan.md, section 6",
        "benchmark_model": plan.get("benchmark_model"),
        "servers": servers,
        "models": {},
    }
    root = args.out / "transfer"
    for directory in sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []:
        other = load_dir(directory, servers)
        if not other:
            continue
        missing = sorted(set(servers) - set(other))
        model = next(iter(other.values()))["run"]["model"]
        tests = []
        for slug in sorted(set(base) & set(other)):
            shared, improved, regressed = paired_accuracy(base[slug], other[slug])
            tests.append({"server": slug, "paired_trials": shared, "other_better": improved,
                          "other_worse": regressed, "p": mcnemar_exact(improved, regressed)})
        if tests:
            _, q, _, _ = multipletests([t["p"] for t in tests], method="fdr_bh")
            for test, value in zip(tests, q, strict=True):
                test["q"] = float(value)
        primary = correlate(base, other, nonzero=False)
        report["models"][model] = {
            "complete": not missing,
            "missing": missing,
            "spearman_all_pairs": primary,
            "spearman_nonzero_pairs": correlate(base, other, nonzero=True),
            "verdict": interpret(primary),
            "top_k": top_k_agreement(base, other),
            "mcnemar_f2": tests,
        }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "transfer.json").write_text(json.dumps(report, indent=2) + "\n")
    for model, entry in report["models"].items():
        flag = "" if entry["complete"] else f" (INCOMPLETE: {len(entry['missing'])} missing)"
        print(f"{model}{flag}: {entry['verdict']}")
    if not report["models"]:
        print("no transfer runs found yet")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=ROOT / "analysis/data/benchmark-v1")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "benchmark/snapshots")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="pick the subset and estimate tokens; no model calls")
    plan.add_argument("--model", action="append", required=True)

    run = sub.add_parser("run", help="re-run the subset on one model")
    run.add_argument("--model", required=True)
    run.add_argument("--yes", action="store_true", help="confirm the calls were approved")
    run.add_argument("--rpm", type=int, default=0)
    run.add_argument("--tpm", type=int, default=0)
    run.add_argument("--budget-calls", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=2)

    sub.add_parser("analyze", help="compare every finished run against the benchmark")

    args = parser.parse_args(argv)
    handler = {"plan": cmd_plan, "run": cmd_run, "analyze": cmd_analyze}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
