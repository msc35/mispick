#!/usr/bin/env python3
"""RQ-c of docs/analysis-plan.md: do hand-written requests give the same results as generated?

    template  write analysis/human_queries/<server>.yaml for the four pre-registered servers:
              each tool's name and description, and empty slots to fill in. No generated
              query is ever written into a template.
    check     validate the hand-written files: at least 2 per tool and 20 per server, no
              blanks, no duplicates. No model call.
    run       measure the hand-written requests with the benchmark's model and settings.
              Refuses without --yes.
    analyze   accuracy difference (human minus generated, primary sets) with a bootstrap CI over
              queries within server, and the overlap of confused pairs.

    uv run python analysis/human_vs_generated.py template
    uv run python analysis/human_vs_generated.py check
    uv run python analysis/human_vs_generated.py run --yes
    uv run python analysis/human_vs_generated.py analyze
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from mispick.metrics import compute
from mispick.models.base import QuotaExhaustedError
from mispick.models.pacing import PacedBackend
from mispick.models.registry import get_backend
from mispick.report import json as json_report
from mispick.select import RunConfig, run_selection
from mispick.sources.snapshot import load_snapshot
from mispick.types import Choice, Query, ToolSet

ROOT = Path(__file__).resolve().parent.parent

#: Section 7 of the plan, chosen before any of their results were seen.
SERVERS = [
    "io-github-modelcontextprotocol-server-filesystem",
    "ai-adeu-adeu",
    "ai-mcpcap-mcpcap",
    "ai-thiri-chord-intelligence",
]
MIN_PER_TOOL = 2
MIN_PER_SERVER = 20
PRIMARY_KINDS = {"straightforward", "paraphrased"}
#: A hand-written request may be marked as a hard case. It is reported, but kept out of the
#: primary comparison, mirroring the generated side (plan, Deviations, 2026-10-16).
HUMAN_KINDS = {"straightforward", "paraphrased", "hard_negative", "no_tool"}

MODEL = "gemini/gemini-3.5-flash-lite"
K = 3
TEMPERATURE = 0.0
RUN_SEED = 1
SEED = 20261016
BOOTSTRAP = 2000
EQUIVALENCE_PP = 10.0
PAIR_MIN_TRIALS = 2


# --- template -----------------------------------------------------------------------------


def _indent(text: str, prefix: str) -> str:
    return "\n".join(f"{prefix}{line}".rstrip() for line in text.splitlines()) or prefix.rstrip()


def render_template(slug: str, tool_set: ToolSet) -> str:
    tools = tool_set.sorted_tools()
    lines = [
        f"# Hand-written requests for {slug}",
        "#",
        "# Write what a real user would type to an assistant that has these tools.",
        f"# - At least {MIN_PER_TOOL} per tool and {MIN_PER_SERVER} in total. Include some hard"
        " cases.",
        "# - Put each request under the tool that SHOULD be chosen.",
        "# - kind: straightforward (default), paraphrased (indirect wording), or hard_negative",
        "#   (sounds like a different tool, but this one is right). For a hard case, set",
        "#   `neighbour:` to the tool it sounds like.",
        "# - Give concrete details (names, paths, values). Never 'these' or 'that'.",
        "# - Do NOT look at generated queries or benchmark results for this server first.",
        "# - Requests no tool can answer go under no_tool at the bottom.",
        "#",
        "# Leave `fingerprint` alone: it ties these requests to this exact tool list.",
        "version: 1",
        f"server: {slug}",
        "tools:",
    ]
    for tool in tools:
        description = (tool.description or "(no description)").strip()
        lines += [
            f"  {tool.qualified_name}:",
            "    # What this tool says it does:",
            _indent(description, "    #   "),
            f"    fingerprint: {tool.fingerprint()}",
            "    queries:",
        ]
        for _ in range(MIN_PER_TOOL):
            lines += ['    - text: ""', "      kind: straightforward"]
    lines += [
        "no_tool:",
        '- text: ""',
        "",
    ]
    return "\n".join(lines)


def cmd_template(args: argparse.Namespace) -> int:
    args.human.mkdir(parents=True, exist_ok=True)
    for slug in args.servers:
        path = args.human / f"{slug}.yaml"
        if path.exists() and not args.force:
            print(f"{path}: exists, left alone (use --force to overwrite)")
            continue
        path.write_text(render_template(slug, load_snapshot(args.snapshots / f"{slug}.json")))
        print(f"{path}: written")
    return 0


# --- reading and checking -----------------------------------------------------------------


def read_human(path: Path, tool_set: ToolSet) -> tuple[list[Query], list[str]]:
    """The filled-in requests as mispick queries, plus every problem found."""
    data = yaml.safe_load(path.read_text()) or {}
    tools = {t.qualified_name: t for t in tool_set.sorted_tools()}
    problems: list[str] = []
    queries: list[Query] = []
    seen: set[str] = set()

    entries = data.get("tools") or {}
    for name in tools:
        if name not in entries:
            problems.append(f"{name}: missing from the file")
    for name, entry in entries.items():
        if name not in tools:
            problems.append(f"{name}: not a tool on this server")
            continue
        if (entry or {}).get("fingerprint") != tools[name].fingerprint():
            problems.append(f"{name}: fingerprint does not match the snapshot")
        written = 0
        for i, item in enumerate((entry or {}).get("queries") or [], 1):
            text = str((item or {}).get("text") or "").strip()
            kind = str((item or {}).get("kind") or "straightforward").strip()
            if not text:
                continue
            if kind not in HUMAN_KINDS - {"no_tool"}:
                problems.append(f"{name} #{i}: unknown kind {kind!r}")
                continue
            neighbour = (item or {}).get("neighbour")
            if kind == "hard_negative" and neighbour not in tools:
                problems.append(f"{name} #{i}: hard_negative needs `neighbour:` set to a tool")
            if text.lower() in seen:
                problems.append(f"{name} #{i}: duplicate request")
            seen.add(text.lower())
            written += 1
            queries.append(
                Query(id=f"human:{name}#{i}", text=text, expected=name, kind=kind,  # type: ignore[arg-type]
                      neighbour=neighbour if kind == "hard_negative" else None)
            )
        if written < MIN_PER_TOOL:
            problems.append(f"{name}: {written} requests (need {MIN_PER_TOOL})")

    for i, item in enumerate(data.get("no_tool") or [], 1):
        text = str((item or {}).get("text") or "").strip()
        if text:
            queries.append(Query(id=f"human:(none)#{i}", text=text, expected=None, kind="no_tool"))

    tool_queries = [q for q in queries if q.expected is not None]
    if len(tool_queries) < MIN_PER_SERVER:
        problems.append(f"{len(tool_queries)} tool requests in total (need {MIN_PER_SERVER})")
    return queries, problems


def cmd_check(args: argparse.Namespace) -> int:
    ready = 0
    for slug in args.servers:
        path = args.human / f"{slug}.yaml"
        if not path.is_file():
            print(f"{slug}: no file yet (run `template`)")
            continue
        queries, problems = read_human(path, load_snapshot(args.snapshots / f"{slug}.json"))
        if problems:
            print(f"{slug}: {len(problems)} problem(s)")
            for problem in problems:
                print(f"  - {problem}")
        else:
            ready += 1
            print(f"{slug}: ready ({len(queries)} requests)")
    print(f"\n{ready}/{len(args.servers)} servers ready")
    return 0 if ready == len(args.servers) else 1


# --- run ----------------------------------------------------------------------------------


def load_trials(path: Path) -> list[Choice]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        try:
            out.append(Choice.model_validate_json(line))
        except ValueError:
            continue
    return out


async def run_all(args: argparse.Namespace) -> int:
    out = args.out / "human"
    out.mkdir(parents=True, exist_ok=True)
    backend = PacedBackend(
        get_backend(args.model), rpm=args.rpm, tpm=args.tpm, max_calls=args.budget_calls
    )
    try:
        for slug in args.servers:
            result_path = out / f"{slug}.json"
            if result_path.is_file():
                print(f"{slug}: done")
                continue
            tool_set = load_snapshot(args.snapshots / f"{slug}.json")
            queries, problems = read_human(args.human / f"{slug}.yaml", tool_set)
            if problems:
                print(f"{slug}: not ready ({len(problems)} problems; run `check`)")
                continue
            log_path = out / f".{slug}.jsonl"
            with log_path.open("a") as log:

                def record(choice: Choice, log: Any = log) -> None:
                    if not choice.error:
                        log.write(choice.model_dump_json() + "\n")
                        log.flush()

                try:
                    result = await run_selection(
                        backend, tool_set, queries,
                        config=RunConfig(n=len(queries), k=K, temperature=TEMPERATURE,
                                         seed=RUN_SEED, concurrency=args.concurrency),
                        completed=load_trials(log_path),
                        on_progress=record,
                    )
                except QuotaExhaustedError as exc:
                    print(f"{slug}: paused ({exc}). Run again to continue.")
                    break
            payload = json_report.build(result, compute(result))
            payload["benchmark"] = {"slug": slug, "queries": "human"}
            result_path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"{slug}: score {payload['score']}")
    finally:
        await backend.aclose()
    print(f"model calls this run: {backend.calls}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.yes:
        print("error: this costs model calls. Run `check`, get approval, then pass --yes.")
        return 2
    return asyncio.run(run_all(args))


# --- analyze ------------------------------------------------------------------------------


def query_outcomes(payload: dict[str, Any], kinds: set[str]) -> dict[str, list[bool]]:
    """Per query, whether each valid trial chose the expected tool."""
    kind_of = {q["id"]: q.get("kind") for q in payload["queries"]}
    out: dict[str, list[bool]] = {}
    for trial in payload["trials"]:
        if trial.get("error") or kind_of.get(trial["query_id"]) not in kinds:
            continue
        out.setdefault(trial["query_id"], []).append(trial.get("chosen") == trial.get("expected"))
    return out


def accuracy(groups: dict[str, dict[str, list[bool]]]) -> float:
    flat = [x for per_server in groups.values() for trials in per_server.values() for x in trials]
    return sum(flat) / len(flat) if flat else float("nan")


def bootstrap_difference(
    human: dict[str, dict[str, list[bool]]], generated: dict[str, dict[str, list[bool]]]
) -> dict[str, Any]:
    """Human minus generated accuracy, resampling queries within each server and set."""
    import numpy as np

    rng = np.random.default_rng(SEED)

    def resample(groups: dict[str, dict[str, list[bool]]]) -> dict[str, dict[str, list[bool]]]:
        out = {}
        for server, per_query in groups.items():
            keys = list(per_query)
            picks = rng.choice(len(keys), size=len(keys), replace=True)
            out[server] = {f"{i}": per_query[keys[j]] for i, j in enumerate(picks)}
        return out

    estimate = (accuracy(human) - accuracy(generated)) * 100
    draws = [(accuracy(resample(human)) - accuracy(resample(generated))) * 100
             for _ in range(BOOTSTRAP)]
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    if lo >= -EQUIVALENCE_PP and hi <= EQUIVALENCE_PP:
        verdict = "equivalent (95% CI within +/-10 points)"
    elif lo > 0 or hi < 0:
        verdict = "different (95% CI excludes 0)"
    else:
        verdict = "inconclusive"
    return {"difference_pp": estimate, "lo": lo, "hi": hi, "resamples": BOOTSTRAP,
            "seed": SEED, "verdict": verdict, "rule": "plan section 9"}


def confused_pairs(payload: dict[str, Any], kinds: set[str]) -> set[tuple[str, str]]:
    """Ordered pairs (expected, chosen) with at least PAIR_MIN_TRIALS confusing trials."""
    kind_of = {q["id"]: q.get("kind") for q in payload["queries"]}
    counts: dict[tuple[str, str], int] = {}
    for trial in payload["trials"]:
        expected, chosen = trial.get("expected"), trial.get("chosen")
        if trial.get("error") or kind_of.get(trial["query_id"]) not in kinds:
            continue
        if expected and chosen and chosen != expected:
            counts[(expected, chosen)] = counts.get((expected, chosen), 0) + 1
    return {pair for pair, n in counts.items() if n >= PAIR_MIN_TRIALS}


def jaccard(a: set[Any], b: set[Any]) -> float | None:
    union = a | b
    return len(a & b) / len(union) if union else None


def cmd_analyze(args: argparse.Namespace) -> int:
    human_dir = args.out / "human"
    human_payloads, generated_payloads = {}, {}
    for slug in args.servers:
        h, g = human_dir / f"{slug}.json", args.data / f"{slug}.json"
        if h.is_file() and g.is_file():
            human_payloads[slug] = json.loads(h.read_text())
            generated_payloads[slug] = json.loads(g.read_text())
    if not human_payloads:
        print("no server has both a human run and a benchmark result yet")
        return 1

    human = {s: query_outcomes(p, PRIMARY_KINDS) for s, p in human_payloads.items()}
    generated = {s: query_outcomes(p, PRIMARY_KINDS) for s, p in generated_payloads.items()}
    hard = {s: query_outcomes(p, {"hard_negative"}) for s, p in human_payloads.items()}

    pairs = {}
    all_h: set[tuple[str, str, str]] = set()
    all_g: set[tuple[str, str, str]] = set()
    for slug in human_payloads:
        hp = confused_pairs(human_payloads[slug], PRIMARY_KINDS)
        gp = confused_pairs(generated_payloads[slug], PRIMARY_KINDS)
        all_h |= {(slug, *p) for p in hp}
        all_g |= {(slug, *p) for p in gp}
        pairs[slug] = {"human": sorted(map(list, hp)), "generated": sorted(map(list, gp)),
                       "shared": sorted(map(list, hp & gp)), "jaccard": jaccard(hp, gp)}

    report = {
        "generated": dt.date.today().isoformat(),
        "plan": "docs/analysis-plan.md, section 7",
        "model": next(iter(human_payloads.values()))["run"]["model"],
        "servers": sorted(human_payloads),
        "complete": len(human_payloads) == len(args.servers),
        "accuracy": {
            "human_primary": accuracy(human),
            "generated_primary": accuracy(generated),
            "human_hard_cases": accuracy(hard) if any(hard.values()) else None,
        },
        "difference": bootstrap_difference(human, generated),
        "confused_pairs": {"per_server": pairs, "pooled_jaccard": jaccard(all_h, all_g),
                           "min_trials": PAIR_MIN_TRIALS},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "human_vs_generated.json").write_text(json.dumps(report, indent=2) + "\n")
    d = report["difference"]
    flag = "" if report["complete"] else " (INCOMPLETE)"
    if math.isfinite(d["difference_pp"]):
        print(f"human - generated: {d['difference_pp']:+.1f} pp [{d['lo']:+.1f}, {d['hi']:+.1f}]"
              f" -> {d['verdict']}{flag}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--human", type=Path, default=ROOT / "analysis/human_queries")
    parser.add_argument("--snapshots", type=Path, default=ROOT / "benchmark/snapshots")
    parser.add_argument("--data", type=Path, default=ROOT / "analysis/data/benchmark-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument(
        "--servers",
        type=lambda v: [s for s in v.split(",") if s],
        default=SERVERS,
        help="Comma-separated snapshot names. Defaults to the four registered servers.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    template = sub.add_parser("template", help="write empty templates to fill in by hand")
    template.add_argument("--force", action="store_true")
    sub.add_parser("check", help="validate the hand-written files")
    run = sub.add_parser("run", help="measure the hand-written requests")
    run.add_argument("--model", default=MODEL)
    run.add_argument("--yes", action="store_true")
    run.add_argument("--rpm", type=int, default=14)
    run.add_argument("--tpm", type=int, default=230000)
    run.add_argument("--budget-calls", type=int, default=0)
    run.add_argument("--concurrency", type=int, default=2)
    sub.add_parser("analyze", help="compare against the benchmark's generated requests")
    args = parser.parse_args(argv)
    handler = {"template": cmd_template, "check": cmd_check, "run": cmd_run,
               "analyze": cmd_analyze}[args.command]
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
