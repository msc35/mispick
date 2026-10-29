#!/usr/bin/env python3
"""Plain-language figures for the write-up, drawn from results/analysis.json only.

analyze.py draws the technical charts. These are the reader-facing versions: plain labels,
one idea per chart, and a light and a dark variant (GitHub, and the blog).

    uv run python analysis/charts.py --analysis results/analysis.json --out results/figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

#: What each registered predictor means, said plainly. A is the tool the request was meant for,
#: B the tool that was picked instead.
PLAIN = {
    "name_jaccard": "The two names share words",
    "cos_sim": "The two descriptions say similar things",
    "when_a": "The right tool says when to use it",
    "when_b": "The wrong tool says when to use it",
    "log_len_a": "The right tool has a longer description",
    "log_len_b": "The wrong tool has a longer description",
    "log_len_diff": "The descriptions differ a lot in length",
    "param_doc_a": "The right tool documents its parameters",
    "param_doc_b": "The wrong tool documents its parameters",
}

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df",
              "series": "#2a78d6"},
    "dark": {"surface": "#101017", "text": "#f4f4f7", "muted": "#9696a4", "grid": "#26262f",
             "series": "#3ee6ff"},
}

#: One-hue sequential ramps, light to dark on the light surface and dark to bright on the dark one.
RAMPS = {
    "light": ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"],
    "dark": ["#101017", "#162238", "#184f95", "#2a78d6", "#6da7ec", "#cde2fb"],
}

#: Example matrices for the README. None of them is one of the four servers reserved for the
#: hand-written comparison, which stays blind until it is done (plan, Deviations).
MATRIX_SERVERS = [
    "ai-wavespeed-mcp",
    "io-github-modelcontextprotocol-server-memory",
    "com-1102tools-acquisition-gov-mcp",
]
PRIMARY_KINDS = {"straightforward", "paraphrased"}


def matrix_counts(payload: dict[str, Any]) -> tuple[list[str], list[str], list[list[int]]]:
    """Rows: the tool each primary request was meant for. Columns: what was picked."""
    kinds = {q["id"]: q.get("kind") for q in payload["queries"]}
    tools = [t["name"] for t in payload["tools"]]
    cols = [*tools, "(no tool)"]
    index = {name: i for i, name in enumerate(cols)}
    grid = [[0] * len(cols) for _ in tools]
    for trial in payload["trials"]:
        if trial.get("error") or kinds.get(trial["query_id"]) not in PRIMARY_KINDS:
            continue
        expected = trial.get("expected")
        if expected not in index:
            continue
        chosen = trial.get("chosen") or "(no tool)"
        grid[tools.index(expected)][index.get(chosen, index["(no tool)"])] += 1
    return tools, cols, grid


def matrix_chart(payload: dict[str, Any], title: str, theme_name: str, path: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    theme = THEMES[theme_name]
    rows, cols, grid = matrix_counts(payload)
    cmap = LinearSegmentedColormap.from_list("ramp", RAMPS[theme_name])
    shares = [[v / max(1, sum(row)) for v in row] for row in grid]
    total = sum(map(sum, grid))
    correct = sum(grid[i][i] for i in range(len(rows)))

    size = 0.62 * len(cols) + 3.4
    fig, ax = plt.subplots(figsize=(size, 0.55 * len(rows) + 2.6), facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])
    ax.imshow(shares, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, value in enumerate(row):
            if not value:
                continue
            r, g, b, _ = cmap(shares[i][j])
            ink = "#0b0b0b" if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.5 else "#f4f4f7"
            ax.text(j, i, str(value), ha="center", va="center", fontsize=8.5, color=ink,
                    fontweight="bold" if i != j else "normal")
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right", fontsize=8,
                  color=theme["text"])
    ax.set_yticks(range(len(rows)), rows, fontsize=8, color=theme["text"])
    ax.set_xticks([x - 0.5 for x in range(1, len(cols))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(rows))], minor=True)
    ax.grid(which="minor", color=theme["surface"], linewidth=2)
    ax.tick_params(which="both", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_xlabel("Tool the AI picked", color=theme["muted"], fontsize=9)
    ax.set_ylabel("Tool the request was meant for", color=theme["muted"], fontsize=9)
    share = correct / total if total else math.nan
    fig.suptitle(f"{title}: {share:.0%} of requests went to the right tool", x=0.02,
                 ha="left", fontsize=12, fontweight="bold", color=theme["text"])
    ax.set_title("Numbers are test requests. The diagonal is correct; anything off it is a "
                 "mix-up.", loc="left", fontsize=8.5, color=theme["muted"], pad=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=theme["surface"])
    plt.close(fig)


def headline(data: Path) -> dict[str, Any]:
    """Pooled accuracy and stability over every measured server, primary requests only."""
    correct = total = stable = repeated = servers = pairs = confused_pairs = 0
    for path in sorted(data.glob("*.json")):
        if path.name in {"index.json", "MANIFEST.json"}:
            continue
        payload = json.loads(path.read_text())
        rows, _, grid = matrix_counts(payload)
        requests = sum(map(sum, grid))
        servers += 1 if requests else 0
        total += requests
        correct += sum(grid[i][i] for i in range(len(grid)))
        pairs += len(rows) * (len(rows) - 1)
        confused_pairs += sum(
            1 for i in range(len(rows)) for j in range(len(rows)) if i != j and grid[i][j]
        )
        stability = (payload.get("summary") or {}).get("stability") or {}
        stable += int(stability.get("hits") or 0)
        repeated += int(stability.get("total") or 0)
    return {
        "servers_with_requests": servers,
        "requests": total,
        "tool_pairs": pairs,
        "tool_pairs_ever_confused": confused_pairs,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "stability": {"same_answer_all_runs": stable, "queries": repeated,
                      "share": stable / repeated if repeated else None},
        "note": "accuracy: primary requests (straightforward and paraphrased) on every measured "
                "server, errored trials excluded, so a server whose every call errored adds "
                "nothing. stability: queries whose K=3 runs all chose the same tool, from each "
                "server's own summary (all request kinds)",
    }


def robust(primary: dict[str, Any], bootstrap: dict[str, Any]) -> set[str]:
    """Predictors whose whole-server bootstrap CI (S1) also excludes 1, in the same direction."""
    by_name = {e["predictor"]: e for e in bootstrap.get("effects", [])}
    keep = set()
    for effect in primary["effects"]:
        b = by_name.get(effect["predictor"])
        if not b or b["lo"] is None or b["hi"] is None:
            continue
        same_side = (effect["or"] > 1) == (b["or"] > 1)
        if same_side and (b["lo"] > 1 or b["hi"] < 1):
            keep.add(effect["predictor"])
    return keep


def _style(ax: Any, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0)


def effects_chart(analysis: dict[str, Any], theme: dict[str, str], path: Path) -> None:
    import matplotlib.pyplot as plt

    effects = sorted(analysis["primary"]["effects"], key=lambda e: e["or"])
    strong = robust(analysis["primary"], analysis["sensitivity"]["S1_cluster_bootstrap"])
    fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=theme["surface"])
    _style(ax, theme)

    for y, e in enumerate(effects):
        ax.plot([e["lo"], e["hi"]], [y, y], color=theme["series"], linewidth=2,
                solid_capstyle="round")
        ax.plot(e["or"], y, "o", markersize=8, color=theme["series"],
                markeredgecolor=theme["surface"], markeredgewidth=2)
        change = (e["or"] - 1) * 100
        label = f"{change:+.0f}%" + ("  ✓" if e["predictor"] in strong else "")
        if e["or"] >= 1:
            ax.text(e["hi"] * 1.03, y, label, va="center", ha="left", fontsize=9,
                    color=theme["text"])
        else:
            ax.text(e["lo"] / 1.03, y, label, va="center", ha="right", fontsize=9,
                    color=theme["text"])

    ax.axvline(1.0, color=theme["muted"], linewidth=1, linestyle=(0, (3, 3)))
    ax.set_xscale("log")
    ticks = [0.6, 0.8, 1.0, 1.25, 1.5]
    ax.set_xticks(ticks, [f"{(t - 1) * 100:+.0f}%" if t != 1 else "no change" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.5, 1.95)
    ax.set_yticks(range(len(effects)), [PLAIN.get(e["predictor"], e["predictor"])
                                        for e in effects], color=theme["text"], fontsize=9.5)
    ax.set_xlabel("Change in the odds of the wrong tool being picked (per typical difference)",
                  color=theme["muted"], fontsize=9)
    model = analysis["benchmark"]["model"]
    counts = analysis["counts"]
    fig.suptitle("What goes with an AI picking the wrong tool", x=0.02, ha="left",
                 fontsize=13, fontweight="bold", color=theme["text"])
    ax.set_title(
        f"{counts['servers']} MCP servers, {counts['trials']:,} test requests, {model}.\n"
        "Lines show the 95% range.  ✓ = still clear in the strict check that resamples whole "
        "servers.",
        loc="left", fontsize=8.5, color=theme["muted"], pad=10)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=theme["surface"])
    plt.close(fig)


def similarity_chart(analysis: dict[str, Any], theme: dict[str, str], path: Path) -> None:
    import matplotlib.pyplot as plt

    bins = analysis["descriptive"]["confusion_by_similarity"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4), facecolor=theme["surface"])
    _style(ax, theme)
    xs = range(len(bins))
    rates = [b["rate"] * 100 for b in bins]
    lows = [(b["rate"] - b["lo"]) * 100 for b in bins]
    highs = [(b["hi"] - b["rate"]) * 100 for b in bins]
    ax.bar(xs, rates, width=0.62, color=theme["series"], edgecolor=theme["surface"],
           linewidth=2)
    ax.errorbar(xs, rates, yerr=[lows, highs], fmt="none", ecolor=theme["muted"],
                elinewidth=1.2, capsize=3)
    for x, b in zip(xs, bins, strict=True):
        ax.text(x, (b["hi"] * 100) + 0.04, f"{b['rate'] * 100:.2f}%", ha="center",
                fontsize=9.5, color=theme["text"])
    names = ["Least\nsimilar", "", "", "", "Most\nsimilar"]
    ax.set_xticks(list(xs), [n if n else f"{i + 1}" for i, n in enumerate(names)],
                  color=theme["text"], fontsize=9)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    ratio = bins[-1]["rate"] / bins[0]["rate"] if bins and bins[0]["rate"] else math.nan
    fig.suptitle(f"Pairs with the most similar descriptions were confused {ratio:.1f}× as often",  # noqa: RUF001
                 x=0.02, ha="left", fontsize=13, fontweight="bold", color=theme["text"])
    ax.set_title(
        "Share of test requests that went to the other tool of the pair, by how similar the two "
        "descriptions are (fifths of all pairs). Bars show the 95% range. A pattern, not proof "
        "of cause.", loc="left", fontsize=8.5, color=theme["muted"], pad=10, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, facecolor=theme["surface"])
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = ["DejaVu Sans"]
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--analysis", type=Path, default=Path("results/analysis.json"))
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    parser.add_argument("--data", type=Path, default=Path("analysis/data/benchmark-v1"))
    args = parser.parse_args(argv)

    analysis = json.loads(args.analysis.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        effects_chart(analysis, theme, args.out / f"what-matters-{name}.png")
        similarity_chart(analysis, theme, args.out / f"similarity-{name}.png")
    index = json.loads((args.data / "index.json").read_text())
    titles = {row["slug"]: row["title"] for row in index.get("servers") or []}
    for slug in MATRIX_SERVERS:
        payload = json.loads((args.data / f"{slug}.json").read_text())
        for name in THEMES:
            matrix_chart(payload, titles.get(slug, slug), name,
                         args.out / f"matrix-{slug}-{name}.png")
    summary = headline(args.data)
    (args.out.parent / "headline.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {4 + 2 * len(MATRIX_SERVERS)} figures -> {args.out}; headline -> "
          f"{args.out.parent / 'headline.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
