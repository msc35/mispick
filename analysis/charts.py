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
    fig.suptitle(f"Pairs with the most similar descriptions were confused {ratio:.1f}× as often",
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
    args = parser.parse_args(argv)

    analysis = json.loads(args.analysis.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        effects_chart(analysis, theme, args.out / f"what-matters-{name}.png")
        similarity_chart(analysis, theme, args.out / f"similarity-{name}.png")
    print(f"wrote 4 figures -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
