#!/usr/bin/env python3
"""Fit the pre-registered RQ-a model of docs/analysis-plan.md and write results/analysis.json.

Inputs: the pair table from features.py, and the benchmark's per-server result files (trial
level). Applies the plan's exclusions, fits the mixed-effects logistic model and the three
sensitivity checks, applies Benjamini-Hochberg, and draws the charts.

Every verdict printed here is computed mechanically from the rules in the plan, section 9.

    uv sync --all-extras --group analysis
    uv run python analysis/analyze.py --features results/features.csv \
        --results analysis/data/benchmark-v1 --out results
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

#: Section 5 of the plan. The covariate is adjusted for but not tested.
PREDICTORS = [
    "cos_sim",
    "name_jaccard",
    "log_len_a",
    "log_len_b",
    "log_len_diff",
    "param_doc_a",
    "param_doc_b",
    "when_a",
    "when_b",
]
CONFIRMATORY = ["cos_sim", "name_jaccard"]
COVARIATE = "log_n_tools"

#: Section 4 of the plan.
PRIMARY_KINDS = {"straightforward", "paraphrased"}
SENSITIVITY_KINDS = PRIMARY_KINDS | {"hard_negative"}
MIN_VALID_TRIALS = 12
MAX_SERVER_ERROR_SHARE = 0.20

ALPHA = 0.05
Z95 = 1.959963984540054
BOOTSTRAP_SEED = 20261016


# --- loading ------------------------------------------------------------------------------


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_results(directory: Path) -> dict[str, dict[str, Any]]:
    """Per-server result files, keyed by snapshot slug."""
    results = {}
    for path in sorted(directory.glob("*.json")):
        if path.name == "index.json":
            continue
        payload = json.loads(path.read_text())
        if "trials" not in payload or "queries" not in payload:
            continue
        slug = (payload.get("benchmark") or {}).get("slug") or path.stem
        payload["_path"] = str(path)
        payload["_sha256"] = sha256(path)
        results[slug] = payload
    return results


def trial_frame(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """One row per trial, with the kind of query it came from."""
    records = []
    for server, payload in results.items():
        kinds = {q["id"]: q.get("kind") for q in payload["queries"]}
        for trial in payload["trials"]:
            records.append(
                {
                    "server": server,
                    "query_id": trial["query_id"],
                    "run": trial.get("run", 0),
                    "kind": kinds.get(trial["query_id"]),
                    "expected": trial.get("expected"),
                    "chosen": trial.get("chosen"),
                    "error": bool(trial.get("error")),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["server", "query_id", "run", "kind", "expected", "chosen", "error"],
    )


# --- exclusions ---------------------------------------------------------------------------


def apply_exclusions(
    features: pd.DataFrame, trials: pd.DataFrame, kinds: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Section 4 of the plan. Returns the kept pair table, kept trials and a full account."""
    report: dict[str, Any] = {"servers": {}, "tools": {}, "trials": {}}
    pairs = features[features["same_server"] == 1].copy()

    measured = set(trials["server"].unique())
    for server in sorted(set(pairs["server"].unique()) - measured):
        report["servers"][server] = "no benchmark result"
    pairs = pairs[pairs["server"].isin(measured)]

    for server, group in trials.groupby("server"):
        share = float(group["error"].mean()) if len(group) else 0.0
        if share > MAX_SERVER_ERROR_SHARE:
            report["servers"][server] = f"{share:.0%} of trials errored (limit 20%)"
    bad_servers = set(report["servers"])
    pairs = pairs[~pairs["server"].isin(bad_servers)]
    trials = trials[~trials["server"].isin(bad_servers)]

    report["trials"]["errored"] = int(trials["error"].sum())
    report["trials"]["other_kinds"] = int((~trials["kind"].isin(kinds) & ~trials["error"]).sum())
    trials = trials[~trials["error"] & trials["kind"].isin(kinds)]

    empty = set(
        zip(pairs.loc[pairs["empty_a"] == 1, "server"], pairs.loc[pairs["empty_a"] == 1, "tool_a"],
            strict=True)
    )
    primary = trials[trials["kind"].isin(PRIMARY_KINDS)]
    counts = primary.groupby(["server", "expected"]).size()
    tools = set(zip(pairs["server"], pairs["tool_a"], strict=True))
    excluded: dict[tuple[str, str], str] = {}
    for key in sorted(tools):
        if key in empty:
            excluded[key] = "empty description"
        elif int(counts.get(key, 0)) < MIN_VALID_TRIALS:
            excluded[key] = (
                f"{int(counts.get(key, 0))} valid primary trials (minimum {MIN_VALID_TRIALS})"
            )
    report["tools"] = {f"{s}::{t}": why for (s, t), why in excluded.items()}

    keep_a = [(s, a) not in excluded for s, a in zip(pairs["server"], pairs["tool_a"], strict=True)]
    keep_b = [(s, b) not in excluded for s, b in zip(pairs["server"], pairs["tool_b"], strict=True)]

    pairs = pairs[np.array(keep_a, dtype=bool) & np.array(keep_b, dtype=bool)]
    keep_t = [
        (s, e) not in excluded and (s, e) in tools
        for s, e in zip(trials["server"], trials["expected"], strict=True)
    ]
    trials = trials[np.array(keep_t, dtype=bool)] if len(trials) else trials

    for server in sorted(set(pairs["server"].unique())):
        if pairs[pairs["server"] == server]["tool_a"].nunique() < 2:
            report["servers"][server] = "fewer than 2 tools left after exclusions"
    pairs = pairs[~pairs["server"].isin(set(report["servers"]))]
    trials = trials[~trials["server"].isin(set(report["servers"]))]
    return pairs, trials, report


def outcome_rows(pairs: pd.DataFrame, trials: pd.DataFrame) -> pd.DataFrame:
    """One row per (trial, other tool B): y = 1 if the trial chose B."""
    joined = trials.merge(
        pairs, left_on=["server", "expected"], right_on=["server", "tool_a"], how="inner"
    )
    joined["y"] = (joined["chosen"] == joined["tool_b"]).astype(int)
    joined["server_tool"] = joined["server"] + "::" + joined["tool_a"]
    return joined


# --- models -------------------------------------------------------------------------------


def standardise(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    """z-score each column. A constant column cannot be estimated and is reported as such."""
    df = df.copy()
    kept, constant = [], []
    for column in columns:
        values = df[column].astype(float)
        sd = float(values.std(ddof=0))
        if not math.isfinite(sd) or sd == 0:
            constant.append(column)
            continue
        df[column] = (values - values.mean()) / sd
        kept.append(column)
    return df, kept, constant


def fit_mixed(rows: pd.DataFrame) -> dict[str, Any]:
    """The primary model: random intercepts for server and for expected tool within server."""
    from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

    data, predictors, constant = standardise(rows, [*PREDICTORS, COVARIATE])
    tested = [p for p in predictors if p != COVARIATE]
    formula = "y ~ " + (" + ".join(predictors) if predictors else "1")
    vc = {"tool": "0 + C(server_tool)"}
    if data["server"].nunique() > 1:
        vc["server"] = "0 + C(server)"
    model = BinomialBayesMixedGLM.from_formula(formula, vc, data)
    fit = model.fit_vb()

    names = list(model.exog_names)
    table = []
    for predictor in tested:
        i = names.index(predictor)
        mean, sd = float(fit.fe_mean[i]), float(fit.fe_sd[i])
        z = mean / sd if sd > 0 else float("nan")
        table.append(
            {
                "predictor": predictor,
                "log_or": mean,
                "or": math.exp(mean),
                "lo": math.exp(mean - Z95 * sd),
                "hi": math.exp(mean + Z95 * sd),
                "p": _two_sided_p(z),
                "confirmatory": predictor in CONFIRMATORY,
            }
        )
    return {
        "method": "statsmodels BinomialBayesMixedGLM, fit_vb (variational Bayes)",
        "interval": "posterior mean +/- 1.96 posterior SD, exponentiated",
        "random_effects": sorted(vc),
        "rows": len(data),
        "events": int(data["y"].sum()),
        "not_estimable": [c for c in constant if c != COVARIATE],
        "covariate_estimable": COVARIATE not in constant,
        "effects": table,
    }


def _two_sided_p(z: float) -> float:
    if not math.isfinite(z):
        return float("nan")
    return math.erfc(abs(z) / math.sqrt(2))


def fit_cluster_bootstrap(rows: pd.DataFrame, resamples: int) -> dict[str, Any]:
    """S1: ordinary logistic regression, percentile CIs from resampling whole servers."""
    import statsmodels.api as sm

    data, predictors, constant = standardise(rows, [*PREDICTORS, COVARIATE])
    tested = [p for p in predictors if p != COVARIATE]
    if not predictors:
        return {"effects": [], "not_estimable": constant, "resamples": 0, "failed": 0}

    def fit(frame: pd.DataFrame) -> np.ndarray | None:
        exog = sm.add_constant(frame[predictors].astype(float), has_constant="add")
        # A resample can lose all variation in a predictor. Its coefficients are then not
        # determined, and any number reported for them would be arbitrary.
        if np.linalg.matrix_rank(exog.to_numpy()) < exog.shape[1]:
            return None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = sm.GLM(
                    frame["y"].astype(float), exog, family=sm.families.Binomial()
                ).fit()
        except Exception:
            return None
        params = np.asarray(result.params[tested], dtype=float)
        return params if np.all(np.isfinite(params)) else None

    point = fit(data)
    servers = data["server"].unique()
    groups = {s: data[data["server"] == s] for s in servers}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws, failed = [], 0
    for _ in range(resamples):
        pick = rng.choice(servers, size=len(servers), replace=True)
        sample = pd.concat([groups[s] for s in pick], ignore_index=True)
        if sample["y"].nunique() < 2:
            failed += 1
            continue
        params = fit(sample)
        if params is None:
            failed += 1
            continue
        draws.append(params)

    table = []
    stacked = np.array(draws) if draws else np.empty((0, len(tested)))
    for i, predictor in enumerate(tested):
        column = stacked[:, i] if len(stacked) else np.array([])
        lo, hi = (
            (float(np.percentile(column, 2.5)), float(np.percentile(column, 97.5)))
            if len(column)
            else (float("nan"), float("nan"))
        )
        estimate = float(point[i]) if point is not None else float("nan")
        table.append(
            {
                "predictor": predictor,
                "or": math.exp(estimate) if math.isfinite(estimate) else None,
                "lo": math.exp(lo) if math.isfinite(lo) else None,
                "hi": math.exp(hi) if math.isfinite(hi) else None,
            }
        )
    return {
        "method": "GLM binomial, cluster bootstrap over servers, percentile CI",
        "resamples": resamples,
        "failed": failed,
        "seed": BOOTSTRAP_SEED,
        "not_estimable": [c for c in constant if c != COVARIATE],
        "effects": table,
    }


def benjamini_hochberg(effects: list[dict[str, Any]]) -> None:
    """Family F1. Adds `q` to every effect with a finite p-value, in place."""
    from statsmodels.stats.multitest import multipletests

    finite = [e for e in effects if e["p"] is not None and math.isfinite(e["p"])]
    if not finite:
        return
    _, q, _, _ = multipletests([e["p"] for e in finite], alpha=ALPHA, method="fdr_bh")
    for effect, value in zip(finite, q, strict=True):
        effect["q"] = float(value)


def verdict_rq_a(primary: dict[str, Any]) -> dict[str, Any]:
    """Section 9: null unless cos_sim or name_jaccard has BH q < 0.05."""
    by_name = {e["predictor"]: e for e in primary["effects"]}
    tested = [p for p in CONFIRMATORY if p in by_name and "q" in by_name[p]]
    if not tested:
        return {"rq": "a", "verdict": "not estimable", "rule": "plan section 9"}
    significant = [p for p in tested if by_name[p]["q"] < ALPHA]
    return {
        "rq": "a",
        "verdict": "effect" if significant else "null",
        "significant": significant,
        "tested": tested,
        "rule": "null unless cos_sim or name_jaccard has BH q < 0.05 (plan section 9)",
    }


# --- descriptives and charts --------------------------------------------------------------


def wilson(hits: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = hits / total
    denom = 1 + Z95**2 / total
    centre = (p + Z95**2 / (2 * total)) / denom
    half = Z95 * math.sqrt(p * (1 - p) / total + Z95**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def server_accuracy(trials: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for server, group in trials[trials["kind"].isin(PRIMARY_KINDS)].groupby("server"):
        hits = int((group["chosen"] == group["expected"]).sum())
        lo, hi = wilson(hits, len(group))
        rows.append(
            {"server": server, "trials": len(group), "correct": hits, "accuracy": hits / len(group),
             "lo": lo, "hi": hi}
        )
    return rows


def confusion_by_similarity(rows: pd.DataFrame, bins: int = 5) -> list[dict[str, Any]]:
    """Pooled pairwise confusion rate within cos_sim quantile bins."""
    usable = rows.dropna(subset=["cos_sim"])
    if usable["cos_sim"].nunique() < 2:
        return []
    labels = pd.qcut(usable["cos_sim"], q=bins, duplicates="drop")
    out = []
    for interval, group in usable.groupby(labels, observed=True):
        hits, total = int(group["y"].sum()), len(group)
        lo, hi = wilson(hits, total)
        out.append(
            {"cos_sim_from": float(interval.left), "cos_sim_to": float(interval.right),
             "rows": total, "confusions": hits, "rate": hits / total, "lo": lo, "hi": hi}
        )
    return out


def draw_charts(analysis: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written = []
    model = analysis["benchmark"].get("model", "unknown model")

    effects = [e for e in analysis["primary"]["effects"] if math.isfinite(e["lo"])]
    if effects:
        fig, ax = plt.subplots(figsize=(7, 0.5 * len(effects) + 1.6))
        names = [e["predictor"] for e in effects][::-1]
        ors = [e["or"] for e in effects][::-1]
        lows = [e["or"] - e["lo"] for e in effects][::-1]
        highs = [e["hi"] - e["or"] for e in effects][::-1]
        ax.errorbar(ors, range(len(names)), xerr=[lows, highs], fmt="o", color="#2b6cb0",
                    capsize=3)
        ax.axvline(1.0, color="#888", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Odds ratio per 1 SD (95% interval)")
        ax.set_title(f"Predictors of pairwise confusion ({model})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "effects.png", dpi=150)
        plt.close(fig)
        written.append("effects.png")

    bins = analysis["descriptive"]["confusion_by_similarity"]
    if bins:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        mids = [(b["cos_sim_from"] + b["cos_sim_to"]) / 2 for b in bins]
        rates = [b["rate"] * 100 for b in bins]
        err = [
            [(b["rate"] - b["lo"]) * 100 for b in bins],
            [(b["hi"] - b["rate"]) * 100 for b in bins],
        ]
        ax.errorbar(mids, rates, yerr=err, fmt="o-", color="#2b6cb0", capsize=3)
        ax.set_xlabel("Description cosine similarity (quantile bins)")
        ax.set_ylabel("Pairwise confusion rate (%)")
        ax.set_title(f"Confusion by description similarity ({model})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "confusion_by_similarity.png", dpi=150)
        plt.close(fig)
        written.append("confusion_by_similarity.png")

    servers = sorted(analysis["descriptive"]["server_accuracy"], key=lambda r: r["accuracy"])
    if servers:
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(servers) + 1.4))
        acc = [s["accuracy"] * 100 for s in servers]
        err = [[(s["accuracy"] - s["lo"]) * 100 for s in servers],
               [(s["hi"] - s["accuracy"]) * 100 for s in servers]]
        ax.errorbar(acc, range(len(servers)), xerr=err, fmt="o", color="#2b6cb0", capsize=3)
        ax.set_yticks(range(len(servers)), [s["server"] for s in servers], fontsize=7)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Accuracy on primary requests (%, Wilson 95%)")
        ax.set_title(f"Per-server accuracy ({model})", fontsize=10)
        fig.tight_layout()
        fig.savefig(out / "server_accuracy.png", dpi=150)
        plt.close(fig)
        written.append("server_accuracy.png")
    return written


# --- main ---------------------------------------------------------------------------------


def run_model_set(
    features: pd.DataFrame, trials: pd.DataFrame, kinds: set[str]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pairs, kept, exclusions = apply_exclusions(features, trials, kinds)
    return outcome_rows(pairs, kept), kept, exclusions


def analyse(
    features_path: Path, results_dir: Path, out: Path, resamples: int
) -> dict[str, Any]:
    import statsmodels

    features = pd.read_csv(features_path)
    features["cos_sim"] = pd.to_numeric(features["cos_sim"], errors="coerce")
    results = load_results(results_dir)
    if not results:
        raise SystemExit(f"error: no benchmark result files in {results_dir}")
    trials = trial_frame(results)

    rows, kept, exclusions = run_model_set(features, trials, PRIMARY_KINDS)
    if rows.empty or rows["y"].nunique() < 2:
        raise SystemExit("error: after exclusions there is no variation in the outcome to model")

    primary = fit_mixed(rows)
    benjamini_hochberg(primary["effects"])

    s1 = fit_cluster_bootstrap(rows, resamples)
    rows_hn, _, exclusions_hn = run_model_set(features, trials, SENSITIVITY_KINDS)
    s2 = fit_mixed(rows_hn)
    largest = rows.groupby("server")["tool_a"].nunique().idxmax()
    s3_rows = rows[rows["server"] != largest]
    s3: dict[str, Any] = (
        fit_mixed(s3_rows)
        if s3_rows["server"].nunique() >= 1 and s3_rows["y"].nunique() == 2
        else {"skipped": "nothing left to fit without the largest server"}
    )
    s3["dropped_server"] = largest

    runs = [p.get("run") or {} for p in results.values()]
    analysis: dict[str, Any] = {
        "generated": dt.date.today().isoformat(),
        "plan": "docs/analysis-plan.md",
        "benchmark": {
            "model": runs[0].get("model"),
            "n": runs[0].get("n"),
            "k": runs[0].get("k"),
            "temperature": runs[0].get("temperature"),
            "dates": sorted({str(r.get("date")) for r in runs}),
            # One model and one setting across every server, or the pooled model is invalid.
            "consistent": len({(r.get("model"), r.get("n"), r.get("k")) for r in runs}) == 1,
        },
        "inputs": {
            "features": {"path": str(features_path), "sha256": sha256(features_path)},
            "results": {
                s: {"path": p["_path"], "sha256": p["_sha256"]} for s, p in results.items()
            },
        },
        "rules": {
            "primary_kinds": sorted(PRIMARY_KINDS),
            "min_valid_trials": MIN_VALID_TRIALS,
            "max_server_error_share": MAX_SERVER_ERROR_SHARE,
            "alpha": ALPHA,
        },
        "exclusions": exclusions,
        "counts": {
            "servers": int(rows["server"].nunique()),
            "tools": int(rows["server_tool"].nunique()),
            "pairs": int(rows[["server", "tool_a", "tool_b"]].drop_duplicates().shape[0]),
            "trials": len(kept),
            "rows": len(rows),
            "confusions": int(rows["y"].sum()),
        },
        "primary": primary,
        "verdicts": [verdict_rq_a(primary)],
        "sensitivity": {
            "S1_cluster_bootstrap": s1,
            "S2_with_hard_negatives": {**s2, "exclusions": exclusions_hn},
            "S3_without_largest_server": s3,
        },
        "descriptive": {
            "server_accuracy": server_accuracy(kept),
            "confusion_by_similarity": confusion_by_similarity(rows),
        },
        "versions": {
            "statsmodels": statsmodels.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    analysis["charts"] = draw_charts(analysis, out)
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2, default=_json_safe) + "\n")
    return analysis


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return str(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--features", type=Path, default=Path("results/features.csv"))
    parser.add_argument("--results", type=Path, default=Path("analysis/data/benchmark-v1"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--bootstrap", type=int, default=2000, help="S1 resamples. The plan registers 2000."
    )
    args = parser.parse_args(argv)
    analysis = analyse(args.features, args.results, args.out, args.bootstrap)
    counts = analysis["counts"]
    print(
        f"{counts['servers']} servers, {counts['tools']} tools, {counts['rows']} rows -> "
        f"{args.out / 'analysis.json'}"
    )
    for verdict in analysis["verdicts"]:
        print(f"RQ-{verdict['rq']}: {verdict['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
