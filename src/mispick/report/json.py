"""The machine-readable report. This is the contract other tools consume."""

from __future__ import annotations

import json
from typing import Any

from mispick.crossserver import analyse
from mispick.metrics import Metrics, top_confused_pairs
from mispick.report.provenance import provenance_of
from mispick.select import RunResult

#: Bump when a field changes meaning, so consumers can tell.
SCHEMA_VERSION = 1


def build(result: RunResult, metrics: Metrics) -> dict[str, Any]:
    """The full report as plain data."""
    prov = provenance_of(result, metrics.token_estimate)
    by_query = {q.id: q for q in result.queries}

    return {
        "schemaVersion": SCHEMA_VERSION,
        "tool": "mispick",
        "run": {
            "model": prov.model,
            "n": prov.n,
            "k": prov.k,
            "date": prov.date,
            "temperature": prov.temperature,
            "seed": prov.seed,
            "deterministic": prov.deterministic,
            "caveat": prov.caveat,
        },
        "servers": [info.model_dump() for info in result.tool_set.servers],
        "tools": [
            {
                "name": t.qualified_name,
                "server": t.server,
                "title": t.title,
                "description": t.description,
                # The schema is part of what the model was shown, and the token estimate is
                # computed from it - without it a re-rendered report would not match.
                "inputSchema": t.input_schema,
                "annotations": t.annotations,
            }
            for t in result.tools
        ],
        "score": metrics.score,
        "summary": {
            "accuracy": _rate(metrics.accuracy),
            "stability": _rate(metrics.stability),
            "argumentValidity": _rate(metrics.arg_validity),
            "overTrigger": _rate(metrics.over_trigger),
            "phantomRate": _rate(metrics.phantom_rate),
            "trials": metrics.trials,
            "erroredTrials": metrics.errored_trials,
            "toolListTokensEstimate": metrics.token_estimate,
        },
        "confusionMatrix": {
            "rows": metrics.row_labels,
            "columns": metrics.column_labels,
            "counts": metrics.matrix,
        },
        "perTool": {
            name: {
                "recall": _rate(tool.recall),
                "precision": _rate(tool.precision),
                "f1": round(tool.f1, 4),
            }
            for name, tool in metrics.per_tool.items()
        },
        "confusedPairs": [
            {
                "expected": p.expected,
                "chosen": p.chosen,
                "count": p.count,
                "share": round(p.share, 4),
            }
            for p in metrics.confused_pairs
        ],
        "topConfusedPairs": [
            {"expected": p.expected, "chosen": p.chosen, "count": p.count}
            for p in top_confused_pairs(metrics, limit=3)
        ],
        "crossServer": _cross_server(result, metrics),
        "unstableQueries": metrics.unstable_queries,
        "queries": [q.model_dump() for q in result.queries],
        "trials": [
            {
                **c.model_dump(),
                "expected": by_query[c.query_id].expected if c.query_id in by_query else None,
            }
            for c in result.choices
        ],
        "backendErrors": result.errors,
    }


def _cross_server(result: RunResult, metrics: Metrics) -> dict[str, Any]:
    cross = analyse(result, metrics)
    return {
        "isMultiServer": cross.is_multi_server,
        "crossServerRate": _rate(cross.cross_server_rate),
        "nameCollisions": [
            {
                "name": c.name,
                "servers": c.servers,
                "identicalDescriptions": c.identical_descriptions,
            }
            for c in cross.collisions
        ],
        "servers": [
            {
                "label": s.label,
                "toolCount": s.tool_count,
                "accuracy": _rate(s.accuracy),
                "lost": s.lost,
                "stolen": s.stolen,
            }
            for s in cross.servers
        ],
        "leaks": [
            {"expected": e, "chosen": c, "count": n} for e, c, n in cross.leaks
        ],
    }


def _rate(rate: Any) -> dict[str, Any]:
    low, high = rate.interval
    return {
        "hits": rate.hits,
        "total": rate.total,
        "value": round(rate.value, 4) if rate.total else None,
        "ci95": [round(low, 4), round(high, 4)] if rate.total else None,
    }


def render(result: RunResult, metrics: Metrics) -> str:
    return json.dumps(build(result, metrics), indent=2) + "\n"
