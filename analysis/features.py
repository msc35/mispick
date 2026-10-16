#!/usr/bin/env python3
"""Build the pair-level predictor table for RQ-a of docs/analysis-plan.md.

One row per ordered tool pair (A, B) on the same snapshot. Everything here comes from the
committed snapshots alone: no model call, no benchmark result. Exclusions are not applied here;
analyze.py applies them, so they live in one place.

For a snapshot that holds several servers (a config run), pairs across servers are included and
flagged with `same_server = 0`; the pre-registered model does not use them, because the
benchmark measures each server alone.

    uv sync --all-extras --group analysis
    uv run python analysis/features.py --snapshots benchmark/snapshots --out results
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from mispick.sources.snapshot import load_snapshot
from mispick.types import Tool

#: The embedding model, pinned to an exact revision and file hash (analysis plan, section 5).
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_REPO = "Qdrant/all-MiniLM-L6-v2-onnx"
EMBED_REVISION = "d13954661f83248295ba75c1ed411eef3b7b936e"
EMBED_FILE = "model.onnx"
EMBED_SHA256 = "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5"

#: "States when to use the tool", exactly as registered in the analysis plan, section 5.
WHEN_PATTERNS = [
    r"\buse (?:this|it)(?: tool)? (?:when|to|for|if)\b",
    r"\bwhen (?:you|the user)\b",
    r"\bonly (?:when|if|for)\b",
    r"\bdo not use\b",
    r"\bdon't use\b",
    r"\binstead of\b",
    r"\bprefer\b",
]
_WHEN = re.compile("|".join(WHEN_PATTERNS), re.IGNORECASE)

#: Tokens in at least this share of a server's tool names are dropped before name overlap.
COMMON_TOKEN_SHARE = 0.5

COLUMNS = [
    "server",
    "tool_a",
    "tool_b",
    "server_a",
    "server_b",
    "same_server",
    "name_collision",
    "n_tools",
    "log_n_tools",
    "empty_a",
    "empty_b",
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


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    def describe(self) -> dict[str, Any]: ...


class MiniLMEmbedder:
    """The registered model, run on CPU through fastembed's ONNX runtime."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        from fastembed import TextEmbedding
        from huggingface_hub import snapshot_download

        path = Path(
            snapshot_download(
                EMBED_REPO,
                revision=EMBED_REVISION,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        digest = hashlib.sha256((path / EMBED_FILE).read_bytes()).hexdigest()
        if digest != EMBED_SHA256:
            raise SystemExit(
                f"error: {EMBED_FILE} at revision {EMBED_REVISION} hashes to {digest}, not the "
                f"registered {EMBED_SHA256}. Refusing to compute features with a different model."
            )
        self._model = TextEmbedding(EMBED_MODEL, specific_model_path=str(path))

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._model.embed(list(texts))]

    def describe(self) -> dict[str, Any]:
        import fastembed

        return {
            "kind": "minilm",
            "model": EMBED_MODEL,
            "repo": EMBED_REPO,
            "revision": EMBED_REVISION,
            "sha256": EMBED_SHA256,
            "fastembed": fastembed.__version__,
        }


class HashEmbedder:
    """A deterministic bag-of-words stand-in, for offline tests only. Never for results."""

    DIM = 256

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vector = [0.0] * self.DIM
            for word in re.findall(r"[a-z0-9]+", text.lower()):
                slot = int(hashlib.sha256(word.encode()).hexdigest(), 16) % self.DIM
                vector[slot] += 1.0
            vectors.append(vector)
        return vectors

    def describe(self) -> dict[str, Any]:
        return {"kind": "hash", "note": "test stand-in; not the registered model"}


def name_tokens(name: str) -> list[str]:
    """Split on `_`, `-`, `.` and camelCase, lowercased."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    return [t.lower() for t in re.split(r"[_\-.\s]+", spaced) if t]


def common_tokens(names: Sequence[str]) -> set[str]:
    """Tokens in at least half of a server's tool names, such as a shared prefix."""
    if len(names) < 2:
        return set()
    counts: dict[str, int] = {}
    for name in names:
        for token in set(name_tokens(name)):
            counts[token] = counts.get(token, 0) + 1
    return {t for t, c in counts.items() if c / len(names) >= COMMON_TOKEN_SHARE}


def name_jaccard(a: str, b: str, common: set[str]) -> float:
    left = set(name_tokens(a)) - common
    right = set(name_tokens(b)) - common
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def param_doc_share(schema: dict[str, Any] | None) -> float:
    """Share of top-level properties with a non-empty description; 1.0 when there are none."""
    properties = (schema or {}).get("properties") or {}
    if not isinstance(properties, dict) or not properties:
        return 1.0
    documented = sum(
        1
        for spec in properties.values()
        if isinstance(spec, dict) and str(spec.get("description") or "").strip()
    )
    return documented / len(properties)


def states_when(description: str) -> int:
    return 1 if _WHEN.search(description or "") else 0


def log_len(description: str) -> float:
    return math.log1p(len(description or ""))


def cosine(u: Sequence[float], v: Sequence[float]) -> float | None:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0 or nv == 0:
        return None
    return sum(x * y for x, y in zip(u, v, strict=True)) / (nu * nv)


def pair_rows(server: str, tools: list[Tool], embedder: Embedder) -> list[dict[str, Any]]:
    """Every ordered pair of distinct tools in one snapshot."""
    descriptions = [(t.description or "").strip() for t in tools]
    vectors = embedder.embed(descriptions)
    by_server: dict[str | None, list[str]] = {}
    for tool in tools:
        by_server.setdefault(tool.server, []).append(tool.name)
    common = {key: common_tokens(names) for key, names in by_server.items()}
    n = len(tools)

    rows = []
    for i, a in enumerate(tools):
        for j, b in enumerate(tools):
            if i == j:
                continue
            same = a.server == b.server
            shared = common[a.server] | common[b.server] if not same else common[a.server]
            sim = cosine(vectors[i], vectors[j]) if descriptions[i] and descriptions[j] else None
            rows.append(
                {
                    "server": server,
                    "tool_a": a.qualified_name,
                    "tool_b": b.qualified_name,
                    "server_a": a.server or "",
                    "server_b": b.server or "",
                    "same_server": int(same),
                    "name_collision": int(a.name == b.name),
                    "n_tools": n,
                    "log_n_tools": math.log(n),
                    "empty_a": int(not descriptions[i]),
                    "empty_b": int(not descriptions[j]),
                    "cos_sim": sim,
                    "name_jaccard": name_jaccard(a.name, b.name, shared),
                    "log_len_a": log_len(descriptions[i]),
                    "log_len_b": log_len(descriptions[j]),
                    "log_len_diff": abs(log_len(descriptions[i]) - log_len(descriptions[j])),
                    "param_doc_a": param_doc_share(a.input_schema),
                    "param_doc_b": param_doc_share(b.input_schema),
                    "when_a": states_when(descriptions[i]),
                    "when_b": states_when(descriptions[j]),
                }
            )
    return rows


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(
    snapshots: Path, out: Path, embedder: Embedder, max_tools: int
) -> tuple[Path, dict[str, Any]]:
    files = sorted(p for p in snapshots.glob("*.json") if p.name != "index.json")
    if not files:
        raise SystemExit(f"error: no snapshots in {snapshots}")

    rows: list[dict[str, Any]] = []
    servers: dict[str, Any] = {}
    for path in files:
        tool_set = load_snapshot(path)
        tools = tool_set.sorted_tools()
        record: dict[str, Any] = {"tools": len(tools), "sha256": sha256(path)}
        if len(tools) < 2:
            record["skipped"] = "one tool, so no pairs"
        elif max_tools and len(tools) > max_tools:
            record["skipped"] = f"over --max-tools {max_tools}, not measured by the benchmark"
        else:
            new = pair_rows(path.stem, tools, embedder)
            record["pairs"] = len(new)
            rows.extend(new)
        servers[path.stem] = record

    out.mkdir(parents=True, exist_ok=True)
    table = out / "features.csv"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if row[k] is None else row[k]) for k in COLUMNS})

    meta = {
        "plan": "docs/analysis-plan.md",
        "snapshots": str(snapshots),
        "embedder": embedder.describe(),
        "when_patterns": WHEN_PATTERNS,
        "common_token_share": COMMON_TOKEN_SHARE,
        "max_tools": max_tools,
        "servers": servers,
        "pairs": len(rows),
        "features_sha256": sha256(table),
    }
    (out / "features_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return table, meta


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshots", type=Path, default=Path("benchmark/snapshots"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument(
        "--embedder",
        choices=["minilm", "hash"],
        default="minilm",
        help="minilm is the registered model. hash is an offline stand-in for tests only.",
    )
    parser.add_argument("--max-tools", type=int, default=120, help="Match the benchmark's limit.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    embedder: Embedder = HashEmbedder() if args.embedder == "hash" else MiniLMEmbedder(
        args.cache_dir
    )
    table, meta = build(args.snapshots, args.out, embedder, args.max_tools)
    print(f"{meta['pairs']} pairs from {len(meta['servers'])} snapshots -> {table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
