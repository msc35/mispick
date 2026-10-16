#!/usr/bin/env python3
"""Freeze the finished benchmark into analysis/data/ (docs/analysis-plan.md, section 1).

Downloads the newest `benchmark-progress` artifact with the GitHub CLI, refuses it unless its
index.json says the run is complete, and writes MANIFEST.json with the artifact id and a SHA-256
of every file. Commit the result: the artifact itself expires after 90 days.

    uv run python analysis/fetch_results.py --out analysis/data/benchmark-v1
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

ARTIFACT = "benchmark-progress"


def gh(*args: str) -> bytes:
    return subprocess.run(["gh", *args], check=True, capture_output=True).stdout


def latest_artifact(repo: str) -> dict[str, Any]:
    listing = json.loads(gh("api", f"repos/{repo}/actions/artifacts?name={ARTIFACT}&per_page=1"))
    artifacts = [a for a in listing.get("artifacts", []) if not a.get("expired")]
    if not artifacts:
        raise SystemExit(f"error: no unexpired {ARTIFACT} artifact in {repo}")
    return dict(artifacts[0])


def manifest_for(directory: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    files = {
        str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and p.name != "MANIFEST.json"
    }
    return {
        "artifact": {
            "id": artifact.get("id"),
            "name": artifact.get("name"),
            "created_at": artifact.get("created_at"),
            "workflow_run": (artifact.get("workflow_run") or {}).get("id"),
        },
        "files": files,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--repo", default="msc35/mispick")
    parser.add_argument("--out", type=Path, default=Path("analysis/data/benchmark-v1"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Freeze a run that has not finished. For checking the pipeline, never for results.",
    )
    args = parser.parse_args(argv)

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"error: {args.out} is not empty. A frozen copy is never overwritten.")

    artifact = latest_artifact(args.repo)
    archive = zipfile.ZipFile(
        io.BytesIO(gh("api", f"repos/{args.repo}/actions/artifacts/{artifact['id']}/zip"))
    )
    index = json.loads(archive.read("index.json"))
    if not index.get("complete") and not args.allow_incomplete:
        raise SystemExit(
            f"error: artifact {artifact['id']} is not complete; pending: "
            f"{', '.join(index.get('pending') or [])}. Wait for the run to finish."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    archive.extractall(args.out)
    manifest = manifest_for(args.out, artifact)
    manifest["complete"] = bool(index.get("complete"))
    (args.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"froze artifact {artifact['id']} ({len(manifest['files'])} files) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
