#!/usr/bin/env python3
"""Verify every `uses:` reference in our workflows and action resolves to a real tag.

A missing tag fails GitHub's *job setup*, before any step runs - so it produces a red run
with no log and no annotation pointing at the cause. Worse, `release.yml` and `pages.yml` only
run at release time, so a stale pin there stays invisible until the moment it matters most.

Not every repository publishes moving major tags: `actions/*` do, while `astral-sh/setup-uv`
and `marocchino/sticky-pull-request-comment` publish only exact versions. Reading the latest
release name is not enough; the tag has to exist.

Needs network, so this is a CI/manual check and not part of the offline test suite.
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

USES = re.compile(r"^\s*-?\s*uses:\s*([^\s#]+)", re.MULTILINE)


def refs() -> dict[str, set[Path]]:
    found: dict[str, set[Path]] = {}
    targets = list(Path(".github/workflows").glob("*.yml"))
    if Path("action.yml").is_file():
        targets.append(Path("action.yml"))
    for path in targets:
        for ref in USES.findall(path.read_text()):
            # Local actions and docker images are not tag refs.
            if ref.startswith(("./", "docker://")):
                continue
            found.setdefault(ref, set()).add(path)
    return found


def _headers() -> dict[str, str]:
    headers = {"User-Agent": "mispick-pin-check"}
    # Unauthenticated GitHub API allows 60 requests an hour per IP, and CI runner IPs are
    # shared, so without a token this check flakes rather than fails honestly.
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def resolves(ref: str) -> tuple[bool, str]:
    repo, _, version = ref.partition("@")
    if not version:
        return False, "no version pinned - pin it"
    # A branch-shaped ref (e.g. pypa's release/v1) is valid but not a tag.
    kinds = ["tags", "heads"] if "/" in version else ["tags"]
    for kind in kinds:
        url = f"https://api.github.com/repos/{repo}/git/ref/{kind}/{version}"
        request = urllib.request.Request(url, headers=_headers())
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                json.load(response)
            return True, kind.rstrip("s")
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                continue
            return False, f"HTTP {exc.code} (could not check)"
        except Exception as exc:  # pragma: no cover - network flake
            return False, f"{type(exc).__name__} (could not check)"
    return False, "no such tag or branch"


def main() -> int:
    problems = 0
    for ref, paths in sorted(refs().items()):
        ok, detail = resolves(ref)
        where = ", ".join(sorted(str(p) for p in paths))
        if ok:
            print(f"ok      {ref}  ({detail})")
        else:
            problems += 1
            print(f"BROKEN  {ref}  -> {detail}   [{where}]")
    if problems:
        print(
            f"\n{problems} unresolvable reference(s). A missing tag fails job setup with no "
            "log, so fix these before pushing."
        )
        return 1
    print("\nall references resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())
