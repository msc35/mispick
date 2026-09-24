# Committed tool-list snapshots

One `<slug>.json` per server that started, plus `index.json` recording the outcome for every
candidate — including the ones that did not start, and why.

Captured 2026-09-24 by the `snapshot` workflow on a GitHub runner, not on anyone's laptop:
capturing launches other people's packages through `npx` / `uvx`, which is arbitrary code
execution. Only `initialize` and `tools/list` are ever sent — mispick cannot call a tool, which is
enforced in `mispick.sources.server` and tested in `tests/test_never_calls_tools.py`.

Committed so a published result can be reproduced exactly later, even after a server changes.

## This capture

30 candidates: the four official reference servers plus the 26 most-starred registry entries with
a launchable stdio package.

| outcome | count | meaning |
|---|---|---|
| `captured` | 16 | started and returned a tool list |
| `needs_credentials` | 5 | asked for a key or env var, or tried to open a browser to authorise |
| `failed` | 7 | did not start, for a reason not otherwise classified |
| `needs_arguments` | 1 | printed its usage; the registry's install command is incomplete |
| `package_unavailable` | 1 | the registry advertises a version npm does not have |

Only `failed` suggests a problem with the server itself. Needing a credential is a limitation of
this benchmark, not a defect, and a registry entry pointing at a nonexistent package version is a
registry problem.

## Refreshing

Run the `snapshot` workflow and download its `snapshots` artifact over this directory. Do not run
`benchmark/capture.py` on a machine you care about.
