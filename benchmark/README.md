# The public benchmark

A reproducible run over popular MCP servers, published as a static leaderboard.

## Rules

1. **Only `initialize` and `tools/list`.** No tool is ever called. This is what makes it safe to
   point at someone else's server.
2. **Only servers that start without API keys.** Anything needing a credential is skipped and
   recorded as skipped, not guessed at.
3. **Snapshots are committed.** `snapshots/` holds the captured `tools/list` for each server, so
   a result can be reproduced exactly later even if the server changes.
4. **One fixed model, fixed seed, temperature 0.** Stated on every page.
5. **Neutral wording.** "Most confused pair", never "worst server". Every server links to its
   own repository. A low score is a finding about one model's reading of a tool surface, not a
   judgement of the people who wrote it.

## Steps

```bash
# 1. pick servers from the official registry that need no keys
uv run python benchmark/pick.py --limit 30 --out benchmark/servers.json

# 2. capture their tool lists (initialize + tools/list only)
uv run python benchmark/capture.py --servers benchmark/servers.json --out benchmark/snapshots

# 3. measure every snapshot with one fixed model
uv run python benchmark/measure.py --snapshots benchmark/snapshots --out benchmark/results \
    --model ollama/qwen3.5:4b --seed 1

# 4. build the static site
uv run python benchmark/build_site.py --results benchmark/results --out benchmark/site
```

`benchmark/site/` is plain HTML with no framework and no external requests, ready for GitHub
Pages.

## Re-runs

If a server's result looks wrong, open an issue with the
[re-run template](../.github/ISSUE_TEMPLATE/benchmark-rerun.md). Servers change; a snapshot from
last month may not describe today's tools.
