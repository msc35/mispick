# Using mispick in CI

Copy one of these into `.github/workflows/`.

## The free check, with no model

With no model key, mispick cannot measure anything, so the Action **skips with an explanation
rather than failing**. That is deliberate: a check that fails because a secret is missing
teaches people to ignore it.

```yaml
name: mispick
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  mispick:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: msc35/mispick@v0.1.0
        with:
          snapshot: tools.json
```

## The real check

```yaml
name: mispick
on: pull_request

permissions:
  contents: read
  pull-requests: write

jobs:
  mispick:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0      # needed for compare-base
      - uses: msc35/mispick@v0.1.0
        with:
          snapshot: tools.json
          model: anthropic/claude-haiku-4-5-20251001
          queries: '8'
          runs: '3'
          temperature: '0'
          seed: '1'
          fail-under: '80'
          compare-base: 'true'
          badge: mispick-badge.svg
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

## Why a committed snapshot, and a committed query cache

Two things make a CI run mean something:

**Commit `tools.json`.** Capture it with
`mispick snapshot --cmd "python -m my_server" -o tools.json`. Starting your real server on a
runner adds a dependency on every service it talks to; a snapshot is the tool surface, which is
all mispick looks at. It also lets `compare-base` diff the surface between branches.

**Commit `.mispick/queries.yaml`.** Generated queries are a test set. If CI regenerates them on
every run, the test set changes under you and a score change might mean the queries got harder
rather than the descriptions got worse. Commit them, read them, edit the ones that are wrong.
They regenerate only when a tool's name, description or schema changes.

## Why Ollama is not the CI default

The local default is what makes mispick free to run on your laptop, and it is the wrong choice
on a shared runner: a small model takes seconds per selection call, and a realistic run makes
hundreds of them. Set a cloud model for CI, or point `OLLAMA_HOST` at a self-hosted runner.

## Determinism

Use `temperature: '0'` and a `seed` in CI. A score that moves because of sampling noise is a
flaky check, and a flaky check is worse than no check. Note that the Anthropic backend takes no
seed and reports itself as non-deterministic — with it, lean on `runs` (K) and read the
confidence intervals rather than the point estimate.
