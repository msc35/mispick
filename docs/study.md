# Which MCP tools do models confuse, and why?

**Status:** in progress. The setup and methods sections are final. Every results section is a
placeholder until its numbers exist in `results/`. No finding is stated here before the data
shows it.

## Question

AI assistants choose a tool by reading short descriptions. When two descriptions sound alike,
the model can quietly pick the wrong tool. This study measures that on real MCP servers and asks
three questions:

1. **What makes two tools easy to confuse?** Candidate causes are similar descriptions,
   overlapping names, description length, undocumented parameters, and missing "use this
   when…" guidance.
2. **Does a cheap model's confusion predict a stronger model's?**
3. **Do auto-generated test requests give the same answers as hand-written ones?**

## Pre-registration

The analysis was fixed in [analysis-plan.md](analysis-plan.md) and committed on 2026-10-16,
before any benchmark result was analysed. It sets out the research questions, metrics,
exclusion rules, statistical models, multiple-comparison families and what counts as a null
result. Any change is listed, with a date, in that file's *Deviations* section.

## Setup

**Servers.**
- **Source.** The servers come from the official MCP registry. They were ranked by GitHub stars
  and capped per owner, and only servers that start without an API key were kept. The four MCP
  reference servers were added.
- **Snapshots.** Each server's tool list was captured once on a GitHub runner, sending only
  `initialize` and `tools/list`, and committed to `benchmark/snapshots/`. No tool was ever
  called.
- **Coverage.** Servers that could not be captured are listed in
  `benchmark/snapshots/index.json` with the reason.
- **Size limit.** A server with more than 120 tools is not measured. It is reported by size only.

**Measurement.** The tool is [mispick](../README.md).
- **Requests.** For each tool, mispick generates realistic user requests in three kinds:
  straightforward, paraphrased, and *hard negatives* written to sound like the tool's nearest
  neighbour. Each server also gets a few "no tool fits" requests. The generator is told to spell
  out concrete details rather than refer to missing context.
- **Selection.** For each request, the model sees the server's full tool list, as a client
  would send it, and picks one tool. Each request is asked K times.
- **Settings.** Model `gemini/gemini-3.5-flash-lite`, N=8 requests per tool, K=3, temperature 0.
  These come from `.github/workflows/benchmark.yml`. Gemini does not accept a seed, and the
  reports say so.
- **Schedule.** The free API tier allows a limited number of calls per day, so the run is
  spread over several days. It resumes from saved progress, and no call is repeated.
- **Errors.** A trial that errored, from a rate limit or a server error, is recorded as an error.
  It is never recorded as "the model chose nothing".

**Data freeze.** When the run completes, its output is committed to `analysis/data/benchmark-v1/`
with a checksum of every file (`analysis/fetch_results.py`). All analysis reads only that copy.

## Methods

**Primary metric.** The *pairwise confusion rate* of an ordered pair (A, B) is the share of
valid trials meant for tool A in which the model chose tool B.

**Q1: what predicts confusion.**
- **Model.** A mixed-effects logistic regression on trial-level outcomes, with random intercepts
  for server and for the intended tool within its server. It is fit with variational Bayes
  (`statsmodels`).
- **Predictors** (definitions in the plan, section 5):
  - embedding similarity of the two descriptions (`all-MiniLM-L6-v2` at a pinned revision)
  - name-token overlap, after dropping prefixes that most of a server's tools share
  - description lengths and their difference
  - the share of documented parameters
  - whether each description says when to use the tool
  - the server's tool count, as a covariate
- **Trials used.** Hard-negative requests are left out of the main model, because they are
  written to sound like the tool most similar by wording, and that would inflate the
  similarity predictors.
- **Sensitivity checks.**
  - a cluster bootstrap over servers
  - the model with hard negatives included
  - the model without the largest server
- **Multiple testing.** Benjamini-Hochberg across the predictor tests.

**Q2: cross-model transfer.**
- **Subset.** A stratified subset of servers: score thirds, a fixed seed, and a tool cap for
  cost.
- **Re-run.** The subset is re-run on one or two stronger models with the same snapshots and
  the exact same frozen requests.
- **Comparison.** A Spearman correlation of pairwise confusion rates with a bootstrap CI over
  servers, agreement on each server's top-3 confused pairs, and per-server McNemar tests with
  Benjamini-Hochberg.

**Q3: generated vs hand-written.**
- **Requests.** Four servers with distinct domains were chosen in advance. About 20 requests
  per server were written by hand, without seeing that server's generated requests or results,
  and measured with the same model and settings.
- **Comparison.** The accuracy difference with a bootstrap CI, judged against a ±10-point
  equivalence margin, and the overlap of confused pairs.

**Null results** are defined in advance in the plan, section 9, and are reported in the same way
as positive ones.

## Results

*Placeholder. Filled from `results/analysis.json` once the benchmark run is frozen.*

## Cross-model transfer

*Placeholder. Filled from `results/transfer.json`.*

## Generated vs hand-written requests

*Placeholder. Filled from `results/human_vs_generated.json`.*

## Limitations

These are known in advance and come from the design, not from the data.

- **One benchmark model.** The main results describe one model's confusions on one date. Q2
  tests how far they generalise.
- **Few servers.** After exclusions, few servers remain, so server-level variation is estimated
  poorly and many tool pairs have few trials.
- **Generated requests.** The benchmark's requests are written by the model being measured.
  Q3 checks how much that matters.
- **First call only.** mispick scores the first tool the model calls. A model that sensibly
  looks something up before acting counts as confused.
- **Repeats are not independent.** At temperature 0, the K repeats of a request mostly agree.
  The bootstrap over servers is the more conservative read.
- **Who gets measured.** Only servers that start without credentials are included, so
  popular services that need an API key are missing.

## Reproduce

```bash
uv sync --all-extras --group analysis
uv run python analysis/features.py --snapshots benchmark/snapshots --out results
uv run python analysis/analyze.py --features results/features.csv \
    --results analysis/data/benchmark-v1 --out results
uv run python analysis/transfer.py analyze
uv run python analysis/human_vs_generated.py analyze
```

[analysis/README.md](../analysis/README.md) has every step, including re-running the benchmark
itself.
