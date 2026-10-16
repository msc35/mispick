# Analysis plan (pre-registered)

**Registered:** 2026-10-16, before any benchmark result was analysed.
**Status:** frozen. Changes go in [Deviations](#deviations), with a date and a reason, and never
replace the text above them.

This plan describes how the benchmark data in `benchmark/` will be analysed. It was committed
while the scheduled run was still in progress. Only one pilot server
(`io-github-modelcontextprotocol-server-memory`) had been looked at, and only to debug the run.
[docs/benchmark-plan.md](benchmark-plan.md) records what that pilot showed.

## 1. Data

- **Benchmark run.** Model `gemini/gemini-3.5-flash-lite`, N=8 (4 straightforward, 2
  paraphrased and 2 hard-negative requests per tool), plus 3 "no tool fits" requests per server.
  K=3, temperature 0. The seed is sent nowhere because Gemini does not accept it. The input is
  the snapshots committed in `benchmark/snapshots/`.
- **Freeze.** When `index.json` reports `"complete": true`, the final `benchmark-progress`
  artifact is downloaded to `analysis/data/benchmark-v1/` and committed, with its artifact ID
  and a SHA-256 of every file. All analysis reads that frozen copy only. Every number in any
  document comes from a file under `results/` produced from it.
- **Unit.** One row per (trial, other tool): for a trial whose expected tool is A, there is one
  row for each other tool B on the same server. The outcome is 1 if the model chose B.

## 2. Research questions

- **RQ-a.** Which description and tool-set properties predict pairwise confusion across real MCP
  servers?
- **RQ-b.** Does confusion measured on the benchmark model transfer to stronger models?
- **RQ-c.** Do generated requests give the same results as hand-written ones?

## 3. Metrics

- **Primary.** The **pairwise confusion rate** of an ordered pair (A, B): the share of valid
  trials whose expected tool is A that chose B.
- **Secondary.**
  - per-tool recall
  - per-server accuracy
  - cross-server confusion: **not estimable here**, because each server is measured alone. It is
    out of scope for this registration.

## 4. Exclusions (decided now)

| Level | Excluded when | Reason |
|---|---|---|
| Server | Not captured (failed to start, needs credentials, package missing) | No tool list |
| Server | Skipped by `--max-tools 120` | Not measured |
| Server | Has one tool | No pairs |
| Server | More than 20% of its trials errored | Measurement not trustworthy |
| Tool | Empty description | No text to embed |
| Tool | **Fewer than 12** valid primary trials, out of 18 possible (6 queries × K=3) | Too few to estimate a rate. 12 needs at least 4 of 6 queries fully answered |
| Trial | Errored (rate limit after retries, 5xx, truncated tool list) | Not a model decision |

A pair is excluded if either of its tools is excluded.

Two kinds of answer that are not a tool on the server are kept:
- A **phantom** tool name (one that doesn't exist) stays in the denominator and counts toward
  no pair.
- **"No tool"** is treated the same way.

Every exclusion is counted and reported.

**Primary trial set:** straightforward and paraphrased requests only. Hard-negative requests are
written to sound like the expected tool's nearest neighbour, and that neighbour is chosen by
word overlap, so they push confusion up exactly where the predictors are high. They are
**excluded from the primary model** and used only in sensitivity check S2.

## 5. RQ-a: predictors and model

Predictors are computed from the snapshot alone and need no model call. Continuous predictors are
z-scored, so odds ratios are per 1 SD.

| Predictor | Definition |
|---|---|
| `cos_sim` | Cosine similarity of the A and B descriptions. Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, run on CPU, with its exact revision recorded by `features.py` |
| `name_jaccard` | Jaccard overlap of name tokens (split on `_`, `-`, `.` and camelCase, lowercased). Tokens found in ≥50% of a server's tool names are dropped first, so a shared prefix like `workix_` does not count as overlap |
| `log_len_a`, `log_len_b` | log(1 + characters) of each description |
| `log_len_diff` | Absolute difference of the two log lengths |
| `param_doc_a`, `param_doc_b` | Share of top-level `inputSchema` properties that have a non-empty description. A tool with no parameters scores 1.0 |
| `when_a`, `when_b` | 1 if the description matches, case-insensitively, any of: `use (this\|it)( tool)? (when\|to\|for\|if)`, `when (you\|the user)`, `only (when\|if\|for)`, `do not use`, `don't use`, `instead of`, `prefer` |
| `log_n_tools` (covariate) | log of the server's tool count. Controls for the number of tools a choice is spread across. Not tested |

**Dropped:**
- **Exact name collision.** Names are unique within a server, so it is constant at 0.
- **Same-server vs cross-server.** There are no cross-server trials in this data.

**Model.** Logistic regression with random intercepts for **server** and for **expected tool A
within server**:

```
logit P(choose B | expected A) = b0 + b·predictors + log_n_tools + u_server + u_server:A
```

- **Fitting.** Python has no maximum-likelihood GLMM in the repo's toolset, so the model is fit
  with `statsmodels` `BinomialBayesMixedGLM` (variational Bayes).
- **Reporting.** Each fixed effect is reported as an odds ratio with a 95% interval: posterior
  mean ± 1.96 posterior SD, exponentiated. A two-sided p-value comes from z = mean / SD.
- **Repeats.** The K=3 repeats of a query are correlated (temperature 0). S1 accounts for this.

**Hypotheses.**
- **Confirmatory.** H1: `cos_sim` OR > 1. H2: `name_jaccard` OR > 1. Tests are two-sided.
- **Exploratory.** All other predictors.

**Sensitivity checks.** All are reported, and none replaces the primary result.
- **S1.** Ordinary logistic regression with a **cluster bootstrap over servers** (2,000
  resamples, percentile CIs).
- **S2.** Primary model with the hard-negative trials included.
- **S3.** Primary model without the largest server. One 64-tool server contributes about two
  thirds of all pairs.

## 6. RQ-b: cross-model transfer

- **Subset.** Servers are stratified by benchmark score tertile, with **4 per tertile (12
  servers)**, drawn at random with seed 20261016 among eligible servers with ≤ 40 tools. The tool
  cap is for cost, since every call resends the whole list. If a tertile has fewer than 4,
  all of it is taken.
- **Models.** One or two models stronger than the benchmark model. They must support tool
  calling, run at temperature 0, and use the same snapshots and the **same frozen query files**.
  The choice is fixed in a dated addendum committed before the run.
- **Analysis.**
  - **Spearman ρ** of pairwise confusion rates between the benchmark model and each other model,
    over all ordered pairs in the subset. A secondary ρ uses only the pairs with non-zero
    confusion in either model. Each gets a 95% CI from a bootstrap over servers (2,000 resamples).
  - **Top-k agreement.** For each server, the overlap of the k=3 most-confused pairs (|∩| / 3),
    averaged over servers, plus the share of servers whose single worst pair matches.
  - Per-server accuracy compared with an exact McNemar test, which is paired because the queries
    are the same.

## 7. RQ-c: generated vs hand-written requests

- **Servers.** Chosen now, for distinct domains a person can write requests for without
  studying them:
  - `io-github-modelcontextprotocol-server-filesystem` (files)
  - `ai-adeu-adeu` (documents and email)
  - `ai-mcpcap-mcpcap` (packet captures)
  - `ai-thiri-chord-intelligence` (music harmony)

  `server-memory` is excluded, because its pilot results were already seen.
- **Requests.** At least 20 per server and at least 2 per tool, including some hard cases. Each
  is written by hand in `analysis/human_queries/`. The writer **must not look at the generated
  queries or at any benchmark result for these servers** before the file is committed.
- **Run.** Same model and settings as the benchmark.
- **Analysis.**
  - **Accuracy difference** (human minus generated primary set), with a 95% CI from a bootstrap
    over queries within server.
  - **Confused-pair overlap**: Jaccard of the sets of ordered pairs with ≥ 2 confusing trials in
    each set.

## 8. Multiple comparisons

Benjamini-Hochberg, q = 0.05, is applied separately within each family:

| Family | Tests |
|---|---|
| F1 | RQ-a: the 9 predictor tests of the primary model |
| F2 | RQ-b: the per-server McNemar accuracy tests, per model |
| F3 | Fix mode: the McNemar tests of every rewrite tried in the upstream-fix phase |

## 9. What counts as a null result (stated in advance)

- **RQ-a.** Null if neither `cos_sim` nor `name_jaccard` has a BH-adjusted q < 0.05 in the
  primary model.
- **RQ-b.** No evidence of transfer if the 95% CI of the primary ρ includes 0. The size of ρ is
  read as weak below 0.4, moderate from 0.4 to 0.7, and strong above 0.7, judged by the CI and
  not the point estimate.
- **RQ-c.**
  - **Equivalent** if the 95% CI of the accuracy difference lies within ±10 percentage points.
  - **Different** if it excludes 0.
  - **Inconclusive** otherwise.

All of these outcomes are publishable and will be reported in the same way.

## 10. Known limits, accepted in advance

- **One benchmark model, one date.** RQ-a describes Gemini 3.5 Flash-Lite's confusions, which is
  why RQ-b exists.
- **Few servers.** About 13 servers survive the exclusions, so server-level variance is poorly
  estimated and most pair cells are small.
- **Generated requests.** The requests are generated by the same model that is measured. RQ-c
  checks how much that matters.
- **First call only.** mispick scores the first tool call. A model that looks something up
  before acting, such as `search_nodes` before `delete_observations`, counts as confused. The
  pilot showed this happens.

## Deviations

None yet.
