# analysis/

These scripts implement the study pre-registered in
[docs/analysis-plan.md](../docs/analysis-plan.md). None of them changes the benchmark or its
workflow.

```bash
uv sync --all-extras --group analysis      # statsmodels, matplotlib, fastembed (pinned)

# 1. freeze the finished benchmark run into the repo (refuses an unfinished run)
uv run python analysis/fetch_results.py --out analysis/data/benchmark-v1

# 2. pair-level predictors from the committed snapshots (no model calls)
uv run python analysis/features.py --snapshots benchmark/snapshots --out results

# 3. the pre-registered model, sensitivity checks, BH, and charts
uv run python analysis/analyze.py --features results/features.csv \
    --results analysis/data/benchmark-v1 --out results
```

**RQ-b, cross-model transfer** (`transfer.py`) takes three steps. Each step stops before the next
one costs anything:

```bash
uv run python analysis/transfer.py plan --model <provider/model>    # subset + token estimate
uv run python analysis/transfer.py run --model <provider/model> --yes   # only after approval
uv run python analysis/transfer.py analyze                          # writes results/transfer.json
```

- **No new questions.** `run` reuses the benchmark's frozen queries and refuses a server where
  any are missing, so the other model never generates questions of its own.
- **Resumable.** Like the benchmark, an interrupted `run` continues where it stopped.
- **Fixed in advance.** The model choice goes into a dated addendum in the analysis plan before
  the run.

**Outputs** go to `results/`:
- `features.csv` and `features_meta.json`
- `analysis.json`
- `effects.png`, `confusion_by_similarity.png` and `server_accuracy.png`

Every number quoted in a document comes from one of these files.

**Embedding model.** `features.py` loads `sentence-transformers/all-MiniLM-L6-v2` at a pinned
Hugging Face revision. It checks the model file's SHA-256 and refuses to run if the file differs.
`--embedder hash` is a stand-in for offline tests only, and the output says so.

**Tests.** `tests/test_analysis.py` runs the full pipeline on the test fixtures with the offline
`mock` model. Without the `analysis` group installed, the model-fitting tests are skipped.
