# Benchmark plan: Gemini free tier

Status: **Steps 1-3 done (2026-10-16). Next: add the GEMINI_API_KEY repo secret, push, and let the daily job run.**

Goal: measure the 16 committed snapshots in `benchmark/snapshots/` against one cloud model,
for $0, and publish a leaderboard only after a manual review.

## The numbers that shape the plan

**Model.** `gemini/gemini-3.5-flash-lite`, the cheapest Gemini model with function calling. On
the free tier, Google may use the requests for training. That is acceptable here because every
tool description in the corpus is already public.

**Quota (read from the AI Studio console on 2026-10-16):** Gemini 3.5 Flash-Lite allows
**15 RPM, 250K TPM and 500 RPD**. The key was verified working on the same day, and
`gemini-3.5-flash-lite` is available to it.

**Workload at N=8, K=3.** A server costs `tools × 24` selection calls, plus about `tools + 1`
calls to generate queries.

| Group | Servers | Tools | Calls (approx.) |
|---|---|---|---|
| Small servers | 13 | 152 | 3,820 |
| `co-workix-mcp` | 1 | 64 | 1,600 (each call sends ~20k tokens) |
| `sequential-thinking` | 1 | 1 | ~30 (a single tool, so nothing can be confused) |
| `affiliate-networks-mcp` | 1 | 682 | 0 (skipped by `--max-tools 120`, reported by size only) |
| **Total** | | | **~5,450** |

The daily quota is the bottleneck. At 500 RPD the run takes **about 11 days**, or about 7.5 days
at K=2. One day's 500 calls take about 35 minutes at 15 RPM.

**Tokens per minute.** One `co-workix` call sends ~20k tokens. A 250k TPM cap therefore allows
only about 12 of its calls per minute, so the pacing must respect both RPM and TPM.
To stay under the limits, pace at 14 RPM and 230K TPM, and pass `--budget-calls 480` per day.

## What mispick lacks today

`measure.py` currently assumes one uninterrupted run, which a free-tier run cannot be:

1. **No resume.** Every run re-measures every server. Only the generated queries are cached.
2. **No per-trial checkpoint.** `co-workix` needs more calls than one day's quota, so it must be
   able to stop partway and continue the next day.
3. **No rate limiting.** Nothing paces requests against RPM and TPM.
4. **No handling of 429 errors.** A per-minute 429 means wait and retry. A daily-quota 429 means
   save progress and exit cleanly. Neither case should ever be recorded as "the model chose
   nothing".
5. **Thinking tokens not verified.** The Gemini backend sends no thinking control. If Flash-Lite
   thinks by default, that adds latency and output tokens. Check this in the pilot.

## Steps

### Step 1: Setup (you, ~5 minutes)
- Put the key in `~/.config/mispick/env` as `GEMINI_API_KEY=...` and run `chmod 600` on it.
  Never paste the key into chat.
- Open aistudio.google.com/rate-limit, note the real RPM, TPM and RPD for Flash-Lite, and tell
  Claude the numbers.
- Make sure billing is **not** enabled on that Google Cloud project. Enabling it switches the
  project to paid rates.

### Step 2: Make the run resumable (Claude, ~1 session)
- `measure.py` skips servers that already have a finished result file.
- Selection trials are written to `benchmark/results/.trials/<server>.jsonl` as they finish.
  When a run restarts, it replays those trials instead of making the calls again.
- A pacer limits requests per minute (`--rpm`) and tokens per minute (`--tpm`) at the same time.
- 429 handling: a per-minute 429 backs off and retries, honouring `Retry-After`. A daily-quota
  429 flushes progress, prints "quota reached, rerun tomorrow" and exits with code 0.
- Add an `--only <slug>` flag to measure one server.
- Add a `--budget-calls` flag that stops cleanly after a given number of calls, so one day's run
  stays under the quota.
- Tests run on the mock backend, offline. They cover resume, the pacer, both kinds of 429, and
  "a quota stop never counts as a miss".
- Commit the work.

### Step 3: Pilot (~250 calls, day 1)
Run one small reference server, `server-memory` (9 tools):
```
set -a; source ~/.config/mispick/env; set +a
uv run python benchmark/measure.py --model gemini/gemini-3.5-flash-lite \
  --queries 8 --runs 3 --temperature 0 --seed 1 --max-tools 120 \
  --only io-github-modelcontextprotocol-server-memory
```
The pilot checks five things:
- Real seconds per call and real requests per minute.
- Whether usage reports thinking tokens. If it does, test `reasoning_effort` and keep it only
  if Gemini accepts it.
- That `prompt_tokens` matches the token estimate, so no truncation.
- Generated query quality: read the YAML cache by eye.
- Repeatability: rerun the server once, and K=3 should show high stability at temperature 0.
  Gemini does not honour `seed`, and reports already say so.

**Stop and review together here** before spending the rest of the quota.

### Step 4: Small servers (days 2 to ~5)
Run the same command each day without `--only`, and with `--budget-calls` set just under the
daily quota. Each daily run resumes where the previous one stopped. Expected: the 13 small
servers and `sequential-thinking` finish first.

### Step 5: `co-workix-mcp` (days ~5 to ~8)
This server takes its 1,600 calls over two or more days, using the trial checkpoint. Watch the
TPM pacing on it.

### Step 6: Review the results (you and Claude)
- Check each server's top confused pair to see whether it makes sense or is a generator artifact.
- Rerun any server with an error rate above 0. Don't publish errors as misses.
- Every report must state the model, N, K, temperature, date, and "seed not honoured".

### Step 7: Publish (only on your explicit go-ahead)
- `build_site.py` builds the leaderboard. Review it locally first, because it names other
  people's projects.
- Enable GitHub Pages and run the `pages` workflow.
- Update the README with a link and one headline finding.
- Then use it for promotion: the LinkedIn post, Reddit, and the awesome-list PR already drafted
  in `docs/awesome-list-entries.md`.

## If it takes too long

These options are listed in the order to try them:
1. Drop to **K=2**. That cuts about a third of the calls and still measures stability.
2. Publish the small servers first and add `co-workix` later.
3. Spend about $1 on the paid tier, for `co-workix` only.

## Later, optional
- Add a second model through GitHub Models (free with the Student Pack) for a two-model
  comparison. The `compatible/<model>` backend should work, but test it first.
- Add prompt caching to the Anthropic and OpenAI backends, which makes paid runs 5 to 10x
  cheaper.

## Pilot findings (2026-10-16, `server-memory`)

- Gemini rejected every request that carried `seed`. Fixed: seed is only sent where supported.
- A laptop sleep cost 37 trials and Google's "high demand" 503 cost 4. Fixed: remote
  connection drops, timeouts and 5xx are retried.
- Vague generated requests ("insert these nodes") sent 13 of 30 create requests to
  `read_graph`. Fixed: requests must spell out concrete details.
- No thinking tokens (10 completion tokens per call), and `prompt_tokens` matched the
  estimate (956 vs 991), so nothing was truncated.
- The daily quota ran out at ~450 calls including retries, and the run stopped cleanly with
  190 trials saved. Retries count against the quota, which is why the budget is 480, not 500.
- With concrete requests, 57% of 190 trials were correct. The main pattern: for edits and
  deletes, Flash-Lite calls `search_nodes` first, looking an entity up before changing it.
  mispick scores the first tool call, so this counts as a miss. That is a real behaviour of
  this model on this server, but the leaderboard text should say "first tool chosen".
