#!/usr/bin/env bash
# Measures the base branch's snapshot and rewrites the PR comment as a comparison.
set -euo pipefail

if [ -n "${MISPICK_BIN:-}" ]; then
  read -r -a MISPICK <<< "$MISPICK_BIN"
else
  MISPICK=(uvx --from "${MISPICK_VERSION:-mispick}" mispick)
fi

if [ -z "${MISPICK_SNAPSHOT:-}" ]; then
  echo "::notice title=mispick::compare-base needs a snapshot path; skipping the comparison"
  exit 0
fi

# The base revision may not have the snapshot at all - a new server, for instance.
if ! git cat-file -e "${BASE_SHA}:${MISPICK_SNAPSHOT}" 2>/dev/null; then
  echo "::notice title=mispick::${MISPICK_SNAPSHOT} does not exist on the base revision; nothing to compare"
  exit 0
fi

git show "${BASE_SHA}:${MISPICK_SNAPSHOT}" > mispick-base-snapshot.json

seed=()
[ -n "${MISPICK_SEED:-}" ] && seed=(--seed "$MISPICK_SEED")

echo "::group::mispick run (base)"
"${MISPICK[@]}" run \
  --snapshot mispick-base-snapshot.json \
  --model "$MISPICK_MODEL" \
  --queries "${MISPICK_N:-8}" \
  --runs "${MISPICK_K:-3}" \
  --temperature "${MISPICK_TEMPERATURE:-0}" \
  "${seed[@]}" \
  --cache-dir "${MISPICK_CACHE_DIR:-.mispick}" \
  --format json --out mispick-base-report.json
echo "::endgroup::"

"${MISPICK[@]}" compare \
  mispick-base-report.json mispick-report.json --out mispick-comment.md
