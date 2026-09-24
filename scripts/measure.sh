#!/usr/bin/env bash
# Runs mispick for the GitHub Action. Kept as a script rather than inline YAML so it can be
# shellchecked and read.
set -euo pipefail

out() { echo "$1=$2" >> "$GITHUB_OUTPUT"; }

# How to invoke mispick. Overridable so the Action can be exercised against a local build
# (MISPICK_BIN="uv run mispick") instead of only against a published release.
if [ -n "${MISPICK_BIN:-}" ]; then
  read -r -a MISPICK <<< "$MISPICK_BIN"
else
  MISPICK=(uvx --from "${MISPICK_VERSION:-mispick}" mispick)
fi

# Decide the target.
target=()
if   [ -n "${MISPICK_SNAPSHOT:-}" ]; then target=(--snapshot "$MISPICK_SNAPSHOT")
elif [ -n "${MISPICK_CMD:-}" ];      then target=(--cmd "$MISPICK_CMD")
elif [ -n "${MISPICK_URL:-}" ];      then target=(--url "$MISPICK_URL")
elif [ -n "${MISPICK_CONFIG:-}" ];   then target=(--config "$MISPICK_CONFIG")
else
  echo "::error title=mispick::No target. Set one of snapshot, cmd, url or config."
  exit 1
fi

# A model is required, and Ollama is not a realistic CI backend - a 4B model takes minutes per
# dozen calls on a shared runner. Rather than burn 30 minutes and time out, skip with a clear
# message and let the rest of the workflow pass.
model="${MISPICK_MODEL:-}"
if [ -z "$model" ]; then
  if   [ -n "${ANTHROPIC_API_KEY:-}" ]; then model="anthropic/claude-haiku-4-5-20251001"
  elif [ -n "${OPENAI_API_KEY:-}" ];    then model="openai/gpt-4.1-mini"
  fi
fi

case "$model" in
  ollama/*|"") needs_key=1 ;;
  *)           needs_key=0 ;;
esac

if [ -z "$model" ] || { [ "$needs_key" = 1 ] && [ -z "${OLLAMA_HOST:-}" ]; }; then
  {
    echo "## mispick: skipped"
    echo
    echo "No cloud model key was available, so no measurement ran."
    echo
    echo "mispick needs a model to measure tool selection. The default backend is a local"
    echo "Ollama model, which is too slow for a shared CI runner - a small model takes"
    echo "seconds per selection call, and a realistic run makes hundreds."
    echo
    echo "To enable this check, set one repository secret and pass it in:"
    echo
    echo '```yaml'
    echo "      - uses: mispick/mispick@v0.1.0"
    echo "        with:"
    echo "          snapshot: tools.json"
    echo "          model: anthropic/claude-haiku-4-5-20251001"
    echo "        env:"
    echo "          ANTHROPIC_API_KEY: \${{ secrets.ANTHROPIC_API_KEY }}"
    echo '```'
    echo
    echo "Or point \`OLLAMA_HOST\` at a self-hosted runner that has Ollama."
  } > mispick-comment.md
  out skipped true
  out score ""
  out accuracy ""
  out report ""
  echo "::notice title=mispick::skipped - no model available (see the job summary)"
  exit 0
fi

seed=()
[ -n "${MISPICK_SEED:-}" ] && seed=(--seed "$MISPICK_SEED")

echo "::group::mispick run"
"${MISPICK[@]}" run \
  "${target[@]}" \
  --model "$model" \
  --queries "${MISPICK_N:-8}" \
  --runs "${MISPICK_K:-3}" \
  --temperature "${MISPICK_TEMPERATURE:-0}" \
  "${seed[@]}" \
  --cache-dir "${MISPICK_CACHE_DIR:-.mispick}" \
  --format json --out mispick-report.json
echo "::endgroup::"

# Re-render the same run rather than measuring twice.
"${MISPICK[@]}" report mispick-report.json \
  --format md --out mispick-comment.md
"${MISPICK[@]}" report mispick-report.json \
  --format html --out mispick-report.html

if [ -n "${MISPICK_BADGE:-}" ]; then
  # stdout is the terminal report, which we already have; only the badge file is wanted here.
  "${MISPICK[@]}" report mispick-report.json --badge "$MISPICK_BADGE" > /dev/null
fi

score=$(python3 -c 'import json;print(json.load(open("mispick-report.json"))["score"])')
accuracy=$(python3 -c 'import json;v=json.load(open("mispick-report.json"))["summary"]["accuracy"]["value"];print(v if v is not None else "")')

out skipped false
out score "$score"
out accuracy "$accuracy"
out report mispick-report.json
echo "::notice title=mispick::score $score"
