#!/usr/bin/env bash
#
# Run the OpenHands agent meta-reviewer over every target model.
#
# Each conversation produces both axis labels (the agent's own judgment)
# and a prediction_of_expert_judgments (10-class) in a single pass.
#
# Agent runs are slow and expensive — one paper is one OpenHands
# conversation with ~10-30 tool calls. Use LIMIT aggressively during
# iteration.
#
# Usage:
#   cd metareview_bench/expert_annotation_meta_review
#   PAPER_ROOT=../../papers ./run_meta_review_agent.sh          # all 3 models
#   PAPER_ROOT=../../papers LIMIT=3 ./run_meta_review_agent.sh  # 3 papers each

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

if [ -z "${PAPER_ROOT:-}" ]; then
  echo "ERROR: set PAPER_ROOT to the directory with paper{N}/ subdirectories" >&2
  exit 1
fi

# Agent baseline only makes sense with models that can reason over files + images
ALL_MODELS=(
  "litellm_proxy/azure_ai/gpt-5.4"
  "litellm_proxy/gemini/gemini-3.1-pro-preview"
  "litellm_proxy/anthropic/claude-opus-4-6"
)

MODELS=("$@")
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("${ALL_MODELS[@]}")
fi

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "Using --limit $LIMIT"
fi

echo "Models: ${MODELS[*]}"
echo "Paper root: $PAPER_ROOT"

for MODEL in "${MODELS[@]}"; do
  echo ""
  echo "========================================================================"
  echo "  $MODEL [agent]"
  echo "========================================================================"
  python3 run_meta_review_agent.py \
    --model-name "$MODEL" \
    --paper-root "$PAPER_ROOT" \
    $LIMIT_ARG
done

echo ""
echo "=== Done ==="
echo "Results are in ../results/"
