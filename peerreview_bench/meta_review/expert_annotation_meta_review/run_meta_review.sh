#!/usr/bin/env bash
#
# Run the meta-reviewer benchmark over every target model × every prompt mode.
#
# Targets: 2 baselines + 7 LiteLLM models × 2 prompt modes (axis, tenclass) = 18 runs
#
# Requirements:
#   - `LITELLM_API_KEY` env var (or `./key_litellm.txt`)
#   - The `meta_reviewer` HF config on prometheus-eval/peerreview-bench
#
# Usage:
#   cd meta_review/expert_annotation_meta_review
#   ./run_meta_review.sh                         # every model × both modes
#   ./run_meta_review.sh axis                    # only axis mode
#   ./run_meta_review.sh tenclass                # only tenclass mode
#   LIMIT=20 ./run_meta_review.sh                # smoke test: 20 rows each
#   ./run_meta_review.sh majority random         # only those two models (both modes)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

# Full list: 2 baselines + 7 LiteLLM-proxied models
ALL_MODELS=(
  "majority"
  "random"
  "litellm_proxy/azure_ai/gpt-5.4"
  "litellm_proxy/azure_ai/grok-4-1-fast-reasoning"
  "litellm_proxy/azure_ai/Kimi-K2.5"
  "litellm_proxy/gemini/gemini-3.1-pro-preview"
  "litellm_proxy/anthropic/claude-opus-4-6"
  "litellm_proxy/fireworks_ai/accounts/fireworks/models/qwen3p6-plus"
)

# -----------------------------------------------------------------------
# Parse mode and model list from args
# -----------------------------------------------------------------------
MODES=("axis" "tenclass")
MODELS=()

if [ $# -gt 0 ]; then
  if [ "$1" = "axis" ] || [ "$1" = "tenclass" ]; then
    MODES=("$1")
    shift
  fi
  if [ $# -gt 0 ]; then
    MODELS=("$@")
  fi
fi
if [ ${#MODELS[@]} -eq 0 ]; then
  MODELS=("${ALL_MODELS[@]}")
fi

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "Using --limit $LIMIT"
fi

CONCURRENCY_ARG="--concurrency ${CONCURRENCY:-16}"

echo "Modes : ${MODES[*]}"
echo "Models: ${MODELS[*]}"
echo "Concurrency: ${CONCURRENCY:-16}"

for MODE in "${MODES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  $MODEL  (mode=$MODE)"
    echo "========================================================================"
    python3 run_meta_review.py \
      --model "$MODEL" \
      --prompt-mode "$MODE" \
      $LIMIT_ARG $CONCURRENCY_ARG
  done
done

echo ""
echo "=== Done ==="
echo "Outputs are in ../../outputs/expert_annotation_meta_review/"
