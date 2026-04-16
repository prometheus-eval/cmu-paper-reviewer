#!/usr/bin/env bash
#
# Run the OpenHands agent meta-reviewer baseline over every multimodal
# target model × every prompt mode.
#
# Agent runs are slow and expensive — one item is one OpenHands
# conversation with ~10-30 tool calls. Use LIMIT aggressively during
# iteration.
#
# Usage:
#   cd meta_review/expert_annotation_meta_review
#   ./run_meta_review_agent.sh                              # 3 models × 2 modes
#   ./run_meta_review_agent.sh axis                         # only axis mode
#   ./run_meta_review_agent.sh tenclass                     # only tenclass mode
#   LIMIT=3 ./run_meta_review_agent.sh                      # 3 items each
#   PAPER_ROOT=/some/path ./run_meta_review_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

if [ -z "${PAPER_ROOT:-}" ]; then
  echo "ERROR: set PAPER_ROOT to the local directory holding raw paper trees" >&2
  echo "       (each subdirectory is an author folder with one subdir per paper)" >&2
  exit 1
fi

# Agent baseline only makes sense with models that can reason over files + images
ALL_MODELS=(
  "litellm_proxy/azure_ai/gpt-5.4"
  "litellm_proxy/gemini/gemini-3.1-pro-preview"
  "litellm_proxy/anthropic/claude-opus-4-6"
)

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

echo "Modes : ${MODES[*]}"
echo "Models: ${MODELS[*]}"
echo "Paper root: $PAPER_ROOT"

for MODE in "${MODES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  $MODEL  (mode=$MODE) [agent]"
    echo "========================================================================"
    python3 run_meta_review_agent.py \
      --model-name "$MODEL" \
      --prompt-mode "$MODE" \
      --paper-root "$PAPER_ROOT" \
      $LIMIT_ARG
  done
done

echo ""
echo "=== Done ==="
echo "Outputs are in ../../outputs/expert_annotation_meta_review/"
