#!/usr/bin/env bash
#
# Agent meta-reviewer over the FULL expert_annotation HF config.
#
# 3 agent models × 2 prompt modes = 6 runs.
# Agent runs are slow; use LIMIT aggressively while iterating.
#
# Usage:
#   cd meta_review/full_metareview
#   PAPER_ROOT=/path ./run_full_metareview_agent.sh
#   PAPER_ROOT=/path ./run_full_metareview_agent.sh axis
#   LIMIT=3 PAPER_ROOT=/path ./run_full_metareview_agent.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

if [ -z "${PAPER_ROOT:-}" ]; then
  echo "ERROR: set PAPER_ROOT" >&2
  exit 1
fi

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
fi

echo "Modes : ${MODES[*]}"
echo "Models: ${MODELS[*]}"
echo "Paper root: $PAPER_ROOT"

for MODE in "${MODES[@]}"; do
  for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  [full/agent] $MODEL  (mode=$MODE)"
    echo "========================================================================"
    python3 run_full_metareview_agent.py \
      --model-name "$MODEL" \
      --prompt-mode "$MODE" \
      --paper-root "$PAPER_ROOT" \
      $LIMIT_ARG
  done
done

echo ""
echo "=== Done ==="
echo "Outputs are in ../../outputs/full_metareview/"
