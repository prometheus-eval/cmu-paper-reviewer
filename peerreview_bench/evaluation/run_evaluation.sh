#!/usr/bin/env bash
#
# PeerReview Bench evaluation pipeline.
#
# Usage:
#   cd peerreview_bench/evaluation
#
#   # Full pipeline: generate reviews + evaluate (both metrics)
#   ./run_evaluation.sh
#
#   # Smoke test (5 papers)
#   LIMIT=5 ./run_evaluation.sh
#
#   # BYOJ (bring your own JSON — skip review generation)
#   BYOJ=1 ./run_evaluation.sh
#
#   # Custom model
#   MODEL=litellm_proxy/gemini/gemini-3.1-pro-preview ./run_evaluation.sh
#
#   # Only recall or only precision
#   RECALL_ONLY=1 ./run_evaluation.sh
#   PRECISION_ONLY=1 ./run_evaluation.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------

# AI reviewer model
MODEL="${MODEL:-litellm_proxy/anthropic/claude-opus-4-6}"

# Paper root
PAPER_ROOT="${PAPER_ROOT:-../papers}"

# Max review items per paper
MAX_ITEMS="${MAX_ITEMS:-5}"

# Evaluation criteria preset
CRITERIA="${CRITERIA:-nature}"

# LLM judge for similarity (recall)
SIMILARITY_MODEL="${SIMILARITY_MODEL:-litellm_proxy/anthropic/claude-opus-4-6}"

# LLM judge for meta-review (precision)
JUDGE_MODEL="${JUDGE_MODEL:-litellm_proxy/anthropic/claude-opus-4-6}"
JUDGE_MODE="${JUDGE_MODE:-llm}"

# Concurrency
CONCURRENCY="${CONCURRENCY:-16}"

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-../outputs/evaluation}"

# HF Hub workarounds
export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

# ---------------------------------------------------------------------------
# Build command
# ---------------------------------------------------------------------------

CMD=(python3 evaluate.py
     --paper-root "$PAPER_ROOT"
     --similarity-model "$SIMILARITY_MODEL"
     --judge-model "$JUDGE_MODEL"
     --judge-mode "$JUDGE_MODE"
     --concurrency "$CONCURRENCY"
     --output-dir "$OUTPUT_DIR"
)

# Model (only if not BYOJ)
if [ -z "${BYOJ:-}" ]; then
    CMD+=(--model-name "$MODEL"
          --max-items "$MAX_ITEMS"
          --criteria-preset "$CRITERIA")
else
    CMD+=(--byoj)
fi

# Prepare papers
if [ -n "${PREPARE:-}" ]; then
    CMD+=(--prepare)
fi

# Limit
if [ -n "${LIMIT:-}" ]; then
    CMD+=(--limit "$LIMIT")
fi

# Recall/precision only
if [ -n "${RECALL_ONLY:-}" ]; then
    CMD+=(--recall-only)
fi
if [ -n "${PRECISION_ONLY:-}" ]; then
    CMD+=(--precision-only)
fi

echo "========================================================================"
echo "  PeerReview Bench Evaluation"
echo "========================================================================"
echo "  Model:            $MODEL"
echo "  Paper root:       $PAPER_ROOT"
echo "  Max items:        $MAX_ITEMS"
echo "  Criteria:         $CRITERIA"
echo "  Similarity judge: $SIMILARITY_MODEL"
echo "  Meta-review judge: $JUDGE_MODEL ($JUDGE_MODE)"
echo "  Concurrency:      $CONCURRENCY"
echo "  Output:           $OUTPUT_DIR"
if [ -n "${BYOJ:-}" ]; then
    echo "  Mode:             BYOJ (bring your own JSON)"
fi
if [ -n "${LIMIT:-}" ]; then
    echo "  Limit:            $LIMIT papers"
fi
echo "========================================================================"
echo ""

"${CMD[@]}"
