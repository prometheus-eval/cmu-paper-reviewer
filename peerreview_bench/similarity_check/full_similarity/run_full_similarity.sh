#!/usr/bin/env bash
#
# Run the full_similarity pipeline.
#
# There are TWO scoring paths; this script can run either or both:
#
#   (A) Embedding path — compute_full_similarity_embedding.py
#        - Cosine similarity on Azure / Gemini text-embedding models
#        - ~2-3 minutes per backend (fast, cheap)
#        - Continuous score per pair → cosine distributions, ranking,
#          nearest-neighbor, etc.
#
#   (B) LLM-judge path — compute_full_similarity_llm.py
#        - Thinking-mode 4-way classification (same prompt as the
#          164-pair expert_annotation_similarity judge)
#        - Hours per model at --concurrency 16 (slow, expensive)
#        - Categorical label per pair → similar-rate, category mix, etc.
#        - **Resumable**: crashed/interrupted runs can be restarted and
#          will pick up where they left off.
#
# The two paths share the same pair set (every within-paper pair in the
# expert_annotation HF config) and the same analyze step is split into
# analyze_embedding.py and analyze_llm.py.
#
# Usage:
#   cd similarity_check/full_similarity
#   ./run_full_similarity.sh                 # BOTH paths + both analyses
#   ./run_full_similarity.sh embedding       # only embedding + analyze_embedding
#   ./run_full_similarity.sh llm             # only LLM + analyze_llm
#   ./run_full_similarity.sh analyze         # re-run both analyses on existing pairs
#   LIMIT=5 ./run_full_similarity.sh         # smoke test on the first 5 papers

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

# HF Hub Xet workaround — see peerreview_bench/load_data.py.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

OUTPUT_DIR="../../outputs/full_similarity"
mkdir -p "$OUTPUT_DIR"

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "Using --limit $LIMIT (smoke-test mode)"
fi

# Concurrency for the LLM-judge path. 16 is a safe-aggressive default.
# Each worker has its own retry-with-backoff on 429 / RESOURCE_EXHAUSTED.
CONCURRENCY_ARG="--concurrency ${CONCURRENCY:-16}"

# Embedding backends.
EMBEDDING_BACKENDS=(
  "litellm_proxy/azure_ai/text-embedding-3-large"
  "litellm_proxy/gemini/gemini-embedding-001"
)

# LLM-as-judge models. Same three models as the expert_annotation_similarity
# baseline so the paper can compare judged rates across subset sizes.
LLM_MODELS=(
  "litellm_proxy/azure_ai/gpt-5.4"
  "litellm_proxy/gemini/gemini-3.1-pro-preview"
  "litellm_proxy/anthropic/claude-opus-4-6"
)

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
RUN_EMBED=1
RUN_LLM=1
RUN_ANALYZE=1
if [ $# -gt 0 ]; then
  case "$1" in
    embedding|embed)   RUN_LLM=0 ;;
    llm|judge)         RUN_EMBED=0 ;;
    analyze|eval)      RUN_EMBED=0; RUN_LLM=0 ;;
    all)               ;;
    *) echo "Unknown stage: $1 (expected: embedding | llm | analyze | all)" >&2; exit 1 ;;
  esac
fi

# ---------------------------------------------------------------------------
# 1. Embedding compute
# ---------------------------------------------------------------------------
if [ "$RUN_EMBED" = "1" ]; then
  for BACKEND in "${EMBEDDING_BACKENDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  EMBED compute: $BACKEND"
    echo "========================================================================"
    python3 compute_full_similarity_embedding.py \
      --backend "$BACKEND" \
      --output-dir "$OUTPUT_DIR" \
      $LIMIT_ARG
  done
fi

# ---------------------------------------------------------------------------
# 2. LLM-judge compute (resumable — safe to rerun after a crash)
# ---------------------------------------------------------------------------
if [ "$RUN_LLM" = "1" ]; then
  for MODEL in "${LLM_MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  LLM judge compute: $MODEL (resumable)"
    echo "========================================================================"
    python3 compute_full_similarity_llm.py \
      --model "$MODEL" \
      --output-dir "$OUTPUT_DIR" \
      $LIMIT_ARG \
      $CONCURRENCY_ARG
  done
fi

# ---------------------------------------------------------------------------
# 3. Analyze
# ---------------------------------------------------------------------------
if [ "$RUN_ANALYZE" = "1" ]; then
  echo ""
  echo "========================================================================"
  echo "  Analyze embedding pairs (pairs_embedding_*.jsonl in $OUTPUT_DIR)"
  echo "========================================================================"
  for PAIRS in "$OUTPUT_DIR"/pairs_embedding_*.jsonl; do
    [ -e "$PAIRS" ] || continue
    echo ""
    echo "--- $PAIRS ---"
    python3 analyze_embedding.py "$PAIRS" || echo "  (analyze_embedding failed for $PAIRS)"
  done

  echo ""
  echo "========================================================================"
  echo "  Analyze LLM-judge pairs (pairs_llm_*.jsonl in $OUTPUT_DIR)"
  echo "========================================================================"
  for PAIRS in "$OUTPUT_DIR"/pairs_llm_*.jsonl; do
    [ -e "$PAIRS" ] || continue
    echo ""
    echo "--- $PAIRS ---"
    python3 analyze_llm.py "$PAIRS" || echo "  (analyze_llm failed for $PAIRS)"
  done
fi

echo ""
echo "=== Done ==="
echo "Outputs are in $OUTPUT_DIR/"
