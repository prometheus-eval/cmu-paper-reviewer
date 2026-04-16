#!/usr/bin/env bash
#
# Run the 238-pair similarity_check benchmark.
#
# Two kinds of baselines are scored:
#
#   1. Embedding-based: cosine similarity between the two review-item texts
#      (see baselines/embedding_classifier.py).
#   2. LLM-as-a-judge:  4-way classification asking the model to tag the
#      pair as one of {near-paraphrase, convergent, topical neighbor,
#      unrelated} (see baselines/llm_classifier.py).
#
# Both kinds are routed through the CMU LiteLLM proxy — the `litellm_proxy/`
# prefix tells embeddings.py / litellm_client.py to use the proxy.
#
# Requirements:
#   - LITELLM_API_KEY env var, or <peerreview_bench>/api_key/litellm.txt
#   - The `similarity_check` HF config pushed to prometheus-eval/peerreview-bench
#   - The `reviewer` HF config (for file_refs) and `submitted_papers` config
#     (for raw image bytes), used by the LLM baseline's multimodal path.
#
# Usage:
#   cd similarity_check
#   ./run_similarity.sh                # run every embedding + every LLM baseline
#   LIMIT=5 ./run_similarity.sh        # smoke test: only the first 5 pairs each
#   ./run_similarity.sh embeddings     # only embedding baselines
#   ./run_similarity.sh llm            # only LLM baselines
#   ./run_similarity.sh evaluate       # only re-score existing outputs
#
# All outputs land in ../outputs/similarity_check/ (shared across baselines).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export LITELLM_BASE_URL="${LITELLM_BASE_URL:-https://cmu.litellm.ai}"

# HF Hub Xet workaround — see ../load_data.py for the full context.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
# HF_HUB_ETAG_TIMEOUT governs HEAD (metadata) requests. Default is 10s,
# which is too short for the large submitted_papers config.
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

OUTPUT_DIR="../../outputs/similarity_check"
mkdir -p "$OUTPUT_DIR"

LIMIT_ARG=""
if [ -n "${LIMIT:-}" ]; then
  LIMIT_ARG="--limit $LIMIT"
  echo "Using --limit $LIMIT (smoke-test mode)"
fi

# Concurrent LLM workers. 16 is a safe-aggressive default; bump higher
# (CONCURRENCY=64 ./run_similarity.sh llm) if you have headroom on the
# LiteLLM proxy and your provider quotas allow it. Each worker has its
# own retry-with-backoff on 429 / RESOURCE_EXHAUSTED.
CONCURRENCY_ARG="--concurrency ${CONCURRENCY:-16}"
echo "Using $CONCURRENCY_ARG"

# Embedding backends: both routed via LiteLLM proxy.
EMBEDDING_BACKENDS=(
  "litellm_proxy/azure_ai/text-embedding-3-large"
  "litellm_proxy/gemini/gemini-embedding-001"
)

# LLM-as-judge models: routed via LiteLLM proxy. Thinking mode is enabled
# per-model inside llm_classifier.py (reasoning_effort="high" plus the
# Anthropic-specific `thinking` kwarg for Claude).
LLM_MODELS=(
  "litellm_proxy/azure_ai/gpt-5.4"
  "litellm_proxy/gemini/gemini-3.1-pro-preview"
  "litellm_proxy/anthropic/claude-opus-4-6"
)

# ---------------------------------------------------------------------------
# Dispatch: which stages to run
# ---------------------------------------------------------------------------
RUN_EMBED=1
RUN_LLM=1
RUN_EVAL=1
if [ $# -gt 0 ]; then
  case "$1" in
    embeddings|embed) RUN_LLM=0 ;;
    llm|judge)        RUN_EMBED=0 ;;
    evaluate|eval)    RUN_EMBED=0; RUN_LLM=0 ;;
    all)              ;;
    *) echo "Unknown stage: $1 (expected: embeddings | llm | evaluate | all)" >&2; exit 1 ;;
  esac
fi

# ---------------------------------------------------------------------------
# 1. Embedding baselines
# ---------------------------------------------------------------------------
if [ "$RUN_EMBED" = "1" ]; then
  for BACKEND in "${EMBEDDING_BACKENDS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  Embedding baseline: $BACKEND"
    echo "========================================================================"
    python3 baselines/embedding_classifier.py \
      --backend "$BACKEND" \
      --output-dir "$OUTPUT_DIR" \
      $LIMIT_ARG
  done
fi

# ---------------------------------------------------------------------------
# 2. LLM-as-judge baselines (4-way classification, thinking mode on)
# ---------------------------------------------------------------------------
if [ "$RUN_LLM" = "1" ]; then
  # Pre-warm the HF local cache for submitted_papers before spawning the
  # per-model subprocesses. Each subprocess loads submitted_papers
  # independently; if the cache isn't populated yet, the first subprocess
  # pays the full ~2 GB download cost and sometimes times out, while the
  # later subprocesses succeed on the warm cache. Pre-warming here means
  # every subprocess starts from the same cached state, which matters for
  # the FIRST model in the loop (gpt-5.4 today) — without this step, its
  # images get silently disabled by the in-process retry fallback.
  #
  # submitted_papers is EXEMPT from HF_FORCE_REDOWNLOAD because it's
  # ~2 GB of stable-schema blob storage — the HF hub cache handles it
  # correctly once populated.
  echo ""
  echo "========================================================================"
  echo "  Pre-warming HF cache: submitted_papers (for image bytes)"
  echo "========================================================================"
  python3 - <<'PYEOF'
import os, sys, time
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')
sys.path.insert(0, '../..')
from load_data import load_submitted_papers
t0 = time.time()
last_err = None
# 5 attempts with exponential backoff capped at 5 min, total worst-case
# wait ~10 min. Patient enough to survive most transient HF hiccups and
# cold-cache cold-starts without giving up.
BACKOFFS = [30, 60, 120, 240, 300]
for attempt in range(5):
    try:
        h = load_submitted_papers()
        print(f'Cached {len(h)} file blobs in {time.time()-t0:.1f}s', flush=True)
        break
    except Exception as e:
        last_err = e
        wait = BACKOFFS[attempt]
        print(f'Attempt {attempt+1}/5 failed ({type(e).__name__}: {e}); '
              f'retrying in {wait}s...', flush=True)
        if attempt < 4:
            time.sleep(wait)
else:
    print(f'WARNING: could not pre-warm submitted_papers '
          f'({type(last_err).__name__}: {last_err}). '
          f'LLM baselines will fall back to text-only when image loading fails.',
          flush=True)
PYEOF

  for MODEL in "${LLM_MODELS[@]}"; do
    echo ""
    echo "========================================================================"
    echo "  LLM judge baseline: $MODEL (4-way, thinking mode, multimodal)"
    echo "========================================================================"
    python3 baselines/llm_classifier.py \
      --model "$MODEL" \
      --output-dir "$OUTPUT_DIR" \
      $LIMIT_ARG \
      $CONCURRENCY_ARG
  done
fi

# ---------------------------------------------------------------------------
# 3. Aggregate metrics
# ---------------------------------------------------------------------------
if [ "$RUN_EVAL" = "1" ]; then
  echo ""
  echo "========================================================================"
  echo "  Aggregating metrics (evaluate.py on every output in $OUTPUT_DIR)"
  echo "========================================================================"
  for OUT in "$OUTPUT_DIR"/embedding_*.json "$OUTPUT_DIR"/llm_judge_*.json; do
    [ -e "$OUT" ] || continue
    echo ""
    echo "--- $OUT ---"
    python3 evaluate.py "$OUT" || echo "  (evaluate failed for $OUT)"
  done
fi

echo ""
echo "=== Done ==="
echo "Outputs are in $OUTPUT_DIR/"
