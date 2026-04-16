#!/usr/bin/env bash
# Run the full PeerReview Bench analysis pipeline.
#
# Data is loaded from the HuggingFace dataset `prometheus-eval/peerreview-bench`
# (config `expert_annotation`, split `eval`). Requires the dataset to have
# been pushed — there is no local-JSON fallback.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# HF Hub Xet workaround — see ../load_data.py for the full context.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

OUTPUTS_ROOT="../outputs/analysis"
mkdir -p "$OUTPUTS_ROOT"

echo "=== Primary paired analysis ==="
python3 peerreview_analysis.py --output-dir "$OUTPUTS_ROOT/analysis_output"

echo ""
echo "=== Per-paper paired analysis ==="
python3 peerreview_analysis_per_paper.py --output-dir "$OUTPUTS_ROOT/analysis_output_per_paper"

echo ""
echo "=== Logistic GLMM sensitivity analysis ==="
python3 peerreview_analysis_glmm.py --output-dir "$OUTPUTS_ROOT/glmm_output"

echo ""
echo "=== Inter-rater reliability (primary vs secondary) ==="
python3 inter_rater_reliability.py --output-dir "$OUTPUTS_ROOT/irr_output"

echo ""
echo "=== Analysis pipeline complete ==="
