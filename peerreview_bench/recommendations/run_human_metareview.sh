#!/usr/bin/env bash
# Run GPT-5.4 meta-reviewer on human review items for the 55 gap papers.
# Uses api_key and base_url from peerreview_bench/api_key/.
#
# Usage:
#   ./run_human_metareview.sh              # full run
#   ./run_human_metareview.sh --limit 3    # test on 3 papers first

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python3 recommendations/run_human_metareview.py "$@"
