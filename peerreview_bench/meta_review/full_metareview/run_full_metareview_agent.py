#!/usr/bin/env python3
"""
Agent meta-reviewer over ALL 85 papers (full expert_annotation set).

Thin wrapper around the expert_annotation_meta_review agent runner — uses
the same per-paper conversation design, same prompts, same output format.
The only difference is that it runs on all papers (not just the 27 overlap
papers in the meta_reviewer config) and outputs to a different directory.

Usage:
    python3 run_full_metareview_agent.py \
        --model-name litellm_proxy/anthropic/claude-opus-4-6 \
        --paper-root ../../papers --prompt-mode axis --limit 3
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_EA_META_DIR = _META_REVIEW_DIR / 'expert_annotation_meta_review'

for p in (_HERE, _EA_META_DIR, _META_REVIEW_DIR, _META_REVIEW_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the full agent runner — the only difference is the default output dir
from run_meta_review_agent import main as _agent_main  # noqa: E402
from run_meta_review_agent import DEFAULT_OUTPUT_ROOT as _EA_OUTPUT_ROOT  # noqa: E402

# Override the default output root for full_metareview
import run_meta_review_agent as _agent_mod  # noqa: E402
_agent_mod.DEFAULT_OUTPUT_ROOT = _META_REVIEW_DIR.parent / 'outputs' / 'full_metareview'

if __name__ == '__main__':
    _agent_main()
