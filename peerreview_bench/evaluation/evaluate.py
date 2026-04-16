#!/usr/bin/env python3
"""
PeerReview Bench unified evaluation entry point.

Runs the full pipeline:
  1. (optional) Prepare papers — download from HF
  2. (optional) Generate reviews — run AI agent on each paper
  3. Parse reviews — extract structured items from markdown
  4. Evaluate recall — coverage of fully-good human items
  5. Evaluate precision — quality of AI items via meta-review

Users can also run individual steps via the component scripts, or provide
their own review items JSON (BYOJ mode) and skip steps 2-3.

Usage:
    # Full pipeline: generate + evaluate
    python3 evaluate.py \
        --model-name litellm_proxy/anthropic/claude-opus-4-6 \
        --paper-root papers/ \
        --limit 5

    # BYOJ: bring your own review items JSON, just evaluate
    python3 evaluate.py \
        --byoj \
        --paper-root papers/ \
        --limit 5

    # Only recall (skip precision)
    python3 evaluate.py --paper-root papers/ --byoj --recall-only

    # Only precision (skip recall)
    python3 evaluate.py --paper-root papers/ --byoj --precision-only
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_BENCH_DIR))


def main():
    parser = argparse.ArgumentParser(
        description='PeerReview Bench: evaluate an AI reviewer'
    )

    # Mode
    parser.add_argument('--byoj', action='store_true',
                        help='Bring Your Own JSON: skip review generation, '
                             'use review_items_*.json already in paper dirs')
    parser.add_argument('--recall-only', action='store_true',
                        help='Only run recall evaluation (skip precision)')
    parser.add_argument('--precision-only', action='store_true',
                        help='Only run precision evaluation (skip recall)')

    # Paper prep
    parser.add_argument('--paper-root', type=Path,
                        default=_BENCH_DIR / 'papers',
                        help='Root dir with paper{N}/ subdirectories')
    parser.add_argument('--prepare', action='store_true',
                        help='Download papers from HF (runs download_papers.py)')

    # Review generation
    parser.add_argument('--model-name', type=str, default=None,
                        help='Model name for AI reviewer agent')
    parser.add_argument('--max-items', type=int, default=5,
                        help='Max review items per paper (default 5)')
    parser.add_argument('--criteria-preset', type=str, default='nature',
                        choices=('nature', 'neurips'),
                        help='Evaluation criteria preset')

    # Judge configuration
    parser.add_argument('--similarity-model', type=str,
                        default='litellm_proxy/anthropic/claude-opus-4-6',
                        help='LLM judge for recall (similarity)')
    parser.add_argument('--judge-model', type=str,
                        default='litellm_proxy/anthropic/claude-opus-4-6',
                        help='LLM judge for precision (meta-review)')
    parser.add_argument('--judge-mode', type=str, default='llm',
                        choices=('llm', 'agent'),
                        help='Meta-review judge mode')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)

    # General
    parser.add_argument('--limit', type=int, default=None,
                        help='Only evaluate first N papers')
    parser.add_argument('--output-dir', type=Path,
                        default=_BENCH_DIR / 'outputs' / 'evaluation',
                        help='Output directory for results')

    args = parser.parse_args()

    t0 = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Prepare papers ----
    if args.prepare:
        print('=' * 60)
        print('Step 1: Downloading papers from HuggingFace')
        print('=' * 60)
        from prepare_papers import prepare_papers
        prepare_papers(
            output_dir=args.paper_root,
            limit=args.limit,
            skip_existing=True,
        )

    # ---- Step 2: Generate reviews (skip if BYOJ) ----
    if not args.byoj and args.model_name:
        print('\n' + '=' * 60)
        print('Step 2: Generating reviews with AI agent')
        print('=' * 60)
        from generate_reviews import generate_reviews
        generate_reviews(
            paper_root=args.paper_root,
            model_name=args.model_name,
            max_items=args.max_items,
            criteria_preset=args.criteria_preset,
            limit=args.limit,
            skip_existing=args.byoj,
        )

    # ---- Step 3: Parse reviews ----
    if not args.byoj:
        print('\n' + '=' * 60)
        print('Step 3: Parsing review markdowns into structured items')
        print('=' * 60)
        from parse_review import load_review_items
        paper_dirs = sorted(args.paper_root.glob('paper*'))
        n_parsed = 0
        for pd in paper_dirs:
            items = load_review_items(pd, model_name=args.model_name)
            if items:
                n_parsed += 1
        print(f'  Parsed reviews for {n_parsed} papers')

    # ---- Step 4: Recall ----
    recall_result = None
    if not args.precision_only:
        print('\n' + '=' * 60)
        print('Step 4: Evaluating RECALL (coverage of human rubric items)')
        print('=' * 60)
        from evaluate_recall import main as recall_main
        recall_output = args.output_dir / 'recall.json'
        # Build the args for evaluate_recall
        recall_args = [
            '--paper-root', str(args.paper_root),
            '--similarity-model', args.similarity_model,
            '--concurrency', str(args.concurrency),
            '--temperature', str(args.temperature),
        ]
        if args.model_name:
            recall_args += ['--model-name', args.model_name]
        if args.byoj:
            recall_args += ['--byoj']
        if args.limit:
            recall_args += ['--limit', str(args.limit)]
        recall_args += ['--output', str(recall_output)]

        # Re-parse args for the recall module
        sys.argv = ['evaluate_recall.py'] + recall_args
        recall_main()

    # ---- Step 5: Precision ----
    precision_result = None
    if not args.recall_only:
        print('\n' + '=' * 60)
        print('Step 5: Evaluating PRECISION (quality of AI items via meta-review)')
        print('=' * 60)
        from evaluate_precision import main as precision_main
        precision_output = args.output_dir / 'precision.json'
        precision_args = [
            '--paper-root', str(args.paper_root),
            '--judge-model', args.judge_model,
            '--judge-mode', args.judge_mode,
            '--concurrency', str(args.concurrency),
            '--temperature', str(args.temperature),
        ]
        if args.model_name:
            precision_args += ['--model-name', args.model_name]
        if args.byoj:
            precision_args += ['--byoj']
        if args.limit:
            precision_args += ['--limit', str(args.limit)]
        precision_args += ['--output', str(precision_output)]

        sys.argv = ['evaluate_precision.py'] + precision_args
        precision_main()

    # ---- Summary ----
    elapsed = time.time() - t0
    print('\n' + '=' * 60)
    print('PEERREVIEW BENCH EVALUATION COMPLETE')
    print('=' * 60)

    # Load results if they were saved
    recall_path = args.output_dir / 'recall.json'
    precision_path = args.output_dir / 'precision.json'

    if recall_path.exists():
        r = json.loads(recall_path.read_text())
        print(f'  Recall:    {r.get("overall_recall", 0):.2%} '
              f'({r.get("total_covered", 0)}/{r.get("total_rubric_items", 0)} rubric items covered)')

    if precision_path.exists():
        p = json.loads(precision_path.read_text())
        print(f'  Precision: {p.get("precision", 0):.2%} '
              f'({p.get("n_fully_good", 0)}/{p.get("n_items", 0)} AI items fully good)')

    if recall_path.exists() and precision_path.exists():
        r = json.loads(recall_path.read_text())
        p = json.loads(precision_path.read_text())
        recall = r.get('overall_recall', 0)
        precision = p.get('precision', 0)
        if recall + precision > 0:
            f1 = 2 * recall * precision / (recall + precision)
            print(f'  F1:        {f1:.2%}')

    print(f'\n  Total time: {elapsed:.0f}s')
    print(f'  Results in: {args.output_dir}/')


if __name__ == '__main__':
    main()
