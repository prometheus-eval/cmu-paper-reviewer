#!/usr/bin/env python3
"""
Evaluation script for agent meta-review predictions.

Reads agent output: agent_trajectories/{mode}/paper{N}/prediction.json
(nested JSON per paper: {reviewers: [{items: [...]}]}).

Flattens predictions into per-item rows, loads matching ground-truth
from HuggingFace, and runs the appropriate scorer (axis or tenclass).

Usage:
    # Score a single model's results
    python3 evaluate.py ../results/azure_ai__gpt_5_4/

    # Compare multiple models
    python3 evaluate.py ../results/*/

    # Only axis mode
    python3 evaluate.py ../results/*/ --mode axis
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent

for p in (_HERE, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import load_meta_reviewer  # noqa: E402
from prompts import TENCLASS_LABEL_TO_ID  # noqa: E402
from metrics import (  # noqa: E402
    evaluate_axis_predictions,
    evaluate_tenclass_predictions,
    format_mode_report,
)


# ---------------------------------------------------------------------------
# Ground-truth loading and indexing
# ---------------------------------------------------------------------------

def _load_ground_truth() -> Dict[Tuple[int, str, int], Dict[str, Any]]:
    """Load ground-truth rows from the meta_reviewer HF config (27 overlap
    papers, 908 items). Each row has both primary + secondary annotator
    labels and a 10-class label_id.

    Returns dict indexed by (paper_id, reviewer_id, item_number).
    """
    mr_rows = load_meta_reviewer()
    gt_index: Dict[Tuple[int, str, int], Dict[str, Any]] = {}

    for r in mr_rows:
        pid = int(r['paper_id'])
        rid = r['reviewer_id']
        inum = int(r.get('review_item_number') or 0)
        key = (pid, rid, inum)
        if key not in gt_index:
            gt_index[key] = dict(r)

    return gt_index


# ---------------------------------------------------------------------------
# Reviewer name normalization
# ---------------------------------------------------------------------------

def _normalize_reviewer_id(rid: str) -> str:
    """Map review-file stems to GT reviewer_ids."""
    if rid.startswith('Human_'):
        return rid
    if rid.lower().startswith('review_'):
        return rid
    low = rid.lower()
    if 'claude' in low:
        return 'Claude'
    if 'gpt' in low:
        return 'GPT'
    if 'gemini' in low:
        return 'Gemini'
    return rid


def _reviewer_type(rid: str) -> str:
    """Return 'Human' or 'AI' based on the (normalized) reviewer_id."""
    return 'Human' if rid.startswith('Human_') else 'AI'


# ---------------------------------------------------------------------------
# Prediction loading: Agent nested JSON format
# ---------------------------------------------------------------------------

def _load_agent_predictions(
    pred_files: List[Path],
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str, int]]]:
    """Load predictions from *_metareview.json files.
    Returns (predictions, keys) — flattened to per-item."""
    preds: List[Dict[str, Any]] = []
    keys: List[Tuple[int, str, int]] = []

    for pf in pred_files:
        try:
            data = json.loads(pf.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            print(f'  warn: skipping {pf}: {e}', file=sys.stderr)
            continue

        pid = int(data.get('paper_id', -1))
        # Extract paper_id from filename like slug_paper12_metareview.json
        if pid == -1:
            import re
            m = re.search(r'paper(\d+)', pf.name)
            if m:
                pid = int(m.group(1))

        for rev_block in data.get('reviewers', []):
            rid = _normalize_reviewer_id(rev_block.get('reviewer_id', ''))
            for item_block in rev_block.get('items', []):
                inum = int(item_block.get('item_number', 0))
                pred = dict(item_block)
                # Map prediction_of_expert_judgments → label/label_id
                # (metrics.py expects 'label' and 'label_id')
                expert_label = pred.get('prediction_of_expert_judgments')
                if expert_label and 'label' not in pred:
                    pred['label'] = expert_label
                if 'label' in pred and 'label_id' not in pred:
                    pred['label_id'] = TENCLASS_LABEL_TO_ID.get(pred['label'])
                preds.append(pred)
                keys.append((pid, rid, inum))

    return preds, keys


# ---------------------------------------------------------------------------
# Auto-detection: discover prediction files in a model output directory
# ---------------------------------------------------------------------------

_METAREVIEW_RE = re.compile(r'_paper\d+_metareview\.json$')


def _discover_prediction_files(
    path: Path,
) -> List[Path]:
    """Given a model results directory, find all *_metareview.json files."""
    if not path.is_dir():
        return []
    return sorted(f for f in path.glob('*_metareview.json')
                  if _METAREVIEW_RE.search(f.name))


# ---------------------------------------------------------------------------
# Main scoring logic
# ---------------------------------------------------------------------------

def score_predictions(
    preds: List[Dict[str, Any]],
    pred_keys: List[Tuple[int, str, int]],
    gt_index: Dict[Tuple[int, str, int], Dict[str, Any]],
    mode: str,
    source_name: str,
) -> Dict[str, Any]:
    """Match predictions to ground truth and score."""
    matched_preds: List[Dict[str, Any]] = []
    matched_gt: List[Dict[str, Any]] = []
    n_unmatched = 0

    for pred, key in zip(preds, pred_keys):
        gt_row = gt_index.get(key)
        if gt_row is None:
            n_unmatched += 1
            continue
        matched_preds.append(pred)
        matched_gt.append(gt_row)

    if not matched_preds:
        return {
            'source': source_name,
            'mode': mode,
            'n_predictions': len(preds),
            'n_matched': 0,
            'n_unmatched': n_unmatched,
            'error': 'no predictions matched ground truth',
        }

    if mode == 'axis':
        metrics = evaluate_axis_predictions(matched_preds, matched_gt)
    else:
        metrics = evaluate_tenclass_predictions(matched_preds, matched_gt)

    metrics['source'] = source_name
    metrics['n_predictions_total'] = len(preds)
    metrics['n_matched'] = len(matched_preds)
    metrics['n_unmatched'] = n_unmatched
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate agent meta-review predictions'
    )
    parser.add_argument(
        'paths', nargs='+', type=Path,
        help='Model results directory(ies) to score. Each should contain '
             '*_{mode}_metareview.json files.',
    )
    parser.add_argument(
        '--mode', type=str, choices=('axis', 'tenclass'), default=None,
        help='Only score this mode (default: auto-discover both).',
    )
    parser.add_argument(
        '--output', type=Path, default=None,
        help='Write combined metrics JSON to this file.',
    )
    args = parser.parse_args()

    # Load ground truth once
    print('Loading ground truth from HuggingFace...')
    gt_index = _load_ground_truth()
    print(f'  {len(gt_index)} ground-truth items indexed')

    all_results: List[Dict[str, Any]] = []

    for input_path in args.paths:
        input_path = input_path.resolve()
        print(f'\n{"=" * 70}')
        print(f'  Scoring: {input_path}')
        print(f'{"=" * 70}')

        if not input_path.exists():
            print(f'  ERROR: path does not exist')
            continue

        pred_files = _discover_prediction_files(input_path)
        if not pred_files:
            print(f'  No *_metareview.json files found')
            continue

        preds, pred_keys = _load_agent_predictions(pred_files)
        print(f'  Loaded {len(preds)} predictions from {len(pred_files)} files')

        if not preds:
            print(f'  No predictions to score')
            continue

        source_name = input_path.name

        # Build per-type subsets
        subsets = [('overall', list(range(len(preds))))]
        human_idx = [i for i, k in enumerate(pred_keys) if _reviewer_type(k[1]) == 'Human']
        ai_idx = [i for i, k in enumerate(pred_keys) if _reviewer_type(k[1]) == 'AI']
        if human_idx:
            subsets.append(('human_reviewers', human_idx))
        if ai_idx:
            subsets.append(('ai_reviewers', ai_idx))

        for subset_name, indices in subsets:
            sub_preds = [preds[i] for i in indices]
            sub_keys = [pred_keys[i] for i in indices]

            for mode in ('axis', 'tenclass'):
                if args.mode and mode != args.mode:
                    continue

                label = f'{source_name}:{subset_name}:{mode}'
                print(f'\n  --- {label} ({len(sub_preds)} items) ---')
                result = score_predictions(sub_preds, sub_keys, gt_index, mode, label)
                all_results.append(result)

                report = format_mode_report(result, predictor_name=label)
                print(report)

                if result.get('n_unmatched', 0) > 0:
                    print(f'  NOTE: {result["n_unmatched"]} predictions had no matching '
                          f'ground-truth row')

    # Summary table
    if len(all_results) > 1:
        print(f'\n{"=" * 70}')
        print('  COMPARISON SUMMARY')
        print(f'{"=" * 70}')
        print(f'{"source":<50} {"mode":<10} {"matched":>8} {"key_metric":>12}')
        print('-' * 85)
        for r in all_results:
            source = r.get('source', '?')
            mode = r.get('mode', '?')
            n = r.get('n_matched', r.get('n', 0))
            if mode == 'axis':
                axes = r.get('label_accuracy_per_axis', {})
                corr_acc = axes.get('correctness', {}).get('accuracy')
                key = f'corr={corr_acc:.3f}' if corr_acc is not None else 'N/A'
            else:
                acc = r.get('overall_accuracy')
                key = f'10cl={acc:.3f}' if acc is not None else 'N/A'
            print(f'{source:<50} {mode:<10} {n:>8} {key:>12}')

    # Write combined output
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(all_results, indent=2, default=str))
        print(f'\nCombined metrics written to {args.output}')


if __name__ == '__main__':
    main()
