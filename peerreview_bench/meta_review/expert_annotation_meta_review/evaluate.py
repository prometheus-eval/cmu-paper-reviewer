#!/usr/bin/env python3
"""
Unified evaluation script for meta-review predictions.

Handles BOTH output formats:
  - LLM path:   predictions_{mode}.jsonl   (one JSON line per item)
  - Agent path:  agent_trajectories/{mode}/paper{N}/prediction.json
                 (nested JSON per paper: {reviewers: [{items: [...]}]})

The script auto-detects the format, flattens everything into per-item
predictions, loads the matching ground-truth rows from HuggingFace, and
runs the appropriate scorer (axis or tenclass).

Usage:
    # Score a single LLM predictions file
    python3 evaluate.py predictions_axis.jsonl --mode axis

    # Score an agent's output directory (reads all paper*/prediction.json)
    python3 evaluate.py agent_trajectories/axis/ --mode axis

    # Score everything in a model's output directory (auto-discovers files)
    python3 evaluate.py ../../outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview/

    # Compare multiple models
    python3 evaluate.py ../../outputs/expert_annotation_meta_review/*/
"""

from __future__ import annotations

import argparse
import json
import os
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

from load_data import load_expert_annotation_rows, load_meta_reviewer  # noqa: E402
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
    """Load ground-truth rows indexed by (paper_id, reviewer_id, item_number).

    Uses meta_reviewer config (has both primary + secondary labels) for the
    27 overlap papers, and expert_annotation for all 85 papers. Meta_reviewer
    rows take precedence where available since they carry the 10-class label.
    """
    # Start with expert_annotation (all papers, single or dual annotator)
    ea_rows = load_expert_annotation_rows()
    gt_index: Dict[Tuple[int, str, int], Dict[str, Any]] = {}

    # Deduplicate EA rows: keep first (primary) per (paper_id, reviewer_id, item_number)
    for r in ea_rows:
        pid = int(r['paper_id'])
        rid = r['reviewer_id']
        inum = int(r.get('review_item_number') or 0)
        key = (pid, rid, inum)
        if key not in gt_index:
            gt_row = dict(r)
            # Normalize into _primary/_secondary shape for the scorers
            src = r.get('annotator_source', 'primary')
            suffix = '_primary' if src == 'primary' else '_secondary'
            gt_row.setdefault('correctness_primary', None)
            gt_row.setdefault('correctness_secondary', None)
            gt_row.setdefault('significance_primary', None)
            gt_row.setdefault('significance_secondary', None)
            gt_row.setdefault('evidence_primary', None)
            gt_row.setdefault('evidence_secondary', None)
            gt_row[f'correctness{suffix}'] = r.get('correctness')
            gt_row[f'significance{suffix}'] = r.get('significance')
            gt_row[f'evidence{suffix}'] = r.get('evidence')
            gt_index[key] = gt_row
        else:
            # Second annotator on the same item (overlap paper)
            existing = gt_index[key]
            src = r.get('annotator_source', 'secondary')
            suffix = '_primary' if src == 'primary' else '_secondary'
            existing[f'correctness{suffix}'] = r.get('correctness')
            existing[f'significance{suffix}'] = r.get('significance')
            existing[f'evidence{suffix}'] = r.get('evidence')

    # Overlay meta_reviewer rows where available (they carry label_id)
    try:
        mr_rows = load_meta_reviewer()
        for r in mr_rows:
            pid = int(r['paper_id'])
            rid = r['reviewer_id']
            inum = int(r.get('review_item_number') or r.get('item_number') or 0)
            key = (pid, rid, inum)
            if key in gt_index:
                gt_index[key]['label_id'] = r.get('label_id')
                gt_index[key]['label'] = r.get('label')
                # Also fill in primary/secondary from MR (more reliable)
                for axis in ('correctness', 'significance', 'evidence'):
                    for suffix in ('_primary', '_secondary'):
                        val = r.get(f'{axis}{suffix}')
                        if val is not None:
                            gt_index[key][f'{axis}{suffix}'] = val
    except Exception:
        pass  # meta_reviewer not available — use EA labels only

    return gt_index


# ---------------------------------------------------------------------------
# Prediction loading: LLM JSONL format
# ---------------------------------------------------------------------------

def _load_llm_predictions(
    jsonl_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str, int]]]:
    """Load predictions from a JSONL file (one line per item).
    Returns (predictions, keys) where keys are (paper_id, reviewer_id, item_number)."""
    preds: List[Dict[str, Any]] = []
    keys: List[Tuple[int, str, int]] = []

    with jsonl_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f'  warn: dropping truncated line {line_num}', file=sys.stderr)
                continue

            pred = rec.get('prediction', rec)  # handle both wrapped and bare
            pid = int(rec.get('paper_id', -1))
            review = rec.get('review', {})
            rid = review.get('reviewer_id', rec.get('reviewer_id', ''))
            inum = int(rec.get('item_number') or rec.get('review_item_number')
                       or review.get('review_item_number') or 0)

            preds.append(pred)
            keys.append((pid, rid, inum))

    return preds, keys


# ---------------------------------------------------------------------------
# Prediction loading: Agent nested JSON format
# ---------------------------------------------------------------------------

def _load_agent_predictions(
    agent_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str, int]]]:
    """Load predictions from agent_trajectories/{mode}/paper{N}/prediction.json files.
    Returns (predictions, keys) — flattened to per-item."""
    preds: List[Dict[str, Any]] = []
    keys: List[Tuple[int, str, int]] = []

    # Find all prediction.json files under the agent dir
    pred_files = sorted(agent_dir.glob('*/prediction.json'))
    if not pred_files:
        # Maybe the dir IS the mode dir (agent_trajectories/axis/)
        pred_files = sorted(agent_dir.glob('paper*/prediction.json'))
    if not pred_files:
        print(f'  warn: no prediction.json files found under {agent_dir}',
              file=sys.stderr)
        return preds, keys

    for pf in pred_files:
        try:
            data = json.loads(pf.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError) as e:
            print(f'  warn: skipping {pf}: {e}', file=sys.stderr)
            continue

        pid = int(data.get('paper_id', -1))
        # Extract paper_id from dir name if not in JSON
        if pid == -1:
            dir_name = pf.parent.name  # e.g. "paper12"
            try:
                pid = int(dir_name.replace('paper', ''))
            except ValueError:
                pass

        for rev_block in data.get('reviewers', []):
            rid = rev_block.get('reviewer_id', '')
            for item_block in rev_block.get('items', []):
                inum = int(item_block.get('item_number', 0))
                pred = dict(item_block)
                # Add label_id for tenclass if missing
                if 'label' in pred and 'label_id' not in pred:
                    pred['label_id'] = TENCLASS_LABEL_TO_ID.get(pred['label'])
                preds.append(pred)
                keys.append((pid, rid, inum))

    return preds, keys


# ---------------------------------------------------------------------------
# Auto-detection: discover prediction files in a model output directory
# ---------------------------------------------------------------------------

def _discover_prediction_sources(
    path: Path,
) -> List[Tuple[str, Path, str]]:
    """Given a path (file or directory), discover all scorable prediction
    sources. Returns list of (source_name, path, mode)."""
    sources: List[Tuple[str, Path, str]] = []

    if path.is_file() and path.suffix == '.jsonl':
        # Single JSONL file — infer mode from filename
        mode = 'tenclass' if 'tenclass' in path.name else 'axis'
        sources.append((path.name, path, mode))
        return sources

    if path.is_dir():
        # Look for JSONL files
        for jsonl in sorted(path.glob('predictions_*.jsonl')):
            mode = 'tenclass' if 'tenclass' in jsonl.name else 'axis'
            sources.append((f'llm:{jsonl.name}', jsonl, mode))

        # Look for agent directories
        for mode_dir in sorted(path.glob('agent_trajectories/*')):
            if mode_dir.is_dir() and any(mode_dir.glob('paper*/prediction.json')):
                mode = mode_dir.name  # 'axis' or 'tenclass'
                if mode in ('axis', 'tenclass'):
                    sources.append((f'agent:{mode}', mode_dir, mode))

    return sources


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
        description='Unified evaluation for meta-review predictions '
                    '(handles both LLM JSONL and agent nested JSON)'
    )
    parser.add_argument(
        'paths', nargs='+', type=Path,
        help='Prediction file(s) or directory(ies) to score. '
             'Accepts: a .jsonl file, an agent_trajectories/{mode}/ dir, '
             'or a model output dir (auto-discovers both LLM and agent files).',
    )
    parser.add_argument(
        '--mode', type=str, choices=('axis', 'tenclass'), default=None,
        help='Override auto-detected mode. Required when scoring a single '
             'file whose name doesn\'t contain "axis" or "tenclass".',
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

        sources = _discover_prediction_sources(input_path)
        if not sources:
            # Maybe it's an agent dir directly
            if input_path.is_dir() and any(input_path.glob('paper*/prediction.json')):
                mode = args.mode or ('tenclass' if 'tenclass' in str(input_path) else 'axis')
                sources = [(f'agent:{input_path.name}', input_path, mode)]

        if not sources:
            print(f'  No prediction sources found')
            continue

        for source_name, source_path, mode in sources:
            if args.mode:
                mode = args.mode

            print(f'\n  --- {source_name} (mode={mode}) ---')

            # Load predictions
            if source_path.is_file() and source_path.suffix == '.jsonl':
                preds, pred_keys = _load_llm_predictions(source_path)
                print(f'  Loaded {len(preds)} predictions from JSONL')
            elif source_path.is_dir():
                preds, pred_keys = _load_agent_predictions(source_path)
                print(f'  Loaded {len(preds)} predictions from agent JSONs')
            else:
                print(f'  Unknown source type: {source_path}')
                continue

            if not preds:
                print(f'  No predictions to score')
                continue

            # Score
            result = score_predictions(preds, pred_keys, gt_index, mode, source_name)
            all_results.append(result)

            # Print report
            report = format_mode_report(result, predictor_name=source_name)
            print(report)

            if result.get('n_unmatched', 0) > 0:
                print(f'  NOTE: {result["n_unmatched"]} predictions had no matching '
                      f'ground-truth row (wrong paper_id/reviewer_id/item_number?)')

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
