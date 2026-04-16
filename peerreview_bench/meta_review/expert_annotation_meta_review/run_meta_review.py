#!/usr/bin/env python3
"""
Run a meta-reviewer over the `meta_reviewer` HuggingFace config and score it.

Two prompt modes are supported via `--prompt-mode`:
    axis     — LLM plays a single expert meta-reviewer; outputs per-axis
               cascade labels (correctness, significance, evidence)
    tenclass — LLM predicts what a PAIR of expert meta-reviewers would
               jointly produce; outputs one of 10 collapsed class labels

The `--model` arg is either:
  - a baseline name (`random`, `majority`, `constant`), or
  - a LiteLLM model id, with or without the `litellm_proxy/` prefix
    (e.g. `litellm_proxy/azure_ai/gpt-5.4`).

This runner is resumable: each result is flushed to a JSONL file as soon
as it is computed, and on restart the same command will skip rows that
have already been scored (detected by `(paper_id, reviewer_id, item_number)`
tuples in the existing file).

Usage:
    python3 run_meta_review.py --model majority --prompt-mode axis
    python3 run_meta_review.py --model litellm_proxy/azure_ai/gpt-5.4 --prompt-mode axis --concurrency 16
    python3 run_meta_review.py --model litellm_proxy/anthropic/claude-opus-4-6 --prompt-mode tenclass --concurrency 16
    LIMIT=20 python3 run_meta_review.py --model litellm_proxy/gemini/gemini-3.1-pro-preview --prompt-mode axis
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- HF Hub timeouts ---------------------------------------------------------
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

# --- Path plumbing -----------------------------------------------------------
# Layout:
#   peerreview_bench/meta_review/expert_annotation_meta_review/run_meta_review.py
#   peerreview_bench/meta_review/{litellm_client, model_config, image_mapping, metrics}.py
#   peerreview_bench/load_data.py
_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent

for p in (_HERE, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import load_meta_reviewer, load_submitted_papers  # noqa: E402

from predictors import (  # noqa: E402
    get_predictor,
    BASELINE_REGISTRY,
    LiteLLMMetaReviewer,
    _slug,
)
from prompts import (  # noqa: E402
    TENCLASS_LABELS,
    TENCLASS_LABEL_TO_ID,
)
from metrics import (  # noqa: E402
    evaluate_axis_predictions,
    evaluate_tenclass_predictions,
    format_mode_report,
    META_LABEL_ID_TO_NAME,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


DEFAULT_OUTPUT_ROOT = _BENCH_DIR / 'outputs' / 'expert_annotation_meta_review'


# ---------------------------------------------------------------------------
# Per-row record construction
# ---------------------------------------------------------------------------

def _build_per_item_record(
    row: Dict[str, Any],
    pred: Dict[str, Any],
    *,
    meta_reviewer_id: str,
    prompt_mode: str,
) -> Dict[str, Any]:
    """Rich per-item output record — mirrors the similarity_check JSONL
    format so it's easy to grep / join to the HF rows later."""
    base = {
        'paper_id': row.get('paper_id'),
        'paper_title': row.get('paper_title'),
        'item_number': row.get('item_number') or row.get('review_item_number'),
        'meta_reviewer': meta_reviewer_id,
        'prompt_mode': prompt_mode,
        'review': {
            'reviewer_id': row.get('reviewer_id'),
            'reviewer_type': row.get('reviewer_type'),
            'review_content': row.get('review_content'),
            'review_claim': row.get('review_claim'),
            'review_evidence': row.get('review_evidence'),
            'review_cited_references': row.get('review_cited_references'),
            'review_item_merged': row.get('review_item'),
        },
        'prediction': {
            k: v for k, v in pred.items()
            if not k.startswith('_')
        },
        'parsed': pred.get('_parsed', True),
        'error': pred.get('_error'),
        'elapsed_seconds': pred.get('_elapsed_seconds'),
        'n_images': pred.get('_n_images', 0),
        'n_artifacts': pred.get('_n_artifacts', 0),
        'ground_truth': {
            'correctness_primary':     row.get('correctness_primary'),
            'correctness_secondary':   row.get('correctness_secondary'),
            'significance_primary':    row.get('significance_primary'),
            'significance_secondary':  row.get('significance_secondary'),
            'evidence_primary':        row.get('evidence_primary'),
            'evidence_secondary':      row.get('evidence_secondary'),
            'label_id':                row.get('label_id'),
            'label':                   row.get('label'),
        },
        'additional_context': {
            'hf_config': 'submitted_papers',
            'hf_repo': 'prometheus-eval/peerreview-bench',
            'paper_content_length': len(row.get('paper_content') or ''),
        },
    }
    return base


def _row_key(row: Dict[str, Any]) -> Tuple:
    """Stable pair key to detect already-scored rows on resume."""
    return (
        int(row.get('paper_id', -1)),
        str(row.get('reviewer_id', '')),
        int(row.get('item_number') or row.get('review_item_number') or 0),
    )


def _load_existing_keys(pairs_path: Path) -> set:
    """Read the resume file and return already-scored (paper_id, reviewer, item) keys."""
    seen: set = set()
    if not pairs_path.exists():
        return seen
    with pairs_path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f'  note: dropping truncated line {line_num} in {pairs_path.name}',
                      file=sys.stderr)
                continue
            try:
                key = (
                    int(rec.get('paper_id', -1)),
                    str(rec['review']['reviewer_id']),
                    int(rec['item_number'] or 0),
                )
                seen.add(key)
            except (KeyError, TypeError, ValueError):
                continue
    return seen


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run a meta-reviewer benchmark over the `meta_reviewer` HF config'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help="Baseline name (random/majority/constant) OR a LiteLLM model id.",
    )
    parser.add_argument(
        '--prompt-mode', type=str, choices=('axis', 'tenclass'), required=True,
        help="Prompt family: 'axis' (LLM as meta-reviewer) or 'tenclass' "
             "(LLM predicts the collapsed 10-class label).",
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help=f'Default: {DEFAULT_OUTPUT_ROOT}/<model_slug>/',
    )
    parser.add_argument('--limit', type=int, default=None,
                        help='Score only the first N rows (smoke tests).')
    parser.add_argument('--no-images', action='store_true',
                        help='Disable image attachment for multimodal models.')
    parser.add_argument('--no-artifacts', action='store_true',
                        help='Disable code/text artifact attachment.')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Max images per item (default unbounded).')
    parser.add_argument('--max-tokens', type=int, default=None,
                        help="Completion budget override. Default: model's "
                             "max_output_tokens from model_config.py.")
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature. Default 1.0 — required by '
                             'Anthropic extended thinking, recommended for '
                             'Gemini 3 reasoning.')
    parser.add_argument('--concurrency', type=int, default=16,
                        help='Number of concurrent LLM workers. Default 16. '
                             'Baselines ignore this (run sequentially).')
    parser.add_argument('--progress-every', type=int, default=5,
                        help='Print progress every N rows (text-only bar fallback).')
    args = parser.parse_args()

    is_baseline = args.model.lower() in BASELINE_REGISTRY

    # Output paths
    if args.output_dir:
        out_root = Path(args.output_dir)
    else:
        out_root = DEFAULT_OUTPUT_ROOT / _slug(args.model)
    out_root.mkdir(parents=True, exist_ok=True)
    pairs_path = out_root / f'predictions_{args.prompt_mode}.jsonl'
    metrics_path = out_root / f'metrics_{args.prompt_mode}.json'
    report_path = out_root / f'report_{args.prompt_mode}.txt'

    # Load data
    print('Loading meta_reviewer config...')
    rows = load_meta_reviewer()
    print(f'  {len(rows)} rows loaded')
    if args.limit:
        rows = rows[:args.limit]
        print(f'  (truncated to {len(rows)} rows via --limit)')

    # Instantiate predictor
    if is_baseline:
        print(f'\nBaseline predictor: {args.model}  (mode={args.prompt_mode})')
        predictor = get_predictor(args.model, mode=args.prompt_mode)
        concurrency = 1  # baselines are instantaneous — don't spin up a pool
    else:
        print(f'\nLiteLLM predictor: {args.model}  (mode={args.prompt_mode})')
        # Lazy hash-map loader so we only download submitted_papers when
        # images or artifacts are actually being attached.
        _loader_cache: Dict[str, Any] = {}
        def _load_hash_map():
            if 'hash_map' not in _loader_cache:
                print('  Loading submitted_papers config (first use)...')
                _loader_cache['hash_map'] = load_submitted_papers()
            return _loader_cache['hash_map']

        attach_images = None if not args.no_images else False
        attach_artifacts = not args.no_artifacts
        predictor = LiteLLMMetaReviewer(
            model=args.model,
            mode=args.prompt_mode,
            attach_images=attach_images,
            attach_artifacts=attach_artifacts,
            hash_to_bytes_loader=(_load_hash_map if (not args.no_images or attach_artifacts) else None),
            max_images=args.max_images,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        concurrency = args.concurrency

    print(f'  name: {predictor.name}')
    print(f'  output: {pairs_path}')
    if not is_baseline:
        print(f'  max_tokens: {predictor.max_tokens}')
        print(f'  reasoning_kwargs: {predictor.reasoning_kwargs}')
        print(f'  attach_images: {predictor.attach_images}')
        print(f'  attach_artifacts: {predictor.attach_artifacts}')
        print(f'  concurrency: {concurrency}')

    # ---- Resume: skip already-scored rows ---------------------------------
    already = _load_existing_keys(pairs_path)
    if already:
        print(f'\n  resuming: {len(already)} rows already in {pairs_path.name}')
    todo_rows: List[Tuple[int, Dict[str, Any]]] = []
    for idx, row in enumerate(rows):
        if _row_key(row) in already:
            continue
        todo_rows.append((idx, row))
    n_total = len(rows)
    n_done = n_total - len(todo_rows)
    print(f'  {len(todo_rows)} rows to score ({n_done} done, {n_total} total)')

    # ---- Run ---------------------------------------------------------------
    predictions_so_far: List[Optional[Dict[str, Any]]] = [None] * n_total
    # Pre-populate predictions_so_far with whatever's already on disk so the
    # final scoring pass has a complete list.
    if already:
        with pairs_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Map back to the row index by key
                rec_key = (
                    int(rec.get('paper_id', -1)),
                    str(rec['review']['reviewer_id']),
                    int(rec['item_number'] or 0),
                )
                for idx, row in enumerate(rows):
                    if _row_key(row) == rec_key:
                        # Re-materialize the prediction dict (without _underscore fields)
                        predictions_so_far[idx] = rec.get('prediction') or {}
                        break

    lock = threading.Lock()
    n_scored = 0
    n_errors = 0
    n_parsed = 0

    pbar = tqdm(
        total=n_total,
        initial=n_done,
        desc=f'{predictor.name} [{args.prompt_mode}]',
        unit='row',
        dynamic_ncols=True,
        mininterval=1.0,
    )

    pairs_fh = pairs_path.open('a', buffering=1)

    def _on_result(idx: int, row: Dict[str, Any], pred: Dict[str, Any]) -> None:
        nonlocal n_scored, n_errors, n_parsed
        record = _build_per_item_record(
            row, pred,
            meta_reviewer_id=args.model,
            prompt_mode=args.prompt_mode,
        )
        with lock:
            pairs_fh.write(json.dumps(record, default=str) + '\n')
            pairs_fh.flush()
            predictions_so_far[idx] = record['prediction']
            n_scored += 1
            if pred.get('_parsed', True):
                n_parsed += 1
            if pred.get('_error'):
                n_errors += 1
                try:
                    tqdm.write(f"  [{idx+1}/{n_total}] ERROR: {pred['_error']}")
                except Exception:
                    pass
            try:
                pbar.set_postfix({
                    'parsed': f'{n_parsed}/{n_scored}',
                    'err': n_errors,
                })
                pbar.update(1)
            except Exception:
                pass

    def _worker(idx_row: Tuple[int, Dict[str, Any]]) -> None:
        idx, row = idx_row
        try:
            pred = predictor.predict(row)
        except Exception as e:
            pred = {
                'reasoning': f'(unhandled worker error: {type(e).__name__}: {e})',
                '_parsed': False,
                '_error': f'{type(e).__name__}: {e}',
                '_elapsed_seconds': 0.0,
                '_n_images': 0,
                '_n_artifacts': 0,
            }
            if args.prompt_mode == 'axis':
                pred.update({'correctness': 'Correct', 'significance': 'Significant', 'evidence': 'Sufficient'})
            else:
                pred.update({'label': 'correct_significant_sufficient', 'label_id': 1})
        _on_result(idx, row, pred)

    try:
        if concurrency <= 1 or is_baseline:
            for idx_row in todo_rows:
                _worker(idx_row)
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(_worker, r) for r in todo_rows]
                for _ in as_completed(futures):
                    pass
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        pairs_fh.close()

    print(f'\nScored {n_scored} new rows ({n_parsed} parsed, {n_errors} errors)')

    # ---- Score final predictions (across all rows, including the resumed set) ----
    final_predictions: List[Dict[str, Any]] = []
    for p in predictions_so_far:
        if p is None:
            # Row was never scored (shouldn't happen unless --limit reduced the set)
            final_predictions.append(
                {'reasoning': '(no prediction)', 'correctness': 'Correct',
                 'significance': 'Significant', 'evidence': 'Sufficient'}
                if args.prompt_mode == 'axis'
                else {'reasoning': '(no prediction)', 'label': 'correct_significant_sufficient', 'label_id': 1}
            )
        else:
            final_predictions.append(p)

    print('\nScoring...')
    if args.prompt_mode == 'axis':
        metrics = evaluate_axis_predictions(final_predictions, rows)
    else:
        metrics = evaluate_tenclass_predictions(final_predictions, rows)

    report = format_mode_report(metrics, predictor_name=predictor.name)
    print('\n' + report)

    # ---- Write artifacts ---------------------------------------------------
    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2, default=str)
    with report_path.open('w') as f:
        f.write(report)
    print(f'\nPer-item predictions: {pairs_path}')
    print(f'Metrics:              {metrics_path}')
    print(f'Report:               {report_path}')


if __name__ == '__main__':
    main()
