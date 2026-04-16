#!/usr/bin/env python3
"""
Run a meta-reviewer over the FULL `expert_annotation` HuggingFace config —
all 85 papers, not just the 27 overlap papers that the `meta_reviewer`
config carves out.

Design parallels run_meta_review.py (in the sister expert_annotation_meta_review
subdirectory) but with three differences:

  1. Loads `expert_annotation` instead of `meta_reviewer`. The rows carry
     per-item single-annotator labels (`correctness`, `significance`,
     `evidence`) for all 85 papers, plus primary/secondary overlap on 27
     papers.

  2. Deduplicates by (paper_id, reviewer_id, review_item_number) so the
     primary/secondary duplicates on the overlap papers count as one
     unique item — we want exactly one prediction per review item.

  3. Ground-truth scoring handles the mixed annotator situation:
       - AXIS mode: score per-axis against the single-annotator labels
         (on all rows). On the 27 overlap papers, we prefer the consensus
         label where both annotators agreed, and skip the axis otherwise.
       - TENCLASS mode: score against the derived 10-class label on rows
         where both annotators exist; report prediction distribution on
         the rest (no accuracy number on non-overlap papers).

The predictors, prompts, parser, and retry logic are imported from the
sister `expert_annotation_meta_review/` subdirectory so there's exactly
one source of truth.

Usage:
    python3 run_full_metareview.py --model majority --prompt-mode axis
    python3 run_full_metareview.py --model litellm_proxy/azure_ai/gpt-5.4 \
        --prompt-mode axis --concurrency 16
    LIMIT=5 python3 run_full_metareview.py \
        --model litellm_proxy/anthropic/claude-opus-4-6 --prompt-mode tenclass
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
#   peerreview_bench/meta_review/full_metareview/run_full_metareview.py
#   peerreview_bench/meta_review/expert_annotation_meta_review/{prompts,predictors}.py
#   peerreview_bench/meta_review/{model_config,image_mapping,metrics,litellm_client}.py
#   peerreview_bench/load_data.py
_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent
_EA_META_DIR = _META_REVIEW_DIR / 'expert_annotation_meta_review'

for p in (_HERE, _EA_META_DIR, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import load_expert_annotation_rows, load_submitted_papers  # noqa: E402

# Import from the sister subdir (the single source of truth for prompts,
# predictors, and per-row record builder).
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
)
from run_meta_review import _build_per_item_record  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


DEFAULT_OUTPUT_ROOT = _BENCH_DIR / 'outputs' / 'full_metareview'


# ---------------------------------------------------------------------------
# Data prep: dedupe expert_annotation rows to one-per-item
# ---------------------------------------------------------------------------
#
# The expert_annotation config has primary AND secondary rows for the 27
# overlap papers. We want one prediction per review item, so we dedupe by
# (paper_id, reviewer_id, review_item_number). To preserve the ground-truth
# information carried by both annotator sources, we also AGGREGATE the
# per-annotator labels into the _primary/_secondary shape that the axis
# and tenclass scorers already know how to interpret.

def _dedupe_and_merge(rows: List[Dict]) -> List[Dict]:
    """Dedupe expert_annotation rows by (paper_id, reviewer_id,
    review_item_number). For rows with both primary and secondary
    annotations (27 overlap papers), merge them into a single dict with
    *_primary / *_secondary fields so the downstream scorers can use the
    consensus logic. For rows with only one annotator, the _primary field
    is populated and the _secondary field is left None.

    Also maps `review_item_number` → `item_number` so run_meta_review's
    _build_per_item_record keeps working unchanged.
    """
    merged: Dict[Tuple[int, str, int], Dict[str, Any]] = {}

    for r in rows:
        key = (
            int(r.get('paper_id', -1)),
            str(r.get('reviewer_id', '')),
            int(r.get('review_item_number') or 0),
        )
        src = r.get('annotator_source') or 'primary'
        suffix = '_primary' if src == 'primary' else '_secondary'

        if key not in merged:
            merged[key] = {
                'paper_id': r.get('paper_id'),
                'paper_title': r.get('paper_title'),
                'paper_content': r.get('paper_content'),
                'file_refs': r.get('file_refs') or [],
                'reviewer_id': r.get('reviewer_id'),
                'reviewer_type': r.get('reviewer_type'),
                'review_item_number': r.get('review_item_number'),
                'item_number': r.get('review_item_number'),
                'review_item': r.get('review_item'),
                # For the predictor we only have the merged text
                'review_content': r.get('review_item'),
                'review_claim': None,
                'review_evidence': None,
                'review_cited_references': None,
                # Placeholders for per-annotator ground-truth labels
                'correctness_primary': None,
                'correctness_secondary': None,
                'significance_primary': None,
                'significance_secondary': None,
                'evidence_primary': None,
                'evidence_secondary': None,
                'label_id': None,
                'label': None,
            }
        # Write the labels into the correct annotator slot
        merged[key][f'correctness{suffix}'] = r.get('correctness')
        merged[key][f'significance{suffix}'] = r.get('significance')
        merged[key][f'evidence{suffix}'] = r.get('evidence')

    return list(merged.values())


# ---------------------------------------------------------------------------
# Resume: same helpers as run_meta_review.py
# ---------------------------------------------------------------------------

def _row_key(row: Dict[str, Any]) -> Tuple:
    return (
        int(row.get('paper_id', -1)),
        str(row.get('reviewer_id', '')),
        int(row.get('item_number') or row.get('review_item_number') or 0),
    )


def _load_existing_keys(pairs_path: Path) -> set:
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
        description='Run a meta-reviewer over the FULL expert_annotation HF config'
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Baseline name or LiteLLM model id.')
    parser.add_argument('--prompt-mode', type=str, choices=('axis', 'tenclass'),
                        required=True,
                        help="Prompt family: 'axis' or 'tenclass'")
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Default: {DEFAULT_OUTPUT_ROOT}/<model_slug>/')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only score rows from the first N papers (smoke test).')
    parser.add_argument('--no-images', action='store_true',
                        help='Disable image attachment for multimodal models.')
    parser.add_argument('--no-artifacts', action='store_true',
                        help='Disable code/text artifact attachment.')
    parser.add_argument('--max-images', type=int, default=None)
    parser.add_argument('--max-tokens', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--concurrency', type=int, default=16)
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

    print('Loading expert_annotation config...')
    raw_rows = load_expert_annotation_rows()
    print(f'  {len(raw_rows)} raw rows (primary + secondary)')

    rows = _dedupe_and_merge(raw_rows)
    # Drop rows with empty review text
    rows = [r for r in rows if (r.get('review_item') or '').strip()]
    print(f'  {len(rows)} unique non-empty items after dedupe')

    if args.limit:
        # smoke test: keep only the first N papers (by sorted paper_id)
        all_pids = sorted({int(r['paper_id']) for r in rows})
        keep = set(all_pids[:args.limit])
        rows = [r for r in rows if int(r['paper_id']) in keep]
        print(f'  --limit {args.limit} papers: {len(rows)} items retained')

    # Instantiate predictor
    if is_baseline:
        print(f'\nBaseline predictor: {args.model}  (mode={args.prompt_mode})')
        predictor = get_predictor(args.model, mode=args.prompt_mode)
        concurrency = 1
    else:
        print(f'\nLiteLLM predictor: {args.model}  (mode={args.prompt_mode})')
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
        print(f'  concurrency: {concurrency}')

    # Resume
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

    predictions_so_far: List[Optional[Dict[str, Any]]] = [None] * n_total

    # Materialize existing predictions for final scoring
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
                rec_key = (
                    int(rec.get('paper_id', -1)),
                    str(rec['review']['reviewer_id']),
                    int(rec['item_number'] or 0),
                )
                for idx, row in enumerate(rows):
                    if _row_key(row) == rec_key:
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

    # Final scoring (includes rows that were resumed from disk)
    final_predictions: List[Dict[str, Any]] = []
    for p in predictions_so_far:
        if p is None:
            final_predictions.append(
                {'reasoning': '(no prediction)', 'correctness': 'Correct',
                 'significance': 'Significant', 'evidence': 'Sufficient'}
                if args.prompt_mode == 'axis'
                else {'reasoning': '(no prediction)',
                      'label': 'correct_significant_sufficient', 'label_id': 1}
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

    with metrics_path.open('w') as f:
        json.dump(metrics, f, indent=2, default=str)
    with report_path.open('w') as f:
        f.write(report)

    print(f'\nPer-item predictions: {pairs_path}')
    print(f'Metrics:              {metrics_path}')
    print(f'Report:               {report_path}')


if __name__ == '__main__':
    main()
