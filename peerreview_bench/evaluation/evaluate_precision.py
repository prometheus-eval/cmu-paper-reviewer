#!/usr/bin/env python3
"""
Precision metric: "How good are the AI reviewer's items in terms of
correctness, significance, and sufficiency of evidence?"

For each AI review item, run the LLM meta-reviewer (axis mode) to predict
correctness → significance → evidence. An item is "fully good" if the judge
says Correct + Significant + Sufficient.

    Precision = fully-good AI items / total AI items

Two judge modes:
  - "agent": OpenHands agent meta-reviewer (navigates paper files, same as
    meta_review/expert_annotation_meta_review/run_meta_review_agent.py)
  - "llm": Direct LLM call meta-reviewer (paper content in prompt, same as
    meta_review/expert_annotation_meta_review/run_meta_review.py)

Usage:
    python3 evaluate_precision.py --paper-root papers/ --model-name my-agent --judge-mode llm
    python3 evaluate_precision.py --paper-root papers/ --byoj --judge-mode agent
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_EA_META_DIR = _BENCH_DIR / 'meta_review' / 'expert_annotation_meta_review'

for p in (_HERE, _BENCH_DIR, _EA_META_DIR, _BENCH_DIR / 'meta_review'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parse_review import load_review_items  # noqa: E402

# Reuse the meta-review predictor from expert_annotation_meta_review
from predictors import LiteLLMMetaReviewer  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


def _is_fully_good_prediction(pred: Dict[str, Any]) -> bool:
    """Check if the meta-reviewer's axis-mode prediction says the item is
    Correct + Significant + Sufficient."""
    return (
        pred.get('correctness') == 'Correct'
        and pred.get('significance') == 'Significant'
        and pred.get('evidence') == 'Sufficient'
    )


def evaluate_precision_llm(
    paper_ids: List[int],
    paper_root: Path,
    ai_model_name: Optional[str],
    *,
    judge_model: str,
    concurrency: int = 16,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """Evaluate precision via direct LLM meta-reviewer (axis mode).

    Each AI item gets one LLM call that judges its correctness, significance,
    and evidence.
    """
    from load_data import load_submitted_papers, load_reviewer  # noqa: E402

    # Load paper content for prompts
    reviewer_rows = load_reviewer()
    paper_content_by_id = {int(r['paper_id']): r for r in reviewer_rows}

    # Lazy hash-map for images/artifacts
    _cache: Dict[str, Any] = {}
    def _load_hash_map():
        if 'hm' not in _cache:
            print('  Loading submitted_papers for images/artifacts...')
            _cache['hm'] = load_submitted_papers()
        return _cache['hm']

    predictor = LiteLLMMetaReviewer(
        model=judge_model,
        mode='axis',
        attach_images=True,
        attach_artifacts=True,
        hash_to_bytes_loader=_load_hash_map,
        max_images=3,
        temperature=temperature,
    )

    # Collect all items across papers
    all_items: List[Dict[str, Any]] = []  # {paper_id, item, paper_row}
    for pid in paper_ids:
        paper_dir = paper_root / f'paper{pid}'
        ai_items = load_review_items(paper_dir, model_name=ai_model_name)
        paper_row = paper_content_by_id.get(pid, {})
        for item in ai_items:
            all_items.append({
                'paper_id': pid,
                'item': item,
                'paper_row': paper_row,
            })

    if not all_items:
        return {
            'n_papers': len(paper_ids),
            'n_items': 0,
            'n_fully_good': 0,
            'precision': 0.0,
        }

    print(f'  Scoring {len(all_items)} AI items with LLM judge ({judge_model})...')

    results: List[Dict[str, Any]] = []
    lock = threading.Lock()
    n_good = 0

    pbar = tqdm(total=len(all_items), desc='precision', unit='item')

    def _score_one(entry: Dict[str, Any]) -> None:
        nonlocal n_good
        item = entry['item']
        paper_row = entry['paper_row']

        # Build a pseudo-row that the predictor expects
        row = {
            'paper_title': paper_row.get('paper_title', ''),
            'paper_content': paper_row.get('paper_content', ''),
            'file_refs': paper_row.get('file_refs', []),
            'review_content': item.get('main_point') or item.get('text', ''),
            'review_item': item.get('text', ''),
            'review_claim': item.get('claim_full'),
            'review_evidence': item.get('evidence_full'),
            'review_item_number': item.get('item_number', 0),
            'reviewer_id': 'AI_Reviewer',
            'reviewer_type': 'AI',
        }

        pred = predictor.predict(row)

        is_good = _is_fully_good_prediction(pred)
        result = {
            'paper_id': entry['paper_id'],
            'item_number': item.get('item_number'),
            'correctness': pred.get('correctness'),
            'significance': pred.get('significance'),
            'evidence': pred.get('evidence'),
            'is_fully_good': is_good,
            'parsed': pred.get('_parsed', True),
            'error': pred.get('_error'),
        }

        with lock:
            results.append(result)
            if is_good:
                n_good += 1
            pbar.update(1)
            pbar.set_postfix({'good': n_good, 'total': len(results)})

    if concurrency <= 1:
        for entry in all_items:
            _score_one(entry)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_score_one, e) for e in all_items]
            for _ in as_completed(futures):
                pass

    pbar.close()

    precision = n_good / len(results) if results else 0.0
    return {
        'n_papers': len(paper_ids),
        'n_items': len(results),
        'n_fully_good': n_good,
        'precision': precision,
        'per_item': results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Precision metric: quality of AI reviewer items via LLM meta-review'
    )
    parser.add_argument('--paper-root', type=Path, required=True,
                        help='Root dir with paper{N}/ subdirectories')
    parser.add_argument('--model-name', type=str, default=None,
                        help='AI reviewer model name (to find review files)')
    parser.add_argument('--byoj', action='store_true',
                        help='Use any review_items_*.json found in review/ dirs')
    parser.add_argument('--judge-model', type=str,
                        default='litellm_proxy/anthropic/claude-opus-4-6',
                        help='LLM model for meta-review judging')
    parser.add_argument('--judge-mode', type=str, default='llm',
                        choices=('llm', 'agent'),
                        help="'llm' = direct LLM call (default), 'agent' = OpenHands agent")
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()

    # Find papers with AI reviews, excluding empty-rubric papers
    paper_root = args.paper_root.resolve()

    from build_rubric import build_rubric  # noqa: E402
    _rubric, dropped = build_rubric()
    dropped_set = set(dropped)

    paper_dirs = sorted(paper_root.glob('paper*'))
    paper_ids = []
    for pd in paper_dirs:
        try:
            pid = int(pd.name.replace('paper', ''))
        except ValueError:
            continue
        if pid in dropped_set:
            continue  # skip papers with no fully-good human items
        review_dir = pd / 'review'
        if review_dir.exists() and any(review_dir.iterdir()):
            paper_ids.append(pid)

    if args.limit:
        paper_ids = paper_ids[:args.limit]

    print(f'Found {len(paper_ids)} papers with reviews '
          f'(excluded {len(dropped_set)} empty-rubric papers)')

    if args.judge_mode == 'agent':
        print('Agent-based precision evaluation not yet implemented.')
        print('Use --judge-mode llm for now.')
        sys.exit(1)

    t0 = time.time()
    result = evaluate_precision_llm(
        paper_ids=paper_ids,
        paper_root=paper_root,
        ai_model_name=args.model_name if not args.byoj else None,
        judge_model=args.judge_model,
        concurrency=args.concurrency,
        temperature=args.temperature,
    )
    elapsed = time.time() - t0

    print(f'\n{"=" * 60}')
    print(f'PRECISION RESULTS')
    print(f'{"=" * 60}')
    print(f'Papers scored: {result["n_papers"]}')
    print(f'Total AI items: {result["n_items"]}')
    print(f'Fully good: {result["n_fully_good"]}')
    print(f'Precision: {result["precision"]:.2%}')
    print(f'Elapsed: {elapsed:.0f}s')

    # Per-axis breakdown
    if result.get('per_item'):
        from collections import Counter
        corr = Counter(r['correctness'] for r in result['per_item'])
        sig = Counter(r['significance'] for r in result['per_item'] if r['correctness'] == 'Correct')
        evi = Counter(r['evidence'] for r in result['per_item']
                      if r['correctness'] == 'Correct'
                      and r['significance'] in ('Significant', 'Marginally Significant'))
        print(f'\nPer-axis breakdown:')
        print(f'  Correctness: {dict(corr)}')
        print(f'  Significance (if Correct): {dict(sig)}')
        print(f'  Evidence (if Correct + Sig/Marg): {dict(evi)}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result['judge_model'] = args.judge_model
        result['judge_mode'] = args.judge_mode
        result['ai_reviewer_model'] = args.model_name
        result['elapsed_seconds'] = round(elapsed, 1)
        args.output.write_text(json.dumps(result, indent=2, default=str))
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
