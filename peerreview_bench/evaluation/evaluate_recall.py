#!/usr/bin/env python3
"""
Recall metric: "How many fully-good human review items did the AI reviewer
cover?"

For each paper:
  1. Load the rubric (fully-good human items).
  2. Load the AI reviewer's parsed items.
  3. For each rubric item × AI item pair: run the 4-way LLM similarity
     judge (same prompt as expert_annotation_similarity/).
  4. If the judge returns "near-paraphrase" or "convergent conclusion"
     (the two "similar" classes), the rubric item is considered covered.
  5. Recall = covered rubric items / total rubric items.

Usage:
    python3 evaluate_recall.py --paper-root papers/ --model-name my-agent
    python3 evaluate_recall.py --paper-root papers/ --byoj  # uses review_items_*.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_EA_SIM_DIR = _BENCH_DIR / 'similarity_check' / 'expert_annotation_similarity'

for p in (_HERE, _BENCH_DIR, _EA_SIM_DIR, _EA_SIM_DIR / 'baselines',
          _BENCH_DIR / 'meta_review'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from build_rubric import build_rubric_with_texts  # noqa: E402
from parse_review import load_review_items  # noqa: E402

# Reuse the 4-way similarity prompt and LLM call machinery from
# expert_annotation_similarity
from prompts import (  # noqa: E402
    FOURWAY_SYSTEM_PROMPT,
    FOURWAY_USER_PROMPT_TEMPLATE,
    fourway_to_binary,
)
from llm_classifier import (  # noqa: E402
    _call_llm_with_reasoning,
    build_reasoning_kwargs,
    extract_4way_answer,
)
from model_config import get_max_output_tokens  # noqa: E402

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, **kwargs):  # type: ignore
        return iterable


# ---------------------------------------------------------------------------
# Similarity judge (reuses the 4-way prompt from similarity_check)
# ---------------------------------------------------------------------------

def _judge_pair(
    rubric_text: str,
    ai_text: str,
    paper_content: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the 4-way similarity judge on one (rubric_item, ai_item) pair.
    Returns {parsed_answer, parsed_binary, error}."""
    user_text = FOURWAY_USER_PROMPT_TEMPLATE.format(
        paper_text=paper_content,
        reviewer_a='Human (rubric)',
        reviewer_b='AI Reviewer',
        item_a=rubric_text,
        item_b=ai_text,
    )
    messages = [
        {'role': 'system', 'content': FOURWAY_SYSTEM_PROMPT},
        {'role': 'user', 'content': user_text},
    ]

    try:
        result = _call_llm_with_reasoning(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_kwargs=reasoning_kwargs,
        )
        response_text = result.get('content', '')
        parsed = extract_4way_answer(response_text)
        if parsed:
            binary = fourway_to_binary(parsed)
            return {'parsed_answer': parsed, 'parsed_binary': binary, 'error': None}
        return {'parsed_answer': None, 'parsed_binary': None, 'error': 'parse_failed'}
    except Exception as e:
        return {'parsed_answer': None, 'parsed_binary': None,
                'error': f'{type(e).__name__}: {e}'}


# ---------------------------------------------------------------------------
# Per-paper recall computation
# ---------------------------------------------------------------------------

def compute_paper_recall(
    paper_id: int,
    rubric_items: List[Dict[str, Any]],
    ai_items: List[Dict[str, Any]],
    paper_content: str,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    reasoning_kwargs: Dict[str, Any],
    concurrency: int = 16,
) -> Dict[str, Any]:
    """Compute recall for one paper.

    For each rubric item, check if ANY AI item is similar (near-paraphrase
    or convergent). The rubric item is "covered" if at least one AI item
    matches.

    Returns:
        {paper_id, n_rubric, n_ai, n_covered, recall, pair_details}
    """
    if not rubric_items or not ai_items:
        return {
            'paper_id': paper_id,
            'n_rubric': len(rubric_items),
            'n_ai': len(ai_items),
            'n_covered': 0,
            'recall': 0.0,
            'pair_details': [],
        }

    # Generate all rubric × AI pairs
    pairs: List[Tuple[int, int]] = []  # (rubric_idx, ai_idx)
    for ri in range(len(rubric_items)):
        for ai in range(len(ai_items)):
            pairs.append((ri, ai))

    # Score all pairs
    pair_results: Dict[Tuple[int, int], Dict] = {}
    lock = threading.Lock()

    def _score_pair(pair: Tuple[int, int]) -> None:
        ri, ai = pair
        result = _judge_pair(
            rubric_text=rubric_items[ri]['text'],
            ai_text=ai_items[ai]['text'],
            paper_content=paper_content,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_kwargs=reasoning_kwargs,
        )
        with lock:
            pair_results[(ri, ai)] = result

    if concurrency <= 1:
        for pair in pairs:
            _score_pair(pair)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(_score_pair, p) for p in pairs]
            for _ in as_completed(futures):
                pass

    # Determine coverage: rubric item is covered if ANY AI item is "similar"
    covered = set()
    pair_details = []
    for (ri, ai), result in pair_results.items():
        is_similar = result.get('parsed_binary') == 'similar'
        pair_details.append({
            'rubric_idx': ri,
            'rubric_reviewer': rubric_items[ri].get('reviewer_id'),
            'rubric_item_number': rubric_items[ri].get('item_number'),
            'ai_item_number': ai_items[ai].get('item_number'),
            'parsed_answer': result.get('parsed_answer'),
            'parsed_binary': result.get('parsed_binary'),
            'is_similar': is_similar,
            'error': result.get('error'),
        })
        if is_similar:
            covered.add(ri)

    n_rubric = len(rubric_items)
    n_covered = len(covered)
    recall = n_covered / n_rubric if n_rubric > 0 else 0.0

    return {
        'paper_id': paper_id,
        'n_rubric': n_rubric,
        'n_ai': len(ai_items),
        'n_pairs_scored': len(pair_results),
        'n_covered': n_covered,
        'recall': recall,
        'pair_details': pair_details,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Recall metric: coverage of fully-good human items by AI reviewer'
    )
    parser.add_argument('--paper-root', type=Path, required=True,
                        help='Root dir with paper{N}/ subdirectories')
    parser.add_argument('--model-name', type=str, default=None,
                        help='AI reviewer model name (to find review files). '
                             'Not needed if using --byoj.')
    parser.add_argument('--byoj', action='store_true',
                        help='Use any review_items_*.json found in review/ dirs')
    parser.add_argument('--similarity-model', type=str,
                        default='litellm_proxy/anthropic/claude-opus-4-6',
                        help='LLM judge for 4-way similarity')
    parser.add_argument('--concurrency', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results JSON')
    args = parser.parse_args()

    # Build rubric
    print('Building rubric from fully-good human items...')
    rubric, dropped = build_rubric_with_texts()
    total_rubric = sum(len(v) for v in rubric.values())
    print(f'  {len(rubric)} papers, {total_rubric} rubric items '
          f'({len(dropped)} papers dropped)')

    # Set up similarity judge
    bare_model = args.similarity_model
    if bare_model.startswith('litellm_proxy/'):
        bare_model = bare_model[len('litellm_proxy/'):]
    max_tokens = get_max_output_tokens(bare_model)
    reasoning_kwargs = build_reasoning_kwargs(args.similarity_model, max_tokens)
    print(f'  Similarity judge: {args.similarity_model}')
    print(f'  Concurrency: {args.concurrency}')

    # Load paper content for prompts
    from load_data import load_reviewer  # noqa: E402
    reviewer_rows = load_reviewer()
    paper_content_by_id = {int(r['paper_id']): r.get('paper_content', '')
                           for r in reviewer_rows}

    # Iterate over papers
    paper_ids = sorted(rubric.keys())
    if args.limit:
        paper_ids = paper_ids[:args.limit]

    all_results: List[Dict[str, Any]] = []
    total_covered = 0
    total_rubric_items = 0
    t0 = time.time()

    for i, pid in enumerate(paper_ids):
        paper_dir = args.paper_root / f'paper{pid}'
        rubric_items = rubric[pid]

        # Load AI reviewer items
        ai_items = load_review_items(paper_dir, model_name=args.model_name)
        if not ai_items:
            print(f'  [{i+1}/{len(paper_ids)}] paper{pid}: no AI review items found, skipping')
            continue

        paper_content = paper_content_by_id.get(pid, '')

        print(f'  [{i+1}/{len(paper_ids)}] paper{pid}: '
              f'{len(rubric_items)} rubric × {len(ai_items)} AI items '
              f'= {len(rubric_items) * len(ai_items)} pairs...',
              end='', flush=True)

        result = compute_paper_recall(
            paper_id=pid,
            rubric_items=rubric_items,
            ai_items=ai_items,
            paper_content=paper_content,
            model=args.similarity_model,
            max_tokens=max_tokens,
            temperature=args.temperature,
            reasoning_kwargs=reasoning_kwargs,
            concurrency=args.concurrency,
        )
        all_results.append(result)
        total_covered += result['n_covered']
        total_rubric_items += result['n_rubric']

        print(f' recall={result["recall"]:.2%} '
              f'({result["n_covered"]}/{result["n_rubric"]})')

    elapsed = time.time() - t0
    overall_recall = total_covered / total_rubric_items if total_rubric_items > 0 else 0.0

    print(f'\n{"=" * 60}')
    print(f'RECALL RESULTS')
    print(f'{"=" * 60}')
    print(f'Papers scored: {len(all_results)}')
    print(f'Total rubric items: {total_rubric_items}')
    print(f'Total covered: {total_covered}')
    print(f'Overall recall: {overall_recall:.2%}')
    print(f'Elapsed: {elapsed:.0f}s')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {
            'metric': 'recall',
            'similarity_model': args.similarity_model,
            'ai_reviewer_model': args.model_name,
            'n_papers': len(all_results),
            'total_rubric_items': total_rubric_items,
            'total_covered': total_covered,
            'overall_recall': overall_recall,
            'elapsed_seconds': round(elapsed, 1),
            'per_paper': all_results,
        }
        args.output.write_text(json.dumps(out, indent=2, default=str))
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
