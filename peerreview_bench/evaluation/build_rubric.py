#!/usr/bin/env python3
"""
Build the recall rubric from "fully good" human review items.

A review item is "fully good" if the primary human meta-reviewer annotated
it as Correct (1) + Significant (2) + Sufficient evidence (1). This
matches the `is_fully_good()` function in `analysis/data_filter.py` and
the definition used in Table 5 of the paper.

The rubric is a per-paper list of fully-good human review item texts.
Papers with zero fully-good human items are dropped from the evaluation.

Usage:
    python3 build_rubric.py                          # print rubric stats
    python3 build_rubric.py --save rubric.json       # save rubric to file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
sys.path.insert(0, str(_BENCH_DIR))
sys.path.insert(0, str(_BENCH_DIR / 'analysis'))

from load_data import load_annotations  # noqa: E402


def _is_fully_good(item: Any) -> bool:
    """Correct (1) + Significant (2) + Sufficient (1).
    Matches analysis/data_filter.py::is_fully_good()."""
    corr = getattr(item, 'correctness_numeric', None)
    sig = getattr(item, 'significance_numeric', None)
    evi = getattr(item, 'evidence_numeric', None)
    if corr is None or sig is None or evi is None:
        return False
    return corr == 1 and sig == 2 and evi == 1


def build_rubric(
    annotator_source: str = 'primary',
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[int]]:
    """Build the per-paper rubric of fully-good human review items.

    Args:
        annotator_source: Which annotator labels to use. Default 'primary'
            (matches the paper's Table 5).

    Returns:
        (rubric, dropped_paper_ids) where:
          rubric = {paper_id: [{reviewer_id, item_number, text}, ...]}
          dropped_paper_ids = paper IDs with zero fully-good human items
    """
    items, _rankings = load_annotations(annotator_source=annotator_source)

    # Filter to human reviewers only
    human_items = [i for i in items if i.reviewer_type == 'Human']

    # Group fully-good human items by paper
    rubric: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    all_paper_ids = set()
    for item in human_items:
        all_paper_ids.add(item.paper_id)
        if _is_fully_good(item):
            rubric[item.paper_id].append({
                'reviewer_id': item.reviewer_id,
                'item_number': item.item_number,
                'item_id': item.item_id,
            })

    # Papers with zero fully-good human items get dropped
    dropped = sorted(all_paper_ids - set(rubric.keys()))

    return dict(rubric), dropped


def build_rubric_with_texts(
    annotator_source: str = 'primary',
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[int]]:
    """Same as build_rubric but includes the review_item text (from
    expert_annotation HF config) so each rubric entry is self-contained
    for similarity comparison.

    Returns:
        (rubric, dropped_paper_ids) where each rubric entry has:
          {reviewer_id, item_number, item_id, text}
    """
    from load_data import load_expert_annotation_rows  # noqa: E402

    rubric, dropped = build_rubric(annotator_source)

    # Load expert_annotation to get the review_item text
    ea_rows = load_expert_annotation_rows()
    text_index: Dict[Tuple[int, str, int], str] = {}
    for r in ea_rows:
        pid = int(r['paper_id'])
        rid = r['reviewer_id']
        inum = int(r.get('review_item_number') or 0)
        text = r.get('review_item') or ''
        key = (pid, rid, inum)
        if key not in text_index:
            text_index[key] = text

    # Attach texts to rubric entries
    for pid, entries in rubric.items():
        for entry in entries:
            key = (pid, entry['reviewer_id'], entry['item_number'])
            entry['text'] = text_index.get(key, '')

    # Drop entries with empty text (shouldn't happen but be safe)
    for pid in list(rubric.keys()):
        rubric[pid] = [e for e in rubric[pid] if e.get('text', '').strip()]
        if not rubric[pid]:
            del rubric[pid]
            if pid not in dropped:
                dropped.append(pid)
    dropped.sort()

    return rubric, dropped


def main():
    parser = argparse.ArgumentParser(description='Build recall rubric from fully-good human items')
    parser.add_argument('--annotator-source', default='primary',
                        choices=('primary', 'secondary', 'both'),
                        help="Which annotator labels to use (default: primary)")
    parser.add_argument('--save', type=Path, default=None,
                        help='Save rubric JSON to this file')
    args = parser.parse_args()

    print(f'Building rubric (annotator_source={args.annotator_source})...')
    rubric, dropped = build_rubric_with_texts(args.annotator_source)

    total_items = sum(len(v) for v in rubric.values())
    print(f'  Papers with rubric items: {len(rubric)}')
    print(f'  Papers dropped (0 fully-good human items): {len(dropped)} → {dropped}')
    print(f'  Total rubric items: {total_items}')
    print(f'  Mean items per paper: {total_items / max(len(rubric), 1):.1f}')

    # Show distribution
    from collections import Counter
    dist = Counter(len(v) for v in rubric.values())
    print(f'  Distribution of rubric items per paper:')
    for k in sorted(dist.keys()):
        print(f'    {k} items: {dist[k]} papers')

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        # Convert to serializable format
        out = {
            'annotator_source': args.annotator_source,
            'n_papers': len(rubric),
            'n_dropped': len(dropped),
            'dropped_paper_ids': dropped,
            'n_rubric_items': total_items,
            'rubric': {str(k): v for k, v in rubric.items()},
        }
        args.save.write_text(json.dumps(out, indent=2))
        print(f'\n  Saved to {args.save}')


if __name__ == '__main__':
    main()
