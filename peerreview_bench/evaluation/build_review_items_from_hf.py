#!/usr/bin/env python3
"""
Build review_items_*.json files for the pre-existing AI reviewers
(GPT-5.2, Claude Opus 4.5, Gemini 3.0 Pro) from the HuggingFace
expert_annotation config.

The newer models (gpt-5.4, claude-opus-4-7, gemini-3.1-pro-preview,
gemini-3-flash-preview) already have these files generated during their
review runs. The pre-existing models only have the raw markdown
(e.g., gpt-5.2.md) without the review_items_*.json sidecar.

This script reconstructs the JSON from the HF data so that
evaluate_recall.py and evaluate_precision.py can find them via
--model-name.

Usage:
    python3 build_review_items_from_hf.py --paper-root ../papers/
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from load_data import load_expert_annotation_rows

# HF reviewer_id -> file slug for the pre-existing models
REVIEWER_TO_SLUG = {
    'GPT': 'gpt-5.2',
    'Claude': 'claude-opus-4-5',
    'Gemini': 'gemini-3.0-pro-preview',
}


def main():
    parser = argparse.ArgumentParser(
        description='Build review_items JSON from HF expert_annotation')
    parser.add_argument('--paper-root', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()

    print('Loading expert_annotation rows from HuggingFace...')
    rows = load_expert_annotation_rows()

    # Group by (paper_id, reviewer_id), deduplicate across annotator sources
    # (primary + secondary have the same review text, different annotations)
    items_by_paper_reviewer = defaultdict(dict)
    for r in rows:
        if r['reviewer_type'] != 'AI':
            continue
        rid = r['reviewer_id']
        if rid not in REVIEWER_TO_SLUG:
            continue
        pid = r['paper_id']
        item_num = r['review_item_number']
        # Deduplicate: same (paper, reviewer, item_number) may appear
        # in both primary and secondary, but review_item text is identical
        key = (pid, rid, item_num)
        if key not in items_by_paper_reviewer:
            items_by_paper_reviewer[key] = {
                'paper_id': pid,
                'reviewer_id': rid,
                'item_number': item_num,
                'text': r.get('review_item', ''),
            }

    # Group into per-(paper, reviewer)
    by_paper_reviewer = defaultdict(list)
    for item in items_by_paper_reviewer.values():
        by_paper_reviewer[(item['paper_id'], item['reviewer_id'])].append(item)

    written = 0
    skipped = 0
    for (pid, rid), items in sorted(by_paper_reviewer.items()):
        slug = REVIEWER_TO_SLUG[rid]
        review_dir = args.paper_root / f'paper{pid}' / 'reviews'
        if not review_dir.exists():
            skipped += 1
            continue

        out_path = review_dir / f'review_items_{slug}.json'
        if out_path.exists():
            skipped += 1
            continue

        # Sort by item_number
        items.sort(key=lambda x: x['item_number'])

        # Build output in the same format as parse_review.py
        out_items = []
        for it in items:
            out_items.append({
                'item_number': it['item_number'],
                'title': '',
                'main_point': '',
                'claim_full': '',
                'evidence_full': '',
                'text': it['text'],
            })

        if args.dry_run:
            print(f'  [dry-run] paper{pid}/{slug}: {len(out_items)} items -> {out_path.name}')
        else:
            out_path.write_text(json.dumps(out_items, indent=2, ensure_ascii=False),
                                encoding='utf-8')
            written += 1

    print(f'\nWritten: {written}, Skipped (already exist or no dir): {skipped}')


if __name__ == '__main__':
    main()
