#!/usr/bin/env python3
"""
Download all 85 papers from HuggingFace and reconstruct per-reviewer
review files on disk.

Creates the following directory structure:

    papers/
    ├── paper1/
    │   ├── preprint/
    │   │   ├── preprint.md
    │   │   ├── images/
    │   │   ├── images_list.json
    │   │   ├── supplementary/
    │   │   └── code/
    │   └── reviews/
    │       ├── Claude.md
    │       ├── GPT.md
    │       ├── Gemini.md
    │       ├── reviewer_1.md
    │       ├── reviewer_2.md
    │       └── reviewer_3.md
    ├── paper2/
    │   ├── preprint/
    │   └── reviews/
    ...
    └── paper85/

Each review file aggregates ALL items from that reviewer into a single
document. Items are ordered by review_item_number. Cited references are
collected and deduplicated at the end (when structured data is available
from the meta_reviewer config).

Usage:
    python3 download_papers.py                          # default: ./papers/
    python3 download_papers.py --output-dir /path/to/papers
    python3 download_papers.py --limit 5                # first 5 papers only
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

sys.path.insert(0, str(Path(__file__).resolve().parent))

from load_data import (
    load_expert_annotation_rows,
    load_meta_reviewer,
    load_submitted_papers,
)


# Map the anonymized HF reviewer_ids to full model names used in the paper.
# Human reviewers keep their IDs as-is (Human_1, Human_2, Human_3).
REVIEWER_ID_TO_FULL_NAME = {
    'Claude': 'claude-opus-4-5',
    'GPT': 'gpt-5.2',
    'Gemini': 'gemini-3.0-pro-preview',
}


# ---------------------------------------------------------------------------
# Paper file resolution
# ---------------------------------------------------------------------------

def _write_paper_files(
    paper_dir: Path,
    file_refs: List[Dict[str, Any]],
    hash_to_bytes: Dict[str, Any],
) -> int:
    """Resolve file_refs via hash_to_bytes and write to paper_dir/preprint/.
    Returns the number of files written."""
    preprint_dir = paper_dir / 'preprint'
    n_written = 0
    for ref in file_refs:
        path = ref.get('path')
        if not path:
            continue
        blob = hash_to_bytes.get(ref.get('content_hash'))
        if blob is None:
            continue
        content = blob['content_bytes'] if isinstance(blob, dict) else blob
        if not isinstance(content, (bytes, bytearray)):
            continue

        out_path = preprint_dir / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(bytes(content))
        n_written += 1
    return n_written


# ---------------------------------------------------------------------------
# Review reconstruction
# ---------------------------------------------------------------------------

def _build_review_markdown(
    reviewer_id: str,
    items: List[Dict[str, Any]],
    structured_items: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Build a single markdown review file for one reviewer.

    `items` is a list of expert_annotation rows for this reviewer, sorted
    by review_item_number. `structured_items` is an optional dict keyed by
    item_number with structured fields from the meta_reviewer config
    (review_content, review_claim, review_evidence, review_cited_references).

    If structured data is available, format each item with separate
    claim/evidence sections and collect references at the end. Otherwise,
    use the merged review_item text directly.
    """
    lines = [f'# Review by {reviewer_id}\n']
    all_refs: List[str] = []
    seen_refs: Set[str] = set()

    for item in items:
        item_num = int(item.get('review_item_number') or 0)
        lines.append(f'\n## Item {item_num}\n')

        struct = (structured_items or {}).get(item_num)
        if struct and (struct.get('review_claim') or struct.get('review_evidence')):
            # Structured format
            main_point = struct.get('review_content') or ''
            if main_point:
                lines.append(f'**Main point of criticism:** {main_point}\n')

            claim = struct.get('review_claim')
            if claim:
                lines.append(f'\n**Claim:**\n{claim}\n')

            evidence = struct.get('review_evidence')
            if evidence:
                lines.append(f'\n**Evidence:**\n{evidence}\n')

            cited = struct.get('review_cited_references') or []
            for ref in cited:
                ref_str = str(ref).strip()
                if ref_str and ref_str not in seen_refs:
                    seen_refs.add(ref_str)
                    all_refs.append(ref_str)
        else:
            # Merged text only
            review_text = item.get('review_item') or ''
            lines.append(f'{review_text}\n')

    # Deduplicated references at the end
    if all_refs:
        lines.append('\n---\n')
        lines.append('\n## References\n')
        for i, ref in enumerate(all_refs, 1):
            lines.append(f'{i}. {ref}')
        lines.append('')

    return '\n'.join(lines)


def _write_reviews(
    paper_dir: Path,
    paper_id: int,
    ea_rows: List[Dict[str, Any]],
    mr_rows_by_item: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
) -> int:
    """Write per-reviewer review files to paper_dir/reviews/.
    Returns the number of review files written."""
    reviews_dir = paper_dir / 'reviews'
    reviews_dir.mkdir(parents=True, exist_ok=True)

    # Group EA rows by reviewer_id, deduplicating by (reviewer_id, item_number)
    by_reviewer: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in ea_rows:
        rid = row['reviewer_id']
        inum = int(row.get('review_item_number') or 0)
        if inum not in by_reviewer[rid]:
            by_reviewer[rid][inum] = row

    n_written = 0
    for reviewer_id, items_dict in sorted(by_reviewer.items()):
        # Sort by item number
        items_sorted = [items_dict[k] for k in sorted(items_dict.keys())]

        # Get structured data from meta_reviewer if available
        structured: Optional[Dict[int, Dict[str, Any]]] = None
        if mr_rows_by_item:
            structured = {}
            for inum in items_dict:
                key = (reviewer_id, inum)
                if key in mr_rows_by_item:
                    structured[inum] = mr_rows_by_item[key]

        # Use full model name for AI reviewers (e.g., gpt-5.2 instead of GPT)
        display_name = REVIEWER_ID_TO_FULL_NAME.get(reviewer_id, reviewer_id)
        md = _build_review_markdown(display_name, items_sorted, structured)

        safe_name = display_name.replace('/', '_').replace(' ', '_')
        review_path = reviews_dir / f'{safe_name}.md'
        review_path.write_text(md, encoding='utf-8')
        n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Download papers and reconstruct reviews from HuggingFace'
    )
    parser.add_argument(
        '--output-dir', type=Path,
        default=Path(__file__).resolve().parent / 'papers',
        help='Root directory for paper{N}/ subdirectories. Default: ./papers/',
    )
    parser.add_argument(
        '--limit', type=int, default=None,
        help='Only download the first N papers (smoke test).',
    )
    parser.add_argument(
        '--skip-existing', action='store_true',
        help='Skip papers whose directory already exists.',
    )
    args = parser.parse_args()

    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Load all data from HF
    print('Loading expert_annotation config (review items + file_refs)...')
    ea_rows = load_expert_annotation_rows()
    print(f'  {len(ea_rows)} rows')

    print('Loading meta_reviewer config (structured review fields for 27 papers)...')
    try:
        mr_rows = load_meta_reviewer()
        print(f'  {len(mr_rows)} rows')
    except Exception as e:
        print(f'  WARNING: failed to load meta_reviewer ({e}); structured fields unavailable')
        mr_rows = []

    print('Loading submitted_papers (file bytes)...')
    hash_to_bytes = load_submitted_papers()
    print(f'  {len(hash_to_bytes)} blobs')

    # Index EA rows by paper_id
    ea_by_paper: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in ea_rows:
        ea_by_paper[int(row['paper_id'])].append(row)

    # Index MR rows by (paper_id, reviewer_id, item_number) for structured lookup
    mr_by_paper_reviewer_item: Dict[int, Dict[Tuple[str, int], Dict[str, Any]]] = defaultdict(dict)
    for row in mr_rows:
        pid = int(row['paper_id'])
        rid = row['reviewer_id']
        inum = int(row.get('review_item_number') or row.get('item_number') or 0)
        mr_by_paper_reviewer_item[pid][(rid, inum)] = row

    # Build unique paper list from expert_annotation (one entry per paper_id)
    papers_by_pid: Dict[int, Dict[str, Any]] = {}
    for row in ea_rows:
        pid = int(row['paper_id'])
        if pid not in papers_by_pid:
            papers_by_pid[pid] = row
    papers = sorted(papers_by_pid.values(), key=lambda r: int(r['paper_id']))
    print(f'  {len(papers)} unique papers')
    if args.limit:
        papers = papers[:args.limit]

    print(f'\nDownloading {len(papers)} papers to {output_root}/\n')

    for i, paper_row in enumerate(papers):
        pid = int(paper_row['paper_id'])
        title = paper_row.get('paper_title', '')[:60]
        paper_dir = output_root / f'paper{pid}'

        if args.skip_existing and paper_dir.exists():
            print(f'[{i+1}/{len(papers)}] paper{pid}: skipping (already exists)')
            continue

        paper_dir.mkdir(parents=True, exist_ok=True)

        # Write paper files
        file_refs = paper_row.get('file_refs') or []
        n_files = _write_paper_files(paper_dir, file_refs, hash_to_bytes)

        # Write reviews
        paper_ea = ea_by_paper.get(pid, [])
        paper_mr = mr_by_paper_reviewer_item.get(pid)
        n_reviews = _write_reviews(paper_dir, pid, paper_ea, paper_mr)

        # Count reviewers and items
        reviewers = set(r['reviewer_id'] for r in paper_ea)
        n_items = len(set(
            (r['reviewer_id'], int(r.get('review_item_number') or 0))
            for r in paper_ea
        ))

        print(f'[{i+1}/{len(papers)}] paper{pid}: {n_files} files, '
              f'{n_reviews} reviews ({len(reviewers)} reviewers, {n_items} items)  '
              f'"{title}"')

    print(f'\nDone. Papers written to {output_root}/')


if __name__ == '__main__':
    main()
