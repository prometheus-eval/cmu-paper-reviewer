#!/usr/bin/env python3
"""
Parse a generated review markdown into structured review items.

Handles the PeerReview Bench review format:

    ## Item {n}: <title>

    #### Claim
    * Main point of criticism: <text>
    * Evaluation criteria: <text>

    #### Evidence
    * Quote: <text>
       * Comment: <text>
    ...

Each item is extracted into a dict with:
    {item_number, title, main_point, claim_full, evidence_full, text}

where `text` is the merged representation used for similarity comparison
(main_point + evidence, matching the format in the HF dataset).

Usage:
    python3 parse_review.py papers/paper1/review/review_claude-opus-4-6.md
    python3 parse_review.py papers/paper1/review/review_claude-opus-4-6.md --save
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# Regex patterns for the structured review format
# Matches both "## Item 1: Title" (agent format) and "## Item 1" (HF format)
_ITEM_HEADER_RE = re.compile(
    r'^##\s*Item\s+(\d+)\s*(?::\s*(.*?))?\s*$',
    re.MULTILINE | re.IGNORECASE,
)

_SECTION_RE = re.compile(
    r'^####\s+(Claim|Evidence|Concrete Action Item)\s*$',
    re.MULTILINE | re.IGNORECASE,
)

_MAIN_POINT_RE = re.compile(
    r'\*\s*Main\s+point\s+of\s+criticism\s*:\s*(.*?)(?=\n\s*\*\s|\n\s*####|\Z)',
    re.DOTALL | re.IGNORECASE,
)


def parse_review_markdown(text: str) -> List[Dict[str, Any]]:
    """Parse a PeerReview Bench–formatted review into structured items.

    Args:
        text: Full review markdown text.

    Returns:
        List of dicts, each with keys:
          item_number, title, main_point, claim_full, evidence_full, text
    """
    items: List[Dict[str, Any]] = []

    # Find all item headers and their positions
    headers = list(_ITEM_HEADER_RE.finditer(text))
    if not headers:
        return items

    for i, header in enumerate(headers):
        item_num = int(header.group(1))
        title = (header.group(2) or '').strip()

        # Extract the full text of this item (until next item header or end)
        start = header.end()
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        item_text = text[start:end].strip()

        # Split into sections
        claim_full = ''
        evidence_full = ''
        action_full = ''

        sections = list(_SECTION_RE.finditer(item_text))
        for j, sec in enumerate(sections):
            sec_name = sec.group(1).lower()
            sec_start = sec.end()
            sec_end = sections[j + 1].start() if j + 1 < len(sections) else len(item_text)
            sec_text = item_text[sec_start:sec_end].strip()

            if sec_name == 'claim':
                claim_full = sec_text
            elif sec_name == 'evidence':
                evidence_full = sec_text
            elif 'action' in sec_name:
                action_full = sec_text

        # If no sections found, treat entire item_text as claim+evidence
        if not sections:
            claim_full = item_text

        # Extract the main point from the claim section
        main_point = ''
        mp_match = _MAIN_POINT_RE.search(claim_full)
        if mp_match:
            main_point = mp_match.group(1).strip()
        elif claim_full:
            # Fallback: use the first paragraph of the claim
            main_point = claim_full.split('\n\n')[0].strip()
            # Strip markdown bullet prefix if present
            main_point = re.sub(r'^\*\s*', '', main_point)

        # Build the merged text representation for similarity comparison.
        # This mirrors how the HF dataset's review_item field is constructed:
        # main_point + evidence, with structural markup stripped.
        parts = []
        if main_point:
            parts.append(main_point)
        if evidence_full:
            # Strip quote/comment markup for the merged text
            ev_clean = evidence_full
            ev_clean = re.sub(r'^\s*\*\s*Quote:\s*', '', ev_clean, flags=re.MULTILINE | re.IGNORECASE)
            ev_clean = re.sub(r'^\s*\*\s*Comment:\s*', '', ev_clean, flags=re.MULTILINE | re.IGNORECASE)
            ev_clean = re.sub(r'\n{3,}', '\n\n', ev_clean).strip()
            if ev_clean:
                parts.append(ev_clean)
        merged_text = '\n\n'.join(parts)

        items.append({
            'item_number': item_num,
            'title': title,
            'main_point': main_point,
            'claim_full': claim_full,
            'evidence_full': evidence_full,
            'text': merged_text,
        })

    return items


def parse_review_file(
    review_path: Path,
    save: bool = False,
) -> List[Dict[str, Any]]:
    """Parse a review markdown file and optionally save the items JSON.

    If save=True, writes to the same directory as review_items_{stem}.json.
    """
    text = review_path.read_text(encoding='utf-8')
    items = parse_review_markdown(text)

    if save and items:
        stem = review_path.stem  # e.g. "review_claude-opus-4-6"
        out_name = f'review_items_{stem.replace("review_", "")}.json'
        out_path = review_path.parent / out_name
        out_path.write_text(json.dumps(items, indent=2))
        print(f'  Saved {len(items)} items → {out_path}')

    return items


def load_review_items(
    paper_dir: Path,
    model_name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load review items for a paper.

    Tries in order:
    1. review_items_{model}.json (pre-parsed JSON — BYOJ or previously parsed)
    2. review_{model}.md (parse on the fly)
    3. Any review_items_*.json in the review/ dir
    4. Any review_*.md in the review/ dir

    Returns list of item dicts, or empty list if nothing found.
    """
    review_dir = paper_dir / 'review'
    if not review_dir.exists():
        return []

    # Try exact model match first
    if model_name:
        slug = model_name.split('/')[-1]
        # Try JSON
        json_path = review_dir / f'review_items_{slug}.json'
        if json_path.exists():
            return json.loads(json_path.read_text(encoding='utf-8'))
        # Try markdown
        md_path = review_dir / f'review_{slug}.md'
        if md_path.exists():
            return parse_review_file(md_path, save=True)

    # Try any review_items JSON
    for jp in sorted(review_dir.glob('review_items_*.json')):
        try:
            return json.loads(jp.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            continue

    # Try any review markdown
    for mp in sorted(review_dir.glob('review_*.md')):
        items = parse_review_file(mp, save=True)
        if items:
            return items

    return []


def main():
    parser = argparse.ArgumentParser(description='Parse review markdown into structured items')
    parser.add_argument('review_file', type=Path,
                        help='Path to review markdown file')
    parser.add_argument('--save', action='store_true',
                        help='Save items JSON next to the markdown file')
    args = parser.parse_args()

    items = parse_review_file(args.review_file, save=args.save)
    print(f'Parsed {len(items)} items from {args.review_file.name}')
    for item in items:
        print(f'  Item {item["item_number"]}: {item["title"][:60]}')
        print(f'    main_point: {item["main_point"][:80]}...')
        print(f'    text length: {len(item["text"])} chars')


if __name__ == '__main__':
    main()
