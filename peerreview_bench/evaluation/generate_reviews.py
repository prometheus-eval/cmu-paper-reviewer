#!/usr/bin/env python3
"""
Run an AI reviewer agent on each paper to generate reviews.

Uses the same OpenHands agent pattern as the backend (ReviewService), with
the configurable reviewer prompt from backend/reviewer_prompt.py.

The agent writes reviews to papers/paper{N}/review/review_{model}.md.
After generation, parse_review.py extracts the structured items.

Usage:
    python3 generate_reviews.py \
        --model-name litellm_proxy/anthropic/claude-opus-4-6 \
        --paper-root papers/ \
        --limit 5

    # With custom settings
    python3 generate_reviews.py \
        --model-name litellm_proxy/gemini/gemini-3.1-pro-preview \
        --paper-root papers/ \
        --max-items 3 \
        --criteria-preset neurips
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_BACKEND_DIR = _BENCH_DIR.parent / 'backend'

for p in (_HERE, _BENCH_DIR, _BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the backend's reviewer prompt builder
from reviewer_prompt import build_reviewer_prompt, get_default_settings  # noqa: E402
from parse_review import parse_review_file  # noqa: E402


def _extract_paper_id(paper_dir: Path) -> int:
    """Extract the integer paper ID from a directory name like 'paper42'."""
    try:
        return int(paper_dir.name.replace('paper', ''))
    except ValueError:
        return -1


def _validate_review(review_path: Path) -> bool:
    """Check that the review contains at least one '## Item N:' header."""
    if not review_path.exists():
        return False
    content = review_path.read_text(encoding='utf-8')
    return bool(re.search(r'^##\s*Item\s+\d+\s*:', content, re.MULTILINE | re.IGNORECASE))


def generate_review_for_paper(
    paper_dir: Path,
    model_name: str,
    *,
    review_settings: Dict[str, Any],
    api_key: str,
    base_url: str,
    max_iterations: int = 5000,
) -> Optional[Path]:
    """Run the OpenHands agent reviewer on one paper.

    Returns the path to the generated review markdown, or None if generation
    failed.
    """
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal.definition import TerminalTool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    preprint_dir = paper_dir / 'preprint'
    review_dir = paper_dir / 'review'
    review_dir.mkdir(parents=True, exist_ok=True)

    model_short = model_name.split('/')[-1]
    review_path = review_dir / f'review_{model_short}.md'

    link_to_paper = str(preprint_dir)

    # Build prompt
    prompt = build_reviewer_prompt(review_settings)
    prompt = prompt.replace('[LINK TO THE PAPER]', link_to_paper)
    prompt = prompt.replace('[MODEL NAME]', model_short)

    # OpenHands setup (matching backend/review_service.py)
    llm = LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
    )
    condenser = LLMSummarizingCondenser(
        llm=llm.model_copy(update={'usage_id': 'condenser'}),
        max_size=200,
        keep_first=3,
    )
    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ],
        condenser=condenser,
    )

    readable_id = f'{model_name.replace("/", "_")}_{paper_dir.name}'.replace('.', '_').replace('-', '_')
    conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

    conversation = Conversation(
        agent=agent,
        workspace=str(paper_dir.parent),
        persistence_dir=str(review_dir / f'{model_short}_trajectory'),
        conversation_id=conv_uuid,
        max_iteration_per_run=max_iterations,
    )

    conversation.send_message(prompt)
    conversation.run()

    cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
    del conversation

    # Validate
    if _validate_review(review_path):
        return review_path

    return None


def generate_reviews(
    paper_root: Path,
    model_name: str,
    *,
    max_items: int = 5,
    criteria_preset: str = 'nature',
    limit: Optional[int] = None,
    skip_existing: bool = True,
    api_key: Optional[str] = None,
    base_url: str = 'https://cmu.litellm.ai',
    max_iterations: int = 5000,
):
    """Generate reviews for all papers in paper_root."""
    import time

    # Resolve API key
    if not api_key:
        api_key = os.environ.get('LITELLM_API_KEY')
    if not api_key:
        key_file = Path(os.environ.get('LITELLM_KEY_FILE', ''))
        if key_file.is_file():
            api_key = key_file.read_text(encoding='utf-8').strip()
    if not api_key:
        fallback = _BENCH_DIR / 'api_key' / 'litellm.txt'
        if fallback.is_file():
            api_key = fallback.read_text(encoding='utf-8').strip()
    if not api_key:
        print('ERROR: no API key. Set LITELLM_API_KEY or provide a key file.',
              file=sys.stderr)
        sys.exit(1)

    # Build review settings
    review_settings = get_default_settings()
    review_settings['max_items'] = max_items
    review_settings['reviewer_criteria_preset'] = criteria_preset

    # Find papers, excluding those with empty rubrics (no fully-good human
    # items → no recall ground truth → not worth generating a review for).
    from build_rubric import build_rubric  # noqa: E402
    rubric, dropped = build_rubric()
    dropped_set = set(dropped)

    paper_dirs = sorted(paper_root.glob('paper*'))
    paper_dirs = [
        pd for pd in paper_dirs
        if _extract_paper_id(pd) not in dropped_set
    ]
    if limit:
        paper_dirs = paper_dirs[:limit]

    model_short = model_name.split('/')[-1]

    print(f'Generating reviews for {len(paper_dirs)} papers')
    print(f'  (skipped {len(dropped_set)} papers with empty rubric: {sorted(dropped_set)})')
    print(f'  model: {model_name}')
    print(f'  max_items: {max_items}')
    print(f'  criteria: {criteria_preset}')

    for i, pd in enumerate(paper_dirs):
        review_path = pd / 'review' / f'review_{model_short}.md'
        if skip_existing and review_path.exists() and _validate_review(review_path):
            print(f'  [{i+1}/{len(paper_dirs)}] {pd.name}: skipping (review exists)')
            continue

        print(f'  [{i+1}/{len(paper_dirs)}] {pd.name}: generating...',
              end='', flush=True)
        t0 = time.time()
        result = generate_review_for_paper(
            pd, model_name,
            review_settings=review_settings,
            api_key=api_key,
            base_url=base_url,
            max_iterations=max_iterations,
        )
        elapsed = time.time() - t0

        if result:
            # Parse immediately
            items = parse_review_file(result, save=True)
            print(f' done ({elapsed:.0f}s, {len(items)} items)')
        else:
            print(f' FAILED ({elapsed:.0f}s)')


def main():
    parser = argparse.ArgumentParser(
        description='Generate AI reviews for papers using an OpenHands agent'
    )
    parser.add_argument('--model-name', type=str, required=True,
                        help='LLM model name for the reviewer agent')
    parser.add_argument('--paper-root', type=Path, required=True,
                        help='Root dir with paper{N}/ subdirectories')
    parser.add_argument('--max-items', type=int, default=5,
                        help='Max review items per paper (default 5)')
    parser.add_argument('--criteria-preset', type=str, default='nature',
                        choices=('nature', 'neurips'),
                        help='Evaluation criteria preset')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only generate for first N papers')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                        help='Skip papers that already have reviews')
    parser.add_argument('--max-iterations', type=int, default=5000,
                        help='Max OpenHands iterations per paper')
    parser.add_argument('--base-url', type=str, default='https://cmu.litellm.ai')
    args = parser.parse_args()

    generate_reviews(
        paper_root=args.paper_root,
        model_name=args.model_name,
        max_items=args.max_items,
        criteria_preset=args.criteria_preset,
        limit=args.limit,
        skip_existing=args.skip_existing,
        max_iterations=args.max_iterations,
        base_url=args.base_url,
    )


if __name__ == '__main__':
    main()
