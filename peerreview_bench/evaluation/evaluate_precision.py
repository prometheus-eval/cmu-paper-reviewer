#!/usr/bin/env python3
"""
Precision metric: "How good are the AI reviewer's items in terms of
correctness, significance, and sufficiency of evidence?"

Uses an OpenHands agent meta-reviewer (axis mode) that navigates the
paper's files and judges each AI review item. One conversation per paper.
An item is "fully good" if the judge says Correct + Significant + Sufficient.

    Precision = fully-good AI items / total AI items

Usage:
    python3 evaluate_precision.py --paper-root papers/ --model-name my-agent \
        --judge-model litellm_proxy/azure_ai/gpt-5.4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_EA_META_DIR = _BENCH_DIR / 'metareview_bench' / 'expert_annotation_meta_review'

for p in (_HERE, _BENCH_DIR, _EA_META_DIR, _BENCH_DIR / 'metareview_bench'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from parse_review import load_review_items  # noqa: E402
from prompts import _DIMENSION_DEFINITIONS  # noqa: E402


def _is_fully_good(pred: Dict[str, Any]) -> bool:
    return (
        pred.get('correctness') == 'Correct'
        and pred.get('significance') == 'Significant'
        and pred.get('evidence') == 'Sufficient'
    )


# ============================================================================
# Agent system + user prompts for precision meta-review
# ============================================================================

PRECISION_SYSTEM_PROMPT = dedent("""\
    You are a meta-reviewer agent for scientific papers. You will be given
    a paper's source files on disk AND a set of review items written by an
    AI reviewer. Your task is to judge EVERY review item along three axes:
    correctness, significance, and evidence sufficiency.

    You are NOT writing a new review. You are judging the quality of
    existing review items by verifying their claims against the paper.
    """) + "\n" + _DIMENSION_DEFINITIONS


PRECISION_USER_PROMPT = dedent("""\
    ### Paper location
    The paper's source files are at: {paper_preprint_dir}

    The directory structure is:
    ```
    preprint/
    ├── preprint.md        (main paper, markdown)
    ├── images_list.json   (list of figure images with captions)
    ├── images/            (figure files referenced by the paper)
    ├── supplementary/     (optional supplementary materials)
    └── code/              (optional source code)
    ```

    ### Review items to meta-review
    The following are review items written by an AI reviewer. Judge each one.

    {review_items_text}

    ### Decision procedure (for EACH item)
    Step 1 (Understand). Read the item. What is the main point?
    Step 2 (Correctness). Verify against the paper. Is the core claim correct?
      - If YES → "Correct". Continue to Step 3.
      - If NO  → "Not Correct". Set significance and evidence to null. Done.
    Step 3 (Significance). Would addressing this improve the paper?
      - Insightful and helpful → "Significant"
      - Not helpful but worth keeping → "Marginally Significant"
      - Should be removed → "Not Significant". Set evidence to null. Done.
    Step 4 (Evidence). Does the reviewer provide enough justification?
      - Yes → "Sufficient"
      - No  → "Requires More"

    ### Output format (STRICT)
    Write a single JSON file to {output_file} with this exact shape:

    ```json
    {{
      "items": [
        {{"item_number": 1, "reasoning": "...", "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient"}},
        {{"item_number": 2, "reasoning": "...", "correctness": "Not Correct", "significance": null, "evidence": null}}
      ]
    }}
    ```

    Rules:
    - Include ALL {n_items} items — no exceptions.
    - Each item must have: item_number, reasoning, correctness, significance, evidence.
    - Use exact label strings.
    - null for cascade-skipped fields.

    ### CRITICAL: Verification before finishing
    After writing the JSON file:
    1. Read it back and verify valid JSON.
    2. Count items — must be exactly {n_items}.
    3. Check all label strings are exact matches.
    4. Fix and re-verify if any check fails.

    Only after all checks pass, print:
    "Verification complete. All {n_items} items judged."
    Then stop.

    ### Guidelines
    - The paper is the source of truth.
    - OCR errors in preprint.md are possible — infer from context.
    - Figures are at preprint/images/figure1.png, figure2.png, etc.
    - When a review item references code, open the file and verify.
    - "Significant" means the criticism would genuinely improve the paper.

    ### Filesystem boundaries
    - Only read files under {paper_preprint_dir}.
    - Only write to {output_file}. Do not create any other files.
    - Do not navigate to parent or sibling directories.
    """)


# ============================================================================
# Agent execution
# ============================================================================

def _format_review_items_for_prompt(items: List[Dict[str, Any]]) -> str:
    parts = []
    for item in items:
        num = item.get('item_number', '?')
        text = item.get('text') or item.get('main_point') or ''
        parts.append(f"#### Item {num}\n{text}\n")
    return '\n'.join(parts)


def run_precision_agent_on_paper(
    paper_id: int,
    paper_dir: Path,
    ai_items: List[Dict[str, Any]],
    output_dir: Path,
    *,
    model_name: str,
    reviewer_model_name: str = '',
    api_key: str,
    base_url: str,
    max_iterations: int = 5000,
) -> Dict[str, Any]:
    """Run the agent meta-reviewer on one paper's AI review items.
    Returns {predictions: [{item_number, correctness, significance, evidence}, ...], status}."""
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal.definition import TerminalTool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    preprint_dir = paper_dir / 'preprint'
    if not preprint_dir.exists():
        return {'predictions': [], 'status': 'no_preprint_dir'}

    model_short = model_name.split('/')[-1]
    reviewer_short = reviewer_model_name.split('/')[-1] if reviewer_model_name else 'unknown'
    traj_name = f'reviewer_{reviewer_short}_meta_reviewer_{model_short}_precision_trajectories'
    trajectory_dir = output_dir / traj_name / f'paper{paper_id}'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = trajectory_dir / 'prediction.json'

    # Build prompt
    review_items_text = _format_review_items_for_prompt(ai_items)
    user_prompt = PRECISION_USER_PROMPT.format(
        paper_preprint_dir=str(preprint_dir),
        review_items_text=review_items_text,
        output_file=str(prediction_file),
        n_items=len(ai_items),
    )

    llm = LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        reasoning_effort='high',
        extended_thinking_budget=200000,
        temperature=1.0,
        drop_params=True,
        timeout=600,
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

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        # Clean up stale conversation from a previous failed attempt
        conv_dir = trajectory_dir / 'conversation'
        if conv_dir.exists():
            import shutil
            shutil.rmtree(conv_dir)

        readable_id = f'precision_{model_short}_paper{paper_id}_attempt{attempt}'
        conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

        conversation = Conversation(
            agent=agent,
            workspace=str(paper_dir),
            persistence_dir=str(conv_dir),
            conversation_id=conv_uuid,
            max_iteration_per_run=max_iterations,
        )

        conversation.send_message(PRECISION_SYSTEM_PROMPT + '\n\n---\n\n' + user_prompt)
        try:
            conversation.run()
        except Exception as e:
            if attempt < max_retries:
                print(f' attempt {attempt} failed ({type(e).__name__}), retrying...',
                      end='', flush=True)
                continue
            return {'predictions': [], 'status': f'exception:{type(e).__name__}'}

        # Check if prediction file was produced
        if prediction_file.exists():
            break
        elif attempt < max_retries:
            print(f' attempt {attempt} produced no file, retrying...', end='', flush=True)
        else:
            return {'predictions': [], 'status': 'no_file'}

    # Parse output
    if not prediction_file.exists():
        return {'predictions': [], 'status': 'no_file'}

    try:
        data = json.loads(prediction_file.read_text(encoding='utf-8'))
    except json.JSONDecodeError:
        return {'predictions': [], 'status': 'malformed_json'}

    predictions = []
    for item_block in data.get('items', []):
        predictions.append({
            'item_number': item_block.get('item_number'),
            'reasoning': item_block.get('reasoning', ''),
            'correctness': item_block.get('correctness'),
            'significance': item_block.get('significance'),
            'evidence': item_block.get('evidence'),
        })

    return {'predictions': predictions, 'status': 'ok'}


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Precision metric: quality of AI reviewer items via agent meta-review'
    )
    parser.add_argument('--paper-root', type=Path, required=True,
                        help='Root dir with paper{N}/ subdirectories')
    parser.add_argument('--model-name', type=str, default=None,
                        help='AI reviewer model name (to find review files)')
    parser.add_argument('--byoj', action='store_true',
                        help='Use any review_items_*.json found in reviews/ dirs')
    parser.add_argument('--judge-model', type=str,
                        default=None,
                        help='LLM model for agent meta-review judging (default: auto-detected from base URL)')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--output', type=Path, default=None,
                        help='Save results JSON')
    parser.add_argument('--output-dir', type=Path,
                        default=_BENCH_DIR / 'outputs' / 'eval',
                        help='Directory for agent trajectories')
    args = parser.parse_args()

    # Resolve API key
    api_key = os.environ.get('LITELLM_API_KEY')
    if not api_key:
        key_file = _BENCH_DIR / 'api_key' / 'litellm.txt'
        if key_file.is_file():
            api_key = key_file.read_text(encoding='utf-8').strip()
    if not api_key:
        print('ERROR: no API key.', file=sys.stderr)
        sys.exit(1)

    base_url = os.environ.get('LITELLM_BASE_URL')
    if not base_url:
        base_url_file = _BENCH_DIR / 'api_key' / 'base_url.txt'
        if base_url_file.is_file():
            base_url = base_url_file.read_text(encoding='utf-8').strip()
    if not base_url:
        base_url = 'https://cmu.litellm.ai'

    # Resolve default judge model based on active base URL
    if args.judge_model is None:
        if 'cmu.litellm.ai' in base_url:
            args.judge_model = 'litellm_proxy/azure_ai/gpt-5.4'
        else:
            args.judge_model = 'openai/gpt-5.4'

    # Find papers with AI reviews, excluding empty-rubric papers
    paper_root = args.paper_root.resolve()

    from build_rubric import build_rubric  # noqa: E402
    _rubric, dropped = build_rubric()
    dropped_set = set(dropped)

    paper_ids = []
    for pd in sorted(paper_root.glob('paper*'), key=lambda p: int(p.name.replace('paper', '')) if p.name.replace('paper', '').isdigit() else 0):
        try:
            pid = int(pd.name.replace('paper', ''))
        except ValueError:
            continue
        if pid in dropped_set:
            continue
        # Check if review items exist
        ai_items = load_review_items(pd, model_name=args.model_name if not args.byoj else None)
        if ai_items:
            paper_ids.append(pid)

    if args.limit:
        paper_ids = paper_ids[:args.limit]

    print(f'Found {len(paper_ids)} papers with AI review items')
    print(f'  Judge model: {args.judge_model}')

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_per_item: List[Dict[str, Any]] = []
    n_good = 0
    n_total = 0
    t0 = time.time()

    for i, pid in enumerate(paper_ids):
        paper_dir = paper_root / f'paper{pid}'
        ai_items = load_review_items(paper_dir, model_name=args.model_name if not args.byoj else None)

        # Resume logic: check for existing valid prediction.json
        reviewer_short = (args.model_name or 'byoj').split('/')[-1]
        judge_short = args.judge_model.split('/')[-1]
        traj_name = f'reviewer_{reviewer_short}_meta_reviewer_{judge_short}_precision_trajectories'
        prediction_file = args.output_dir / traj_name / f'paper{pid}' / 'prediction.json'

        skip = False
        if prediction_file.exists():
            try:
                existing = json.loads(prediction_file.read_text(encoding='utf-8'))
                existing_preds = existing.get('items', [])
                # Check: all items evaluated with non-null correctness?
                existing_by_num = {p['item_number']: p for p in existing_preds}
                all_valid = all(
                    existing_by_num.get(item.get('item_number'), {}).get('correctness') is not None
                    for item in ai_items
                )
                if all_valid and len(existing_preds) >= len(ai_items):
                    skip = True
            except (json.JSONDecodeError, KeyError):
                pass  # invalid file, re-run

        if skip:
            print(f'\n  [{i+1}/{len(paper_ids)}] paper{pid} ({len(ai_items)} items)...',
                  end='', flush=True)
            # Use existing predictions
            result = {'predictions': existing_preds, 'status': 'cached'}
            status = 'cached'
            preds = existing_preds
            print(f' cached ({sum(1 for p in preds if _is_fully_good(p))}/{len(ai_items)} good)')
        else:
            # Remove stale prediction file if it exists (incomplete run)
            if prediction_file.exists():
                prediction_file.unlink()

            print(f'\n  [{i+1}/{len(paper_ids)}] paper{pid} ({len(ai_items)} items)...',
                  end='', flush=True)

            result = run_precision_agent_on_paper(
                paper_id=pid,
                paper_dir=paper_dir,
                ai_items=ai_items,
                output_dir=args.output_dir,
                model_name=args.judge_model,
                reviewer_model_name=args.model_name or 'byoj',
                api_key=api_key,
                base_url=base_url,
            )

            status = result['status']
            preds = result['predictions']

        # Match predictions to AI items by item_number
        pred_by_num = {p['item_number']: p for p in preds}
        for item in ai_items:
            inum = item.get('item_number')
            pred = pred_by_num.get(inum, {})
            is_good = _is_fully_good(pred)
            n_total += 1
            if is_good:
                n_good += 1
            all_per_item.append({
                'paper_id': pid,
                'item_number': inum,
                'correctness': pred.get('correctness'),
                'significance': pred.get('significance'),
                'evidence': pred.get('evidence'),
                'is_fully_good': is_good,
            })

        if status != 'cached':
            elapsed = time.time() - t0
            paper_good = sum(1 for p in preds if _is_fully_good(p))
            print(f' {status} ({paper_good}/{len(ai_items)} good, {elapsed:.0f}s total)')

    precision = n_good / n_total if n_total > 0 else 0.0

    print(f'\n{"=" * 60}')
    print(f'PRECISION RESULTS')
    print(f'{"=" * 60}')
    print(f'Papers scored: {len(paper_ids)}')
    print(f'Total AI items: {n_total}')
    print(f'Fully good: {n_good}')
    print(f'Precision: {precision:.2%}')
    print(f'Elapsed: {time.time() - t0:.0f}s')

    if all_per_item:
        from collections import Counter
        corr = Counter(r['correctness'] for r in all_per_item)
        sig = Counter(r['significance'] for r in all_per_item if r['correctness'] == 'Correct')
        evi = Counter(r['evidence'] for r in all_per_item
                      if r['correctness'] == 'Correct'
                      and r['significance'] in ('Significant', 'Marginally Significant'))
        print(f'\nPer-axis breakdown:')
        print(f'  Correctness: {dict(corr)}')
        print(f'  Significance (if Correct): {dict(sig)}')
        print(f'  Evidence (if Correct + Sig/Marg): {dict(evi)}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out = {
            'metric': 'precision',
            'judge_model': args.judge_model,
            'ai_reviewer_model': args.model_name,
            'n_papers': len(paper_ids),
            'n_items': n_total,
            'n_fully_good': n_good,
            'precision': precision,
            'elapsed_seconds': round(time.time() - t0, 1),
            'per_item': all_per_item,
        }
        args.output.write_text(json.dumps(out, indent=2, default=str))
        print(f'\nSaved to {args.output}')


if __name__ == '__main__':
    main()
