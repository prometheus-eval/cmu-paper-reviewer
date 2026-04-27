#!/usr/bin/env python3
"""
Run GPT-5.4 meta-reviewer on HUMAN review items only, for papers NOT
already covered by the meta-review bench.

Only evaluates Human_1, Human_2, Human_3 reviews — skips AI reviews
since those are already covered by the precision eval trajectories.

Uses api_key/litellm.txt and api_key/base_url.txt for credentials.

Usage:
    python3 run_human_metareview.py
    python3 run_human_metareview.py --limit 3
    python3 run_human_metareview.py --model openai/gpt-5.4
"""

import argparse
import json
import os
import sys
import shutil
import time
import uuid
from collections import defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Any

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_FORCE_REDOWNLOAD', '0')

_HERE = Path(__file__).resolve().parent
_BENCH = _HERE.parent

sys.path.insert(0, str(_BENCH))
sys.path.insert(0, str(_BENCH / 'metareview_bench'))
sys.path.insert(0, str(_BENCH / 'metareview_bench' / 'expert_annotation_meta_review'))

from load_data import load_annotations, load_meta_reviewer
from prompts import _DIMENSION_DEFINITIONS, _TENCLASS_TAXONOMY

DROPPED_PAPERS = {11, 20, 22}
HUMAN_REVIEWERS = {'Human_1', 'Human_2', 'Human_3'}
OUTPUT_DIR = _HERE / 'human_metareview_results'

# Import the 10-class labels from the meta-review bench prompts
from run_meta_review_agent import TENCLASS_LABELS, AGENT_SYSTEM_PROMPT, _slug


# ── Prompt (adapted to only evaluate human reviewers) ──────────────────

HUMAN_ONLY_USER_PROMPT = dedent("""\
    You are given a scientific paper and review items written by HUMAN
    reviewers only. Judge EVERY review item on three axes: correctness,
    significance, and evidence sufficiency.

    You are NOT writing a new review. You are judging existing review items.

    {dimension_definitions}

    {tenclass_taxonomy}

    ### Paper location
    The paper's source files are at: {paper_preprint_dir}

    Directory structure:
    ```
    preprint/
    ├── preprint.md        (main paper)
    ├── images_list.json   (figure captions)
    ├── images/            (figures)
    ├── supplementary/     (optional)
    └── code/              (optional)
    ```

    ### Human review files to meta-review
    The following human reviewer files are at: {paper_reviews_dir}
    Reviewers: {reviewer_list}

    Read each reviewer's .md file and judge every "## Item" section.

    ### Output format (STRICT)
    Write a single JSON file to {output_file} with this exact shape:

    ```json
    {{
      "paper_id": {paper_id},
      "reviewers": [
        {{
          "reviewer_id": "Human_1",
          "items": [
            {{
              "item_number": 1,
              "reasoning": "...",
              "correctness": "Correct",
              "significance": "Significant",
              "evidence": "Sufficient",
              "prediction_of_expert_judgments": "correct_significant_sufficient"
            }}
          ]
        }}
      ]
    }}
    ```

    Rules:
    - Include ALL human reviewers and ALL their items — no exceptions.
    - Each item must have: item_number, reasoning, correctness, significance,
      evidence, prediction_of_expert_judgments.
    - Use exact label strings.
    - null for cascade-skipped fields.
    - prediction_of_expert_judgments must be one of:
      {tenclass_labels_str}

    ### CRITICAL: Verification before finishing
    After writing the JSON file:
    1. Read it back and verify valid JSON.
    2. Count reviewers — must match the human reviewer .md files.
    3. Count items per reviewer — must match "## Item" sections.
    4. Check all label strings are exact matches.
    5. Fix and re-verify if any check fails.

    Only after all checks pass, print:
    "Verification complete. All reviewers and items included."
    Then stop.

    ### Guidelines
    - The paper is the source of truth.
    - OCR errors in preprint.md are possible — infer from context.
    - Figures are at preprint/images/figure1.png, figure2.png, etc.
    - When a review item references code, open the file and verify.
    - "Significant" means the criticism would genuinely improve the paper.

    ### Filesystem boundaries
    - READ from {paper_preprint_dir} and {paper_reviews_dir}.
    - WRITE only to {output_file}. Do not create any other files.
    - Do not navigate to parent or sibling directories.
    """)


def resolve_api_key():
    key = os.environ.get('LITELLM_API_KEY')
    if not key:
        key_file = _BENCH / 'api_key' / 'litellm.txt'
        if key_file.is_file():
            key = key_file.read_text(encoding='utf-8').strip()
    if not key:
        raise RuntimeError("No API key. Set LITELLM_API_KEY or put key in api_key/litellm.txt")
    return key


def resolve_base_url():
    url = os.environ.get('LITELLM_BASE_URL')
    if not url:
        url_file = _BENCH / 'api_key' / 'base_url.txt'
        if url_file.is_file():
            url = url_file.read_text(encoding='utf-8').strip()
    return (url or 'https://cmu.litellm.ai').rstrip('/')


def get_gap_papers():
    """Find papers with human reviewers not in meta-review bench."""
    mr_rows = load_meta_reviewer()
    mr_papers = set(r['paper_id'] for r in mr_rows)

    items, _ = load_annotations('primary')
    papers_with_humans = set()
    for it in items:
        if it.paper_id not in DROPPED_PAPERS and it.reviewer_type == 'Human':
            papers_with_humans.add(it.paper_id)

    return sorted(papers_with_humans - mr_papers)


def run_human_metareview_on_paper(
    paper_id: int,
    paper_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    api_key: str,
    base_url: str,
    max_iterations: int = 5000,
) -> Dict[str, Any]:
    """Run meta-reviewer on human review items only for one paper."""
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal.definition import TerminalTool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    preprint_dir = paper_dir / 'preprint'
    reviews_dir = paper_dir / 'reviews'

    if not preprint_dir.exists():
        return {'status': 'no_preprint_dir'}

    # Find only human reviewer files
    human_files = sorted(
        f for f in reviews_dir.glob('*.md')
        if f.stem in HUMAN_REVIEWERS
    ) if reviews_dir.exists() else []

    if not human_files:
        return {'status': 'no_human_reviews'}

    reviewer_list = ', '.join(f.stem for f in human_files)

    model_slug = _slug(model_name)
    model_out_dir = output_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = model_out_dir / f'{model_slug}_paper{paper_id}_metareview.json'
    conversation_dir = model_out_dir / 'conversations' / f'paper{paper_id}'

    tenclass_labels_str = '\n      '.join(f'"{lab}"' for lab in TENCLASS_LABELS)

    user_prompt = HUMAN_ONLY_USER_PROMPT.format(
        dimension_definitions=_DIMENSION_DEFINITIONS,
        tenclass_taxonomy=_TENCLASS_TAXONOMY,
        paper_preprint_dir=str(preprint_dir),
        paper_reviews_dir=str(reviews_dir),
        reviewer_list=reviewer_list,
        output_file=str(prediction_file),
        paper_id=paper_id,
        tenclass_labels_str=tenclass_labels_str,
    )

    llm = LLM(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        timeout=600,
        reasoning_effort='high',
        extended_thinking_budget=200000,
        temperature=1.0,
        drop_params=True,
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

    readable_id = f'human_metareview_{model_slug}_paper{paper_id}'
    conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

    conversation = Conversation(
        agent=agent,
        workspace=str(model_out_dir),
        persistence_dir=str(conversation_dir),
        conversation_id=conv_uuid,
        max_iteration_per_run=max_iterations,
    )

    conversation.send_message(AGENT_SYSTEM_PROMPT + '\n\n---\n\n' + user_prompt)
    try:
        conversation.run()
    except Exception as e:
        return {'status': f'exception:{type(e).__name__}'}

    if not prediction_file.exists():
        return {'status': 'no_file'}

    try:
        parsed = json.loads(prediction_file.read_text(encoding='utf-8'))
        n_items = sum(len(rev.get('items', [])) for rev in parsed.get('reviewers', []))
        return {'status': 'ok', 'n_items': n_items}
    except json.JSONDecodeError:
        return {'status': 'malformed_json'}


def main():
    parser = argparse.ArgumentParser(description='Meta-review human items for gap papers')
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--max-iterations', type=int, default=5000)
    args = parser.parse_args()

    api_key = resolve_api_key()
    base_url = resolve_base_url()

    if args.model is None:
        if 'cmu.litellm.ai' in base_url:
            args.model = 'litellm_proxy/azure_ai/gpt-5.4'
        else:
            args.model = 'openai/gpt-5.4'

    print(f"Model: {args.model}")
    print(f"Base URL: {base_url}")

    gap_papers = get_gap_papers()
    if args.limit:
        gap_papers = gap_papers[:args.limit]
    print(f"Papers to process: {len(gap_papers)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_slug = _slug(args.model)
    model_out_dir = OUTPUT_DIR / model_slug

    t0 = time.time()
    for i, pid in enumerate(gap_papers):
        paper_dir = _BENCH / 'papers' / f'paper{pid}'

        # Resume: check for existing valid prediction
        prediction_file = model_out_dir / f'{model_slug}_paper{pid}_metareview.json'
        if prediction_file.exists():
            try:
                existing = json.loads(prediction_file.read_text(encoding='utf-8'))
                reviewers = existing.get('reviewers', [])
                all_valid = all(
                    all(item.get('correctness') is not None for item in rev.get('items', []))
                    for rev in reviewers
                ) and len(reviewers) > 0
                if all_valid:
                    n = sum(len(rev['items']) for rev in reviewers)
                    print(f"  [{i+1}/{len(gap_papers)}] paper{pid}: cached ({n} items)")
                    continue
            except (json.JSONDecodeError, KeyError):
                pass
            prediction_file.unlink()

        # Count human items
        reviews_dir = paper_dir / 'reviews'
        n_human = len([f for f in reviews_dir.glob('*.md') if f.stem in HUMAN_REVIEWERS]) if reviews_dir.exists() else 0
        print(f"  [{i+1}/{len(gap_papers)}] paper{pid} ({n_human} human reviewers)...",
              end='', flush=True)

        # Retry loop
        max_retries = 3
        result = {'status': 'not_started'}
        for attempt in range(1, max_retries + 1):
            conv_dir = model_out_dir / 'conversations' / f'paper{pid}' if model_out_dir.exists() else None
            if conv_dir and conv_dir.exists():
                shutil.rmtree(conv_dir)

            if attempt > 1:
                print(f' retry {attempt}...', end='', flush=True)

            result = run_human_metareview_on_paper(
                paper_id=pid,
                paper_dir=paper_dir,
                output_dir=OUTPUT_DIR,
                model_name=args.model,
                api_key=api_key,
                base_url=base_url,
                max_iterations=args.max_iterations,
            )

            if result.get('status') == 'ok':
                break
            elif attempt == max_retries:
                print(f' all {max_retries} attempts failed', end='')

        elapsed = time.time() - t0
        n_items = result.get('n_items', 0)
        print(f' {result["status"]} ({n_items} items, {elapsed:.0f}s total)')

    print(f"\nDone. Results in {OUTPUT_DIR / model_slug}")


if __name__ == '__main__':
    main()
