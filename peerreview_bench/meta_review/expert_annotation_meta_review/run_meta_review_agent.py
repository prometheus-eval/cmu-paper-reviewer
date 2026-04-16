#!/usr/bin/env python3
"""
Agent-based meta-reviewer using the OpenHands SDK.

One OpenHands conversation per PAPER. The agent reads the paper from the
filesystem AND all reconstructed reviews (from papers/{paperN}/reviews/),
then produces a single JSON file with per-item meta-review labels for
EVERY reviewer × EVERY item on that paper.

Two prompt modes via --prompt-mode:
    axis     — per-item {correctness, significance, evidence}
    tenclass — per-item {label} (one of 10 collapsed classes)

Prerequisites:
    - openhands SDK installed
    - Papers downloaded via `download_papers.py` (creates papers/paper{N}/)
    - LITELLM_API_KEY env var or ./key_litellm.txt

Example:
    python3 run_meta_review_agent.py \
        --model-name litellm_proxy/anthropic/claude-opus-4-6 \
        --paper-root ../../papers \
        --prompt-mode axis --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from textwrap import dedent
from typing import Any, Dict, List, Optional, Tuple

# --- HF Hub timeouts ---------------------------------------------------------
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT', '120')
os.environ.setdefault('HF_HUB_ETAG_TIMEOUT', '120')

# --- Path plumbing -----------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_META_REVIEW_DIR = _HERE.parent
_BENCH_DIR = _META_REVIEW_DIR.parent

for p in (_HERE, _META_REVIEW_DIR, _BENCH_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from load_data import load_expert_annotation_rows  # noqa: E402
from prompts import (  # noqa: E402
    _DIMENSION_DEFINITIONS,
    _TENCLASS_TAXONOMY,
    _TENCLASS_DISAGREEMENT_GUIDANCE,
    TENCLASS_LABELS,
    TENCLASS_LABEL_TO_ID,
    extract_axis_answer,
    extract_tenclass_answer,
)
from predictors import _slug  # noqa: E402
from metrics import (  # noqa: E402
    evaluate_axis_predictions,
    evaluate_tenclass_predictions,
    format_mode_report,
)

DEFAULT_OUTPUT_ROOT = _BENCH_DIR / 'outputs' / 'expert_annotation_meta_review'
DEFAULT_BASE_URL = 'https://cmu.litellm.ai'


# ============================================================================
# Agent system prompt (detailed, matching the level of run.py reviewer prompt)
# ============================================================================

AGENT_SYSTEM_PROMPT_AXIS = dedent("""\
    You are a meta-reviewer agent for scientific papers. You will be given
    a paper's source files on disk AND reconstructed reviews written by
    multiple reviewers (human and AI). Your task is to meta-review EVERY
    review item from EVERY reviewer, judging each item along three axes:
    correctness, significance, and evidence sufficiency.

    You are NOT writing a new review. You are judging the quality of
    existing review items by verifying their claims against the paper.
    """) + "\n" + _DIMENSION_DEFINITIONS


AGENT_SYSTEM_PROMPT_TENCLASS = dedent("""\
    You are a meta-reviewer agent for scientific papers. You will be given
    a paper's source files on disk AND reconstructed reviews written by
    multiple reviewers (human and AI). Your task is to predict, for EVERY
    review item from EVERY reviewer, the collapsed 10-class label that a
    pair of independent expert meta-reviewers would jointly produce.

    You are NOT writing a new review. You are predicting how experts would
    judge the quality of existing review items.
    """) + "\n" + _DIMENSION_DEFINITIONS + "\n" + _TENCLASS_TAXONOMY + "\n" + _TENCLASS_DISAGREEMENT_GUIDANCE


# ============================================================================
# Agent user prompt (per-paper)
# ============================================================================

AGENT_USER_PROMPT_AXIS = dedent("""\
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

    ### Reviews to meta-review
    The reconstructed reviews are at: {paper_reviews_dir}
    Each file is one reviewer's full review with all their items.
    The reviewers are: {reviewer_list}

    ### Principles (ordered by importance)
    1. **Correctness first**: Your meta-review judgments must be
       factually accurate. Verify EVERY reviewer claim against the paper
       before judging it. If you're unsure whether a claim is correct,
       check the paper again rather than guessing.
    2. **Calibrate significance carefully**: "Significant" means the
       criticism would genuinely improve the paper if addressed — not
       that it threatens the paper's validity. Typos and stylistic
       preferences are "Marginally Significant" at best.
    3. **Evidence is about the reviewer's chain of reasoning**, not the
       underlying truth of their claim. A correct claim with no
       supporting evidence gets "Requires More." A wrong claim with
       beautiful evidence still gets "Not Correct."

    ### Decision procedure (follow for EACH item)
    Step 1 (Understand). Read the item. What is the main point?
    Step 2 (Correctness). Verify against the paper. Is the core claim
           factually accurate and clearly stated?
      - If YES → "Correct". Continue to Step 3.
      - If NO  → "Not Correct". Set significance and evidence to null. Done.
    Step 3 (Significance). Would addressing this criticism genuinely
           improve the paper?
      - Insightful and helpful → "Significant"
      - Not helpful but worth keeping → "Marginally Significant"
      - Should be removed from the review → "Not Significant".
        Set evidence to null. Done.
    Step 4 (Evidence). Does the reviewer provide enough justification?
      - Yes → "Sufficient"
      - No  → "Requires More"

    ### Important notes
    - The paper is the source of truth. If the reviewer claims "the paper
      never mentions X" and the paper does mention X, the reviewer is
      Not Correct.
    - Your judgment must be independent of who wrote the review. Do not
      infer reviewer identity (human/AI) from writing style.
    - The three axes are distinct and orthogonal. A review item can be
      Correct with Requires-More evidence. Strong evidence for a wrong
      claim is still Not Correct.

    ### TODO list for your workflow
    - [ ] Read preprint.md thoroughly. Understand the paper's main
          contribution, methodology, and key claims.
    - [ ] Skim images_list.json and open any figures/tables that the
          reviewers mention by name. Verify visual claims.
    - [ ] Read the code/ directory if any reviewer references specific
          files or functions. Check the README first, then look at the
          referenced files. If the code is executable and not resource-
          prohibitive, try running it to verify code-related claims.
          Do not point out minor code issues (formatting, hard-coded
          paths) unless a reviewer explicitly raised them.
    - [ ] For each reviewer's review file, read every item carefully.
    - [ ] For each item, cross-check the reviewer's claims against the
          paper. Follow the decision procedure: correctness first, then
          significance, then evidence.
    - [ ] (Very important) After judging all items, review your
          judgments for consistency. Are you applying the same bar for
          "Significant" across all reviewers? Iterate if needed.
    - [ ] Write the final prediction JSON to: {output_file}
    - [ ] Verify the output (see CRITICAL section below).

    ### Output format (STRICT)
    Write a single JSON file to {output_file} with this exact shape:

    ```json
    {{
      "paper_id": {paper_id},
      "reviewers": [
        {{
          "reviewer_id": "<reviewer name>",
          "items": [
            {{"item_number": 1, "reasoning": "...", "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient"}},
            {{"item_number": 2, "reasoning": "...", "correctness": "Not Correct", "significance": null, "evidence": null}}
          ]
        }},
        ...
      ]
    }}
    ```

    Rules:
    - You MUST include ALL reviewers and ALL items — no exceptions. Every
      single review item from every reviewer file must have a corresponding
      entry in the output JSON. If a reviewer has 5 items, there must be
      exactly 5 entries for that reviewer.
    - Each item must have: item_number, reasoning, correctness, significance, evidence.
    - "reasoning" should be 2-3 sentences explaining your judgment.
    - Use exact label strings: "Correct"/"Not Correct", "Significant"/
      "Marginally Significant"/"Not Significant", "Sufficient"/"Requires More".
    - null for cascade-skipped fields.

    ### CRITICAL: Verification before finishing
    After writing the JSON file, you MUST perform these checks before
    stopping. Do NOT skip this step.

    1. **Read the file back** and verify it is valid JSON (no syntax errors,
       no trailing commas, no truncated content).
    2. **Count reviewers**: the number of reviewer entries in the JSON must
       match the number of .md files in the reviews/ directory.
    3. **Count items per reviewer**: for each reviewer, the number of item
       entries must match the number of "## Item" sections in that
       reviewer's .md file. If any reviewer is missing items, add them.
    4. **Check label strings**: every correctness value must be exactly
       "Correct" or "Not Correct" (not "correct", not "Yes", not "True").
       Every significance value must be exactly one of the three options
       or null. Every evidence value must be exactly one of the two
       options or null.
    5. If any check fails, fix the file and re-verify.

    Only after all checks pass, print:
    "Verification complete. All [N] reviewers and [M] total items included.
     Prediction written to {output_file}"
    Then stop.

    ### Guidelines for reading the paper
    - The paper's markdown may contain OCR errors. Do not assume the paper
      is incorrect solely because of OCR mistakes. Infer content from
      context. Do not penalize reviewers for pointing out things that are
      actually OCR artifacts.
    - Image links in the paper's markdown may be broken. Figures are
      reliably stored at preprint/images/figure1.png, figure2.png, etc.
      Open images_list.json to see their captions.
    - The code/ directory may include a README. Check it before trying to
      run anything. If the code is non-executable or resource-prohibitive,
      read it rather than running it.
    - Do not try to read every file in code/ — focus on the files that
      reviewers explicitly reference.

    ### Guidelines for meta-reviewing
    - Focus on whether the reviewer's main claim is factually correct
      against the paper, not on the reviewer's formatting or writing style.
    - A reviewer's claim may be correct even if their wording is imprecise.
      Judge the substance, not the presentation.
    - When a reviewer references a figure/table, open the actual image
      file and verify the claim against what you see.
    - When a reviewer references code, open the actual file and verify.
    - If you are uncertain about a correctness judgment, re-read the
      relevant section of the paper before deciding.
    - Apply the same significance bar consistently across all reviewers.
      Do not be lenient on one reviewer and strict on another.
    """)


AGENT_USER_PROMPT_TENCLASS = dedent("""\
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

    ### Reviews to meta-review
    The reconstructed reviews are at: {paper_reviews_dir}
    Each file is one reviewer's full review with all their items.
    The reviewers are: {reviewer_list}

    ### Principles (ordered by importance)
    1. **Correctness first**: Verify every reviewer claim against the paper
       before judging. If unsure, check the paper again.
    2. **Calibrate significance**: "Significant" = would improve the paper.
       Typos = "Marginally Significant" at best.
    3. **Predict disagreement honestly**: Only predict "disagree" when you
       are genuinely uncertain about an axis. Agreement is the default.

    ### Your task
    For EACH review item from EACH reviewer, predict the collapsed 10-class
    label that TWO independent expert meta-reviewers would jointly produce.
    This requires reasoning about both (a) the correct per-axis labels and
    (b) which axes experts would likely disagree on.

    The 10 valid labels are:
    {tenclass_labels_str}

    ### Decision procedure (for EACH item)
    Step 1. Read the item. What is the main point?
    Step 2. Verify correctness against the paper.
    Step 3. Would two experts agree on correctness? If borderline → class 10.
    Step 4. If agreed Correct, decide significance. Would experts agree?
            If borderline → class 8.
    Step 5. If agreed on significance ≥ Marginal, decide evidence. Would
            experts agree? If borderline → class 3 or 6.
    Step 6. If all axes agree, pick classes 1/2/4/5/7/9 based on the labels.

    ### Important notes
    - The paper is the source of truth.
    - Judge independently of who wrote the review (human or AI).
    - Default to "agree" — disagreement is the minority outcome.

    ### TODO list
    - [ ] Read preprint.md thoroughly. Understand the paper's contribution.
    - [ ] Skim images_list.json and open referenced figures/tables.
    - [ ] Read code/ if reviewers reference specific files. Check README
          first. Run if feasible and relevant.
    - [ ] For each reviewer's review file, read every item carefully.
    - [ ] For each item, follow the decision procedure above.
    - [ ] After judging all items, review for consistency across reviewers.
    - [ ] Write the final prediction JSON to: {output_file}
    - [ ] Verify the output (see CRITICAL section below).

    ### Output format (STRICT)
    Write a single JSON file to {output_file} with this exact shape:

    ```json
    {{
      "paper_id": {paper_id},
      "reviewers": [
        {{
          "reviewer_id": "<reviewer name>",
          "items": [
            {{"item_number": 1, "reasoning": "...", "label": "correct_significant_sufficient"}},
            {{"item_number": 2, "reasoning": "...", "label": "incorrect"}}
          ]
        }},
        ...
      ]
    }}
    ```

    Rules:
    - You MUST include ALL reviewers and ALL items — no exceptions. Every
      single review item from every reviewer file must have a corresponding
      entry in the output JSON. If a reviewer has 5 items, there must be
      exactly 5 entries for that reviewer.
    - Each item must have: item_number, reasoning, label.
    - "reasoning" should be 2-3 sentences.
    - "label" must be one of the 10 exact strings listed above — exact match,
      no variants, no capitalization changes.

    ### CRITICAL: Verification before finishing
    After writing the JSON file, you MUST perform these checks before
    stopping. Do NOT skip this step.

    1. **Read the file back** and verify it is valid JSON (no syntax errors,
       no trailing commas, no truncated content).
    2. **Count reviewers**: the number of reviewer entries in the JSON must
       match the number of .md files in the reviews/ directory.
    3. **Count items per reviewer**: for each reviewer, the number of item
       entries must match the number of "## Item" sections in that
       reviewer's .md file. If any reviewer is missing items, add them.
    4. **Check label strings**: every "label" value must be exactly one of
       the 10 valid strings listed above. No variants, no typos.
    5. If any check fails, fix the file and re-verify.

    Only after all checks pass, print:
    "Verification complete. All [N] reviewers and [M] total items included.
     Prediction written to {output_file}"
    Then stop.

    ### Guidelines for reading the paper
    - The paper's markdown may contain OCR errors. Infer content from
      context. Do not penalize reviewers for OCR artifacts.
    - Image links may be broken. Figures are at preprint/images/figure1.png,
      figure2.png, etc. Open images_list.json for captions.
    - Check code/ README before trying to run anything. Focus on files
      reviewers explicitly reference.

    ### Guidelines for meta-reviewing
    - Focus on whether the reviewer's main claim is factually correct
      against the paper, not on writing style.
    - When a reviewer references a figure/table/code file, open it and
      verify.
    - Apply the same significance bar consistently across all reviewers.
    - Default to "agree" on ambiguous axes — agreement is the more common
      outcome in the benchmark. Only predict "disagree" when you are
      genuinely uncertain about the axis.
    - If you are uncertain about a correctness judgment, re-read the
      paper before deciding.
    """)


# ============================================================================
# Data prep: group items by paper for scoring
# ============================================================================

def _load_and_group_items(
    limit: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[int]]:
    """Load expert_annotation rows, dedupe, group by paper_id.
    Returns (items_by_paper, sorted_paper_ids)."""
    from collections import defaultdict

    rows = load_expert_annotation_rows()
    # Dedupe by (paper_id, reviewer_id, review_item_number), keep first
    seen: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    for r in rows:
        key = (int(r['paper_id']), r['reviewer_id'],
               int(r.get('review_item_number') or 0))
        if key not in seen:
            seen[key] = r

    by_paper: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in seen.values():
        if (r.get('review_item') or '').strip():
            by_paper[int(r['paper_id'])].append(r)

    all_pids = sorted(by_paper.keys())
    if limit:
        all_pids = all_pids[:limit]
    return dict(by_paper), all_pids


# ============================================================================
# Agent execution
# ============================================================================

def run_agent_on_paper(
    paper_id: int,
    paper_dir: Path,
    output_dir: Path,
    *,
    model_name: str,
    prompt_mode: str,
    api_key: str,
    base_url: str,
    max_iterations: int,
) -> Dict[str, Any]:
    """Run one OpenHands conversation for an entire paper.
    Returns {predictions: [{reviewer_id, item_number, ...}, ...], status, raw_file_path, raw_file_contents}."""
    from openhands.sdk import LLM, Agent, Conversation, Tool
    from openhands.tools.file_editor import FileEditorTool
    from openhands.tools.task_tracker import TaskTrackerTool
    from openhands.tools.terminal.definition import TerminalTool
    from openhands.sdk.context.condenser import LLMSummarizingCondenser

    preprint_dir = paper_dir / 'preprint'
    reviews_dir = paper_dir / 'reviews'

    if not preprint_dir.exists():
        return {
            'predictions': [],
            'status': 'no_preprint_dir',
            'raw_file_path': None,
            'raw_file_contents': None,
        }

    # List reviewers from the reviews/ directory
    reviewer_files = sorted(reviews_dir.glob('*.md')) if reviews_dir.exists() else []
    reviewer_list = ', '.join(f.stem for f in reviewer_files) if reviewer_files else '(no review files found)'

    # Trajectory and output paths
    model_slug = _slug(model_name)
    trajectory_dir = output_dir / 'agent_trajectories' / prompt_mode / f'paper{paper_id}'
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = trajectory_dir / 'prediction.json'

    # Build the prompt
    if prompt_mode == 'axis':
        system_prompt = AGENT_SYSTEM_PROMPT_AXIS
        user_prompt = AGENT_USER_PROMPT_AXIS.format(
            paper_preprint_dir=str(preprint_dir),
            paper_reviews_dir=str(reviews_dir),
            reviewer_list=reviewer_list,
            output_file=str(prediction_file),
            paper_id=paper_id,
        )
    else:
        tenclass_labels_str = '\n    '.join(
            f'"{lab}"' for lab in TENCLASS_LABELS
        )
        system_prompt = AGENT_SYSTEM_PROMPT_TENCLASS
        user_prompt = AGENT_USER_PROMPT_TENCLASS.format(
            paper_preprint_dir=str(preprint_dir),
            paper_reviews_dir=str(reviews_dir),
            reviewer_list=reviewer_list,
            output_file=str(prediction_file),
            paper_id=paper_id,
            tenclass_labels_str=tenclass_labels_str,
        )

    # OpenHands setup (matching run.py hyperparams)
    # The OpenHands V1 SDK already defaults to reasoning_effort="high"
    # and extended_thinking_budget=200000, with drop_params=True so
    # unsupported params are silently dropped per provider. No need for
    # model-specific reasoning logic here — the SDK handles it.
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

    readable_id = f'{model_slug}_{prompt_mode}_paper{paper_id}'
    conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

    conversation = Conversation(
        agent=agent,
        workspace=str(paper_dir.parent),  # parent of paper{N}/ so agent can cd into it
        persistence_dir=str(trajectory_dir / 'conversation'),
        conversation_id=conv_uuid,
        max_iteration_per_run=max_iterations,
    )

    # Send system prompt as first message, then user prompt
    conversation.send_message(system_prompt + '\n\n---\n\n' + user_prompt)
    conversation.run()

    # Parse the agent's output file
    if not prediction_file.exists():
        return {
            'predictions': [],
            'status': 'no_file',
            'raw_file_path': str(prediction_file),
            'raw_file_contents': None,
        }

    raw_contents = prediction_file.read_text(encoding='utf-8')
    try:
        parsed = json.loads(raw_contents)
    except json.JSONDecodeError:
        return {
            'predictions': [],
            'status': 'malformed_json',
            'raw_file_path': str(prediction_file),
            'raw_file_contents': raw_contents,
        }

    if not isinstance(parsed, dict) or 'reviewers' not in parsed:
        return {
            'predictions': [],
            'status': 'wrong_shape',
            'raw_file_path': str(prediction_file),
            'raw_file_contents': raw_contents,
        }

    # Flatten into per-item prediction dicts
    flat_preds: List[Dict[str, Any]] = []
    for rev_block in parsed.get('reviewers', []):
        rid = rev_block.get('reviewer_id', '')
        for item_block in rev_block.get('items', []):
            pred: Dict[str, Any] = {
                'reviewer_id': rid,
                'item_number': item_block.get('item_number'),
                'reasoning': item_block.get('reasoning', ''),
            }
            if prompt_mode == 'axis':
                pred['correctness'] = item_block.get('correctness')
                pred['significance'] = item_block.get('significance')
                pred['evidence'] = item_block.get('evidence')
            else:
                pred['label'] = item_block.get('label')
                pred['label_id'] = TENCLASS_LABEL_TO_ID.get(
                    item_block.get('label'), None
                )
            flat_preds.append(pred)

    return {
        'predictions': flat_preds,
        'status': 'ok',
        'raw_file_path': str(prediction_file),
        'raw_file_contents': raw_contents,
    }


# ============================================================================
# Scoring bridge: match agent predictions to ground-truth rows
# ============================================================================

def _match_predictions_to_gt(
    agent_preds: List[Dict[str, Any]],
    gt_items: List[Dict[str, Any]],
    prompt_mode: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match agent predictions to ground-truth expert_annotation rows by
    (reviewer_id, item_number). Returns (matched_preds, matched_gt_rows)
    in the same order. Unmatched GT rows get a default prediction."""
    pred_index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for p in agent_preds:
        key = (str(p.get('reviewer_id', '')),
               int(p.get('item_number') or 0))
        pred_index[key] = p

    matched_preds: List[Dict[str, Any]] = []
    matched_gt: List[Dict[str, Any]] = []
    for row in gt_items:
        rid = row['reviewer_id']
        inum = int(row.get('review_item_number') or 0)
        key = (rid, inum)
        pred = pred_index.get(key)
        if pred is None:
            # Agent missed this item — use a default
            if prompt_mode == 'axis':
                pred = {
                    'reasoning': '(agent did not produce prediction for this item)',
                    'correctness': 'Correct',
                    'significance': 'Significant',
                    'evidence': 'Sufficient',
                }
            else:
                pred = {
                    'reasoning': '(agent did not produce prediction for this item)',
                    'label': 'correct_significant_sufficient',
                    'label_id': 1,
                }
        matched_preds.append(pred)

        # Build GT row with _primary/_secondary fields for the scorer
        gt_row = dict(row)
        # Single-annotator: put the label into _primary, leave _secondary None
        if 'correctness_primary' not in gt_row:
            gt_row['correctness_primary'] = row.get('correctness')
            gt_row['correctness_secondary'] = None
            gt_row['significance_primary'] = row.get('significance')
            gt_row['significance_secondary'] = None
            gt_row['evidence_primary'] = row.get('evidence')
            gt_row['evidence_secondary'] = None
        matched_gt.append(gt_row)

    return matched_preds, matched_gt


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Agent-based meta-reviewer (one conversation per paper)'
    )
    parser.add_argument('--model-name', type=str, required=True,
                        help="LiteLLM model id, e.g. 'litellm_proxy/azure_ai/gpt-5.4'")
    parser.add_argument('--prompt-mode', type=str, choices=('axis', 'tenclass'),
                        required=True)
    parser.add_argument('--paper-root', type=str, required=True,
                        help='Root dir with paper{N}/ subdirectories (from download_papers.py).')
    parser.add_argument('--base-url', type=str, default=DEFAULT_BASE_URL)
    parser.add_argument('--api-key', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Default: {DEFAULT_OUTPUT_ROOT}/<model>_agent/')
    parser.add_argument('--limit', type=int, default=None,
                        help='Only run on the first N papers.')
    parser.add_argument('--max-iterations', type=int, default=5000,
                        help='Max OpenHands iterations per paper conversation. '
                             'Default 5000 (matching the reviewer agent).')
    args = parser.parse_args()

    # Resolve API key
    api_key = args.api_key or os.environ.get('LITELLM_API_KEY')
    if not api_key:
        key_file = Path(os.environ.get('LITELLM_KEY_FILE', './key_litellm.txt'))
        if key_file.is_file():
            api_key = key_file.read_text(encoding='utf-8').strip()
    if not api_key:
        # Try the peerreview_bench default location
        fallback = _BENCH_DIR / 'api_key' / 'litellm.txt'
        if fallback.is_file():
            api_key = fallback.read_text(encoding='utf-8').strip()
    if not api_key:
        print('ERROR: no LiteLLM API key.', file=sys.stderr)
        sys.exit(1)

    paper_root = Path(args.paper_root).resolve()
    if not paper_root.is_dir():
        print(f'ERROR: --paper-root {paper_root} does not exist', file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = DEFAULT_OUTPUT_ROOT / f'{_slug(args.model_name)}_agent'
    out.mkdir(parents=True, exist_ok=True)

    # Load items grouped by paper
    print('Loading expert_annotation rows...')
    items_by_paper, paper_ids = _load_and_group_items(limit=args.limit)
    total_items = sum(len(v) for pid, v in items_by_paper.items() if pid in paper_ids)
    print(f'  {len(paper_ids)} papers, {total_items} items')

    all_preds: List[Dict[str, Any]] = []
    all_gt: List[Dict[str, Any]] = []
    t0 = time.time()

    for i, pid in enumerate(paper_ids):
        paper_dir = paper_root / f'paper{pid}'
        gt_items = items_by_paper.get(pid, [])
        n_reviewers = len(set(r['reviewer_id'] for r in gt_items))
        n_items = len(gt_items)

        print(f'\n{"=" * 70}')
        print(f'  [{i+1}/{len(paper_ids)}] paper{pid} '
              f'({n_reviewers} reviewers, {n_items} items) '
              f'[mode={args.prompt_mode}]')
        print(f'{"=" * 70}')

        if not paper_dir.exists():
            print(f'  WARNING: {paper_dir} does not exist, skipping')
            continue

        try:
            envelope = run_agent_on_paper(
                paper_id=pid,
                paper_dir=paper_dir,
                output_dir=out,
                model_name=args.model_name,
                prompt_mode=args.prompt_mode,
                api_key=api_key,
                base_url=args.base_url,
                max_iterations=args.max_iterations,
            )
        except Exception as e:
            print(f'  EXCEPTION: {type(e).__name__}: {e}')
            envelope = {
                'predictions': [],
                'status': f'exception:{type(e).__name__}',
                'raw_file_path': None,
                'raw_file_contents': None,
            }

        elapsed = time.time() - t0
        status = envelope['status']
        n_preds = len(envelope['predictions'])
        print(f'  status: {status}')
        print(f'  predictions: {n_preds} (expected {n_items})')
        print(f'  raw_file: {envelope.get("raw_file_path")}')
        print(f'  elapsed total: {elapsed:.0f}s')

        # Match predictions to GT for scoring
        matched_preds, matched_gt = _match_predictions_to_gt(
            envelope['predictions'], gt_items, args.prompt_mode
        )
        all_preds.extend(matched_preds)
        all_gt.extend(matched_gt)

    # Score
    total_time = time.time() - t0
    print(f'\n{"=" * 70}')
    print(f'Finished {len(paper_ids)} papers in {total_time:.0f}s')
    print(f'Total predictions: {len(all_preds)}')
    print(f'{"=" * 70}')

    if all_preds:
        print('\nScoring...')
        if args.prompt_mode == 'axis':
            metrics = evaluate_axis_predictions(all_preds, all_gt)
        else:
            metrics = evaluate_tenclass_predictions(all_preds, all_gt)
        report = format_mode_report(
            metrics, predictor_name=f'{args.model_name} [agent]'
        )
        print('\n' + report)

        # Write artifacts
        metrics_path = out / f'metrics_{args.prompt_mode}.json'
        report_path = out / f'report_{args.prompt_mode}.txt'
        with metrics_path.open('w') as f:
            json.dump(metrics, f, indent=2, default=str)
        with report_path.open('w') as f:
            f.write(report)
        print(f'\nMetrics: {metrics_path}')
        print(f'Report:  {report_path}')

    print(f'\nAgent trajectories: {out / "agent_trajectories" / args.prompt_mode}/')


if __name__ == '__main__':
    main()
