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

# ---------------------------------------------------------------------------
# Monkey-patch: inject exclude_domains into every Tavily MCP search call.
# ---------------------------------------------------------------------------
_TAVILY_EXCLUDE_DOMAINS = [
    'nature.com', 'researchsquare.com', 'springer.com', 'springerlink.com',
]

def _patch_tavily_exclude_domains():
    try:
        import openhands.mcp.utils as mcp_utils
        _original = mcp_utils.call_tool_mcp

        async def _patched(mcp_clients, action):
            if action.name and 'search' in action.name.lower():
                if isinstance(action.arguments, dict):
                    existing = action.arguments.get('exclude_domains', [])
                    merged = list(set(existing + _TAVILY_EXCLUDE_DOMAINS))
                    action.arguments['exclude_domains'] = merged
            return await _original(mcp_clients, action)

        mcp_utils.call_tool_mcp = _patched
    except (ImportError, AttributeError):
        pass

_patch_tavily_exclude_domains()

from load_data import load_meta_reviewer  # noqa: E402
from prompts import (  # noqa: E402
    _DIMENSION_DEFINITIONS,
    _TENCLASS_TAXONOMY,
    _TENCLASS_DISAGREEMENT_GUIDANCE,
    TENCLASS_LABELS,
    TENCLASS_LABEL_TO_ID,
    extract_axis_answer,
    extract_tenclass_answer,
)
from metrics import (  # noqa: E402
    evaluate_axis_predictions,
    evaluate_tenclass_predictions,
    format_mode_report,
)

DEFAULT_OUTPUT_ROOT = _META_REVIEW_DIR / 'results'  # metareview_bench/results/
DEFAULT_BASE_URL = 'https://cmu.litellm.ai'


def _slug(model: str) -> str:
    """Turn a model id into a filesystem-safe slug."""
    return model.replace('/', '__').replace('.', '_').replace('-', '_')


def _normalize_reviewer_id(rid: str) -> str:
    """Map review-file stems to GT reviewer_ids.

    The GT (meta_reviewer HF config) uses short names: Claude, GPT, Gemini,
    Human_1, Human_2, Human_3. But the review files on disk use full model
    names: claude-opus-4-5.md, gpt-5.2.md, gemini-3.0-pro-preview.md.
    New reviewer files (review_claude-opus-4-7.md, etc.) are NOT in the GT
    and should be left as-is so they don't falsely match.
    """
    if rid.startswith('Human_'):
        return rid
    if rid.lower().startswith('review_'):
        return rid  # new reviewer, not in GT
    low = rid.lower()
    if 'claude' in low:
        return 'Claude'
    if 'gpt' in low:
        return 'GPT'
    if 'gemini' in low:
        return 'Gemini'
    return rid


def _reviewer_type(rid: str) -> str:
    """Return 'Human' or 'AI' based on the (normalized) reviewer_id."""
    if rid.startswith('Human_'):
        return 'Human'
    return 'AI'


def _is_self_review(reviewer_id: str, meta_reviewer_model: str) -> bool:
    """Check if the reviewer is the same model family as the meta-reviewer."""
    meta_low = meta_reviewer_model.lower()
    rid_low = reviewer_id.lower()
    if 'claude' in meta_low and rid_low == 'claude':
        return True
    if 'gpt' in meta_low and rid_low == 'gpt':
        return True
    if 'gemini' in meta_low and rid_low == 'gemini':
        return True
    return False


# ============================================================================
# Agent system prompt
# ============================================================================

AGENT_SYSTEM_PROMPT = dedent("""\
    You are a meta-reviewer agent for scientific papers. You will be given
    a paper's source files on disk AND reconstructed reviews written by
    multiple reviewers (human and AI). For EVERY review item from EVERY
    reviewer, you must:

    1. **Judge the item** along three axes: correctness, significance,
       and evidence sufficiency (your own meta-review judgment).
    2. **Predict how two independent expert meta-reviewers would jointly
       judge the item**, expressed as one of 10 collapsed class labels
       that encode both the cascade outcome and inter-expert agreement.

    You are NOT writing a new review. You are judging existing review
    items by verifying their claims against the paper.
    """) + "\n" + _DIMENSION_DEFINITIONS + "\n" + _TENCLASS_TAXONOMY + "\n" + _TENCLASS_DISAGREEMENT_GUIDANCE


# ============================================================================
# Agent user prompt (per-paper)
# ============================================================================

AGENT_USER_PROMPT = dedent("""\
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
    1. **Correctness — judge the main point, not peripheral details.**
       If the reviewer's core concern is valid even though one specific
       supporting claim is inaccurate, the item is still Correct. Only
       mark Not Correct when the main point itself is wrong. A reviewer
       who says "the paper lacks X" when X is partially present may
       still be Correct if the broader concern (inadequate X) holds.
    2. **Significance — the bar is "would this improve the paper?"**
       Any criticism that would genuinely help the paper if addressed is
       Significant — it does NOT need to threaten the paper's validity.
       Missing statistics, undefined figure annotations, unreported
       methodological details, internal inconsistencies between text and
       figures, and missing ablations are typically Significant because
       fixing them would make the paper substantively better. Do not
       treat "presentation" as synonymous with "minor" — an unclear
       figure or an inconsistency between text and data is a
       presentation issue that is Significant if fixing it would affect
       a reader's ability to evaluate the paper's core claims. Only
       pure stylistic preferences (word choice, tone, formatting that
       does not affect understanding) and trivial typos are Marginally
       Significant.
    3. **Evidence — verifiability, not exhaustiveness.** If a
       meta-reviewer can verify the reviewer's claim from what the
       reviewer wrote plus the paper, the evidence is Sufficient — even
       if it is brief. Do not require the reviewer to be exhaustive.
       A single precise citation or a specific pointer to what is absent
       is enough. When the reviewer's criticism IS that something is
       absent (missing analysis, missing statistic, missing comparison),
       the reviewer's identification of the specific absence IS the
       evidence — a meta-reviewer can verify the absence by checking
       the paper. These items are almost always Sufficient. Reserve
       Requires More for cases where the meta-reviewer cannot even
       locate what part of the paper the reviewer is talking about.

    ### Decision procedure (follow for EACH item)

    **Part A — Your initial judgment (axis labels)**
    Step 1 (Understand). Read the item. What is the main point?
    Step 2 (Correctness). Verify the MAIN POINT against the paper.
      - Is the reviewer's core concern valid, even if a peripheral
        detail is slightly off? → "Correct". Continue to Step 3.
      - Is the main point itself factually wrong? → "Not Correct".
        Set significance and evidence to null.
    Step 3 (Significance). Would addressing this criticism genuinely
           improve the paper?
      - Would fixing it make the paper substantively better?
        → "Significant"
      - Is it a minor presentation or stylistic issue that does not
        affect the paper's substance? → "Marginally Significant"
      - Is it so trivial the review would be better without it?
        → "Not Significant". Set evidence to null.
    Step 4 (Evidence). Can a meta-reviewer verify the claim from the
           review item + the paper without doing substantial extra work?
      - Yes → "Sufficient"
      - No  → "Requires More"

    **Part B — Devil's advocate + expert disagreement prediction**
    Expert disagreement is common in meta-review — it is not a rare
    edge case. For each axis, you must argue the other side before
    deciding whether experts would agree.

    Step 5 (Correctness — devil's advocate).
      - If you said "Not Correct" in Step 2: argue the reviewer's
        defense. Could a charitable domain expert still find the
        reviewer's core concern valid? Maybe the paper partially
        addresses the issue but the broader concern still holds, or
        the criticism depends on scope interpretation. If the defense
        is plausible → predict "disagree_on_correctness" (and you may
        revise your Step 2 label to "Correct" if the defense
        convinces you).
      - If you said "Correct" in Step 2: argue the prosecution.
        Could a skeptical expert find the reviewer's claim factually
        wrong? Maybe the paper already addresses the concern, or the
        premise is based on a misreading. If the prosecution is
        plausible → predict "disagree_on_correctness".
      - If neither side is plausible → experts agree on correctness.

    Step 6 (Significance — devil's advocate). Only if Step 5 = agree
           on Correct.
      - If you said "Significant": argue for downgrading. Could an
        expert see this as merely Marginally Significant? Maybe it's
        a real issue but only affects a secondary aspect of the paper,
        or it's mainly about presentation rather than substance.
      - If you said "Marginally Significant": argue for upgrading.
        Could an expert see this as genuinely improving the paper if
        addressed? Maybe fixing it would affect how readers evaluate
        the core claims.
      - If the counter-argument is plausible → predict
        "correct_disagree_on_significance".
      - If the call is clear-cut → experts agree on significance.

      IMPORTANT: Correctness disagreement means the factual truth of
      the claim is ambiguous. Significance disagreement means the fact
      is clear but its importance is debatable. If you are uncertain
      about whether the reviewer's claim is even true, that is
      correctness uncertainty — do not mislabel it as significance
      uncertainty.

    Step 7 (Evidence boundary). Only if Step 6 = agree on ≥ Marginal.
           Is evidence clearly sufficient or clearly lacking, or is it
           borderline?
      - Firm → experts agree. Pick the matching agreement class.
      - Borderline → "correct_significant_disagree_on_evidence" or
        "correct_marginal_disagree_on_evidence".

    The 10 valid prediction_of_expert_judgments labels are:
    {tenclass_labels_str}

    ### CONSISTENCY CONSTRAINT (critical)
    Your prediction_of_expert_judgments MUST be consistent with your
    FINAL axis labels (after any Part B revisions):
    - If your final correctness = "Correct", the prediction must be
      a "correct_*" class OR "disagree_on_correctness". Never "incorrect".
    - If your final correctness = "Not Correct", the prediction must be
      "incorrect" OR "disagree_on_correctness". Never a "correct_*" class.
    - Similar logic applies downstream: your significance and evidence
      labels constrain which agreement/disagreement classes are valid.

    ### Important notes
    - The paper is the source of truth. If the reviewer claims "the paper
      never mentions X" and the paper does mention X, the reviewer is
      Not Correct — UNLESS the broader concern still stands (e.g., X is
      mentioned but inadequately).
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
    - [ ] For each item, follow Part A then Part B. Your reasoning for
          each item must include: (1) your axis judgment with brief
          justification, (2) for correctness and significance, the
          devil's advocate argument and whether it is plausible, (3) the
          resulting prediction_of_expert_judgments.
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
            {{
              "item_number": 1,
              "reasoning": "...",
              "correctness": "Correct",
              "significance": "Significant",
              "evidence": "Sufficient",
              "prediction_of_expert_judgments": "correct_significant_sufficient"
            }},
            {{
              "item_number": 2,
              "reasoning": "...",
              "correctness": "Correct",
              "significance": "Significant",
              "evidence": "Sufficient",
              "prediction_of_expert_judgments": "correct_disagree_on_significance"
            }}
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
    - Each item must have: item_number, reasoning, correctness,
      significance, evidence, prediction_of_expert_judgments.
    - "reasoning" must include: (1) axis judgment + why, (2) devil's
      advocate argument for correctness (and significance if applicable)
      + whether plausible, (3) expert prediction. Aim for 4-6 sentences.
    - Use exact label strings for axis labels: "Correct"/"Not Correct",
      "Significant"/"Marginally Significant"/"Not Significant",
      "Sufficient"/"Requires More". null for cascade-skipped fields.
    - "prediction_of_expert_judgments" must be one of the 10 exact strings
      listed above.

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
       "Correct" or "Not Correct". Every prediction_of_expert_judgments
       must be one of the 10 valid strings.
    5. **Check consistency**: for each item, verify that the
       prediction_of_expert_judgments is consistent with the axis labels
       (see CONSISTENCY CONSTRAINT above).
    6. If any check fails, fix the file and re-verify.

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
    - A reviewer's claim may be correct even if their wording is imprecise
      or a peripheral detail is wrong. Judge the substance of the main
      point, not peripheral supporting claims.
    - When a reviewer references a figure/table, open the actual image
      file and verify the claim against what you see.
    - When a reviewer references code, open the actual file and verify.
    - If you are uncertain about a correctness judgment, re-read the
      relevant section of the paper before deciding. If it is still
      genuinely borderline after re-reading, that uncertainty is itself
      informative — it suggests experts would likely disagree.
    - Apply the same significance bar consistently across all reviewers.
      Do not be lenient on one reviewer and strict on another.

    ### Filesystem boundaries
    - READ from {paper_preprint_dir} and {paper_reviews_dir}. These are
      the paper's source files. Do not modify anything there.
    - WRITE only to {output_file}. Do not create any other files.
    - Do not navigate to parent or sibling directories of the paper.
    - Do NOT access nature.com, researchsquare.com, springer.com, or
      springerlink.com — these host the published versions of benchmark
      papers and may contain reviewer comments. Judge items solely from
      the paper files provided on disk.
    """)


# ============================================================================
# Data prep: group items by paper for scoring
# ============================================================================

def _load_and_group_items(
    limit: Optional[int] = None,
) -> Tuple[Dict[int, List[Dict[str, Any]]], List[int]]:
    """Load meta_reviewer rows (27 overlap papers, 908 items), group by paper_id.
    Returns (items_by_paper, sorted_paper_ids)."""
    from collections import defaultdict

    rows = load_meta_reviewer()
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

    # Only the 6 GT reviewers should be visible to the agent.
    # Hide everything else (new reviewer runs, verification code,
    # trajectories, JSON items) so the agent can't use them.
    GT_REVIEWER_STEMS = {
        'Human_1', 'Human_2', 'Human_3',
        'claude-opus-4-5', 'gemini-3.0-pro-preview', 'gpt-5.2',
    }
    hide_suffix = '._metareview_hidden'
    hidden_items: List[Tuple[Path, Path]] = []
    if reviews_dir.exists():
        for item in reviews_dir.iterdir():
            if item.name.startswith('.') or hide_suffix in item.name:
                continue
            stem = item.stem if item.is_file() else item.name
            if stem not in GT_REVIEWER_STEMS:
                hidden = item.with_name(item.name + hide_suffix)
                item.rename(hidden)
                hidden_items.append((hidden, item))
        if hidden_items:
            print(f'    (hid {len(hidden_items)} non-GT items in reviews/)')

    # List only the GT reviewer files
    reviewer_files = sorted(
        f for f in reviews_dir.glob('*.md')
        if f.stem in GT_REVIEWER_STEMS
    ) if reviews_dir.exists() else []
    reviewer_list = ', '.join(f.stem for f in reviewer_files) if reviewer_files else '(no review files found)'

    # Output paths — all inside output_dir/{model_slug}/, never inside papers/
    model_slug = _slug(model_name)
    model_out_dir = output_dir / model_slug
    model_out_dir.mkdir(parents=True, exist_ok=True)
    prediction_file = model_out_dir / f'{model_slug}_paper{paper_id}_metareview.json'
    conversation_dir = model_out_dir / 'conversations' / f'paper{paper_id}'

    # Build the prompt
    tenclass_labels_str = '\n    '.join(f'"{lab}"' for lab in TENCLASS_LABELS)
    system_prompt = AGENT_SYSTEM_PROMPT
    user_prompt = AGENT_USER_PROMPT.format(
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
        timeout=600,  # 10 min (default 300s times out on large papers)
        reasoning_effort='high',
        extended_thinking_budget=200000,
        temperature=1.0,  # required by Anthropic extended thinking
        drop_params=True,  # silently drop unsupported params per provider
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

    readable_id = f'{model_slug}_paper{paper_id}'
    conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

    conversation = Conversation(
        agent=agent,
        workspace=str(model_out_dir),  # workspace is the output dir, NOT papers/
        persistence_dir=str(conversation_dir),
        conversation_id=conv_uuid,
        max_iteration_per_run=max_iterations,
    )

    # Send system prompt as first message, then user prompt
    conversation.send_message(system_prompt + '\n\n---\n\n' + user_prompt)
    try:
        conversation.run()
    finally:
        # Always restore hidden files, even if the agent crashes
        for hidden, original in hidden_items:
            if hidden.exists():
                hidden.rename(original)
        if hidden_items:
            print(f'    (restored {len(hidden_items)} hidden items)')

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
        rid = _normalize_reviewer_id(rev_block.get('reviewer_id', ''))
        for item_block in rev_block.get('items', []):
            expert_label = item_block.get('prediction_of_expert_judgments')
            pred: Dict[str, Any] = {
                'reviewer_id': rid,
                'item_number': item_block.get('item_number'),
                'reasoning': item_block.get('reasoning', ''),
                # Axis labels (agent's own judgment)
                'correctness': item_block.get('correctness'),
                'significance': item_block.get('significance'),
                'evidence': item_block.get('evidence'),
                # Tenclass prediction (what two experts would produce)
                'label': expert_label,
                'label_id': TENCLASS_LABEL_TO_ID.get(expert_label),
            }
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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Match agent predictions to ground-truth meta_reviewer rows by
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
            pred = {
                'reasoning': '(agent did not produce prediction for this item)',
                'correctness': 'Correct',
                'significance': 'Significant',
                'evidence': 'Sufficient',
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

    # Restore any files left hidden by a previous interrupted run
    hide_suffix = '._metareview_hidden'
    n_restored = 0
    for hidden in paper_root.rglob(f'*{hide_suffix}'):
        original_name = hidden.name[:hidden.name.index(hide_suffix)]
        hidden.rename(hidden.parent / original_name)
        n_restored += 1
    if n_restored:
        print(f'  (restored {n_restored} files left hidden by a previous interrupted run)')

    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = DEFAULT_OUTPUT_ROOT
    out.mkdir(parents=True, exist_ok=True)

    # Load items grouped by paper
    print('Loading meta_reviewer rows (27 overlap papers)...')
    items_by_paper, paper_ids = _load_and_group_items(limit=args.limit)
    total_items = sum(len(v) for pid, v in items_by_paper.items() if pid in paper_ids)
    print(f'  {len(paper_ids)} papers, {total_items} items')

    all_preds: List[Dict[str, Any]] = []
    all_gt: List[Dict[str, Any]] = []
    t0 = time.time()

    def _restore_all_hidden(paper_root_dir: Path):
        """Emergency restore: unhide ALL files hidden by this run."""
        suffix = '._metareview_hidden'
        count = 0
        for hidden in paper_root_dir.rglob(f'*{suffix}'):
            original_name = hidden.name[:hidden.name.index(suffix)]
            hidden.rename(hidden.parent / original_name)
            count += 1
        return count

    try:
      for i, pid in enumerate(paper_ids):
        paper_dir = paper_root / f'paper{pid}'
        gt_items = items_by_paper.get(pid, [])
        n_reviewers = len(set(r['reviewer_id'] for r in gt_items))
        n_items = len(gt_items)

        print(f'\n{"=" * 70}')
        print(f'  [{i+1}/{len(paper_ids)}] paper{pid} '
              f'({n_reviewers} reviewers, {n_items} items)')
        print(f'{"=" * 70}')

        if not paper_dir.exists():
            print(f'  WARNING: {paper_dir} does not exist, skipping')
            continue

        # Resume: skip papers that already have a prediction file,
        # but still load their predictions for end-of-run scoring.
        model_slug = _slug(args.model_name)
        pred_path = out / model_slug / f'{model_slug}_paper{pid}_metareview.json'
        if pred_path.exists():
            print(f'  skipping (prediction exists: {pred_path.name})')
            # Load existing predictions for scoring
            try:
                parsed = json.loads(pred_path.read_text(encoding='utf-8'))
                existing_preds = []
                for rev_block in parsed.get('reviewers', []):
                    rid = _normalize_reviewer_id(rev_block.get('reviewer_id', ''))
                    for item_block in rev_block.get('items', []):
                        expert_label = item_block.get('prediction_of_expert_judgments')
                        existing_preds.append({
                            'reviewer_id': rid,
                            'item_number': item_block.get('item_number'),
                            'reasoning': item_block.get('reasoning', ''),
                            'correctness': item_block.get('correctness'),
                            'significance': item_block.get('significance'),
                            'evidence': item_block.get('evidence'),
                            'label': expert_label,
                            'label_id': TENCLASS_LABEL_TO_ID.get(expert_label),
                        })
                matched_preds, matched_gt = _match_predictions_to_gt(
                    existing_preds, gt_items
                )
                all_preds.extend(matched_preds)
                all_gt.extend(matched_gt)
            except Exception:
                pass  # skip scoring for this paper if file is malformed
            continue

        max_retries = 3
        for attempt in range(1, max_retries + 1):
            # Clean up stale conversation from a previous failed attempt
            # so OpenHands starts fresh instead of resuming a broken state
            conv_dir = out / model_slug / 'conversations' / f'paper{pid}'
            if conv_dir.exists():
                import shutil
                shutil.rmtree(conv_dir)
                if attempt == 1:
                    print(f'  cleaned up stale conversation from previous run')

            attempt_str = f' (attempt {attempt}/{max_retries})' if attempt > 1 else ''
            print(f'  running{attempt_str}...', flush=True)

            try:
                envelope = run_agent_on_paper(
                    paper_id=pid,
                    paper_dir=paper_dir,
                    output_dir=out,
                    model_name=args.model_name,
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

            # Check if prediction file was written
            if pred_path.exists():
                break  # success
            else:
                print(f'  no prediction file produced (status: {envelope["status"]})')
                if attempt == max_retries:
                    print(f'  ⚠ paper{pid}: all {max_retries} attempts failed')

        elapsed = time.time() - t0
        status = envelope['status']
        n_preds = len(envelope['predictions'])
        print(f'  status: {status}')
        print(f'  predictions: {n_preds} (expected {n_items})')
        print(f'  raw_file: {envelope.get("raw_file_path")}')
        print(f'  elapsed total: {elapsed:.0f}s')

        # Match predictions to GT for scoring
        matched_preds, matched_gt = _match_predictions_to_gt(
            envelope['predictions'], gt_items
        )
        all_preds.extend(matched_preds)
        all_gt.extend(matched_gt)
    except KeyboardInterrupt:
        print(f'\n\nInterrupted! Restoring hidden files...')
        n = _restore_all_hidden(paper_root)
        print(f'Restored {n} hidden files. Safe to restart.')
        sys.exit(1)

    # Score
    total_time = time.time() - t0
    print(f'\n{"=" * 70}')
    print(f'Finished {len(paper_ids)} papers in {total_time:.0f}s')
    print(f'Total predictions: {len(all_preds)}')
    print(f'{"=" * 70}')

    if all_preds:
        model_slug = _slug(args.model_name)
        model_out = out / model_slug

        # Build per-type subsets: overall, human, ai, self
        subsets: List[Tuple[str, List[Dict], List[Dict]]] = [
            ('overall', all_preds, all_gt),
        ]
        human_preds, human_gt = [], []
        ai_preds, ai_gt = [], []
        self_preds, self_gt = [], []
        for pred, gt_row in zip(all_preds, all_gt):
            rid = pred.get('reviewer_id', '')
            rtype = _reviewer_type(rid)
            if rtype == 'Human':
                human_preds.append(pred)
                human_gt.append(gt_row)
            else:
                ai_preds.append(pred)
                ai_gt.append(gt_row)
            if _is_self_review(rid, args.model_name):
                self_preds.append(pred)
                self_gt.append(gt_row)
        if human_preds:
            subsets.append(('human_reviewers', human_preds, human_gt))
        if ai_preds:
            subsets.append(('ai_reviewers', ai_preds, ai_gt))
        if self_preds:
            subsets.append(('self_review', self_preds, self_gt))

        all_metrics: Dict[str, Any] = {}
        full_report_parts: List[str] = []

        for subset_name, preds_sub, gt_sub in subsets:
            print(f'\n{"=" * 70}')
            print(f'  Scoring: {subset_name} ({len(preds_sub)} items)')
            print(f'{"=" * 70}')

            axis_m = evaluate_axis_predictions(preds_sub, gt_sub)
            axis_r = format_mode_report(
                axis_m, predictor_name=f'{args.model_name} [{subset_name}, axis]'
            )
            print('\n' + axis_r)

            tenclass_m = evaluate_tenclass_predictions(preds_sub, gt_sub)
            tenclass_r = format_mode_report(
                tenclass_m, predictor_name=f'{args.model_name} [{subset_name}, tenclass]'
            )
            print('\n' + tenclass_r)

            all_metrics[f'{subset_name}_axis'] = axis_m
            all_metrics[f'{subset_name}_tenclass'] = tenclass_m
            full_report_parts.append(axis_r)
            full_report_parts.append(tenclass_r)

        # Write artifacts
        metrics_path = model_out / 'metrics.json'
        report_path = model_out / 'report.txt'
        with metrics_path.open('w') as f:
            json.dump(all_metrics, f, indent=2, default=str)
        with report_path.open('w') as f:
            f.write('\n\n'.join(full_report_parts))
        print(f'\nMetrics: {metrics_path}')
        print(f'Report:  {report_path}')

    print(f'\nResults: {out}/')


if __name__ == '__main__':
    main()
