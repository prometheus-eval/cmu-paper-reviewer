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
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_BENCH_DIR = _HERE.parent
_BACKEND_DIR = _BENCH_DIR.parent / 'backend'


for p in (_HERE, _BENCH_DIR, _BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# ---------------------------------------------------------------------------
# Monkey-patch OpenHands MCP tool executor to inject exclude_domains into
# every Tavily search call. This is a benchmark-level safeguard: the papers
# are sourced from Nature/ResearchSquare, so accessing those sites during
# review would leak published versions and reviewer comments.
# ---------------------------------------------------------------------------
_TAVILY_EXCLUDE_DOMAINS = [
    'nature.com', 'researchsquare.com', 'springer.com', 'springerlink.com',
]

def _patch_tavily_exclude_domains():
    """Inject exclude_domains into every Tavily MCP tool call."""
    try:
        import openhands.mcp.utils as mcp_utils
        _original_call_tool_mcp = mcp_utils.call_tool_mcp

        async def _patched_call_tool_mcp(mcp_clients, action):
            # If this is a tavily search call, inject exclude_domains
            if action.name and 'search' in action.name.lower():
                if isinstance(action.arguments, dict):
                    existing = action.arguments.get('exclude_domains', [])
                    merged = list(set(existing + _TAVILY_EXCLUDE_DOMAINS))
                    action.arguments['exclude_domains'] = merged
            return await _original_call_tool_mcp(mcp_clients, action)

        mcp_utils.call_tool_mcp = _patched_call_tool_mcp
    except (ImportError, AttributeError):
        pass  # OpenHands not installed or API changed — fall back to prompt

_patch_tavily_exclude_domains()

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

    paper_dir = paper_dir.resolve()  # ensure absolute paths in prompts
    preprint_dir = paper_dir / 'preprint'
    reviews_dir = paper_dir / 'reviews'
    reviews_dir.mkdir(parents=True, exist_ok=True)

    model_short = model_name.split('/')[-1]
    review_path = reviews_dir / f'review_{model_short}.md'

    link_to_paper = str(preprint_dir)

    # Build prompt
    prompt = build_reviewer_prompt(review_settings)
    prompt = prompt.replace('[LINK TO THE PAPER]', link_to_paper)
    prompt = prompt.replace('[MODEL NAME]', model_short)
    # The backend prompt writes to "/../review/" but we use "/../reviews/"
    prompt = prompt.replace('/../review/', '/../reviews/')
    # Resolve /../ in paths so the agent sees clean absolute paths
    # e.g., ".../preprint/../reviews/" → ".../reviews/"
    prompt = prompt.replace(f'{link_to_paper}/../reviews/', str(reviews_dir) + '/')
    prompt = prompt.replace(f'{link_to_paper}/../', str(paper_dir) + '/')

    # Add anti-peeking + filesystem boundary instructions (defense-in-depth)
    verification_dir = reviews_dir / f'verification_code_{model_short}'
    prompt += (
        "\n\n### CRITICAL: Do NOT read other reviews\n"
        f"The directory {reviews_dir} may contain review files written by "
        "other reviewers (human or AI). You MUST NOT open, read, or reference "
        "any of these files. Your review must be entirely independent. Only "
        f"read files under {link_to_paper} (the paper's preprint directory) "
        f"and write your output to {review_path}.\n"
        "\n### Filesystem boundaries (STRICT — violations will corrupt the benchmark)\n"
        "You may ONLY create or write files in these two locations:\n"
        f"  1. {review_path}\n"
        f"  2. {verification_dir}/  (for all verification code, scratch scripts, and temporary files)\n"
        "Before creating ANY file, verify its path starts with one of these.\n"
        "Do NOT create files anywhere else — no temp files, no chunk files,\n"
        "no scratch scripts outside the verification_code directory.\n"
        f"Do NOT read or navigate outside {link_to_paper}.\n"
        "IMPORTANT: Always use absolute paths (starting with /) when reading\n"
        f"or writing files. The paper is at {link_to_paper}.\n"
        "\n### File writing instructions\n"
        "When creating or writing files, use the file_editor tool with:\n"
        '  command: "create", path: "<absolute path>", file_text: "<content>"\n'
        "Do NOT pass view_range when creating files. Do NOT use the terminal\n"
        "to write files (e.g., cat, echo, heredoc) — always use file_editor.\n"
        "NEVER use relative paths (../ or ./) — they will fail. Examples of\n"
        "correct absolute paths for this paper:\n"
        f"  Read paper:   {link_to_paper}/preprint.md\n"
        f"  Read images:  {link_to_paper}/images_list.json\n"
        f"  Read code:    {link_to_paper}/code/\n"
        f"  Write review: {review_path}\n"
        f"  Write code:   {verification_dir}/<filename>\n"
        "\n### Restricted domains (STRICT — accessing these will contaminate the benchmark)\n"
        "When searching for literature, you MUST NOT access or retrieve content\n"
        "from these domains:\n"
        "  - nature.com (including nature.com/articles/*)\n"
        "  - researchsquare.com\n"
        "  - springer.com\n"
        "  - springerlink.com\n"
        "These host the published versions of benchmark papers and may contain\n"
        "reviewer comments or post-publication revisions. Accessing them would\n"
        "compromise the independence of your review. Use other sources (arxiv,\n"
        "semantic scholar, google scholar, etc.) for literature search.\n"
        "IMPORTANT: When calling the Tavily search tool, ALWAYS include the\n"
        'exclude_domains parameter: ["nature.com", "researchsquare.com", '
        '"springer.com", "springerlink.com"]\n'
    )

    # Filesystem-level protection: temporarily hide ALL other review artifacts
    # (files AND directories) so the agent can't read other models' reviews,
    # verification code, or trajectories. Only the current model's own output
    # files are left visible.
    hide_suffix = f'._hidden_{model_short}'
    hidden_files: List[Tuple[Path, Path]] = []
    own_names = {
        review_path.name,                         # review_grok-4-1-fast-reasoning.md
        f'verification_code_{model_short}',        # verification_code_grok-4-1-fast-reasoning/
        f'{model_short}_trajectory',               # grok-4-1-fast-reasoning_trajectory/
        f'review_items_{model_short}.json',         # review_items_grok-4-1-fast-reasoning.json
    }
    for item in reviews_dir.iterdir():
        if item.name.startswith('.') or '._hidden' in item.name:
            continue
        if item.name in own_names:
            continue
        hidden = item.with_name(item.name + hide_suffix)
        item.rename(hidden)
        hidden_files.append((hidden, item))

    # OpenHands setup (matching backend/review_service.py)
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

    readable_id = f'{model_name.replace("/", "_")}_{paper_dir.name}'.replace('.', '_').replace('-', '_')
    conv_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, readable_id)

    conversation = Conversation(
        agent=agent,
        workspace=str(paper_dir),  # scope agent to this paper's directory only
        persistence_dir=str(reviews_dir / f'{model_short}_trajectory'),
        conversation_id=conv_uuid,
        max_iteration_per_run=max_iterations,
    )

    conversation.send_message(prompt)
    try:
        conversation.run()
    except Exception as e:
        print(f'\n    Agent error: {type(e).__name__}: {str(e)[:200]}')
        return None
    finally:
        # Always restore hidden files, even if the agent crashes
        for hidden, original in hidden_files:
            if hidden.exists():
                hidden.rename(original)

    try:
        cost = conversation.conversation_stats.get_combined_metrics().accumulated_cost
    except Exception:
        cost = 0
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
    base_url: Optional[str] = None,
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

    # Resolve base URL
    if not base_url:
        base_url = os.environ.get('LITELLM_BASE_URL')
    if not base_url:
        url_file = _BENCH_DIR / 'api_key' / 'base_url.txt'
        if url_file.is_file():
            base_url = url_file.read_text(encoding='utf-8').strip()
    if not base_url:
        base_url = 'https://cmu.litellm.ai'
    base_url = base_url.rstrip('/')

    # Build review settings
    review_settings = get_default_settings()
    review_settings['max_items'] = max_items
    review_settings['reviewer_criteria_preset'] = criteria_preset

    # Find papers, excluding those with empty rubrics (no fully-good human
    # items → no recall ground truth → not worth generating a review for).
    from build_rubric import build_rubric  # noqa: E402
    rubric, dropped = build_rubric()
    dropped_set = set(dropped)

    paper_dirs = [
        pd for pd in paper_root.glob('paper*')
        if _extract_paper_id(pd) > 0 and _extract_paper_id(pd) not in dropped_set
    ]
    paper_dirs.sort(key=lambda pd: _extract_paper_id(pd))  # numeric sort, not lexicographic
    if limit:
        paper_dirs = paper_dirs[:limit]

    model_short = model_name.split('/')[-1]

    print(f'Generating reviews for {len(paper_dirs)} papers')
    print(f'  (skipped {len(dropped_set)} papers with empty rubric: {sorted(dropped_set)})')
    print(f'  model: {model_name}')
    print(f'  max_items: {max_items}')
    print(f'  criteria: {criteria_preset}')

    max_retries = 3

    def _restore_all_hidden(model_slug: str):
        """Emergency restore: unhide ALL files hidden by this model across all papers."""
        suffix = f'._hidden_{model_slug}'
        count = 0
        for hidden in paper_root.rglob(f'*{suffix}'):
            original_name = hidden.name[:hidden.name.index(suffix)]
            hidden.rename(hidden.parent / original_name)
            count += 1
        return count

    # Restore any files left hidden by a previous interrupted run
    n_restored = _restore_all_hidden(model_short)
    if n_restored:
        print(f'  (restored {n_restored} files left hidden by a previous interrupted run)')

    def _review_done(review_path: Path) -> bool:
        """Check if a valid review exists, even if hidden by another model's
        concurrent run."""
        # Direct check
        if review_path.exists() and _validate_review(review_path):
            return True
        # Check if hidden by another model (e.g., ._hidden_claude-opus-4-7)
        reviews_dir = review_path.parent
        for f in reviews_dir.glob(f'{review_path.name}._hidden_*'):
            # Temporarily read the hidden file to validate
            try:
                content = f.read_text(encoding='utf-8')
                import re as _re
                if _re.search(r'^##\s*Item\s+\d+', content, _re.MULTILINE | _re.IGNORECASE):
                    return True
            except Exception:
                pass
        return False

    try:
        for i, pd in enumerate(paper_dirs):
            review_path = pd / 'reviews' / f'review_{model_short}.md'
            if skip_existing and _review_done(review_path):
                print(f'  [{i+1}/{len(paper_dirs)}] {pd.name}: skipping (review exists)')
                continue

            # Retry loop: up to max_retries attempts per paper
            for attempt in range(1, max_retries + 1):
                # Clean up stale trajectory from previous failed attempts
                trajectory_dir = pd / 'reviews' / f'{model_short}_trajectory'
                if trajectory_dir.exists():
                    import shutil
                    shutil.rmtree(trajectory_dir)

                # Also remove any partial/invalid review file
                if review_path.exists() and not _validate_review(review_path):
                    review_path.unlink()

                attempt_str = f' (attempt {attempt}/{max_retries})' if attempt > 1 else ''
                print(f'  [{i+1}/{len(paper_dirs)}] {pd.name}: generating{attempt_str}...',
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
                    items = parse_review_file(result, save=True)
                    print(f' done ({elapsed:.0f}s, {len(items)} items)')
                    break  # success — move to next paper
                else:
                    print(f' FAILED ({elapsed:.0f}s)')
                    if attempt == max_retries:
                        print(f'  ⚠ {pd.name}: all {max_retries} attempts failed')
    except KeyboardInterrupt:
        print(f'\n\nInterrupted! Restoring hidden files...')
        _restore_all_hidden(model_short)
        print('Hidden files restored. Safe to restart.')
        sys.exit(1)


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
    parser.add_argument('--base-url', type=str, default=None)
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
